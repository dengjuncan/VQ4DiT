# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from coding import decode, get_num_centroids
from kmeans import kmeans
from AbstractCompressedLayer import AbstractCompressedLayer
import os
import sys
import global_var
from gaussian_diffusion import mean_flat
from diffusion_utils import custom_loss_function


def custom_loss_function(softmax_I):
    loss = softmax_I * (1 - softmax_I)
    loss = torch.mean(loss)
    return loss


class CompressedLinear(AbstractCompressedLayer):
    """Compressed representation of a linear layer"""

    def __init__(self, codes_matrix: torch.Tensor, codebook: torch.Tensor, weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None):
        super(CompressedLinear, self).__init__()

        self.initialize_codes(codes_matrix, codebook)

        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.bias = None

        self.codebook = nn.Parameter(codebook, requires_grad=False)

        self.use_sim = False

        self.loss_norm = torch.tensor(0., dtype=torch.float32, device=self.codebook.device)
        self.loss_soft = torch.tensor(0., dtype=torch.float32, device=self.codebook.device)
        
        self.weight_o = weight

    @torch.no_grad()
    def prepare_sim(self):
        if self.use_sim:
            return
        
        self.num_output_rows = self.codes_matrix.size(0)
        
        self.top_indices_num = 2
        distances = torch.sqrt(((self.weight_o.unsqueeze(1) - self.codebook.unsqueeze(0)) ** 2).sum(-1))
        codebook_distance, codebook_similarity = torch.topk(distances, self.top_indices_num, dim=1, largest=False)


        codes_matrix_train = torch.zeros(self.codes_matrix.shape[0]*self.codes_matrix.shape[1], codebook_similarity.shape[1], device=self.codes_matrix.device, dtype=torch.float32)
        for i in range(self.top_indices_num - 1):
            codes_matrix_train[torch.arange(len(codes_matrix_train)), i] += torch.log((codebook_distance[:,-1] / (codebook_distance[:,i] + 1e-5))**1)
        
        self.codes_matrix_train = nn.Parameter(codes_matrix_train, requires_grad=True)
        self.mask = nn.Parameter(torch.zeros_like(self.codes_matrix_train[:,0], dtype=torch.bool), requires_grad=False)
        
        self.codes_matrix_similarity = codebook_similarity

        self.use_sim = True

        del self.weight_o   
        del self.codes_matrix



    @torch.no_grad()
    def progressive_freeze(self):
        if not self.use_sim:
            return

        softmax_matrix = F.softmax(self.codes_matrix_train, dim=-1)
        max_vals, max_indices = torch.max(softmax_matrix, dim=1)
        self.mask.data = max_vals > 0.99
        self.codes_matrix_train.data[self.mask] = torch.nn.functional.one_hot(max_indices[self.mask], num_classes=self.top_indices_num).float() * 10000.


        return


    @torch.no_grad()
    def freeze_sim(self):
        if not self.use_sim:
            return
        softmax_matrix = F.softmax(self.codes_matrix_train, dim=-1)
        max_indices = torch.argmax(softmax_matrix, dim=1, keepdim=True)
        index_hard = torch.zeros_like(self.codes_matrix_train)
        index_hard.scatter_(1, max_indices, 1)
        self.codes_matrix = self.codes_matrix_similarity[index_hard.bool()]
        self.codes_matrix = self.codes_matrix.view(self.num_output_rows, -1)
        self.initialize_codes(self.codes_matrix, self.codebook)
        self.use_sim = False
        del self.codes_matrix_train

    
    @torch.no_grad()
    def prepare_repeat(self):
        codebook_clone = self.codebook.detach().clone()
        self.codebook_repeat = nn.ParameterList([nn.Parameter(codebook_clone, requires_grad=True) for i in range(10)])
        self.use_repeat = True


    def _get_uncompressed_weight_soft(self):
        softmax_matrix = F.softmax(self.codes_matrix_train, dim=-1)

        self.loss_soft = self.top_indices_num * custom_loss_function(softmax_matrix[~self.mask, :]) if len(softmax_matrix[~self.mask, :]) > 0 else torch.tensor(0., dtype=torch.float32, device=softmax_matrix.device)
        
        softmax_matrix = softmax_matrix.unsqueeze(1)
        M_hat_unrolled_soft = torch.bmm(softmax_matrix, self.codebook[self.codes_matrix_similarity, :]).squeeze(1)

        self.weight_soft_cur = M_hat_unrolled_soft.detach()

        decoded_weights = M_hat_unrolled_soft.reshape(self.num_output_rows, -1)


        return decoded_weights
    
    def _get_uncompressed_weight(self):
        return decode(self.codes_matrix, self.codebook).float()

    @property
    def weight(self):
        return self._get_uncompressed_weight_soft() if self.use_sim else self._get_uncompressed_weight()

    def forward(self, x):
        return F.linear(input=x, weight=self._get_uncompressed_weight_soft() if self.use_sim else self._get_uncompressed_weight(), bias=self.bias)


    @staticmethod
    def from_uncompressed(
        uncompressed_layer: torch.nn.Linear,
        k: int,
        d: int,
        name: str = "",
        cb_dir: str = None,
    ) -> "CompressedLinear":
        """Given an uncompressed layer, initialize the compressed equivalent according to the specified parameters

        Parameters:
            uncompressed_layer: Linear layer to compress
            k: Size of the codebook
            k_means_n_iters: Number of iterations of k means
            subvector_size: Subvector size for the layer
            name : Name of the layer to print alongside mean-squared error
            cb_dir: Directory to save/load codebooks
        Returns:
            compressed_layer: Initialized compressed layer
        """

        kmeans_fn = kmeans

        weight = uncompressed_layer.weight.detach()

        c_out, c_in = weight.size()

        num_blocks_per_row = c_in // d

        training_set = weight.reshape(-1, d)
        num_centroids = get_num_centroids(num_blocks_per_row, c_out, k)
        log_err = False

        if cb_dir is None:
            cb_dir = 'pre_codebook_'+str(k)+'_'+str(d)

        if os.path.exists(cb_dir+'/'+name+'.pth'):
            codebook = torch.load(cb_dir+'/'+name+'.pth')
            codebook, codes = kmeans_fn(training_set, k=num_centroids, n_iters=100, slow_cb_update=False, codebook=codebook)
        else:
            os.makedirs(cb_dir, exist_ok=True)
            log_err = True
            try:
                codebook, codes = kmeans_fn(training_set, k=num_centroids, n_iters=100, slow_cb_update=False)
            except:
                codebook, codes = kmeans_fn(training_set, k=num_centroids, n_iters=100, slow_cb_update=True)
            torch.save(codebook, cb_dir+'/'+name+'.pth')
        
        codes_matrix = codes.reshape(-1, num_blocks_per_row)


        if log_err:
            decoded_weights = decode(codes_matrix, codebook)
            error = (decoded_weights - weight).pow(2).sum() / (num_blocks_per_row * weight.size(0))
            AbstractCompressedLayer.log_quantization_error(name, 100, error, codebook, codes_matrix)

        return CompressedLinear(codes_matrix, codebook, training_set, uncompressed_layer.bias)
    

