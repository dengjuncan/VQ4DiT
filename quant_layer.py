import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
import copy

import global_var


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()

class UniformAffineQuantizer_fp16(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, low_mem: bool = False):
        super(UniformAffineQuantizer_fp16, self).__init__()
        self.sym = symmetric
        # assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method

    def forward(self, x: torch.Tensor):
        return x
        # if self.n_bits == 16:
        #     # return x
        #     return x.to(torch.float16).to(torch.float32)
        # else:
        #     raise NotImplementedError


class UniformAffineQuantizer_weight(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False):
        super(UniformAffineQuantizer_weight, self).__init__()
        self.sym = symmetric
        # assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            if self.leaf_param:
                delta, zero_point = self.init_quantization_scale(x, self.channel_wise)
                self.delta = torch.nn.Parameter(delta, requires_grad=True)
                self.zero_point = torch.nn.Parameter(zero_point, requires_grad=False)
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        x_int = round_ste(x / self.delta) + self.zero_point
        if self.sym:
            x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        else:
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            delta, zero_point = self.init_quantization_scale(x_clone, channel_wise=False)

            # n_channels = x_clone.shape[0]
            # if len(x.shape) == 4:
            #     x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            # else:
            #     x_max = x_clone.abs().max(dim=-1)[0]
            # delta = x_max.clone()
            # zero_point = x_max.clone()

            # # determine the scale and zero point channel-by-channel
            # for c in range(n_channels):
            #     delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)

        else:
            if 'max' in self.scale_method:
                
                x = x.view(x.size(0), -1)

                # x_min = torch.min(x)
                # x_min = torch.min(x_min, torch.zeros_like(x_min))
                # x_max = torch.max(x)
                # x_max = torch.max(x_max, torch.zeros_like(x_max))

                x_min = torch.min(x, dim=1, keepdim=True)[0]
                x_min = torch.min(x_min, torch.zeros_like(x_min))
                x_max = torch.max(x, dim=1, keepdim=True)[0]
                x_max = torch.max(x_max, torch.zeros_like(x_max))

                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                # x_absmax = max(abs(x_min), x_max)
                x_absmax = torch.max(x_min.abs(), x_max)

                if self.sym:
                    delta = x_absmax / self.n_levels
                else:
                    # delta = float(x_max - x_min) / (self.n_levels - 1)
                    delta = (x_max - x_min) / (self.n_levels - 1)
                

                # if delta < 1e-8:
                #     warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                #     delta = torch.Tensor(1e-8).type_as(x)

                # zero_point = torch.tensor(round(-x_min / delta)).type_as(x)
                # delta = torch.tensor(delta).type_as(x)
                zero_point = torch.round(-x_min / delta).type_as(x) if not self.sym else torch.tensor(0.).type_as(x)
                delta = delta.type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(20):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round()
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        # assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)


class UniformAffineQuantizer_weight_groupwise(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = True, scale_method: str = 'max',
                 leaf_param: bool = True):
        super(UniformAffineQuantizer_weight_groupwise, self).__init__()
        self.sym = symmetric
        # assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        self.group_size = 64
        self.use_group_quant = True

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            if self.leaf_param:
                x_tmp = x.clone().detach()
                shape_0 = x_tmp.shape[0]
                x_tmp = x_tmp.view(-1, self.group_size)
                if x_tmp.shape[0] >= shape_0:
                    delta, self.zero_point = self.init_quantization_scale(x, False, True)
                else:
                    delta, self.zero_point = self.init_quantization_scale(x, True, False)
                    self.use_group_quant = False
                self.delta = torch.nn.Parameter(delta)
                # self.zero_point = torch.nn.Parameter(self.zero_point)
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        # start quantization
        if self.use_group_quant:
            if len(x.shape) == 4:
                x = x.permute(0,2,3,1)
            shape_org = x.shape
            x = x.reshape(-1, self.group_size)
            # print(shape_org, x.shape, self.delta.shape)

        x_int = round_ste(x / self.delta) + self.zero_point
        if self.sym:
            x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        else:
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        if self.use_group_quant:
            x_dequant = x_dequant.view(shape_org)
            if len(shape_org) == 4:
                x_dequant = x_dequant.permute(0,3,1,2)
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False, group_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()

            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        elif group_wise:
            x_clone = x.clone().detach()
            
            if len(x.shape) == 4:
                x_clone = x_clone.permute(0,2,3,1)
            x_clone_group = x_clone.reshape(-1, self.group_size)


            n_elements = x_clone_group.numel()  # 获取总元素数
            n_groups = n_elements // self.group_size  # 计算总共有多少组128元素
            if n_elements % self.group_size != 0:
                n_groups += 1 


            x_min = torch.min(x_clone_group, dim=1, keepdim=True)[0]
            x_min = torch.min(x_min, torch.zeros_like(x_min))
            x_max = torch.max(x_clone_group, dim=1, keepdim=True)[0]
            x_max = torch.max(x_max, torch.zeros_like(x_max))
            
            
            if 'scale' in self.scale_method:
                x_min = x_min * (self.n_bits + 2) / 8
                x_max = x_max * (self.n_bits + 2) / 8

            # x_absmax = max(abs(x_min), x_max)
            x_absmax = torch.max(x_min.abs(), x_max)

            if self.sym:
                delta = x_absmax / self.n_levels
            else:
                # delta = float(x_max - x_min) / (self.n_levels - 1)
                delta = (x_max - x_min) / (self.n_levels - 1)
               
            zero_point = torch.round(-x_min / delta).type_as(x) if not self.sym else torch.zeros_like(delta).type_as(x)
            delta = delta.type_as(x)

        else:
            if 'max' in self.scale_method:
                # x_min = min(x.min().item(), 0)
                # x_max = max(x.max().item(), 0)
                x_min = torch.min(x)
                x_min = torch.min(x_min, torch.zeros_like(x_min))
                x_max = torch.max(x)
                x_max = torch.max(x_max, torch.zeros_like(x_max))
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                # x_absmax = max(abs(x_min), x_max)
                x_absmax = torch.max(x_min.abs(), x_max)

                if self.sym:
                    delta = x_absmax / self.n_levels
                else:
                    # delta = float(x_max - x_min) / (self.n_levels - 1)
                    delta = (x_max - x_min) / (self.n_levels - 1)

                # if delta < 1e-8:
                #     warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                #     delta = torch.Tensor(1e-8).type_as(x)

                # zero_point = torch.tensor(round(-x_min / delta)).type_as(x)
                # delta = torch.tensor(delta).type_as(x)
                zero_point = torch.round(-x_min / delta).type_as(x) if not self.sym else torch.tensor(0.).type_as(x)
                delta = delta.type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(20):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round()
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        # assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)


class UniformAffineQuantizer_weight_bg(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False):
        super(UniformAffineQuantizer_weight_bg, self).__init__()
        self.sym = symmetric
        # assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        self.x_max = None
        self.x_min = None
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            
            x_max, x_min = self.init_quantization_scale(x, self.channel_wise)
            if len(x.shape) == 4:
                x_max = x_max.view(-1, 1, 1, 1)
                x_min = x_min.view(-1, 1, 1, 1)
            else:
                x_max = x_max.view(-1, 1)
                x_min = x_min.view(-1, 1)

            self.x_max = torch.nn.Parameter(x_max, requires_grad=True)
            self.x_min = torch.nn.Parameter(x_min, requires_grad=True)

            # self.gamma = torch.nn.Parameter(torch.ones_like(self.x_max), requires_grad=True)
            # self.beta = torch.nn.Parameter(torch.ones_like(self.x_min), requires_grad=True)
            self.gamma = torch.ones_like(self.x_max)
            self.beta = torch.ones_like(self.x_min)

            if self.sym:
                self.zero_point = torch.zeros_like(self.x_max)

            self.inited = True


        if self.sym:
            x_absmax = torch.max(self.x_min.abs(), self.x_max)
            delta = (self.gamma*x_absmax) / self.n_levels
        else:
            delta = (self.gamma*self.x_max - self.beta*self.x_min) / (self.n_levels - 1)

        zero_point = round_ste((-self.x_min*self.beta) / delta) if not self.sym else self.zero_point

        x_int = round_ste(x / delta) + zero_point
        if self.sym:
            x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        else:
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - zero_point) * delta
        
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            delta, zero_point = self.init_quantization_scale(x_clone, channel_wise=False)

        else:
            if 'max' in self.scale_method:
                
                x = x.view(x.size(0), -1)

                x_min = torch.min(x, dim=1, keepdim=True)[0]
                x_min = torch.min(x_min, torch.zeros_like(x_min))
                x_max = torch.max(x, dim=1, keepdim=True)[0]
                x_max = torch.max(x_max, torch.zeros_like(x_max))

                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8


                return x_max, x_min


                # x_absmax = torch.max(x_min.abs(), x_max)

                # if self.sym:
                #     delta = x_absmax / self.n_levels
                # else:
                #     delta = (x_max - x_min) / (self.n_levels - 1)
                
                # zero_point = torch.round(-x_min / delta).type_as(x) if not self.sym else torch.tensor(0.).type_as(x)
                # delta = delta.type_as(x)


            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(20):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round()
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        # assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)


class UniformAffineQuantizer_bias(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False):
        super(UniformAffineQuantizer_bias, self).__init__()
        self.sym = symmetric
        # assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            if self.leaf_param:
                delta, zero_point = self.init_quantization_scale(x, self.channel_wise)
                self.delta = torch.nn.Parameter(delta, requires_grad=True)
                self.zero_point = torch.nn.Parameter(zero_point, requires_grad=False)
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True


        x_int = round_ste(x / self.delta) + self.zero_point
        if self.sym:
            x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        else:
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            delta, zero_point = self.init_quantization_scale(x_clone, channel_wise=False)

            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()

            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)


        else:
            if 'max' in self.scale_method:
                
                x_min = torch.min(x)
                x_min = torch.min(x_min, torch.zeros_like(x_min))
                x_max = torch.max(x)
                x_max = torch.max(x_max, torch.zeros_like(x_max))

                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                # x_absmax = max(abs(x_min), x_max)
                x_absmax = torch.max(x_min.abs(), x_max)

                if self.sym:
                    delta = x_absmax / self.n_levels
                else:
                    # delta = float(x_max - x_min) / (self.n_levels - 1)
                    delta = (x_max - x_min) / (self.n_levels - 1)
                

                # if delta < 1e-8:
                #     warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                #     delta = torch.Tensor(1e-8).type_as(x)

                # zero_point = torch.tensor(round(-x_min / delta)).type_as(x)
                # delta = torch.tensor(delta).type_as(x)
                zero_point = torch.round(-x_min / delta).type_as(x) if not self.sym else torch.tensor(0.).type_as(x)
                delta = delta.type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(20):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round()
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        # assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)


class UniformAffineQuantizer_bias_bg(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False):
        super(UniformAffineQuantizer_bias_bg, self).__init__()
        self.sym = symmetric
        # assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        self.x_max = None
        self.x_min = None
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            x_max, x_min = self.init_quantization_scale(x, self.channel_wise)
            self.x_max = torch.nn.Parameter(x_max, requires_grad=True)
            self.x_min = torch.nn.Parameter(x_min, requires_grad=True)
            
            # self.gamma = torch.nn.Parameter(torch.ones_like(self.x_max), requires_grad=True)
            # self.beta = torch.nn.Parameter(torch.ones_like(self.x_min), requires_grad=True)
            self.gamma = torch.ones_like(self.x_max)
            self.beta = torch.ones_like(self.x_min)

            if self.sym:
                self.zero_point = torch.zeros_like(self.x_max)

            self.inited = True

        if self.sym:
            x_absmax = torch.max(self.x_min.abs(), self.x_max)
            delta = (self.gamma*x_absmax) / self.n_levels
        else:
            delta = (self.gamma*self.x_max - self.beta*self.x_min) / (self.n_levels - 1)

        zero_point = round_ste((-self.x_min*self.beta) / delta) if not self.sym else self.zero_point

        x_int = round_ste(x / delta) + zero_point
        if self.sym:
            x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        else:
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - zero_point) * delta
        
        return x_dequant


        x_int = round_ste(x / self.delta) + self.zero_point
        if self.sym:
            x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        else:
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            delta, zero_point = self.init_quantization_scale(x_clone, channel_wise=False)

            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()

            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)


        else:
            if 'max' in self.scale_method:
                
                x_min = torch.min(x)
                x_min = torch.min(x_min, torch.zeros_like(x_min))
                x_max = torch.max(x)
                x_max = torch.max(x_max, torch.zeros_like(x_max))

                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                return x_max, x_min

                # x_absmax = torch.max(x_min.abs(), x_max)

                # if self.sym:
                #     delta = x_absmax / self.n_levels
                # else:
                #     # delta = float(x_max - x_min) / (self.n_levels - 1)
                #     delta = (x_max - x_min) / (self.n_levels - 1)
                
                # zero_point = torch.round(-x_min / delta).type_as(x) if not self.sym else torch.tensor(0.).type_as(x)
                # delta = delta.type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(20):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round()
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        # assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)


class UniformAffineQuantizer_act(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, low_mem: bool = False, always_zero: bool = False, learn_zero_point: bool = False):
        super(UniformAffineQuantizer_act, self).__init__()
        self.sym = symmetric
        # assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        self.delta = []
        self.zero_point = []
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        self.low_mem = low_mem
        self.always_zero = always_zero
        self.learn_zero_point = learn_zero_point

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            for i in range(50):
                delta, zero_point = self.init_quantization_scale(x[i], self.channel_wise)
                self.delta.append(delta)
                self.zero_point.append(zero_point)
            self.delta = torch.stack(self.delta)
            self.zero_point = torch.stack(self.zero_point)
            # self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
            # self.delta = self.delta.expand(50)
            # self.zero_point = self.zero_point.expand(50)
            # if len(x.shape) == 4:
            #     self.delta = self.delta.view(-1, 1, 1, 1)
            #     self.zero_point = self.zero_point.view(-1, 1, 1, 1)
            # elif len(x.shape) == 3:
            #     self.delta = self.delta.view(-1, 1, 1)
            #     self.zero_point = self.zero_point.view(-1, 1, 1)
            # elif len(x.shape) == 2:
            #     self.delta = self.delta.view(-1, 1)
            #     self.zero_point = self.zero_point.view(-1, 1)
            self.inited = True

            if not self.low_mem:
                x_int = round_ste(x / self.delta) + self.zero_point
                if self.sym:
                    x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
                else:
                    x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                x_dequant = (x_quant - self.zero_point) * self.delta
            else:
                x_dequant = []
                for i in range(50):
                    x_int = round_ste(x[i] / self.delta[i]) + self.zero_point[i]
                    if self.sym:
                        x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
                    else:
                        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                    x_dequant.append((x_quant - self.zero_point[i]) * self.delta[i])
                x_dequant = torch.stack(x_dequant)
                self.delta = self.delta.float()
                self.zero_point = self.zero_point.float()

            self.delta = torch.nn.Parameter(self.delta, requires_grad=True)
            self.zero_point = torch.nn.Parameter(self.zero_point, requires_grad=False)

            if self.always_zero:
                self.delta.requires_grad_(False)
            if self.learn_zero_point:
                self.zero_point.requires_grad_(True)
            
        else:
            index = global_var.use_global_index()
            delta_cur = self.delta[index]
            zero_point_cur = self.zero_point[index]
            if self.learn_zero_point:
                zero_point_cur = round_ste(zero_point_cur)
            x_int = round_ste(x / delta_cur) + zero_point_cur
            if self.sym:
                x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
            else:
                x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant - zero_point_cur) * delta_cur


        
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if 'max' in self.scale_method:
                # x_min = min(x.min().item(), 0)
                # x_max = max(x.max().item(), 0)
                x_min = torch.min(x)
                x_min = torch.min(x_min, torch.zeros_like(x_min))
                x_max = torch.max(x)
                x_max = torch.max(x_max, torch.zeros_like(x_max))
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                if self.always_zero:
                    x_max = torch.ones_like(x_max)
                    x_min = torch.zeros_like(x_min)

                # x_absmax = max(abs(x_min), x_max)
                x_absmax = torch.max(x_min.abs(), x_max)

                if self.sym:
                    delta = x_absmax / self.n_levels
                else:
                    # delta = float(x_max - x_min) / (self.n_levels - 1)
                    delta = (x_max - x_min) / (self.n_levels - 1)

                # if delta < 1e-8:
                #     warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                #     delta = torch.Tensor(1e-8).type_as(x)

                # zero_point = torch.tensor(round(-x_min / delta)).type_as(x)
                # delta = torch.tensor(delta).type_as(x)
                zero_point = torch.round(-x_min / delta).type_as(x) if not (self.sym or self.always_zero) else torch.tensor(0.).type_as(x)
                delta = delta.type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(20):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round()
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        # assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)


class UniformAffineQuantizer_act_step(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, low_mem: bool = False, always_zero: bool = False, learn_zero_point: bool = False):
        super(UniformAffineQuantizer_act_step, self).__init__()
        self.sym = symmetric
        # assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        self.delta = []
        self.zero_point = []
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        self.low_mem = low_mem
        self.always_zero = always_zero
        self.learn_zero_point = learn_zero_point
        self.index_self = 0

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            index = global_var.use_global_index()
            if index == self.index_self:
                delta, zero_point = self.init_quantization_scale(x, self.channel_wise)
                self.delta.append(delta)
                self.zero_point.append(zero_point)
                self.index_self += 1
            # else:
            #     delta = self.delta[-1]
            #     zero_point = self.zero_point[-1]

            # if index == 49:
            if len(self.delta) == 50:
                self.delta = torch.stack(self.delta)
                self.zero_point = torch.stack(self.zero_point)

                # if len(x.shape) == 4:
                #     self.delta = self.delta.view(-1, 1, 1, 1)
                #     self.zero_point = self.zero_point.view(-1, 1, 1, 1)
                # elif len(x.shape) == 3:
                #     self.delta = self.delta.view(-1, 1, 1)
                #     self.zero_point = self.zero_point.view(-1, 1, 1)
                # elif len(x.shape) == 2:
                #     self.delta = self.delta.view(-1, 1)
                #     self.zero_point = self.zero_point.view(-1, 1)

                self.delta = torch.nn.Parameter(self.delta, requires_grad=True)
                self.zero_point = torch.nn.Parameter(self.zero_point, requires_grad=False)

                if self.always_zero:
                    self.delta.requires_grad_(False)
                if self.learn_zero_point:
                    self.zero_point.requires_grad_(True)

                # print(self.delta.shape, self.zero_point.shape)
                self.inited = True

            x_int = round_ste(x / delta) + zero_point
            if self.sym:
                x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
            else:
                x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant - zero_point) * delta

            
        else:
            index = global_var.use_global_index()
            # print(index)
            delta_cur = self.delta[index]
            zero_point_cur = self.zero_point[index]
            # print(self.delta.shape, self.zero_point.shape, delta_cur.shape, zero_point_cur.shape)
            if self.learn_zero_point:
                zero_point_cur = round_ste(zero_point_cur)
            x_int = round_ste(x / delta_cur) + zero_point_cur
            if self.sym:
                x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
            else:
                x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant - zero_point_cur) * delta_cur


        
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if 'max' in self.scale_method:
                # x_min = min(x.min().item(), 0)
                # x_max = max(x.max().item(), 0)
                x_min = torch.min(x)
                x_min = torch.min(x_min, torch.zeros_like(x_min))
                x_max = torch.max(x)
                x_max = torch.max(x_max, torch.zeros_like(x_max))
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                if self.always_zero:
                    x_max = torch.ones_like(x_max)
                    x_min = torch.ones_like(x_min) * -1.

                # x_absmax = max(abs(x_min), x_max)
                x_absmax = torch.max(x_min.abs(), x_max)

                if self.sym:
                    delta = x_absmax / self.n_levels
                else:
                    # delta = float(x_max - x_min) / (self.n_levels - 1)
                    delta = (x_max - x_min) / (self.n_levels - 1)

                # if delta < 1e-8:
                #     warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                #     delta = torch.Tensor(1e-8).type_as(x)

                # zero_point = torch.tensor(round(-x_min / delta)).type_as(x)
                # delta = torch.tensor(delta).type_as(x)
                zero_point = torch.round(-x_min / delta).type_as(x) if not (self.sym or self.always_zero) else torch.tensor(0.).type_as(x)
                delta = delta.type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(20):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round()
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        # assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)


class UniformAffineQuantizer_act_step_bg(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, low_mem: bool = False, always_zero: bool = False, learn_zero_point: bool = False):
        super(UniformAffineQuantizer_act_step_bg, self).__init__()
        self.sym = symmetric
        # assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        self.x_max = []
        self.x_min = []
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        self.low_mem = low_mem
        self.always_zero = always_zero
        self.learn_zero_point = learn_zero_point
        self.index_self = 0

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            index = global_var.use_global_index()
            if index == self.index_self:
                x_max, x_min = self.init_quantization_scale(x, self.channel_wise)
                self.x_max.append(x_max)
                self.x_min.append(x_min)
                self.index_self += 1


            if len(self.x_max) == 50:
                self.x_max = torch.stack(self.x_max)
                self.x_min = torch.stack(self.x_min)

                # self.gamma = torch.nn.Parameter(torch.ones_like(self.x_max), requires_grad=True)
                # self.beta = torch.nn.Parameter(torch.ones_like(self.x_min), requires_grad=True)
                self.gamma = torch.ones_like(self.x_max)
                self.beta = torch.ones_like(self.x_min)
                self.x_max = torch.nn.Parameter(self.x_max, requires_grad=True)
                self.x_min = torch.nn.Parameter(self.x_min, requires_grad=True)

                if self.sym:
                    self.zero_point = torch.zeros_like(x_max)

                self.inited = True


            if self.sym:
                x_absmax = torch.max(x_min.abs(), x_max)
                delta = x_absmax / self.n_levels
            else:
                delta = (x_max - x_min) / (self.n_levels - 1)

            zero_point = round_ste(-x_min / delta) if not self.sym else torch.zeros_like(delta)

            x_int = round_ste(x / delta) + zero_point
            if self.sym:
                x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
            else:
                x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant - zero_point) * delta

            
        else:
            index = global_var.use_global_index()
            x_max_cur = self.x_max[index]
            x_min_cur = self.x_min[index]
            gamma_cur = self.gamma[index]
            beta_cur = self.beta[index]

            # if self.always_zero:
            #     x_max_cur = torch.clamp(x_max_cur, min=1e-8, max=1.)
            #     x_min_cur = torch.clamp(x_min_cur, min=-1., max=-1e-8)

            if self.sym:
                x_absmax = torch.max(x_min_cur.abs(), x_max_cur)
                delta = (gamma_cur*x_absmax) / self.n_levels
            else:
                delta = (gamma_cur*x_max_cur - beta_cur*x_min_cur) / (self.n_levels - 1)

            zero_point = round_ste((-x_min_cur*beta_cur) / delta) if not self.sym else self.zero_point

            x_int = round_ste(x / delta) + zero_point
            if self.sym:
                x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
            else:
                x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant - zero_point) * delta

        
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if 'max' in self.scale_method:
                x_min = torch.min(x)
                x_min = torch.min(x_min, torch.zeros_like(x_min))
                x_max = torch.max(x)
                x_max = torch.max(x_max, torch.zeros_like(x_max))
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                if self.always_zero:
                    x_max = torch.ones_like(x_max)
                    x_min = torch.ones_like(x_min) * -1.


                return x_max, x_min


                # x_absmax = torch.max(x_min.abs(), x_max)

                # if self.sym:
                #     delta = x_absmax / self.n_levels
                # else:
                #     delta = (x_max - x_min) / (self.n_levels - 1)

                # zero_point = torch.round(-x_min / delta).type_as(x) if not (self.sym or self.always_zero) else torch.tensor(0.).type_as(x)
                # delta = delta.type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(20):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round()
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        # assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)


class UniformAffineQuantizer_weight_step(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, low_mem: bool = False, always_zero: bool = False, learn_zero_point: bool = False):
        super(UniformAffineQuantizer_weight_step, self).__init__()
        self.sym = symmetric
        # assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        self.delta = []
        self.zero_point = []
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        self.low_mem = low_mem
        self.always_zero = always_zero
        self.learn_zero_point = learn_zero_point

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            index = global_var.use_global_index()
            # print(index)
            delta, zero_point = self.init_quantization_scale(x, self.channel_wise)

            for i in range(51):
                self.delta.append(delta)
                self.zero_point.append(zero_point)

            self.delta = torch.stack(self.delta)
            self.zero_point = torch.stack(self.zero_point)

            self.delta = torch.nn.Parameter(self.delta, requires_grad=True)
            self.zero_point = torch.nn.Parameter(self.zero_point, requires_grad=False)

            if self.always_zero:
                self.delta.requires_grad_(False)
            if self.learn_zero_point:
                self.zero_point.requires_grad_(True)

            print(self.delta.shape, self.zero_point.shape)
            self.inited = True

            x_int = round_ste(x / delta) + zero_point
            if self.sym:
                x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
            else:
                x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant - zero_point) * delta

            
        else:
            index = global_var.use_global_index()
            delta_cur = self.delta[index].squeeze(0)
            zero_point_cur = self.zero_point[index].squeeze(0)
            # print(delta_cur.shape, zero_point_cur.shape, x.shape)
            if self.learn_zero_point:
                zero_point_cur = round_ste(zero_point_cur)
            x_int = round_ste(x / delta_cur) + zero_point_cur
            if self.sym:
                x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
            else:
                x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant - zero_point_cur) * delta_cur


        
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if 'max' in self.scale_method:
                # x_min = min(x.min().item(), 0)
                # x_max = max(x.max().item(), 0)
                x_min = torch.min(x)
                x_min = torch.min(x_min, torch.zeros_like(x_min))
                x_max = torch.max(x)
                x_max = torch.max(x_max, torch.zeros_like(x_max))
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                if self.always_zero:
                    x_max = torch.ones_like(x_max)
                    x_min = torch.zeros_like(x_min)

                # x_absmax = max(abs(x_min), x_max)
                x_absmax = torch.max(x_min.abs(), x_max)

                if self.sym:
                    delta = x_absmax / self.n_levels
                else:
                    # delta = float(x_max - x_min) / (self.n_levels - 1)
                    delta = (x_max - x_min) / (self.n_levels - 1)

                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = torch.Tensor(1e-8).type_as(x)

                # zero_point = torch.tensor(round(-x_min / delta)).type_as(x)
                # delta = torch.tensor(delta).type_as(x)
                zero_point = torch.round(-x_min / delta).type_as(x) if not (self.sym or self.always_zero) else torch.tensor(0.).type_as(x)
                delta = delta.type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(20):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round()
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        # assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)






class F_conv2d(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, input, weight, bias, fwd_kwargs):
        return F.conv2d(input, weight, bias, **fwd_kwargs)

class F_linear(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, input, weight, bias, fwd_kwargs):
        return F.linear(input, weight, bias, **fwd_kwargs)

class QuantConv2d(nn.Module):
    def __init__(self, org_module: Union[nn.Conv2d, LoRACompatibleConv], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, se_module=None):
        super(QuantConv2d, self).__init__()
        self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                               dilation=org_module.dilation, groups=org_module.groups)
        self.fwd_func = F_conv2d()
        self.fwd_func.w_bit = weight_quant_params['n_bits']
        
        self.weight = org_module.weight
        self.bias = org_module.bias

        # de-activate the quantized forward default
        self.use_weight_quant = True
        self.use_act_quant = True
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer_weight_bg(**weight_quant_params)
        # if act_quant_params['n_bits'] == 16:
        #     self.act_quantizer = UniformAffineQuantizer_int16(**act_quant_params)
        # else:
        self.act_quantizer = UniformAffineQuantizer_act_step_bg(**act_quant_params)

        if self.bias is not None:
            bias_quant_params = weight_quant_params.copy()
            bias_quant_params['channel_wise'] = False
            self.bias_quantizer = UniformAffineQuantizer_bias_bg(**bias_quant_params)

        self.extra_repr = org_module.extra_repr

    def forward(self, input: torch.Tensor, scale: float = 1.0):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            if self.bias is not None:
                bias = self.bias_quantizer(self.bias)
            else:
                bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        # if self.weight_quantizer.n_bits == 16:
        #     input = input.to(torch.float16).to(torch.float32)

        out = self.fwd_func(input, weight, bias, self.fwd_kwargs)

        if self.use_act_quant:
            out = self.act_quantizer(out)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant


class QuantLinear(nn.Module):
    def __init__(self, org_module: Union[nn.Linear, LoRACompatibleLinear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, se_module=None):
        super(QuantLinear, self).__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F_linear()
        self.fwd_func.w_bit = weight_quant_params['n_bits']

        self.weight = org_module.weight
        self.bias = org_module.bias

        self.weight.requires_grad_(False)

        # de-activate the quantized forward default
        self.use_weight_quant = True
        self.use_act_quant = True
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer_weight(**weight_quant_params)
        # if act_quant_params['n_bits'] == 16:
        #     self.act_quantizer = UniformAffineQuantizer_int16(**act_quant_params)
        # else:
        # self.act_quantizer = UniformAffineQuantizer_act_step_bg(**act_quant_params)

        # if self.bias is not None:
        #     bias_quant_params = weight_quant_params.copy()
        #     bias_quant_params['channel_wise'] = False
        #     self.bias_quantizer = UniformAffineQuantizer_bias_bg(**bias_quant_params)

        self.extra_repr = org_module.extra_repr

        w_dequant = self.weight_quantizer(self.weight)
        error = F.mse_loss(self.weight, w_dequant)
        # print(error)
        global_var.global_error += error.item()
        global_var.global_num +=1

    def forward(self, input: torch.Tensor, scale: float = 1.0):
        # if self.use_weight_quant:
        #     weight = self.weight_quantizer(self.weight)
        #     # if self.bias is not None:
        #     #     bias = self.bias_quantizer(self.bias)
        #     # else:
        #     bias = self.bias
        # else:
        #     weight = self.weight
        #     bias = self.bias

        # if self.weight_quantizer.n_bits == 16:
        #     input = input.to(torch.float16).to(torch.float32)

        weight = self.weight_quantizer(self.weight)
        bias = self.bias
        out = self.fwd_func(input, weight, bias, self.fwd_kwargs)

        # if self.use_act_quant:
        #     out = self.act_quantizer(out)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant


class QuantNorm(nn.Module):
    def __init__(self, org_module, 
                 act_quant_params: dict = {}):
        super(QuantNorm, self).__init__()
        self.norm = org_module

        # de-activate the quantized forward default
        self.use_weight_quant = True
        self.use_act_quant = True
        # initialize quantizer
        self.act_quantizer = UniformAffineQuantizer_act_step(**act_quant_params)

        self.extra_repr = org_module.extra_repr

    def forward(self, input: torch.Tensor):
        out = self.norm(input)

        if self.use_act_quant:
            out = self.act_quantizer(out)
        
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant


class QuantActfun(nn.Module):
    def __init__(self, org_module, 
                 act_quant_params: dict = {}):
        super(QuantActfun, self).__init__()
        self.actfun = org_module

        # de-activate the quantized forward default
        self.use_weight_quant = True
        self.use_act_quant = True
        # initialize quantizer
        self.act_quantizer = UniformAffineQuantizer_act_step(**act_quant_params)

        self.extra_repr = org_module.extra_repr

    def forward(self, input: torch.Tensor):
        out = self.actfun(input)

        if self.use_act_quant:
            out = self.act_quantizer(out)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
