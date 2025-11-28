# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse

from model_size import compute_model_nbits
from model_compression import compress_model
from CompressedLinear import CompressedLinear

import copy
import torch.nn.functional as F
import collections
import random
from tqdm import trange

from diffusion_utils import discretized_gaussian_log_likelihood, normal_kl, add_hook, set_seed, log_compression_ratio, custom_loss_function

from models import DiTBlock

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def main(args):
    # Setup PyTorch:
    # torch.manual_seed(args.seed)
    # torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = torch.load(ckpt_path, map_location='cuda')
    model.load_state_dict(state_dict, strict=True)

 
    model_tea = copy.deepcopy(model)
    requires_grad(model_tea, False)
    model_tea.eval()

    compression_config = {
        "ignored_modules": ["x_embedder", "t_embedder", "y_embedder", "final_layer"],
        "k": args.k,
        "d": args.d,
    }
    uncompressed_model_size_bits, _, __ = compute_model_nbits(model.blocks)
    model = compress_model(model, **compression_config, cb_dir=args.cb_dir).cuda()
    compressed_model_size_bits, compressed_codebook_size_bits, compressed_codes_size_bits = compute_model_nbits(model.blocks)
    log_compression_ratio(uncompressed_model_size_bits, compressed_model_size_bits, compressed_codebook_size_bits, compressed_codes_size_bits)


    for n, m in model.named_modules():
        if isinstance(m, CompressedLinear):
            m.prepare_sim()

    requires_grad(model.x_embedder, False)
    requires_grad(model.t_embedder, False)
    requires_grad(model.y_embedder, False)
    requires_grad(model.final_layer, False)

    mapping_layers = []
    for n, m in model.named_modules():

        if isinstance(m, DiTBlock):
            mapping_layers.append(n)


    print(mapping_layers)

    if len(mapping_layers) > 0:
        acts_tea = {}
        acts_stu = {}   
        mapping_layers_tea = copy.deepcopy(mapping_layers)
        mapping_layers_stu = copy.deepcopy(mapping_layers)
        add_hook(model_tea, acts_tea, mapping_layers_tea)
        add_hook(model, acts_stu, mapping_layers_stu)
        mapping_params = (mapping_layers_tea, mapping_layers_stu, acts_tea, acts_stu)
    else:
        mapping_params = None



    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    params_codebook = []
    params_assign = []
    params_bias = []
    
    for module in model.modules():
        if hasattr(module, 'codebook'):
            params_codebook.append(module.codebook)
        if hasattr(module, 'bias') and module.bias is not None:
            params_bias.append(module.bias)
        if hasattr(module, 'codes_matrix_train'):
            params_assign.append(module.codes_matrix_train)
    
    opt = torch.optim.RMSprop([
        {'params': params_codebook, 'lr': 1e-4},
        {'params': params_bias, 'lr': 1e-4},
        {'params': params_assign, 'lr': 5e-2}
    ], alpha=0.9, foreach=True)

    model.train()
    model.freeze_all_code = False

    sum_loss_step = collections.deque(maxlen=10000)


    # Labels to condition the model with (feel free to change):
    for i in range(1000):
        # Create sampling noise:
        class_labels = random.sample(range(1000), k=4)
        print(class_labels)
        n = len(class_labels)
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = torch.tensor(class_labels, device=device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)


        # Sample images:
        _ = diffusion.p_sample_loop_quant(
            model_tea, model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device, opt=opt, mapping_params=mapping_params, sum_loss_step=sum_loss_step
        )


        print(i)
            

        if (i % 50 == 0 and i > 0) and model.freeze_all_code:
            torch.save(model.state_dict(), 'model_'+str(args.w_bit)+'bit_'+str(i)+'.pt')
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=512)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--w_bit", type=int, default=2)
    parser.add_argument("--cb-dir", type=str, default=None, help="Directory to save/load codebooks")
    args = parser.parse_args()
    if args.w_bit == 2:
        args.k, args.d = 256, 4
    elif args.w_bit == 3:
        args.k, args.d = 64, 2
    set_seed(args.seed)
    main(args)
