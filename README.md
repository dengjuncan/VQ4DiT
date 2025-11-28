# VQ4DiT: Efficient Post-Training Vector Quantization for Diffusion Transformers

This repository contains the official implementation of the paper [VQ4DiT: Efficient Post-Training Vector Quantization for Diffusion Transformers](https://arxiv.org/abs/2408.17131).

**Authors:** Juncan Deng, Shuaiting Li, Zeyu Wang, Hong Gu, Kedong Xu, Kejie Huang

## Abstract

The Diffusion Transformers Models (DiTs) have transitioned the network architecture from traditional UNets to transformers, demonstrating exceptional capabilities in image generation. Although DiTs have been widely applied to high-definition video generation tasks, their large parameter size hinders inference on edge devices. Vector quantization (VQ) can decompose model weight into a codebook and assignments, allowing extreme weight quantization and significantly reducing memory usage. In this paper, we propose VQ4DiT, a fast post-training vector quantization method for DiTs. We found that traditional VQ methods calibrate only the codebook without calibrating the assignments. This leads to weight sub-vectors being incorrectly assigned to the same assignment, providing inconsistent gradients to the codebook and resulting in a suboptimal result. To address this challenge, VQ4DiT calculates the candidate assignment set for each weight sub-vector based on Euclidean distance and reconstructs the sub-vector based on the weighted average. Then, using the zero-data and block-wise calibration method, the optimal assignment from the set is efficiently selected while calibrating the codebook. VQ4DiT quantizes a DiT XL/2 model on a single NVIDIA A100 GPU within 20 minutes to 5 hours depending on the different quantization settings. Experiments show that VQ4DiT establishes a new state-of-the-art in model size and performance trade-offs, quantizing weights to 2-bit precision while retaining acceptable image generation quality.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dengjuncan/VQ4DiT.git
    cd VQ4DiT
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download pre-trained models:**
    Pre-trained DiT models can be downloaded from the official [DiT repository](https://github.com/facebookresearch/DiT). Place the downloaded checkpoint file (e.g., `DiT-XL-2-256x256.pt`) in the root directory of this project.

## Quantization

To run the post-training quantization, you can use the provided shell script. This script will quantize the DiT model and perform sampling.

```bash
bash run.sh
```

You can modify the `run.sh` script to change parameters such as the image size, number of sampling steps, and quantization settings.

## Acknowledgements

This code is based on the following repositories:
*   [facebookresearch/DiT](https://github.com/facebookresearch/DiT)
*   [uber-research/permute-quantize-finetune](https://github.com/uber-research/permute-quantize-finetune)

We thank the authors for their valuable contributions.
