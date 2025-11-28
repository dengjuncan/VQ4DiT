#!/bin/bash

IMAGE_SIZE=256

python sample_quant.py \
    --model DiT-XL/2 \
    --image-size $IMAGE_SIZE \
    --ckpt DiT-XL-2-${IMAGE_SIZE}x${IMAGE_SIZE}.pt \
    --num-sampling-steps 50 \
    --k 256 \
    --d 4 \
    --k-means-type kmeans \
    --k-means-n-iters 100
