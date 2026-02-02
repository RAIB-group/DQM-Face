#!/usr/bin/env bash

python -u IJB_11c.py --model-prefix "/slurm/home/bel/Atten dualFace Project/output/ablation_dual_attention/17_12_25/checkpoint_gpu_0.pt" --model-epoch 0 --gpu 0 --target IJBC --job r100dual > r100dual.log 2>&1 &

