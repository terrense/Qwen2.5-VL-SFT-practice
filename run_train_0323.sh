#!/bin/bash
# 0323 实验：Qwen2.5-VL-32B LoRA 1-epoch 训练
# 用法: ./run_train_0323.sh 或 CUDA_VISIBLE_DEVICES=0 ./run_train_0323.sh
# 输出: runs/qwen25vl32b_full_1epoch_0323/

set -e
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python train_lora_full_0323.py
