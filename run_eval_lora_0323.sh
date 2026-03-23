#!/bin/bash
# 0323 实验：LoRA 微调模型在验证集上评测（需先完成训练）
# 用法: ./run_eval_lora_0323.sh
# 输出: reports/lora_eval_0323/

set -e
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python eval_lora_0323.py
