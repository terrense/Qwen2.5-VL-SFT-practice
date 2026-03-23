#!/bin/bash
# 0323 实验：基座模型（无 LoRA）在验证集上评测
# 用法: ./run_eval_base_0323.sh
# 输出: reports/base_eval_0323/
# 与 bian/Finetune_WHU/eval_base_model.py 一致：在数据目录下运行，使用相对路径

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="/data/bian/Finetune_WHU"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

cd "$DATA_DIR"
python "${SCRIPT_DIR}/eval_base_0323.py"
