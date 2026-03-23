#!/bin/bash
# 0323 实验：评测 merge 后或结构化剪枝后的完整模型
# 用法: ./scripts/run_eval_pruned_0323.sh
# 默认评测: post_sft/pruned/qwen25vl32b_full_1epoch_merged_pruned_0323/
# 输出: reports/pruned_eval_0323/pruned_model_0323/

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="/data/bian/Finetune_WHU"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

cd "$DATA_DIR"
python "${ROOT_DIR}/post_sft/eval_pruned_0323.py"
