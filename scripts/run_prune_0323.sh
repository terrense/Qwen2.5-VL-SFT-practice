#!/bin/bash
# 0323 实验：对 merge 后模型执行 2:4 结构化剪枝
# 用法: ./scripts/run_prune_0323.sh
# 输出: post_sft/pruned/qwen25vl32b_full_1epoch_merged_pruned_0323/

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

python "${ROOT_DIR}/post_sft/prune_2to4_0323.py"
