#!/bin/bash
# 0323 实验：merge 基座模型与 LoRA adapter，输出完整 HuggingFace 模型
# 用法: ./scripts/run_merge_lora_0323.sh
# 输出: post_sft/merged/qwen25vl32b_full_1epoch_merged_0323/

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

python "${ROOT_DIR}/post_sft/merge_lora_0323.py"
