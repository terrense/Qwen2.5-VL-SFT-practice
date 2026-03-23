#!/bin/bash
# 0323 实验：对 merge 后模型做 int4_awq / fp8 量化导出
# 用法: ./scripts/run_quantize_0323.sh
# 依赖: TensorRT-LLM / tensorrt_llm 环境

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

python "${ROOT_DIR}/quantization/quantize_0323.py"
