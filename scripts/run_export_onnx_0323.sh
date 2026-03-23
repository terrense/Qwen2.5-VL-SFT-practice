#!/bin/bash
# 0323 实验：Step1 导出 ONNX 并记录 TensorRT 构建计划
# 用法: ./scripts/run_export_onnx_0323.sh [TensorRT-LLM builder args ...]

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

python "${ROOT_DIR}/trt/step1_export_0323.py" "$@"
