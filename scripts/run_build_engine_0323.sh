#!/bin/bash
# 0323 实验：Step2 从 build_plan_0323.json 构建 TensorRT engine
# 用法: ./scripts/run_build_engine_0323.sh --plan-path /path/to/build_plan_0323.json

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

python "${ROOT_DIR}/trt/step2_build_0323.py" "$@"
