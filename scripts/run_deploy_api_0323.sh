#!/bin/bash
# 0323 实验：启动 OpenAI-compatible 多模态 API 服务
# 用法: ./scripts/run_deploy_api_0323.sh --mode merged

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

python "${ROOT_DIR}/deploy/deploy_api_0323.py" "$@"
