#!/bin/bash
# 0323 实验：单图推理（有 LoRA 则加载，否则仅用基座）
# 用法: ./run_inference_0323.sh
# 默认图片: experiment_config_0323.SAMPLE_IMAGE_PATH_0323

set -e
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python inference_lora_0323.py
