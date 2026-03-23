#!/usr/bin/env python3
"""
0323 实验：将基座模型与 LoRA adapter 合并为完整 HuggingFace 模型。
输出模型可直接用于后续剪枝、量化、TensorRT 构建或 API 部署。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 将实验根目录加入 sys.path，确保从 post_sft/ 子目录运行时也能导入 experiment_config_0323
PROJECT_ROOT_0323 = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_0323) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_0323))

import torch
from modelscope import AutoTokenizer
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from experiment_config_0323 import (
    BASE_MODEL_PATH_0323,
    MERGED_MODEL_ROOT_0323,
    RUN_ROOT_0323,
    ensure_output_dirs_0323,
)


def parse_args_0323() -> argparse.Namespace:
    """解析命令行参数，允许默认使用当前 `_0323` 训练产物。"""
    parser = argparse.ArgumentParser(
        description='Merge Qwen2.5-VL base weights with a LoRA checkpoint for the 0323 experiment.'
    )
    parser.add_argument(
        '--base-model',
        type=Path,
        default=BASE_MODEL_PATH_0323,
        help='基座模型目录；默认使用 experiment_config_0323.py 中的 BASE_MODEL_PATH_0323。',
    )
    parser.add_argument(
        '--lora-checkpoint',
        type=Path,
        default=RUN_ROOT_0323,
        help='LoRA adapter / checkpoint 路径；默认使用当前 0323 训练输出目录。',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=MERGED_MODEL_ROOT_0323,
        help='merge 后完整模型输出目录。',
    )
    parser.add_argument(
        '--dtype',
        choices=['bfloat16', 'float16', 'float32'],
        default='bfloat16',
        help='加载基座模型时使用的精度。一般建议 bfloat16。',
    )
    parser.add_argument(
        '--device-map',
        default='auto',
        help='传给 from_pretrained 的 device_map；单卡可填 cuda:0，多卡推荐 auto。',
    )
    parser.add_argument(
        '--no-safe-serialization',
        action='store_true',
        help='若指定则保存为 PyTorch .bin；默认保存为 safetensors。',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args_0323()
    ensure_output_dirs_0323()

    base_path = args.base_model.expanduser().resolve()
    lora_path = args.lora_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not base_path.exists():
        sys.exit(f'Base model path not found: {base_path}')
    if not lora_path.exists():
        sys.exit(f'LoRA checkpoint path not found: {lora_path}')
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
    }[args.dtype]

    print(f'[1/4] Loading base model from {base_path} (dtype={args.dtype}, device_map={args.device_map}) ...')
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(base_path),
        dtype=torch_dtype,
        trust_remote_code=True,
        device_map=args.device_map,
    )

    print(f'[2/4] Attaching LoRA weights from {lora_path} ...')
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(lora_path),
        torch_dtype=torch_dtype,
        is_trainable=False,
    )

    print('[3/4] Merging LoRA parameters into the base model ...')
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    safe_serialization = not args.no_safe_serialization
    print(f'[4/4] Saving merged model to {output_dir} (safetensors={safe_serialization}) ...')
    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=safe_serialization,
    )

    # merge 后还需同步保存 tokenizer 与 processor，否则后续推理/部署无法直接复用该目录
    print('Saving tokenizer and processor metadata ...')
    tokenizer = AutoTokenizer.from_pretrained(str(base_path), use_fast=False, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))
    processor = AutoProcessor.from_pretrained(str(base_path), trust_remote_code=True)
    processor.save_pretrained(str(output_dir))

    print('Done. Merged model is available at:')
    print(f'    {output_dir}')


if __name__ == '__main__':
    main()
