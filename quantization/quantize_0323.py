#!/usr/bin/env python3
"""
0323 实验：量化导出脚本。
该脚本参考 /data/bian/Finetune_WHU/qwen_tensorrt_llm/quantize.py，
用于把 merge 后模型导出为 int4_awq 或 fp8 量化产物，供后续 TensorRT-LLM 使用。

注意：这条量化链路依赖 tensorrt_llm 及其相关运行环境。
若当前环境未安装 TensorRT-LLM，该脚本不会在 import 阶段崩掉，而会在 main() 里给出明确报错。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT_0323 = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_0323) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_0323))

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from experiment_config_0323 import (
    AWQ_EXPORT_PATH_0323,
    FP8_EXPORT_PATH_0323,
    MERGED_MODEL_ROOT_0323,
    ensure_output_dirs_0323,
)


def parse_args_0323() -> argparse.Namespace:
    """解析量化导出参数。"""
    parser = argparse.ArgumentParser(description='Quantize merged 0323 model for downstream TensorRT-LLM usage.')
    parser.add_argument(
        '--model-dir',
        type=Path,
        default=MERGED_MODEL_ROOT_0323,
        help='待量化的完整模型目录；通常使用 merge 后模型。',
    )
    parser.add_argument(
        '--dtype',
        default='float16',
        help='量化前加载模型的精度。',
    )
    parser.add_argument(
        '--qformat',
        choices=['fp8', 'int4_awq'],
        default='int4_awq',
        help='量化格式。',
    )
    parser.add_argument(
        '--calib-size',
        type=int,
        default=32,
        help='校准样本数。',
    )
    parser.add_argument(
        '--export-path',
        type=Path,
        default=None,
        help='导出文件路径；若为空则按 qformat 选默认路径。',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子。',
    )
    return parser.parse_args()


def get_calib_dataloader_0323(tokenizer, batch_size=1, calib_size=32, block_size=512):
    """
    构造一个轻量校准数据集。
    这里沿用原脚本思路，使用 cnn_dailymail 文本做 PTQ/AWQ 校准。
    """
    print('Loading calibration dataset ...')
    dataset = load_dataset('ccdv/cnn_dailymail', name='3.0.0', split='train')
    dataset = dataset['article'][:calib_size]

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    processed = []
    for text in dataset:
        text = (text + ' TL;DR: ').strip().replace(" n't", "n't")
        processed.append(text)

    batch_encoded = tokenizer.batch_encode_plus(
        processed,
        return_tensors='pt',
        padding=True,
        max_length=block_size,
        truncation=True,
    )
    batch_encoded = batch_encoded['input_ids'].cuda()
    return DataLoader(batch_encoded, batch_size=batch_size, shuffle=False)


def main() -> None:
    args = parse_args_0323()
    ensure_output_dirs_0323()

    try:
        from tensorrt_llm.models.quantized.ammo import quantize_and_export
        from tensorrt_llm._utils import str_dtype_to_torch
        from tensorrt_llm.logger import logger
        from transformers import AutoModelForCausalLM
    except Exception as exc:
        raise RuntimeError(
            'quantize_0323.py 依赖 tensorrt_llm / TensorRT-LLM 环境，请先完成相关安装。'
        ) from exc

    if not torch.cuda.is_available():
        raise EnvironmentError('GPU is required for quantization export.')

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f'Model directory not found: {model_dir}')

    export_path = args.export_path
    if export_path is None:
        export_path = AWQ_EXPORT_PATH_0323 if args.qformat == 'int4_awq' else FP8_EXPORT_PATH_0323
    export_path = export_path.expanduser().resolve()
    export_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Loading tokenizer from {model_dir} ...')
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        padding_side='left',
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f'Loading model from {model_dir} with dtype={args.dtype} ...')
    torch_dtype = str_dtype_to_torch(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.eval()
    model = model.to(memory_format=torch.channels_last)

    calib_dataloader = get_calib_dataloader_0323(
        tokenizer=tokenizer,
        calib_size=args.calib_size,
    )

    print(f'Exporting quantized model ({args.qformat}) to {export_path} ...')
    quantize_and_export(
        model,
        qformat=args.qformat,
        calib_dataloader=calib_dataloader,
        export_path=str(export_path),
    )
    print('Quantization export finished.')


if __name__ == '__main__':
    main()
