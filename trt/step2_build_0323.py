#!/usr/bin/env python3
"""
0323 实验：Step 2 读取 build_plan_0323.json 并真正执行 TensorRT engine 构建。
参考 /data/bian/Finetune_WHU/step2_build.py。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT_0323 = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_0323) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_0323))

import torch

DEFAULT_PLAN_FILENAME_0323 = 'build_plan_0323.json'


def parse_args_0323() -> argparse.Namespace:
    """解析 TensorRT 构建参数。"""
    parser = argparse.ArgumentParser(
        description='Step 2: build TensorRT engines from build_plan_0323.json in a fresh process.'
    )
    parser.add_argument('--plan-path', type=str, help='显式指定 build_plan_0323.json 路径。')
    parser.add_argument('--output-dir', type=str, help='当未指定 --plan-path 时，使用该 engine 目录下的默认计划文件。')
    parser.add_argument('--keep-onnx', action='store_true', help='若指定则保留 ONNX 文件。')
    args = parser.parse_args()
    if not args.plan_path and not args.output_dir:
        parser.error('Provide either --plan-path or --output-dir')
    return args


def normalize_dtype_0323(value: Any) -> torch.dtype:
    """把 JSON 中保存的 dtype 字符串转换回 torch.dtype。"""
    if value is None:
        return torch.float16
    if isinstance(value, str):
        lowered = value.lower().replace('torch.', '')
        mapping = {
            'float16': torch.float16,
            'half': torch.float16,
            'fp16': torch.float16,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
            'float32': torch.float32,
            'fp32': torch.float32,
        }
        if lowered in mapping:
            return mapping[lowered]
        if hasattr(torch, lowered):
            candidate = getattr(torch, lowered)
            if isinstance(candidate, torch.dtype):
                return candidate
    if isinstance(value, torch.dtype):
        return value
    raise ValueError(f'Unsupported dtype representation: {value}')


def load_plan_0323(plan_path: Path) -> List[Dict[str, Any]]:
    """读取 step1 产出的 build_plan_0323.json。"""
    if not plan_path.exists():
        raise FileNotFoundError(f'Plan file not found: {plan_path}')
    data = json.loads(plan_path.read_text())
    if not isinstance(data, list):
        raise ValueError('Plan file must contain a list of build entries')
    return data


def main() -> None:
    try:
        from tensorrt_llm.tools.multimodal_builder import build_trt_engine
    except Exception as exc:
        raise RuntimeError('step2_build_0323.py 依赖 tensorrt_llm / TensorRT 环境。') from exc

    args = parse_args_0323()
    plan_path = Path(args.plan_path).expanduser().resolve() if args.plan_path else (Path(args.output_dir).expanduser().resolve() / DEFAULT_PLAN_FILENAME_0323)
    plan_entries = load_plan_0323(plan_path)
    if not plan_entries:
        raise RuntimeError(f'No build entries stored in {plan_path}')

    print(f'Loaded {len(plan_entries)} build entries from {plan_path}')
    for idx, entry in enumerate(plan_entries, start=1):
        dtype = normalize_dtype_0323(entry.get('dtype'))
        delete_onnx = entry.get('delete_onnx', True) and not args.keep_onnx
        print(f'[{idx}/{len(plan_entries)}] Building engine {entry.get("engine_name", "model.engine")}')
        build_trt_engine(
            model_type=entry['model_type'],
            input_sizes=entry['input_sizes'],
            onnx_dir=entry['onnx_dir'],
            engine_dir=entry['engine_dir'],
            max_batch_size=entry['max_batch_size'],
            dtype=dtype,
            model_params=entry.get('model_params'),
            onnx_name=entry.get('onnx_name', 'model.onnx'),
            engine_name=entry.get('engine_name', 'model.engine'),
            delete_onnx=delete_onnx,
        )

    print('All TensorRT engines built successfully.')


if __name__ == '__main__':
    main()
