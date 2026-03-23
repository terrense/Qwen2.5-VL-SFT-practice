#!/usr/bin/env python3
"""
0323 实验：Step 1 导出 ONNX 并记录 TensorRT 构建计划。
参考 /data/bian/Finetune_WHU/step1_export.py。
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

from experiment_config_0323 import ensure_output_dirs_0323

DEFAULT_PLAN_FILENAME_0323 = 'build_plan_0323.json'


def make_parser_0323() -> argparse.ArgumentParser:
    """构造命令行解析器，并复用 TensorRT-LLM multimodal builder 的参数定义。"""
    try:
        from tensorrt_llm.tools.multimodal_builder import add_multimodal_arguments
    except Exception as exc:
        raise RuntimeError('step1_export_0323.py 依赖 tensorrt_llm / TensorRT 环境。') from exc

    parser = argparse.ArgumentParser(
        description='Step 1: export PyTorch models to ONNX and record params for a later TensorRT build.'
    )
    add_multimodal_arguments(parser)
    parser.add_argument(
        '--plan-filename',
        default=DEFAULT_PLAN_FILENAME_0323,
        help='写在 engine 目录中的构建计划文件名。',
    )
    return parser


def dtype_to_string_0323(dtype: Any) -> str:
    """把 torch dtype 标准化为可写入 JSON 的字符串。"""
    try:
        from tensorrt_llm._utils import torch_dtype_to_str
        return torch_dtype_to_str(dtype)
    except Exception:
        return str(dtype)


def main() -> None:
    ensure_output_dirs_0323()

    try:
        import tensorrt as trt
        import tensorrt_llm.tools.multimodal_builder as mm_builder
        from tensorrt_llm.tools.multimodal_builder import MultimodalEngineBuilder
    except Exception as exc:
        raise RuntimeError('step1_export_0323.py 依赖 tensorrt_llm / TensorRT 环境。') from exc

    parser = make_parser_0323()
    args = parser.parse_args()

    original_build_trt = mm_builder.build_trt_engine
    plan_cache: Dict[Path, List[Dict[str, Any]]] = {}
    initialized_paths: set[Path] = set()

    def record_build_plan(*b_args, **b_kwargs):
        model_type = b_kwargs.get('model_type', b_args[0] if b_args else None)
        input_sizes = b_kwargs.get('input_sizes', b_args[1] if len(b_args) > 1 else None)
        onnx_dir = b_kwargs.get('onnx_dir', b_args[2] if len(b_args) > 2 else None)
        engine_dir = b_kwargs.get('engine_dir', b_args[3] if len(b_args) > 3 else None)
        max_batch_size = b_kwargs.get('max_batch_size', b_args[4] if len(b_args) > 4 else None)
        dtype = b_kwargs.get('dtype')
        model_params = b_kwargs.get('model_params')
        onnx_name = b_kwargs.get('onnx_name', 'model.onnx')
        engine_name = b_kwargs.get('engine_name', 'model.engine')
        delete_onnx = b_kwargs.get('delete_onnx', True)
        logger = b_kwargs.get('logger', trt.Logger(trt.Logger.INFO))

        if engine_dir is None or onnx_dir is None:
            raise RuntimeError('engine_dir and onnx_dir must be provided')

        plan_path = Path(engine_dir).resolve() / args.plan_filename
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        if plan_path not in initialized_paths and plan_path.exists():
            plan_path.unlink()
        initialized_paths.add(plan_path)

        dtype_value = dtype if dtype is not None else torch.float16
        entry = {
            'model_type': model_type,
            'input_sizes': input_sizes,
            'onnx_dir': str(Path(onnx_dir).resolve()),
            'engine_dir': str(Path(engine_dir).resolve()),
            'max_batch_size': max_batch_size,
            'dtype': dtype_to_string_0323(dtype_value),
            'model_params': model_params or {},
            'onnx_name': onnx_name,
            'engine_name': engine_name,
            'delete_onnx': delete_onnx,
        }

        plan_entries = plan_cache.setdefault(plan_path, [])
        plan_entries.append(entry)
        plan_path.write_text(json.dumps(plan_entries, indent=2, ensure_ascii=False))
        logger.log(trt.Logger.INFO, f'Recorded build plan at {plan_path}')
        logger.log(trt.Logger.INFO, 'Skipping TensorRT build in step1; run step2_build_0323.py later.')
        return None

    mm_builder.build_trt_engine = record_build_plan
    try:
        builder = MultimodalEngineBuilder(args)
        builder.build()
    finally:
        mm_builder.build_trt_engine = original_build_trt

    if not plan_cache:
        raise RuntimeError('No build plan was recorded. Ensure the chosen model type is supported.')

    print('ONNX export finished. Build plans recorded in:')
    for plan_path, entries in plan_cache.items():
        print(f'  - Engine dir: {entries[0]["engine_dir"]}')
        print(f'    Plan file: {plan_path}')


if __name__ == '__main__':
    main()
