#!/usr/bin/env python3
"""
0323 实验：Step 1 导出 ONNX 并记录 TensorRT 构建计划。

先解释 `trt` 是干嘛的：
- `TRT` 一般就是 `TensorRT` 的简称
- TensorRT 是 NVIDIA 的高性能推理优化与部署引擎
- TensorRT-LLM 则是在大语言模型 / 多模态模型场景上进一步封装的一套构建与推理工具链

为什么这里要拆成 step1 / step2：
1. Step1 先做 PyTorch -> ONNX 导出，并把“将来怎么 build engine”的参数记下来
2. Step2 再单独在新进程里真正构建 TensorRT engine

这么拆的好处是：
- ONNX 导出和 engine build 分离，排障更清楚
- 某些构建过程对进程状态很敏感，拆进新进程更稳
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# 从 `trt/` 子目录运行时，让 Python 还能找到项目根目录下的配置文件。
PROJECT_ROOT_0323 = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_0323) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_0323))

import torch

from experiment_config_0323 import ensure_output_dirs_0323

# Step1 最终会在 engine 目录里写出一个 JSON 计划文件，供 Step2 读取。
DEFAULT_PLAN_FILENAME_0323 = 'build_plan_0323.json'


def make_parser_0323() -> argparse.ArgumentParser:
    """
    构造命令行解析器，并复用 TensorRT-LLM 官方 multimodal builder 的参数定义。

    这里的思路是：
    - 不自己重复发明一套 builder 参数
    - 直接借用 `add_multimodal_arguments(parser)`，让参数风格和原生 TensorRT-LLM 保持一致
    """

    try:
        from tensorrt_llm.tools.multimodal_builder import add_multimodal_arguments
    except Exception as exc:
        raise RuntimeError('step1_export_0323.py 依赖 tensorrt_llm / TensorRT 环境。') from exc

    parser = argparse.ArgumentParser(
        description='Step 1: export PyTorch models to ONNX and record params for a later TensorRT build.'
    )

    # 把 TensorRT-LLM 原生 builder 所需参数整体挂进来。
    add_multimodal_arguments(parser)

    # 额外补一个本项目自己的参数，用来控制计划文件名。
    parser.add_argument(
        '--plan-filename',
        default=DEFAULT_PLAN_FILENAME_0323,
        help='写在 engine 目录中的构建计划文件名。',
    )
    return parser


def dtype_to_string_0323(dtype: Any) -> str:
    """
    把 torch dtype 标准化成可序列化字符串。

    为什么要做这一步：
    - `torch.float16` 这种对象不能直接优雅地写进 JSON
    - Step2 需要从 JSON 里把 dtype 再读回来
    - 所以 Step1 先把它转换为字符串表示
    """

    try:
        from tensorrt_llm._utils import torch_dtype_to_str
        return torch_dtype_to_str(dtype)
    except Exception:
        # 如果 TensorRT-LLM 工具函数不可用，退化成普通字符串表示，至少保证 JSON 可写。
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

    # 这是这份脚本最关键的设计点之一：
    # 我们先把原始 `build_trt_engine` 函数保存下来。
    # 后面会暂时把它替换成“只记录计划、不真正 build”的版本。
    original_build_trt = mm_builder.build_trt_engine

    # `plan_cache` 用来缓存每个 engine 目录对应的 build 条目列表。
    # 类型写成 `Dict[Path, List[Dict[str, Any]]]`，便于阅读代码时明确数据结构。
    plan_cache: Dict[Path, List[Dict[str, Any]]] = {}

    # `initialized_paths` 用来标记哪些计划文件已经初始化过，
    # 避免第一次写时和旧文件混在一起。
    initialized_paths: set[Path] = set()

    def record_build_plan(*b_args, **b_kwargs):
        """
        这个内部函数是一个“替身函数”。

        原本 TensorRT-LLM builder 在导出 ONNX 之后，会直接调用 `build_trt_engine(...)`。
        但在 Step1，我们不想真的 build engine，只想把参数记下来。

        所以这里使用了一个经典技巧：monkey patch
        - 暂时把官方的 `build_trt_engine` 换成我们自己的函数
        - 让 builder 以为自己在正常工作
        - 实际上我们只记录参数到 JSON，不执行真正构建
        """

        # 这里同时兼容：
        # - 位置参数调用
        # - 关键字参数调用
        # 这样无论底层 builder 用哪种方式调用，我们都能把参数提出来。
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

        # 计划文件总是写到当前 engine 目录下。
        plan_path = Path(engine_dir).resolve() / args.plan_filename
        plan_path.parent.mkdir(parents=True, exist_ok=True)

        # 第一次写某个计划文件时，如果旧文件还在，先删掉，避免脏数据累积。
        if plan_path not in initialized_paths and plan_path.exists():
            plan_path.unlink()
        initialized_paths.add(plan_path)

        # 如果底层没显式传 dtype，这里默认按 float16 记。
        dtype_value = dtype if dtype is not None else torch.float16

        # 这一项 entry 就是 Step2 将来真正 build engine 所需的一条配置记录。
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

        # 同一个 engine 目录下可能有多个 engine 条目，所以用 list 追加。
        plan_entries = plan_cache.setdefault(plan_path, [])
        plan_entries.append(entry)

        # 直接把当前缓存写成 JSON 文件。
        plan_path.write_text(json.dumps(plan_entries, indent=2, ensure_ascii=False))

        logger.log(trt.Logger.INFO, f'Recorded build plan at {plan_path}')
        logger.log(trt.Logger.INFO, 'Skipping TensorRT build in step1; run step2_build_0323.py later.')

        # 返回 None 的含义是：
        # 我们这里只做记录，不产出真正的 TensorRT engine 对象。
        return None

    # 关键步骤：把官方的 build 函数临时替换为我们自己的记录函数。
    mm_builder.build_trt_engine = record_build_plan
    try:
        # `MultimodalEngineBuilder(args)` 是 TensorRT-LLM 提供的多模态 builder。
        # `builder.build()` 内部会走 ONNX 导出以及后续构建流程。
        # 由于 build 函数已被 monkey patch，真正到 build engine 那一步会被我们“截胡”。
        builder = MultimodalEngineBuilder(args)
        builder.build()
    finally:
        # 无论 build 成功还是失败，都把原始函数恢复回去。
        # 这是 monkey patch 的基本礼貌，避免污染后续进程状态。
        mm_builder.build_trt_engine = original_build_trt

    # 如果最后一条 build plan 都没记录到，多半说明：
    # - builder 参数不对
    # - 模型类型不支持
    # - 或前面的导出流程根本没走到 build 计划生成这一步
    if not plan_cache:
        raise RuntimeError('No build plan was recorded. Ensure the chosen model type is supported.')

    print('ONNX export finished. Build plans recorded in:')
    for plan_path, entries in plan_cache.items():
        print(f'  - Engine dir: {entries[0]["engine_dir"]}')
        print(f'    Plan file: {plan_path}')


if __name__ == '__main__':
    main()
