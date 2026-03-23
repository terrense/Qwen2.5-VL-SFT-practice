#!/usr/bin/env python3
"""
0323 实验：Step 2 读取 build_plan_0323.json 并真正执行 TensorRT engine 构建。

Step1 和 Step2 的关系可以这样理解：

- Step1：把“怎么 build”写成计划单
- Step2：按照计划单真正开工

因此 Step2 本身不再负责推断 builder 参数，而是尽量忠实执行 Step1 记录下来的内容。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# 让脚本在 `trt/` 子目录下也能顺利 import 项目其他模块。
PROJECT_ROOT_0323 = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_0323) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_0323))

import torch

DEFAULT_PLAN_FILENAME_0323 = 'build_plan_0323.json'


def parse_args_0323() -> argparse.Namespace:
    """
    解析 TensorRT 构建参数。

    两种用法：
    - `--plan-path /abs/path/to/build_plan_0323.json`
    - `--output-dir /abs/path/to/engine_dir`

    如果只给 `output-dir`，脚本会自动在该目录下寻找默认文件名 `build_plan_0323.json`。
    """

    parser = argparse.ArgumentParser(
        description='Step 2: build TensorRT engines from build_plan_0323.json in a fresh process.'
    )
    parser.add_argument('--plan-path', type=str, help='显式指定 build_plan_0323.json 路径。')
    parser.add_argument('--output-dir', type=str, help='当未指定 --plan-path 时，使用该 engine 目录下的默认计划文件。')
    parser.add_argument('--keep-onnx', action='store_true', help='若指定则保留 ONNX 文件。')
    args = parser.parse_args()

    # `parser.error(...)` 会打印友好错误信息并退出，比我们手写 raise 更符合 argparse 风格。
    if not args.plan_path and not args.output_dir:
        parser.error('Provide either --plan-path or --output-dir')
    return args


def normalize_dtype_0323(value: Any) -> torch.dtype:
    """
    把 JSON 中保存的 dtype 字符串恢复成 `torch.dtype`。

    为什么要恢复：
    - Step1 里为了写 JSON，把 dtype 存成了字符串
    - 但 Step2 真正调用 `build_trt_engine(...)` 时，需要的是 `torch.float16` 这类对象
    """

    if value is None:
        return torch.float16

    if isinstance(value, str):
        # 统一做小写和 `torch.` 前缀清洗，兼容更多字符串写法。
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

        # 有些 dtype 名称可能正好和 torch 属性同名，这里再做一层兜底兼容。
        if hasattr(torch, lowered):
            candidate = getattr(torch, lowered)
            if isinstance(candidate, torch.dtype):
                return candidate

    # 如果传进来的本来就是 torch.dtype，则原样返回。
    if isinstance(value, torch.dtype):
        return value

    raise ValueError(f'Unsupported dtype representation: {value}')


def load_plan_0323(plan_path: Path) -> List[Dict[str, Any]]:
    """
    读取 Step1 产出的 build 计划文件。

    计划文件的预期结构是一个列表，每个元素都是一条 engine build 条目。
    """

    if not plan_path.exists():
        raise FileNotFoundError(f'Plan file not found: {plan_path}')

    # `read_text()` 会直接读取整个文本文件内容。
    data = json.loads(plan_path.read_text())

    if not isinstance(data, list):
        raise ValueError('Plan file must contain a list of build entries')
    return data


def main() -> None:
    try:
        # Step2 只需要一个核心函数：`build_trt_engine(...)`
        # 因为参数已经在 Step1 里记录好了，这里不再需要完整 builder 体系。
        from tensorrt_llm.tools.multimodal_builder import build_trt_engine
    except Exception as exc:
        raise RuntimeError('step2_build_0323.py 依赖 tensorrt_llm / TensorRT 环境。') from exc

    args = parse_args_0323()

    # 三元表达式写法：
    # `A if condition else B`
    # 这里的意思是：
    # - 如果显式给了 `--plan-path`，就直接用它
    # - 否则就在 `--output-dir` 下拼出默认计划文件路径
    plan_path = Path(args.plan_path).expanduser().resolve() if args.plan_path else (Path(args.output_dir).expanduser().resolve() / DEFAULT_PLAN_FILENAME_0323)

    plan_entries = load_plan_0323(plan_path)
    if not plan_entries:
        raise RuntimeError(f'No build entries stored in {plan_path}')

    print(f'Loaded {len(plan_entries)} build entries from {plan_path}')
    for idx, entry in enumerate(plan_entries, start=1):
        # 每条 entry 里的 dtype 是字符串，这里先恢复成 torch dtype。
        dtype = normalize_dtype_0323(entry.get('dtype'))

        # 如果用户指定 `--keep-onnx`，则覆盖计划文件里的 `delete_onnx=True`。
        delete_onnx = entry.get('delete_onnx', True) and not args.keep_onnx

        print(f'[{idx}/{len(plan_entries)}] Building engine {entry.get("engine_name", "model.engine")}')

        # 这里就是真正的 TensorRT engine 构建入口。
        # 参数几乎都来自 Step1 记录的 JSON，保证“导出阶段”和“构建阶段”参数一致。
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
