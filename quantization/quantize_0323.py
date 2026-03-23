#!/usr/bin/env python3
"""
0323 实验：量化导出脚本。

“量化”可以粗略理解为：把模型权重和某些中间数值表示，从更高精度压缩到更低精度，
以换取更小显存、更快推理、更适合后续部署引擎。

当前脚本支持的量化格式：
- `int4_awq`
- `fp8`

典型用途：
- 给 TensorRT-LLM / TensorRT 部署链路准备量化产物
- 作为后续 engine build 的输入之一

注意：
这条量化链路依赖 `tensorrt_llm` 及其运行环境，它不是普通 HuggingFace 训练环境里的常规步骤。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# 从子目录运行时，手动把项目根目录加入 Python 模块搜索路径。
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
    """
    解析量化导出参数。

    关键参数说明：
    - `model-dir`：通常指向 merge 后完整模型目录
    - `dtype`：量化前模型用什么精度加载
    - `qformat`：量化格式
    - `calib-size`：校准样本数量
    - `export-path`：量化产物导出位置
    """

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


def get_calib_dataloader_0323(tokenizer, batch_size: int = 1, calib_size: int = 32, block_size: int = 512):
    """
    构造一个轻量校准数据集。

    为什么量化需要“校准数据”：
    - 低比特量化并不是简单粗暴地把 float 直接截断
    - 往往需要观察一批真实输入的激活分布
    - 从而估计合适的缩放范围、量化区间等

    这里沿用原始脚本思路，使用 `cnn_dailymail` 文本作为校准语料。
    它不一定和你的业务数据完全一致，但作为 PTQ/AWQ 示例足够常见。
    """

    print('Loading calibration dataset ...')

    # `load_dataset(...)` 来自 HuggingFace Datasets。
    # `split='train'` 表示取训练集分片。
    dataset = load_dataset('ccdv/cnn_dailymail', name='3.0.0', split='train')

    # 这里只取文本字段 `article` 的前 `calib_size` 条，避免校准集太大拖慢量化。
    dataset = dataset['article'][:calib_size]

    # 某些 tokenizer 没定义 pad token，会导致 batch padding 出问题。
    # 对生成式模型，常见做法是把 eos_token 兼作 pad_token。
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    processed = []
    for text in dataset:
        # 追加一个非常短的提示尾巴，和原脚本风格保持一致。
        # `.strip()` 去掉两端空白，`.replace(...)` 修正某些 tokenization 细节。
        text = (text + ' TL;DR: ').strip().replace(" n't", "n't")
        processed.append(text)

    # `batch_encode_plus(...)` 会把一组文本转成 token id 张量。
    batch_encoded = tokenizer.batch_encode_plus(
        processed,
        return_tensors='pt',
        padding=True,
        max_length=block_size,
        truncation=True,
    )

    # 量化校准通常直接在 GPU 上跑，所以这里把输入 tensor 放到 CUDA。
    batch_encoded = batch_encoded['input_ids'].cuda()

    # `DataLoader` 可以把整个校准集组织成一个可迭代对象，供量化函数逐 batch 使用。
    return DataLoader(batch_encoded, batch_size=batch_size, shuffle=False)


def main() -> None:
    args = parse_args_0323()
    ensure_output_dirs_0323()

    try:
        # 这里把 TensorRT-LLM 相关 import 放在 main() 里，而不是文件顶层。
        # 好处是：
        # - 没装部署环境时，单纯 import 这个 Python 文件不会立刻炸掉
        # - 只有真正执行量化时，才要求这些依赖存在
        from tensorrt_llm.models.quantized.ammo import quantize_and_export
        from tensorrt_llm._utils import str_dtype_to_torch
        from transformers import AutoModelForCausalLM
    except Exception as exc:
        raise RuntimeError(
            'quantize_0323.py 依赖 tensorrt_llm / TensorRT-LLM 环境，请先完成相关安装。'
        ) from exc

    # 量化导出通常默认假设需要 GPU；没有 GPU 就没必要继续了。
    if not torch.cuda.is_available():
        raise EnvironmentError('GPU is required for quantization export.')

    # 如果用户传了随机种子，就同时固定 Python / NumPy / PyTorch 的随机性来源。
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f'Model directory not found: {model_dir}')

    # 如果用户没自己给导出路径，就按量化格式走默认路径：
    # - int4_awq -> AWQ_EXPORT_PATH_0323
    # - fp8 -> FP8_EXPORT_PATH_0323
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

    # `str_dtype_to_torch(...)` 会把诸如 `'float16'` 这样的字符串映射成 `torch.float16`。
    torch_dtype = str_dtype_to_torch(args.dtype)

    # 这里加载的是一个标准 HuggingFace 因果语言模型接口。
    # 对于后续量化函数而言，重点是“能做前向推理”，不要求它还保留训练态。
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.eval()

    # `channels_last` 常用于优化某些 CUDA kernel / 内存访问模式。
    # 是否一定带来收益依赖具体后端实现，但这也是原始链路中常见的一步。
    model = model.to(memory_format=torch.channels_last)

    calib_dataloader = get_calib_dataloader_0323(
        tokenizer=tokenizer,
        calib_size=args.calib_size,
    )

    print(f'Exporting quantized model ({args.qformat}) to {export_path} ...')

    # `quantize_and_export(...)` 是这份脚本最核心的一步：
    # - 读入模型与校准数据
    # - 执行量化
    # - 直接把量化结果导出到指定路径
    quantize_and_export(
        model,
        qformat=args.qformat,
        calib_dataloader=calib_dataloader,
        export_path=str(export_path),
    )
    print('Quantization export finished.')


if __name__ == '__main__':
    main()
