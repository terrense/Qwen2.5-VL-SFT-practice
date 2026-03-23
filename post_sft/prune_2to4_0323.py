#!/usr/bin/env python3
"""
0323 实验：对 merge 后的 Qwen2.5-VL 模型执行 NVIDIA 2:4 结构化剪枝。
输出模型可继续做剪枝后评测、量化或 TensorRT 构建。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT_0323 = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_0323) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_0323))

import torch
from transformers import Qwen2_5_VLForConditionalGeneration

from experiment_config_0323 import (
    LORA_TARGET_MODULES_0323,
    MERGED_MODEL_ROOT_0323,
    PRUNED_MODEL_ROOT_0323,
    ensure_output_dirs_0323,
)


def parse_args_0323() -> argparse.Namespace:
    """解析结构化剪枝参数。"""
    parser = argparse.ArgumentParser(
        description='Apply NVIDIA 2:4 structured pruning to the merged 0323 Qwen2.5-VL checkpoint.'
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        default=MERGED_MODEL_ROOT_0323,
        help='待剪枝的 merge 后模型目录。',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PRUNED_MODEL_ROOT_0323,
        help='剪枝后模型输出目录。',
    )
    parser.add_argument(
        '--target-modules',
        default=','.join(LORA_TARGET_MODULES_0323),
        help='逗号分隔的模块名子串；只有命中这些子串的 Linear 层会参与 2:4 剪枝。',
    )
    parser.add_argument(
        '--prune-dim',
        type=int,
        default=1,
        help='沿哪个维度执行 2:4；默认 1，即 Linear 输入维。',
    )
    parser.add_argument(
        '--dtype',
        choices=['bf16', 'fp16'],
        default='bf16',
        help='加载权重的精度，通常推荐 bf16。',
    )
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='模型加载设备；auto 允许多卡切分。',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='只扫描并统计将要剪枝的层，不写出模型。',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='若维度不能被 4 整除则报错；默认仅 warning 并跳过。',
    )
    return parser.parse_args()


def chunk_tensor_0323(tensor: torch.Tensor, dim: int):
    """把待剪枝维移动到最后一维，方便 reshape 成 (..., 4) 的 group。"""
    if dim < 0:
        dim += tensor.dim()
    permute_order = list(range(tensor.dim()))
    permute_order.append(permute_order.pop(dim))
    transposed = tensor.permute(*permute_order)
    return transposed, permute_order


def invert_permutation_0323(permute_order: List[int]) -> List[int]:
    """求 permute 的逆排列，用于把张量转回原始维度顺序。"""
    inv = [0] * len(permute_order)
    for idx, target in enumerate(permute_order):
        inv[target] = idx
    return inv


def enforce_two_out_of_four_0323(weight: torch.Tensor, dim: int, strict: bool) -> bool:
    """
    对每组 4 个权重保留绝对值最大的 2 个，置零其余 2 个。
    这是典型的 NVIDIA 2:4 structured sparsity 约束。
    """
    if weight.dim() == 1:
        return False

    transposed, permute_order = chunk_tensor_0323(weight.data, dim)
    last_dim = transposed.size(-1)
    remainder = last_dim % 4
    if remainder != 0:
        msg = f'skip layer: size along dim {dim} is {last_dim}, cannot form 2:4 groups'
        if strict:
            raise ValueError(msg)
        print(f'[WARN] {msg}')
        return False

    view = transposed.reshape(-1, 4)
    abs_view = view.abs()
    keep_idx = torch.topk(abs_view, k=2, dim=-1, largest=True).indices
    mask = torch.zeros_like(view, dtype=torch.bool)
    mask.scatter_(1, keep_idx, True)
    view.mul_(mask.to(view.dtype))

    transposed = view.reshape(transposed.shape)
    weight.data = transposed.permute(*invert_permutation_0323(permute_order))
    return True


def iter_target_modules_0323(model, substrings: Iterable[str]):
    """遍历模型中的 Linear 层，并按名字子串筛选目标模块。"""
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if substrings and not any(token in name for token in substrings):
            continue
        yield name, module


def main() -> None:
    args = parse_args_0323()
    ensure_output_dirs_0323()

    substrings = [token.strip() for token in args.target_modules.split(',') if token.strip()]
    torch_dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16
    device_map = None
    if args.device == 'auto':
        device_map = 'auto'
    elif args.device == 'cuda':
        device_map = {'': 0}

    print(f'Loading merged model from {args.model_path} ...')
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_path),
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    total = 0
    pruned = 0
    for name, module in iter_target_modules_0323(model, substrings):
        total += 1
        changed = enforce_two_out_of_four_0323(module.weight, dim=args.prune_dim, strict=args.strict)
        if changed:
            pruned += 1
            print(f'[OK] Pruned {name}')
        else:
            print(f'[SKIP] {name}')

    print(f'Finished pruning: {pruned}/{total} Linear layers modified.')

    if args.dry_run:
        print('Dry run requested; skipping save.')
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Saving pruned checkpoint to {args.output_dir} ...')
    model.to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.save_pretrained(str(args.output_dir))
    print('Done.')


if __name__ == '__main__':
    main()
