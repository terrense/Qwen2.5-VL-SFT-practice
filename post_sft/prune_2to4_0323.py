#!/usr/bin/env python3
"""
0323 实验：对 merge 后的 Qwen2.5-VL 模型执行 NVIDIA 2:4 结构化剪枝。

这里的“2:4 structured pruning”不是随便把一半权重清零，而是：
- 按 4 个权重为一组
- 每组只保留绝对值最大的 2 个
- 其余 2 个置零

这种规则的意义在于：
- 对硬件更友好
- 某些 NVIDIA 稀疏推理链路可以识别这种结构化稀疏模式
- 后续更容易和 TensorRT / 稀疏加速链路衔接
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

# 让脚本可以从 `post_sft/` 子目录直接运行，同时还能 import 根目录里的配置文件。
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
    """
    解析结构化剪枝参数。

    为什么这里把参数拆得比较细：
    - `model-path`：允许你换不同 merged 模型试验
    - `target-modules`：允许你只剪部分层
    - `prune-dim`：允许你控制按哪个维度执行 2:4
    - `dry-run`：只统计，不真正保存模型
    """

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
    """
    把“要按 4 分组的那个维度”移动到最后一维。

    为什么要这么做：
    - `reshape(-1, 4)` 最容易在“最后一维”上做分组
    - 但原始权重矩阵未必天然就是最后一维适合分组
    - 所以先 `permute()`，把目标维移到最后，再统一处理
    """

    # Python 支持负索引维度，比如 `-1` 表示最后一维。
    # 这里把负维统一换算成正维，后续逻辑更直观。
    if dim < 0:
        dim += tensor.dim()

    # `list(range(tensor.dim()))` 会得到如 `[0, 1]`、`[0, 1, 2]` 这样的维度顺序。
    permute_order = list(range(tensor.dim()))

    # `pop(dim)`：把目标维从原位置取出来。
    # `append(...)`：再把它塞到最后。
    permute_order.append(permute_order.pop(dim))

    # `permute(*permute_order)` 表示按新顺序重排张量维度。
    transposed = tensor.permute(*permute_order)
    return transposed, permute_order


def invert_permutation_0323(permute_order: List[int]) -> List[int]:
    """
    求 `permute` 的逆排列。

    直观理解：
    - 你先把维度顺序改了
    - 最后保存回模型权重前，还得把维度顺序改回去
    - 所以需要一个“逆操作”
    """

    inv = [0] * len(permute_order)
    for idx, target in enumerate(permute_order):
        inv[target] = idx
    return inv


def enforce_two_out_of_four_0323(weight: torch.Tensor, dim: int, strict: bool) -> bool:
    """
    真正执行 2:4 剪枝。

    算法逻辑：
    1. 把要分组的维度移到最后
    2. reshape 成 `(-1, 4)`，这样每一行就是一个 4 元组
    3. 计算每个 4 元组里绝对值最大的 2 个位置
    4. 构造 mask，仅保留这 2 个位置
    5. 把张量 reshape 回去，并 permute 回原始维度顺序

    返回值：
    - True：这一层真的被处理了
    - False：这一层被跳过了
    """

    # 一维张量无法做“每组 4 个”的矩阵式结构化分组，这里直接跳过。
    if weight.dim() == 1:
        return False

    # 这里直接操作 `weight.data`，意味着原地修改参数值。
    # 这类离线模型处理脚本一般就是要真正把权重改掉，所以这样是合理的。
    transposed, permute_order = chunk_tensor_0323(weight.data, dim)

    # 拿到最后一维长度，判断能否被 4 整除。
    last_dim = transposed.size(-1)
    remainder = last_dim % 4
    if remainder != 0:
        msg = f'skip layer: size along dim {dim} is {last_dim}, cannot form 2:4 groups'
        if strict:
            raise ValueError(msg)
        print(f'[WARN] {msg}')
        return False

    # `reshape(-1, 4)` 的意思是：
    # - 前面所有维度压平
    # - 最后一维按 4 一组排列
    view = transposed.reshape(-1, 4)

    # 剪枝通常看绝对值大小，因为我们关心的是“权重贡献强弱”，不关心正负号。
    abs_view = view.abs()

    # `topk(..., k=2)`：取每组里绝对值最大的 2 个元素下标。
    keep_idx = torch.topk(abs_view, k=2, dim=-1, largest=True).indices

    # 先构造一个全 False 的布尔 mask。
    mask = torch.zeros_like(view, dtype=torch.bool)

    # `scatter_`：按 `keep_idx` 把对应位置设成 True。
    # `_` 结尾表示“原地操作”，即直接改当前张量，不返回新副本。
    mask.scatter_(1, keep_idx, True)

    # 把 False 位置乘成 0，True 位置保持原值。
    view.mul_(mask.to(view.dtype))

    # 剪完以后再恢复原始形状。
    transposed = view.reshape(transposed.shape)
    weight.data = transposed.permute(*invert_permutation_0323(permute_order))
    return True


def iter_target_modules_0323(model, substrings: Iterable[str]):
    """
    遍历模型里的目标线性层。

    这里为什么只处理 `torch.nn.Linear`：
    - 2:4 稀疏通常主要作用在线性层权重矩阵
    - 注意力投影层、MLP 投影层都属于这一类
    """

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        # `any(...)` 语义是：只要有一个子串命中，就认为这个模块是目标层。
        if substrings and not any(token in name for token in substrings):
            continue
        yield name, module


def main() -> None:
    args = parse_args_0323()
    ensure_output_dirs_0323()

    # 把逗号分隔字符串变成列表，并去掉每个 token 两端空白。
    substrings = [token.strip() for token in args.target_modules.split(',') if token.strip()]

    # 根据命令行选择的精度，映射到 torch dtype。
    torch_dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16

    # `device_map` 控制模型加载到哪。
    # - None：让模型先留在默认设备/CPU
    # - 'auto'：交给 HF / accelerate 自动切分
    # - {'': 0}：把整个模型映射到 cuda:0
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

    # 逐层遍历目标模块，统计总层数与成功剪枝层数。
    for name, module in iter_target_modules_0323(model, substrings):
        total += 1
        changed = enforce_two_out_of_four_0323(module.weight, dim=args.prune_dim, strict=args.strict)
        if changed:
            pruned += 1
            print(f'[OK] Pruned {name}')
        else:
            print(f'[SKIP] {name}')

    print(f'Finished pruning: {pruned}/{total} Linear layers modified.')

    # dry-run 模式只用于“预检查”：
    # - 看哪些层会被处理
    # - 看是否有层因为尺寸不满足 4 对齐而被跳过
    if args.dry_run:
        print('Dry run requested; skipping save.')
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Saving pruned checkpoint to {args.output_dir} ...')

    # 保存前把模型挪回 CPU，一方面更稳，另一方面可以释放 GPU 显存。
    model.to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # `save_pretrained` 会按 HuggingFace 模型目录格式落盘。
    model.save_pretrained(str(args.output_dir))
    print('Done.')


if __name__ == '__main__':
    main()
