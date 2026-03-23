#!/usr/bin/env python3
"""
0323 实验：将基座模型与 LoRA adapter 合并为完整 HuggingFace 模型。

为什么要做这一步：
1. LoRA 训练后，磁盘上通常只有 adapter 权重，而不是完整模型权重。
2. HuggingFace + PEFT 在推理时可以“基座模型 + adapter”联动加载，但很多后续链路并不喜欢这种双件套。
3. 例如结构化剪枝、量化、TensorRT-LLM、离线部署，通常更希望拿到一个“已经合并好的完整模型目录”。

这份脚本的输出就是一个可独立加载的模型目录，后面可以直接给：
- `post_sft/prune_2to4_0323.py`
- `quantization/quantize_0323.py`
- `trt/step1_export_0323.py`
- `deploy/deploy_api_0323.py`
"""

# `from __future__ import annotations` 的作用：
# 让类型注解延迟求值。简单理解就是：
# - 写 `-> argparse.Namespace`、`list[str]` 之类注解时更灵活
# - 避免某些场景下“函数还没完全定义、类型就被立刻解析”的问题
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# `__file__` 是当前脚本文件路径。
# `resolve()` 会把相对路径、软链接等都解析成绝对路径。
# `parents[1]` 表示：
# - 先取当前文件所在目录 `post_sft/`
# - 再回到上一级项目根目录 `qwen25vl_sft_chain_0323/`
PROJECT_ROOT_0323 = Path(__file__).resolve().parents[1]

# 这里手动把项目根目录塞进 `sys.path`。
# 原因是：这个脚本放在 `post_sft/` 子目录下，如果直接 `python post_sft/merge_lora_0323.py`，
# Python 默认未必能找到项目根目录里的 `experiment_config_0323.py`。
# `sys.path` 可以理解为 Python 的“模块搜索路径列表”。
if str(PROJECT_ROOT_0323) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_0323))

import torch
from modelscope import AutoTokenizer
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# 统一从配置中心导入路径，而不是在脚本里写死。
# 这样后续换目录、换模型时，只需优先改 `experiment_config_0323.py`。
from experiment_config_0323 import (
    BASE_MODEL_PATH_0323,
    MERGED_MODEL_ROOT_0323,
    RUN_ROOT_0323,
    ensure_output_dirs_0323,
)


def parse_args_0323() -> argparse.Namespace:
    """
    使用 `argparse` 解析命令行参数。

    这里返回的是 `argparse.Namespace` 对象，你可以把它理解为：
    一个“属性风格”的参数容器，例如 `args.base_model`、`args.dtype`。
    """

    parser = argparse.ArgumentParser(
        description='Merge Qwen2.5-VL base weights with a LoRA checkpoint for the 0323 experiment.'
    )

    # `type=Path` 的含义：
    # argparse 会自动把命令行字符串转成 `pathlib.Path` 对象，
    # 后面就可以直接用 `.exists()`、`.resolve()`、`/` 拼路径等方法。
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

    # `choices` 可以限制用户只能传这三个值，传别的 argparse 会直接报错。
    # 这是很有用的参数校验手段。
    parser.add_argument(
        '--dtype',
        choices=['bfloat16', 'float16', 'float32'],
        default='bfloat16',
        help='加载基座模型时使用的精度。一般建议 bfloat16。',
    )

    # `device_map` 是 HuggingFace 加载大模型时的关键参数：
    # - `auto`：让 transformers / accelerate 自动分配到 GPU / CPU
    # - `cuda:0`：强制全部放到单卡
    # - 某些高级场景还能传字典做细粒度映射
    parser.add_argument(
        '--device-map',
        default='auto',
        help='传给 from_pretrained 的 device_map；单卡可填 cuda:0，多卡推荐 auto。',
    )

    # `action="store_true"` 的语义：
    # - 命令行里不写这个参数，值就是 False
    # - 命令行里写了 `--no-safe-serialization`，值就变成 True
    parser.add_argument(
        '--no-safe-serialization',
        action='store_true',
        help='若指定则保存为 PyTorch .bin；默认保存为 safetensors。',
    )
    return parser.parse_args()


def main() -> None:
    # 先解析命令行参数。
    args = parse_args_0323()

    # 提前创建项目里约定好的输出目录，防止后续写文件时报“路径不存在”。
    ensure_output_dirs_0323()

    # `expanduser()`：把 `~` 展开为用户 home 目录。
    # `resolve()`：转成绝对路径，便于日志和后续操作稳定。
    base_path = args.base_model.expanduser().resolve()
    lora_path = args.lora_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    # 用存在性校验尽早失败，避免后面加载模型时抛出更难懂的深层异常。
    if not base_path.exists():
        sys.exit(f'Base model path not found: {base_path}')
    if not lora_path.exists():
        sys.exit(f'LoRA checkpoint path not found: {lora_path}')

    # `parents=True`：如果父目录不存在就递归创建。
    # `exist_ok=True`：如果目录已存在，不要报错。
    output_dir.mkdir(parents=True, exist_ok=True)

    # 把字符串形式的 dtype 参数映射成 PyTorch 真实的 dtype 对象。
    # 后续 `from_pretrained` 需要的是 `torch.bfloat16` 这类对象，不是普通字符串。
    torch_dtype = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
    }[args.dtype]

    print(f'[1/4] Loading base model from {base_path} (dtype={args.dtype}, device_map={args.device_map}) ...')

    # `from_pretrained(...)` 是 HuggingFace 加载模型的标准入口。
    # 这里加载的是 Qwen2.5-VL 的“完整基座模型”。
    #
    # 参数说明：
    # - `str(base_path)`：很多 HF 接口接受字符串路径，这里显式转成 str 更稳妥。
    # - `dtype=torch_dtype`：控制权重加载精度，影响显存占用和数值表现。
    # - `trust_remote_code=True`：允许模型仓库提供自定义实现。
    # - `device_map=args.device_map`：决定模型怎么放到 GPU / CPU。
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(base_path),
        dtype=torch_dtype,
        trust_remote_code=True,
        device_map=args.device_map,
    )

    print(f'[2/4] Attaching LoRA weights from {lora_path} ...')

    # `PeftModel.from_pretrained` 的作用不是重新训练，而是：
    # 把磁盘上的 LoRA adapter 权重“挂载”到刚才的 base_model 上。
    #
    # 这里的返回对象仍然像一个模型，但内部结构已经变成：
    # 原始层 + LoRA 增量参数
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(lora_path),
        torch_dtype=torch_dtype,
        is_trainable=False,
    )

    print('[3/4] Merging LoRA parameters into the base model ...')

    # `merge_and_unload()` 是这份脚本最核心的一步：
    # - 把 LoRA 的低秩增量真正加回原始权重矩阵
    # - 返回一个不再依赖 adapter 文件的完整模型
    # - “unload” 表示把 PEFT 包装层卸掉，得到更普通的 HF 模型对象
    merged_model = peft_model.merge_and_unload()

    # `eval()` 把模型切到推理模式：
    # - dropout 等训练期行为会被关闭
    # - 这是保存推理模型前的好习惯
    merged_model.eval()

    # HuggingFace 默认更推荐 `safetensors`，因为：
    # - 更安全
    # - 读写通常更稳定
    # - 社区生态支持度也很好
    safe_serialization = not args.no_safe_serialization
    print(f'[4/4] Saving merged model to {output_dir} (safetensors={safe_serialization}) ...')
    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=safe_serialization,
    )

    # 仅保存模型权重还不够。
    # 一个“可独立复用”的模型目录通常还应包含：
    # - tokenizer：文本分词规则
    # - processor：多模态场景下的图像/文本预处理配置
    #
    # 否则后续推理、部署、评测脚本虽然有模型权重，却不知道怎么把输入转成模型要的张量格式。
    print('Saving tokenizer and processor metadata ...')

    # `use_fast=False` 常用于尽量避开某些 fast tokenizer / fast processor 的兼容问题。
    tokenizer = AutoTokenizer.from_pretrained(str(base_path), use_fast=False, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))

    processor = AutoProcessor.from_pretrained(str(base_path), trust_remote_code=True)
    processor.save_pretrained(str(output_dir))

    print('Done. Merged model is available at:')
    print(f'    {output_dir}')


# Python 文件直接运行时，`__name__` 会等于 `'__main__'`。
# 如果这个文件是被别的模块 `import` 进来的，则不会执行下面的 main()。
# 这是 Python 脚本最常见的入口写法。
if __name__ == '__main__':
    main()
