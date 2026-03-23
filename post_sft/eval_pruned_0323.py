#!/usr/bin/env python3
"""
0323 实验：对 merge 后或 2:4 结构化剪枝后的完整模型做验证集评测。

它和 `eval_lora_0323.py` 的主要区别是：
- `eval_lora_0323.py` 评测的是“基座模型 + LoRA adapter”
- 这份脚本评测的是“已经导出成完整目录的模型”

所以它适用于：
- merge 后模型评测
- prune 后模型评测
- 任何“完整 HuggingFace 目录格式”的 checkpoint 评测
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from pprint import pprint

# 加入项目根目录，保证脚本从 `post_sft/` 直接运行时也能 import 配置文件。
PROJECT_ROOT_0323 = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_0323) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_0323))

import torch
from qwen_vl_utils import process_vision_info
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from experiment_config_0323 import (
    BASE_MODEL_PATH_0323,
    MAX_NEW_TOKENS_0323,
    PRUNED_EVAL_ROOT_0323,
    PRUNED_MODEL_ROOT_0323,
    SOURCE_ROOT_0323,
    VAL_JSON_PATH_0323,
    ensure_output_dirs_0323,
    resolve_image_path_0323,
)

# 这里把设备对象和设备字符串都提前准备好。
# 原因是：
# - `.to(DEVICE_0323)` 需要 `torch.device`
# - `device_map={"": DEVICE_STR_0323}` 这类接口更适合字符串形式
DEVICE_0323 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE_STR_0323 = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 用模块级全局变量保存模型和 processor，便于 `predict_image_0323()` 直接访问。
MODEL_0323 = None
PROCESSOR_0323 = None


def parse_args_0323() -> argparse.Namespace:
    """
    解析评测参数。

    `--model-path`：
    - 默认评测 `PRUNED_MODEL_ROOT_0323`
    - 也可以切换到 merged 模型目录

    `--tag`：
    - 用于控制输出文件名前缀
    - 便于同一个 `reports/pruned_eval_0323/` 目录下保存多份评测结果
    """

    parser = argparse.ArgumentParser(
        description='Evaluate a merged or pruned Qwen2.5-VL checkpoint on the 0323 validation set.'
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        default=PRUNED_MODEL_ROOT_0323,
        help='待评测的完整模型目录；默认使用 PRUNED_MODEL_ROOT_0323。',
    )
    parser.add_argument(
        '--tag',
        type=str,
        default='pruned_model_0323',
        help='输出文件名前缀 tag。',
    )
    return parser.parse_args()


def strip_code_fences_0323(text: str) -> str:
    """
    去掉模型输出外层 markdown 代码块，便于 `json.loads(...)`。

    为什么需要它：
    模型有时会输出这种格式：

    ```json
    {"pain_status": "no_pain", ...}
    ```

    而 `json.loads` 只能吃纯 JSON 文本，不能吃 markdown fence。
    """

    # `^` 匹配行首，`$` 匹配行尾。
    # `\s*` 表示“0 个或多个空白字符”。
    text = re.sub(r'^```[a-zA-Z]*\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text.strip())
    return text.strip()


def predict_image_0323(sample, use_raw_path: bool = False) -> str:
    """
    对单条图文样本做推理，返回模型生成的 JSON 文本。

    参数解释：
    - `sample`：验证集中的一条样本
    - `use_raw_path`：
      - True 时直接使用数据里原始相对路径
      - False 时使用 `resolve_image_path_0323()` 解析成绝对路径

    之所以保留这个选项，是为了兼容原始数据目录运行方式。
    """

    # 按当前数据格式，用户输入 prompt 在 `conversations[0]["value"]`。
    prompt = sample['conversations'][0]['value']

    # 原始 prompt 里包含 `<|vision_start|>图片路径<|vision_end|>`。
    # 这里把中间那段路径截出来。
    raw_path = prompt.split('<|vision_start|>')[1].split('<|vision_end|>')[0]

    # 这里控制是否使用原始相对路径，还是转成绝对路径。
    file_path = raw_path if use_raw_path else resolve_image_path_0323(raw_path)

    # 真正给模型的纯文本 prompt 里，不再保留那段特殊的图片路径标记。
    new_prompt = prompt.replace(f'<|vision_start|>{raw_path}<|vision_end|>', '').strip()

    # Qwen-VL processor 期望的是“结构化 message”格式，而不是简单字符串。
    messages = [
        {
            'role': 'user',
            'content': [
                # `resized_height/width` 会影响图像预处理尺寸。
                {'type': 'image', 'image': file_path, 'resized_height': 480, 'resized_width': 320},
                {'type': 'text', 'text': new_prompt},
            ],
        }
    ]

    # `apply_chat_template` 会把结构化 messages 转成模型实际要吃的文本模板。
    text = PROCESSOR_0323.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # `process_vision_info` 会把多模态 message 里的图像/视频信息拆出来，供 processor 使用。
    image_inputs, video_inputs = process_vision_info(messages)

    # `PROCESSOR_0323(...)` 会把文本、图像一起转成模型输入张量。
    inputs = PROCESSOR_0323(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt',
    ).to(DEVICE_0323)

    # `torch.no_grad()` 表示关闭梯度记录：
    # - 省显存
    # - 更适合纯推理
    with torch.no_grad():
        generated_ids = MODEL_0323.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS_0323)

    # `generate()` 的返回结果包含 prompt 对应的输入 token + 新生成 token。
    # 所以这里按输入长度截掉前面的 prompt 部分，只保留模型新生成内容。
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = PROCESSOR_0323.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text.strip()


def process_result_0323(samples):
    """
    汇总整体评测结果。

    这一步会统计：
    - 每个字段的准确率
    - 四字段全对的 `exact_match`
    - JSON 解析失败率
    - 混淆矩阵
    - 错误样本列表
    """

    fields = ['pain_status', 'facial_asymmetry', 'gender', 'levine_sign']

    # `correct`：记录每个字段预测正确的样本数
    correct = {k: 0 for k in fields}

    # `incorrect_samples`：按字段收集错误样本，便于后续分析
    incorrect_samples = {f: [] for f in fields}

    exact_match = 0
    total_count = 0
    json_error_count = 0

    # 嵌套 defaultdict 的好处是：
    # 不需要先判断 key 是否存在，直接 `+= 1` 即可。
    confusion = {f: defaultdict(lambda: defaultdict(int)) for f in fields}

    for sample in samples:
        total_count += 1

        # 统一转成小写并去掉 markdown fence，提升 JSON 解析稳健性。
        label = strip_code_fences_0323(sample['conversations'][1]['value'].strip().lower())
        response = strip_code_fences_0323(sample.get('response', '').strip().lower())

        prompt = sample['conversations'][0]['value']
        raw_path = prompt.split('<|vision_start|>')[1].split('<|vision_end|>')[0]
        file_path = resolve_image_path_0323(raw_path)

        try:
            pred_json = json.loads(response)
            label_json = json.loads(label)
        except Exception:
            # 一旦模型输出连 JSON 都解析不了，说明这条样本无法进入字段级比较。
            json_error_count += 1
            continue

        all_ok = True
        for field in fields:
            true_val = label_json.get(field, '')
            pred_val = pred_json.get(field, '')

            # 混淆矩阵计数：真实标签 true_val，被预测为 pred_val。
            confusion[field][true_val][pred_val] += 1

            if pred_val == true_val:
                correct[field] += 1
            else:
                all_ok = False
                incorrect_samples[field].append(
                    {
                        'file_path': file_path,
                        'true_label': true_val,
                        'predicted_label': pred_val,
                    }
                )
        if all_ok:
            exact_match += 1

    def _to_dict(obj):
        """
        递归把 `defaultdict` 转成普通 `dict`。

        原因：
        `json.dump(...)` 对普通 dict 很友好，但对嵌套 defaultdict 未必稳定。
        """

        if isinstance(obj, defaultdict):
            return {k: _to_dict(v) for k, v in obj.items()}
        return obj

    # 这里准确率的分母仍然用 total_count。
    # 也就是说，JSON 解析失败样本会体现在 `json_error_rate` 里，
    # 同时对应字段正确率也不会增加。
    metrics = {f: correct[f] / total_count if total_count else 0 for f in fields}
    metrics['exact_match'] = exact_match / total_count if total_count else 0
    metrics['json_error_rate'] = json_error_count / total_count if total_count else 0
    metrics['correct_counts'] = correct
    metrics['total_count'] = total_count
    metrics['confusion_matrices'] = _to_dict(confusion)
    metrics['incorrect_samples'] = incorrect_samples
    return metrics


def evaluate_metrics_0323(samples, field):
    """
    对单个字段输出更细的 sklearn 指标。

    返回内容包括：
    - labels
    - confusion_matrix
    - classification_report
    """

    y_true, y_pred = [], []
    for sample in samples:
        label = strip_code_fences_0323(sample['conversations'][1]['value'].strip().lower())
        response = strip_code_fences_0323(sample.get('response', '').strip().lower())
        try:
            pred_json = json.loads(response)
            label_json = json.loads(label)
        except Exception:
            continue
        y_true.append(label_json.get(field, ''))
        y_pred.append(pred_json.get(field, ''))

    # 如果所有样本都没法形成有效 JSON，这里返回可读提示，而不是让 sklearn 再报错。
    if not y_true and not y_pred:
        return {'field': field, 'labels': [], 'confusion_matrix': [], 'classification_report': 'No valid samples'}

    labels = sorted(list(set(y_true + y_pred)))
    return {
        'field': field,
        'labels': labels,
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        'classification_report': classification_report(
            y_true,
            y_pred,
            digits=3,
            labels=labels,
            zero_division=0,
            output_dict=True,
        ),
    }


def main() -> None:
    # `global` 的作用是告诉 Python：
    # 下面赋值的 `MODEL_0323`、`PROCESSOR_0323` 是模块级全局变量，不是函数局部变量。
    global MODEL_0323, PROCESSOR_0323

    args = parse_args_0323()
    ensure_output_dirs_0323()

    model_path = args.model_path.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f'Model path not found: {model_path}')

    # 结果放在 `reports/pruned_eval_0323/<tag>/` 下，便于多次实验并存。
    output_root = PRUNED_EVAL_ROOT_0323 / args.tag
    output_root.mkdir(parents=True, exist_ok=True)

    print(f'Loading full model from: {model_path}')
    MODEL_0323 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype='auto',
        device_map={'': DEVICE_STR_0323},
        trust_remote_code=True,
    )

    # `use_cache=True` 对生成式推理通常更友好，能够复用 KV cache。
    MODEL_0323.config.use_cache = True

    # 多模态场景里，processor 非常关键。
    # 优先从当前评测模型目录加载，原因是：
    # - merged 模型目录通常已经带有 tokenizer / processor 配置
    # - 如果它没有这些配置，再退回 base model 目录
    processor_source = model_path if (model_path / 'preprocessor_config.json').exists() else BASE_MODEL_PATH_0323
    PROCESSOR_0323 = AutoProcessor.from_pretrained(
        str(processor_source),
        use_fast=False,
        trust_remote_code=True,
    )

    with open(VAL_JSON_PATH_0323, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    # 如果当前工作目录正好就是原始数据目录，则可直接使用样本中的原始相对路径。
    use_raw_path = str(SOURCE_ROOT_0323) == os.getcwd()

    MODEL_0323.eval()
    for sample in tqdm(eval_data, desc=f'Evaluating {args.tag}'):
        sample['response'] = predict_image_0323(sample, use_raw_path=use_raw_path)

    results_path = output_root / f'{args.tag}_predictions_0323.json'
    metrics_path = output_root / f'{args.tag}_metrics_0323.json'
    detailed_path = output_root / f'{args.tag}_detailed_metrics_0323.json'

    # 原始预测结果先完整保存，方便后续复查。
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    processed = process_result_0323(eval_data)
    pprint(processed)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    detailed = [
        evaluate_metrics_0323(eval_data, field)
        for field in ['pain_status', 'facial_asymmetry', 'gender', 'levine_sign']
    ]
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(detailed, f, ensure_ascii=False, indent=2)

    print(f'Pruned / merged model predictions saved to: {results_path}')
    print(f'Pruned / merged model metrics saved to: {metrics_path}')


if __name__ == '__main__':
    main()
