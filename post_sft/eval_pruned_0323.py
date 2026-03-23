#!/usr/bin/env python3
"""
0323 实验：对 merge 后或 2:4 结构化剪枝后的完整模型做验证集评测。
默认评测 PRUNED_MODEL_ROOT_0323，也可通过命令行切换为其它 merge / pruned 模型目录。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT_0323 = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_0323) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_0323))

import torch
from pprint import pprint
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

DEVICE_0323 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE_STR_0323 = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MODEL_0323 = None
PROCESSOR_0323 = None


def parse_args_0323() -> argparse.Namespace:
    """解析评测参数。"""
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
    """去掉模型输出外层 markdown 代码块，便于 json.loads。"""
    text = re.sub(r'^```[a-zA-Z]*\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text.strip())
    return text.strip()


def predict_image_0323(sample, use_raw_path=False):
    """对单条图文样本做推理，返回模型生成的 JSON 文本。"""
    prompt = sample['conversations'][0]['value']
    raw_path = prompt.split('<|vision_start|>')[1].split('<|vision_end|>')[0]
    file_path = raw_path if use_raw_path else resolve_image_path_0323(raw_path)
    new_prompt = prompt.replace(f'<|vision_start|>{raw_path}<|vision_end|>', '').strip()

    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': file_path, 'resized_height': 480, 'resized_width': 320},
                {'type': 'text', 'text': new_prompt},
            ],
        }
    ]

    text = PROCESSOR_0323.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = PROCESSOR_0323(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt',
    ).to(DEVICE_0323)

    with torch.no_grad():
        generated_ids = MODEL_0323.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS_0323)

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
    """统计四字段准确率、exact_match、json_error_rate、混淆矩阵与错误样本。"""
    fields = ['pain_status', 'facial_asymmetry', 'gender', 'levine_sign']
    correct = {k: 0 for k in fields}
    incorrect_samples = {f: [] for f in fields}
    exact_match = 0
    total_count = 0
    json_error_count = 0
    confusion = {f: defaultdict(lambda: defaultdict(int)) for f in fields}

    for sample in samples:
        total_count += 1
        label = strip_code_fences_0323(sample['conversations'][1]['value'].strip().lower())
        response = strip_code_fences_0323(sample.get('response', '').strip().lower())
        prompt = sample['conversations'][0]['value']
        raw_path = prompt.split('<|vision_start|>')[1].split('<|vision_end|>')[0]
        file_path = resolve_image_path_0323(raw_path)

        try:
            pred_json = json.loads(response)
            label_json = json.loads(label)
        except Exception:
            json_error_count += 1
            continue

        all_ok = True
        for field in fields:
            true_val = label_json.get(field, '')
            pred_val = pred_json.get(field, '')
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
        if isinstance(obj, defaultdict):
            return {k: _to_dict(v) for k, v in obj.items()}
        return obj

    metrics = {f: correct[f] / total_count if total_count else 0 for f in fields}
    metrics['exact_match'] = exact_match / total_count if total_count else 0
    metrics['json_error_rate'] = json_error_count / total_count if total_count else 0
    metrics['correct_counts'] = correct
    metrics['total_count'] = total_count
    metrics['confusion_matrices'] = _to_dict(confusion)
    metrics['incorrect_samples'] = incorrect_samples
    return metrics


def evaluate_metrics_0323(samples, field):
    """对单个字段输出 confusion_matrix 与 classification_report。"""
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
    global MODEL_0323, PROCESSOR_0323
    args = parse_args_0323()
    ensure_output_dirs_0323()

    model_path = args.model_path.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f'Model path not found: {model_path}')

    output_root = PRUNED_EVAL_ROOT_0323 / args.tag
    output_root.mkdir(parents=True, exist_ok=True)

    print(f'Loading full model from: {model_path}')
    MODEL_0323 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype='auto',
        device_map={'': DEVICE_STR_0323},
        trust_remote_code=True,
    )
    MODEL_0323.config.use_cache = True

    # processor 优先从评测模型目录加载；若该目录缺失 processor 配置，则回退到基座目录
    processor_source = model_path if (model_path / 'preprocessor_config.json').exists() else BASE_MODEL_PATH_0323
    PROCESSOR_0323 = AutoProcessor.from_pretrained(
        str(processor_source),
        use_fast=False,
        trust_remote_code=True,
    )

    with open(VAL_JSON_PATH_0323, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    use_raw_path = str(SOURCE_ROOT_0323) == os.getcwd()

    MODEL_0323.eval()
    for sample in tqdm(eval_data, desc=f'Evaluating {args.tag}'):
        sample['response'] = predict_image_0323(sample, use_raw_path=use_raw_path)

    results_path = output_root / f'{args.tag}_predictions_0323.json'
    metrics_path = output_root / f'{args.tag}_metrics_0323.json'
    detailed_path = output_root / f'{args.tag}_detailed_metrics_0323.json'

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
