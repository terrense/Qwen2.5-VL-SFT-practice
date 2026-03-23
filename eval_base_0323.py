"""
0323 实验：基座模型（无 LoRA）在验证集上的评测
逐样本推理，解析 JSON 输出，计算各字段准确率、exact_match、混淆矩阵等。
"""

import json
import os
import re
from collections import defaultdict

import torch
from pprint import pprint
from qwen_vl_utils import process_vision_info  # pip install qwen-vl-utils
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from experiment_config_0323 import (
    BASE_EVAL_ROOT_0323,
    BASE_MODEL_PATH_0323,
    MAX_NEW_TOKENS_0323,
    SOURCE_ROOT_0323,
    VAL_JSON_PATH_0323,
    ensure_output_dirs_0323,
    resolve_image_path_0323,
)


DEVICE_0323 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE_STR_0323 = "cuda:0" if torch.cuda.is_available() else "cpu"
# 与 bian/Finetune_WHU/eval.py 一致：device_map={"": "cuda:0"} 固定单卡

MODEL_0323 = None
PROCESSOR_0323 = None


def strip_code_fences_0323(text: str) -> str:
    """
    去掉模型输出中可能包裹的 ```json ... ``` 或 ``` ... ``` 标记。
    模型有时会输出 markdown 代码块，直接 json.loads 会失败。
    正则以 ^ 匹配开头、$ 匹配结尾，\s* 匹配可选空白。
    """
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text.strip())  # 去掉开头 ```json 或 ```
    text = re.sub(r"\s*```$", "", text.strip())            # 去掉结尾 ```
    return text.strip()


def predict_image_0323(sample, use_raw_path=False):
    """
    对单条样本做推理。与 bian/Finetune_WHU/eval_base_model.py 对齐。
    use_raw_path: True 时用原始相对路径（CWD 需为 Finetune_WHU）
    """
    prompt = sample["conversations"][0]["value"]
    raw_path = prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    file_path = raw_path if use_raw_path else resolve_image_path_0323(raw_path)
    new_prompt = prompt.replace(f"<|vision_start|>{raw_path}<|vision_end|>", "").strip()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": file_path, "resized_height": 480, "resized_width": 320},
                {"type": "text", "text": new_prompt},
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
        return_tensors="pt",
    ).to(DEVICE_0323)

    with torch.no_grad():
        generated_ids = MODEL_0323.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS_0323)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    # 去掉 prompt 部分，只保留新生成的 token
    output_text = PROCESSOR_0323.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,   # 不输出 <|im_end|> 等
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text.strip()


def process_result_0323(samples):
    """
    汇总评测结果：各字段准确率、exact_match、JSON 解析失败率、混淆矩阵、错误样本列表。
    samples: 已含 "response" 字段的评测数据
    返回：dict，含 pain_status 等字段准确率、exact_match、confusion_matrices 等
    内部用 defaultdict 构建混淆矩阵，最后转成普通 dict 以便 JSON 序列化。
    """
    fields = ["pain_status", "facial_asymmetry", "gender", "levine_sign"]
    correct = {k: 0 for k in fields}           # 各字段正确数
    incorrect_samples = {f: [] for f in fields}  # 各字段错误样本列表
    exact_match = 0    # 四字段全对的样本数
    total_count = 0
    json_error_count = 0  # 模型输出无法解析为 JSON 的样本数
    confusion = {f: defaultdict(lambda: defaultdict(int)) for f in fields}
    # confusion[field][true][pred] = count

    for sample in samples:
        total_count += 1
        label = strip_code_fences_0323(sample["conversations"][1]["value"].strip().lower())
        response = strip_code_fences_0323(sample.get("response", "").strip().lower())
        prompt = sample["conversations"][0]["value"]
        raw_path = prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
        file_path = resolve_image_path_0323(raw_path)

        try:
            pred_json = json.loads(response)
            label_json = json.loads(label)
        except Exception:
            json_error_count += 1
            continue
        # JSON 解析失败则跳过该样本，不计入准确率分子分母

        all_ok = True
        for field in fields:
            true_val = label_json.get(field, "")
            pred_val = pred_json.get(field, "")
            confusion[field][true_val][pred_val] += 1
            if pred_val == true_val:
                correct[field] += 1
            else:
                all_ok = False
                incorrect_samples[field].append(
                    {
                        "file_path": file_path,
                        "true_label": true_val,
                        "predicted_label": pred_val,
                    }
                )
        if all_ok:
            exact_match += 1

    def _to_dict(obj):
        """递归将 defaultdict 转为普通 dict，否则 json.dump 会报错"""
        if isinstance(obj, defaultdict):
            return {k: _to_dict(v) for k, v in obj.items()}
        return obj

    metrics = {f: correct[f] / total_count if total_count else 0 for f in fields}
    metrics["exact_match"] = exact_match / total_count if total_count else 0
    metrics["json_error_rate"] = json_error_count / total_count if total_count else 0
    metrics["correct_counts"] = correct
    metrics["total_count"] = total_count
    metrics["confusion_matrices"] = _to_dict(confusion)
    metrics["incorrect_samples"] = incorrect_samples
    return metrics


def evaluate_metrics_0323(samples, field):
    """
    针对单个字段计算 sklearn 的 classification_report 与 confusion_matrix。
    field: 如 "pain_status"
    返回：dict 含 labels、confusion_matrix（列表）、classification_report（dict）
    """
    y_true, y_pred = [], []
    for sample in samples:
        label = strip_code_fences_0323(sample["conversations"][1]["value"].strip().lower())
        response = strip_code_fences_0323(sample.get("response", "").strip().lower())
        try:
            pred_json = json.loads(response)
            label_json = json.loads(label)
        except Exception:
            continue
        y_true.append(label_json.get(field, ""))
        y_pred.append(pred_json.get(field, ""))

    if not y_true and not y_pred:
        return {"field": field, "labels": [], "confusion_matrix": [], "classification_report": "No valid samples"}

    labels = sorted(list(set(y_true + y_pred)))
    # 所有出现过的类别，sorted 保证顺序稳定
    return {
        "field": field,
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        # 转为 list 以便 JSON 序列化
        "classification_report": classification_report(
            y_true,
            y_pred,
            digits=3,           # 精度保留 3 位
            labels=labels,
            zero_division=0,    # 除零时用 0
            output_dict=True,   # 返回 dict 而非字符串，便于保存
        ),
    }


def main():
    global MODEL_0323, PROCESSOR_0323
    ensure_output_dirs_0323()

    # 与 bian/Finetune_WHU/eval_base_model.py、eval.py 保持一致
    print("Loading base model...")
    MODEL_0323 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(BASE_MODEL_PATH_0323),
        torch_dtype="auto",
        device_map={"": DEVICE_STR_0323},
    )
    MODEL_0323.config.use_cache = True

    # use_fast=False：强制慢速 image processor，避免 Floating point exception（若无效可尝试 run 在数据目录下执行）
    PROCESSOR_0323 = AutoProcessor.from_pretrained(
        str(BASE_MODEL_PATH_0323),
        use_fast=False,
    )

    with open(VAL_JSON_PATH_0323, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # 若 CWD 为数据目录则用原始相对路径，与 bian/eval_base_model.py 一致
    use_raw_path = str(SOURCE_ROOT_0323) == os.getcwd()

    MODEL_0323.eval()
    for sample in tqdm(eval_data, desc="Evaluating base model"):
        sample["response"] = predict_image_0323(sample, use_raw_path=use_raw_path)
    # 逐条推理，将模型输出写入 sample["response"]

    results_path = BASE_EVAL_ROOT_0323 / "base_predictions_0323.json"
    metrics_path = BASE_EVAL_ROOT_0323 / "base_metrics_0323.json"
    detailed_path = BASE_EVAL_ROOT_0323 / "base_detailed_metrics_0323.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    # ensure_ascii=False：中文等不转 \uXXXX

    processed = process_result_0323(eval_data)
    pprint(processed)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    detailed = [
        evaluate_metrics_0323(eval_data, field)
        for field in ["pain_status", "facial_asymmetry", "gender", "levine_sign"]
    ]
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, ensure_ascii=False, indent=2)

    print(f"Base evaluation predictions saved to: {results_path}")
    print(f"Base evaluation metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
