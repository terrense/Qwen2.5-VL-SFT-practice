"""
0323 实验：LoRA 微调后的模型在验证集上的评测
加载基座 + LoRA adapter，逐样本推理并计算指标。需先运行 train_lora_full_0323.py。
"""

import json
import re
from collections import defaultdict

import torch
from pprint import pprint
from peft import PeftModel
from qwen_vl_utils import process_vision_info  # pip install qwen-vl-utils
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from experiment_config_0323 import (
    BASE_MODEL_PATH_0323,
    LORA_ALPHA_0323,
    LORA_DROPOUT_0323,
    LORA_R_0323,
    LORA_TARGET_MODULES_0323,
    LORA_EVAL_ROOT_0323,
    MAX_NEW_TOKENS_0323,
    RUN_ROOT_0323,
    VAL_JSON_PATH_0323,
    ensure_output_dirs_0323,
    resolve_image_path_0323,
)

DEVICE_0323 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_0323 = None
PROCESSOR_0323 = None


def strip_code_fences_0323(text: str) -> str:
    """去掉模型输出中的 ```json ... ``` 标记，便于 JSON 解析."""
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def predict_image_0323(sample):
    """
    对单条样本做推理，返回模型生成的 JSON 文本。
    逻辑与 eval_base_0323.predict_image_0323 相同，仅 MODEL_0323 为基座+LoRA。
    """
    prompt = sample["conversations"][0]["value"]
    raw_path = prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    file_path = resolve_image_path_0323(raw_path)
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

    pad_id = PROCESSOR_0323.tokenizer.pad_token_id or PROCESSOR_0323.tokenizer.eos_token_id
    with torch.no_grad():  # 推理不计算梯度，省显存
        generated_ids = MODEL_0323.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_0323,
            pad_token_id=pad_id,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    # 去掉 prompt 部分，只保留新生成内容
    output_text = PROCESSOR_0323.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text.strip()


def process_result_0323(samples):
    """
    汇总各字段准确率、exact_match、混淆矩阵等，转为可 JSON 序列化的 dict。
    与 eval_base_0323.process_result_0323 逻辑一致。
    """
    fields = ["pain_status", "facial_asymmetry", "gender", "levine_sign"]
    correct = {k: 0 for k in fields}
    incorrect_samples = {f: [] for f in fields}
    exact_match = 0
    total_count = 0
    json_error_count = 0
    confusion = {f: defaultdict(lambda: defaultdict(int)) for f in fields}

    for sample in samples:
        total_count += 1
        label = strip_code_fences_0323(sample["conversations"][1]["value"].strip().lower())
        response = strip_code_fences_0323(sample.get("response", "").strip().lower())
        prompt = sample["conversations"][0]["value"]
        raw_path = prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
        file_path = resolve_image_path_0323(raw_path)  # 用于错误样本记录

        try:
            pred_json = json.loads(response)   # 模型输出
            label_json = json.loads(label)     # 真实标签
        except Exception:
            json_error_count += 1
            continue

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
                    {"file_path": file_path, "true_label": true_val, "predicted_label": pred_val}
                )
        if all_ok:
            exact_match += 1  # 四字段全对

    def _to_dict(obj):
        """递归将 defaultdict 转为普通 dict，便于 json.dump"""
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


def main():
    global MODEL_0323, PROCESSOR_0323

    lora_ckpt = RUN_ROOT_0323
    if not lora_ckpt.exists():
        raise FileNotFoundError(
            f"LoRA checkpoint not found: {lora_ckpt}. Run train_lora_full_0323.py first."
        )
    # 训练输出目录必须存在，且含 adapter_config.json、adapter_model.safetensors 等

    ensure_output_dirs_0323()
    LORA_EVAL_ROOT_0323.mkdir(parents=True, exist_ok=True)
    # 确保评测结果目录存在

    from peft import LoraConfig, TaskType

    # LoRA 配置必须与训练时完全一致，否则加载会出错
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES_0323,
        inference_mode=True,   # 推理模式：不启用 dropout，且某些实现会做融合优化
        r=LORA_R_0323,
        lora_alpha=LORA_ALPHA_0323,
        lora_dropout=LORA_DROPOUT_0323,
        bias="none",
    )

    print("Loading base model...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(BASE_MODEL_PATH_0323),
        torch_dtype="auto",
        device_map="auto",
    )
    print("Loading LoRA adapter...")
    MODEL_0323 = PeftModel.from_pretrained(
        base_model,
        str(lora_ckpt),
        config=lora_config,
    )
    # 若 checkpoint 内已有 config，可省略 config 参数；显式传入可覆盖
    MODEL_0323.config.use_cache = True
    PROCESSOR_0323 = AutoProcessor.from_pretrained(
        str(BASE_MODEL_PATH_0323),
        use_fast=False,  # 使用慢速 image processor，避免 fast 模式下可能的 Floating point exception
    )

    with open(VAL_JSON_PATH_0323, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    MODEL_0323.eval()
    for sample in tqdm(eval_data, desc="Evaluating LoRA model"):
        sample["response"] = predict_image_0323(sample)

    results_path = LORA_EVAL_ROOT_0323 / "lora_predictions_0323.json"
    metrics_path = LORA_EVAL_ROOT_0323 / "lora_metrics_0323.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    processed = process_result_0323(eval_data)
    pprint(processed)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"LoRA evaluation predictions saved to: {results_path}")
    print(f"LoRA evaluation metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
