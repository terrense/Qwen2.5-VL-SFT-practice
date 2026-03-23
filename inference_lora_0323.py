"""
0323 实验：单图推理脚本
对指定图片做面部症状识别推理。若存在 LoRA checkpoint 则加载 LoRA，否则仅使用基座模型。
默认图片来自 experiment_config_0323.SAMPLE_IMAGE_PATH_0323。
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# 在 import torch 之前设置，确保 PyTorch 只看到指定 GPU

import torch
from peft import LoraConfig, PeftModel, TaskType
from qwen_vl_utils import process_vision_info  # pip install qwen-vl-utils
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from experiment_config_0323 import (
    BASE_MODEL_PATH_0323,
    CUDA_VISIBLE_DEVICES_0323,
    LORA_ALPHA_0323,
    LORA_DROPOUT_0323,
    LORA_R_0323,
    LORA_TARGET_MODULES_0323,
    MAX_NEW_TOKENS_0323,
    RUN_ROOT_0323,
    SAMPLE_IMAGE_PATH_0323,
)

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES_0323
# 显式覆盖环境变量

IMAGE_PATH_0323 = str(SAMPLE_IMAGE_PATH_0323)
# 转为 str，process_vision_info 与 processor 接受路径字符串

LORA_PATH_0323 = str(RUN_ROOT_0323)
# LoRA 保存目录，os.path.exists 和 PeftModel.from_pretrained 均接受 str

# 面部症状识别任务的 prompt 模板：明确要求输出 JSON 及四个字段的取值
PROMPT_TEXT_0323 = (
    "You are an expert in facial symptom recognition.\n"
    "Analyze the image and output a JSON object with the following keys:\n\n"
    '- "pain_status": one of ["no_pain", "pain"]\n'
    '- "facial_asymmetry": one of ["facial_asymmetry", "none"]\n'
    '- "gender": one of ["man", "woman"]\n'
    '- "levine_sign": one of ["levine", "levine_none"]\n\n'
    "Use this format:\n"
    "{\n"
    '  "pain_status": "...",\n'
    '  "facial_asymmetry": "...",\n'
    '  "gender": "...",\n'
    '  "levine_sign": "..."\n'
    "}"
)
# 与训练/评测数据的 prompt 风格一致，约束输出格式


def main():
    print("Loading base model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(BASE_MODEL_PATH_0323),
        torch_dtype="auto",   # 推理可用 float16/bf16 省显存
        device_map="auto",    # 自动分配到可用 GPU
    )

    # 若 LoRA 目录存在则加载 adapter；否则仅用基座
    if os.path.exists(LORA_PATH_0323):
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=LORA_TARGET_MODULES_0323,
            inference_mode=True,
            r=LORA_R_0323,
            lora_alpha=LORA_ALPHA_0323,
            lora_dropout=LORA_DROPOUT_0323,
            bias="none",
        )
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, LORA_PATH_0323, config=config)
    else:
        print("LoRA path not found, using base model only.")
    # 推理时 inference_mode=True，不启用 dropout

    processor = AutoProcessor.from_pretrained(
        str(BASE_MODEL_PATH_0323),
        use_fast=False,  # 使用慢速 image processor，避免 Floating point exception
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": IMAGE_PATH_0323},
                # 单图推理不指定 resized_height/width，用默认或 process_vision_info 自动处理
                {"type": "text", "text": PROMPT_TEXT_0323},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    # 单卡时 "cuda" 等价于 "cuda:0"

    print(f"Inference on: {IMAGE_PATH_0323}")
    print("Generating...")
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_0323,
            pad_token_id=pad_id,
        )
    # 自回归生成，每步取概率最高的 token

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("\n--- Inference Result ---")
    print(output_text)
    print("------------------------\n")


if __name__ == "__main__":
    main()
