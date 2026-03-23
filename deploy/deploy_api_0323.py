#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0323 实验：OpenAI-compatible 多模态 API 服务。
默认加载 merge 后完整模型；也可切换为 LoRA adapter 路径。
接口风格参考 /data/bian/Finetune_WHU/vllm_service_multimodal.py。
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

PROJECT_ROOT_0323 = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_0323) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_0323))

import torch
import uvicorn
from fastapi import FastAPI
from peft import LoraConfig, PeftModel, TaskType
from PIL import Image
from pydantic import BaseModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from experiment_config_0323 import (
    BASE_MODEL_PATH_0323,
    CUDA_VISIBLE_DEVICES_0323,
    LORA_ALPHA_0323,
    LORA_DROPOUT_0323,
    LORA_R_0323,
    LORA_TARGET_MODULES_0323,
    MAX_NEW_TOKENS_0323,
    MERGED_MODEL_ROOT_0323,
    RUN_ROOT_0323,
)

# 运行前优先使用配置中的 GPU 号，方便与训练/评测保持一致
os.environ.setdefault('CUDA_VISIBLE_DEVICES', CUDA_VISIBLE_DEVICES_0323)
DEVICE_0323 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_0323 = None
PROCESSOR_0323 = None


class MessageContent0323(BaseModel):
    type: str
    text: Optional[str] = None
    image: Optional[Dict[str, str]] = None  # 支持 {'base64': '...'} 或 {'image': '/path/to/file'}


class ChatMessage0323(BaseModel):
    role: str
    content: Union[str, List[MessageContent0323]]


class ChatCompletionRequest0323(BaseModel):
    model: str = 'qwen25vl-0323'
    messages: List[ChatMessage0323]
    max_tokens: Optional[int] = MAX_NEW_TOKENS_0323
    temperature: Optional[float] = 0.0


class ChoiceMessage0323(BaseModel):
    role: str
    content: str


class Choice0323(BaseModel):
    index: int
    message: ChoiceMessage0323
    finish_reason: str = 'stop'


class ChatCompletionResponse0323(BaseModel):
    id: str
    object: str = 'chat.completion'
    choices: List[Choice0323]


app = FastAPI(title='Qwen2.5-VL-0323 API')


def parse_args_0323() -> argparse.Namespace:
    """解析部署参数，支持加载 merge 后模型或 LoRA adapter。"""
    parser = argparse.ArgumentParser(description='Serve the 0323 Qwen2.5-VL pipeline as an OpenAI-compatible API.')
    parser.add_argument('--mode', choices=['merged', 'lora'], default='merged', help='使用 merge 后完整模型或基座+LoRA adapter。')
    parser.add_argument('--merged-model-path', type=Path, default=MERGED_MODEL_ROOT_0323, help='merge 后模型目录。')
    parser.add_argument('--lora-path', type=Path, default=RUN_ROOT_0323, help='LoRA adapter 目录。')
    parser.add_argument('--host', default='0.0.0.0', help='服务监听地址。')
    parser.add_argument('--port', type=int, default=9001, help='服务监听端口。')
    return parser.parse_args()


def load_model_and_processor_0323(args: argparse.Namespace):
    """根据 mode 加载 merged 完整模型或 base + LoRA 组合。"""
    if args.mode == 'merged':
        model_path = args.merged_model_path.expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f'Merged model path not found: {model_path}')
        print(f'Loading merged model from {model_path} ...')
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            dtype='auto',
            device_map='auto',
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        return model, processor

    lora_path = args.lora_path.expanduser().resolve()
    if not lora_path.exists():
        raise FileNotFoundError(f'LoRA path not found: {lora_path}')

    print(f'Loading base model from {BASE_MODEL_PATH_0323} ...')
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(BASE_MODEL_PATH_0323),
        dtype='auto',
        device_map='auto',
        trust_remote_code=True,
    )
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES_0323,
        inference_mode=True,
        r=LORA_R_0323,
        lora_alpha=LORA_ALPHA_0323,
        lora_dropout=LORA_DROPOUT_0323,
        bias='none',
    )
    print(f'Applying LoRA adapter from {lora_path} ...')
    model = PeftModel.from_pretrained(base_model, model_id=str(lora_path), config=config)
    processor = AutoProcessor.from_pretrained(str(BASE_MODEL_PATH_0323), trust_remote_code=True)
    return model, processor


def extract_image_and_text_0323(request: ChatCompletionRequest0323):
    """从 OpenAI 风格的 messages 中解析出图像和文本。"""
    image_obj, text_obj = None, None
    for msg in request.messages:
        if isinstance(msg.content, list):
            for item in msg.content:
                if item.type == 'image' and item.image:
                    if 'base64' in item.image:
                        img_data = base64.b64decode(item.image['base64'])
                        image_obj = Image.open(io.BytesIO(img_data)).convert('RGB')
                    elif 'image' in item.image:
                        image_obj = Image.open(item.image['image']).convert('RGB')
                elif item.type == 'text':
                    text_obj = item.text
    return image_obj, text_obj


@app.post('/v1/chat/completions', response_model=ChatCompletionResponse0323)
def create_chat_completion_0323(request: ChatCompletionRequest0323):
    img, prompt = extract_image_and_text_0323(request)
    if img is None or prompt is None:
        return ChatCompletionResponse0323(
            id='chatcmpl-error',
            choices=[
                Choice0323(
                    index=0,
                    message=ChoiceMessage0323(role='assistant', content='输入缺少图像或文本'),
                )
            ],
        )

    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': img},
                {'type': 'text', 'text': prompt},
            ],
        }
    ]

    text_input = PROCESSOR_0323.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = PROCESSOR_0323(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt',
    ).to(DEVICE_0323)

    with torch.no_grad():
        generated_ids = MODEL_0323.generate(**inputs, max_new_tokens=request.max_tokens or MAX_NEW_TOKENS_0323)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = PROCESSOR_0323.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return ChatCompletionResponse0323(
        id='chatcmpl-0323',
        choices=[
            Choice0323(
                index=0,
                message=ChoiceMessage0323(role='assistant', content=output_text),
                finish_reason='stop',
            )
        ],
    )


def main() -> None:
    global MODEL_0323, PROCESSOR_0323
    args = parse_args_0323()
    MODEL_0323, PROCESSOR_0323 = load_model_and_processor_0323(args)
    MODEL_0323.config.use_cache = True
    MODEL_0323.eval()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
