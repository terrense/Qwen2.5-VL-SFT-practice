#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0323 实验：OpenAI-compatible 多模态 API 服务。

这份脚本的定位不是训练，而是“服务化封装”：
- 把本地模型包成一个 HTTP 接口
- 接口形状尽量模仿 OpenAI Chat Completions
- 上层业务系统就可以按类似 `/v1/chat/completions` 的协议访问它

为什么有用：
- 方便前后端联调
- 方便接入已有的 OpenAI 兼容客户端
- 方便后续替换底层模型实现，而不大改业务协议
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

# 让部署脚本从 `deploy/` 子目录直接运行时，也能 import 项目根目录里的配置。
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

# 如果外部没手动设置 `CUDA_VISIBLE_DEVICES`，就默认使用配置中的 GPU 编号。
os.environ.setdefault('CUDA_VISIBLE_DEVICES', CUDA_VISIBLE_DEVICES_0323)

# 部署脚本里通常需要一个“全局设备对象”，后面把输入 batch `.to(DEVICE_0323)` 时会用到。
DEVICE_0323 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 下面两个全局变量会在 main() 中被真正赋值。
MODEL_0323 = None
PROCESSOR_0323 = None


class MessageContent0323(BaseModel):
    """
    定义单个 message content 项的结构。

    对 OpenAI 风格多模态请求来说，一个 message 里可能同时含：
    - 文本
    - 图片
    """

    type: str
    text: Optional[str] = None
    image: Optional[Dict[str, str]] = None  # 支持 {'base64': '...'} 或 {'image': '/path/to/file'}


class ChatMessage0323(BaseModel):
    """
    定义对话消息结构。

    `content` 允许是：
    - 普通字符串
    - 多个结构化 content 项组成的列表
    """

    role: str
    content: Union[str, List[MessageContent0323]]


class ChatCompletionRequest0323(BaseModel):
    """
    定义请求体结构。

    这里有意模仿 OpenAI 的 chat completions 协议，让前端/客户端更容易复用。
    """

    model: str = 'qwen25vl-0323'
    messages: List[ChatMessage0323]
    max_tokens: Optional[int] = MAX_NEW_TOKENS_0323
    temperature: Optional[float] = 0.0


class ChoiceMessage0323(BaseModel):
    """响应里的单条 assistant message。"""

    role: str
    content: str


class Choice0323(BaseModel):
    """响应中的一个候选返回项。这里默认只返回 1 个 choice。"""

    index: int
    message: ChoiceMessage0323
    finish_reason: str = 'stop'


class ChatCompletionResponse0323(BaseModel):
    """整个响应体结构。"""

    id: str
    object: str = 'chat.completion'
    choices: List[Choice0323]


# `FastAPI(...)` 创建 Web 应用实例。
app = FastAPI(title='Qwen2.5-VL-0323 API')


def parse_args_0323() -> argparse.Namespace:
    """
    解析部署参数。

    `mode` 说明：
    - `merged`：直接加载 merge 后完整模型目录
    - `lora`：加载基座模型，再叠加 LoRA adapter

    一般来说：
    - 想部署更简单，优先用 `merged`
    - 想保留“基座 + adapter”加载方式，可用 `lora`
    """

    parser = argparse.ArgumentParser(description='Serve the 0323 Qwen2.5-VL pipeline as an OpenAI-compatible API.')
    parser.add_argument('--mode', choices=['merged', 'lora'], default='merged', help='使用 merge 后完整模型或基座+LoRA adapter。')
    parser.add_argument('--merged-model-path', type=Path, default=MERGED_MODEL_ROOT_0323, help='merge 后模型目录。')
    parser.add_argument('--lora-path', type=Path, default=RUN_ROOT_0323, help='LoRA adapter 目录。')
    parser.add_argument('--host', default='0.0.0.0', help='服务监听地址。')
    parser.add_argument('--port', type=int, default=9001, help='服务监听端口。')
    return parser.parse_args()


def load_model_and_processor_0323(args: argparse.Namespace):
    """
    根据部署模式加载模型和 processor。

    返回值是二元组：
    - `model`
    - `processor`
    """

    if args.mode == 'merged':
        model_path = args.merged_model_path.expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f'Merged model path not found: {model_path}')

        print(f'Loading merged model from {model_path} ...')

        # merge 模式最直接：目录里本来就是完整模型。
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            dtype='auto',
            device_map='auto',
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        return model, processor

    # 如果不是 merged 模式，则走 base + LoRA 路径。
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

    # 这里显式构造 `LoraConfig`，目的主要是让推理阶段的 adapter 加载和训练时配置保持一致。
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
    """
    从 OpenAI 风格请求中解析出图像和文本。

    支持两种图片输入方式：
    - `{"base64": "..."}`
    - `{"image": "/path/to/file"}`
    """

    image_obj, text_obj = None, None
    for msg in request.messages:
        if isinstance(msg.content, list):
            for item in msg.content:
                if item.type == 'image' and item.image:
                    if 'base64' in item.image:
                        # base64 -> bytes -> 内存文件 -> PIL.Image
                        img_data = base64.b64decode(item.image['base64'])
                        image_obj = Image.open(io.BytesIO(img_data)).convert('RGB')
                    elif 'image' in item.image:
                        image_obj = Image.open(item.image['image']).convert('RGB')
                elif item.type == 'text':
                    text_obj = item.text
    return image_obj, text_obj


@app.post('/v1/chat/completions', response_model=ChatCompletionResponse0323)
def create_chat_completion_0323(request: ChatCompletionRequest0323):
    """
    这是 FastAPI 的核心路由函数。

    装饰器 `@app.post(...)` 的含义：
    - 当收到一个 POST 请求
    - 路径是 `/v1/chat/completions`
    - 就执行这个函数

    `response_model=...` 的作用：
    - 让 FastAPI / Pydantic 自动校验和规范化输出结构
    """

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

    # 这里把接口层的输入重新包装成 Qwen-VL processor 能识别的多模态 messages 结构。
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

    # 与前面评测脚本一样：去掉 prompt 对应 token，只保留新生成部分。
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

    # 服务启动前先把模型和 processor 常驻加载进内存。
    MODEL_0323, PROCESSOR_0323 = load_model_and_processor_0323(args)
    MODEL_0323.config.use_cache = True
    MODEL_0323.eval()

    # `uvicorn.run(...)` 会真正启动 ASGI 服务。
    # - `host='0.0.0.0'` 表示允许局域网访问
    # - `port=9001` 是本项目默认端口
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
