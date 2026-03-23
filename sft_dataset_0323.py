"""
0323 实验的数据集与 DataCollator
实现 SupervisedDataset0323 与 DataCollatorForSupervisedDataset0323，用于 Qwen2.5-VL 的多模态对话格式。
依赖：qwen_vl_utils（需 pip install qwen-vl-utils）
"""

import json  # 解析训练/验证 JSON 中的标签

import torch
from torch.utils.data import Dataset  # PyTorch 数据集基类
from qwen_vl_utils import process_vision_info  # 从 messages 中提取图片/视频，转为 processor 所需格式；pip install qwen-vl-utils

from experiment_config_0323 import MAX_LENGTH_0323, resolve_image_path_0323


# 在 loss 计算时，labels 中该位置的 token 不参与梯度；PyTorch 的 CrossEntropyLoss 默认忽略 -100
IGNORE_INDEX_0323 = -100


def pad_sequence_0323(sequences, padding_side="right", padding_value=0):
    """
    对变长序列做 pad，支持左侧或右侧填充。
    用于 DataCollator 中将 batch 内样本的 input_ids/labels 对齐到同一长度。
    参数：
      sequences: list of Tensor，每个 shape (seq_len,) 或 (seq_len, dim)
      padding_side: "right" 右填充（因果 LM 常用），"left" 左填充
      padding_value: 填充用的数值，input_ids 用 pad_token_id，labels 用 IGNORE_INDEX
    返回：
       padded Tensor，shape (batch_size, max_len, ...)
    """
    assert padding_side in ["right", "left"]
    max_size = sequences[0].size()  # 取第一个序列的 shape
    trailing_dims = max_size[1:]    # 除 seq 维外的维度，如 (hidden_dim,)
    max_len = max(len(seq) for seq in sequences)  # batch 内最长序列
    batch_size = len(sequences)
    # new_full: 创建与 sequences[0] 同 dtype/device 的 tensor，填充 padding_value
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == "right":
            # 右填充：有效内容放左边 [0:length]
            output.data[i, :length] = seq
        else:
            # 左填充：有效内容放右边 [-length:]
            output.data[i, -length:] = seq
    return output


class SupervisedDataset0323(Dataset):
    """
    监督学习数据集：从 JSON 读取多轮对话，解析图片路径与文本 prompt，
    构建 Qwen2.5-VL 所需的 input_ids、pixel_values、labels 等。
    每条样本格式：conversations[0]=用户输入（含图片占位符），conversations[1]=助手回复（标签 JSON）
    """

    def __init__(self, json_path, processor, tokenizer, max_length=MAX_LENGTH_0323):
        """
        json_path: 训练 JSON 路径，每行一个样本的列表
        processor: Qwen2.5-VL 的 AutoProcessor，处理图文输入
        tokenizer: 分词器，用于对 output_content 分词得到 labels
        max_length: 最大序列长度，超长截断
        """
        super().__init__()
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)  # 列表，每项为 {"conversations": [...]}
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        conversation = sample["conversations"]
        prompt = conversation[0]["value"]        # 用户输入：含 <|vision_start|>path<|vision_end|> 与文本
        output_content = conversation[1]["value"]  # 助手回复：标签 JSON 字符串

        # 从 prompt 中解析图片路径（Qwen-VL 约定格式）
        raw_path = prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
        file_path = resolve_image_path_0323(raw_path)  # 转为绝对路径
        new_prompt = prompt.replace(f"<|vision_start|>{raw_path}<|vision_end|>", "").strip()  # 去掉占位符，保留纯文本

        # 解析标签 JSON，用于 raw_labels（DataCollator 可能用到）
        try:
            label_json_str = output_content.replace("```json", "").replace("```", "").strip()
            raw_labels = json.loads(label_json_str)
        except (json.JSONDecodeError, AttributeError):
            raw_labels = {}

        # 构建 Qwen-VL 格式的 messages：系统提示 + 用户图文
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Important: do NOT output chain-of-thought, internal reasoning, "
                            "or any tags like <think> or </think>. Only provide the requested final outputs."
                        ),
                    },
                ],
            },  # 系统提示：约束模型只输出最终结果，不输出思考过程
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": file_path,       # 图片路径，process_vision_info 会加载
                        "resized_height": 480,   # Qwen-VL 动态分辨率，指定缩放后高
                        "resized_width": 320,    # 指定缩放后宽，控制显存与速度
                    },
                    {"type": "text", "text": new_prompt},
                ],
            },
        ]

        # 使用 processor 处理图文
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,           # 不在此处分词，processor 会统一处理
            add_generation_prompt=True,  # 添加模型回复的起始 token（如 <|im_start|>assistant）
        )
        image_inputs, video_inputs = process_vision_info(messages)
        # 从 messages 中提取图片/视频，支持路径、URL、PIL、base64 等

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,     # 单样本不 pad，DataCollator 中 batch 时再 pad
            do_resize=False,   # process_vision_info 已处理 resize
            return_tensors="pt",
        )
        # 转为 list 以便后续与 response token 拼接（processor 输出为 tensor）
        inputs = {key: value.tolist() for key, value in inputs.items()}

        # 构建 labels：prompt 部分用 IGNORE_INDEX（不参与 loss），response 部分用真实 token
        response = self.tokenizer(output_content, add_special_tokens=False, padding=False)
        input_ids = inputs["input_ids"][0] + response["input_ids"] + [self.tokenizer.pad_token_id]
        # 完整输入 = prompt tokens + response tokens + pad（EOS 后有时会加 pad）
        attention_mask = inputs["attention_mask"][0] + response["attention_mask"] + [1]
        labels = [-100] * len(inputs["input_ids"][0]) + response["input_ids"] + [self.tokenizer.pad_token_id]
        # labels：prompt 全 -100，response 为真实 id，末尾 pad 也为 -100（若需要）

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            labels = labels[: self.max_length]
        # 超长截断，保证不超过模型最大长度

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels = torch.tensor(labels)
        inputs["pixel_values"] = torch.tensor(inputs["pixel_values"])
        inputs["image_grid_thw"] = torch.tensor(inputs["image_grid_thw"]).squeeze(0)
        # image_grid_thw: (T, H, W) 每个图像块的时间/高/宽，Qwen-VL 动态分辨率需要

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
            "raw_labels": raw_labels,
        }


class DataCollatorForSupervisedDataset0323:
    """
    将 batch 内样本打包成模型输入格式。
    处理：input_ids/labels 的 pad、pixel_values 的 concat、attention_mask 的构造。
    """

    def __init__(self, pad_token_id: int):
        """
        pad_token_id: 用于 pad input_ids 的 token id，通常与 tokenizer.pad_token_id 一致
        """
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        """
        examples: Trainer 传入的 list of dict，每个 dict 为 __getitem__ 的返回值
        返回：拼接、pad 后的 batch dict
        """
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        batch_raw_labels = []

        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                # 视频样本（本实验主要为图像，此处保留兼容）
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            batch_raw_labels.append(example.get("raw_labels", {}))

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

        # 右填充到 batch 内最大长度
        input_ids = pad_sequence_0323(
            batch_input_ids,
            padding_side="right",
            padding_value=self.pad_token_id,
        )
        # attention_mask: pad 位置为 0，其余为 1
        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence_0323(
            batch_label_ids,
            padding_side="right",
            padding_value=IGNORE_INDEX_0323,  # labels 的 pad 用 -100，loss 会忽略
        )

        data_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "raw_labels": batch_raw_labels,
        }

        if len(batch_pixel_values) > 0:
            # 沿 batch 维 concat 所有图像的 pixel_values
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0).view(-1, 3)
            # image_grid_thw: (num_patches, 3)，每行 (T,H,W)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict
