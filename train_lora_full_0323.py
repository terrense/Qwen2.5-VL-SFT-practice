"""
0323 实验：Qwen2.5-VL-32B LoRA 全量 1-epoch 训练
使用 HuggingFace Trainer + PEFT LoRA，在面部症状识别数据上做监督微调。
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# 若环境变量未设置，默认使用 0 号 GPU；必须在 import torch 之前设置

import torch
from modelscope import AutoTokenizer  # 国内镜像友好，与 transformers 兼容
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from experiment_config_0323 import (
    BASE_MODEL_PATH_0323,
    CUDA_VISIBLE_DEVICES_0323,
    LORA_ALPHA_0323,
    LORA_DROPOUT_0323,
    LORA_R_0323,
    LORA_TARGET_MODULES_0323,
    TRAIN_EPOCHS_0323,
    TRAIN_JSON_PATH_0323,
    ensure_output_dirs_0323,
    lora_output_dir_0323,
)
from sft_dataset_0323 import (
    DataCollatorForSupervisedDataset0323,
    SupervisedDataset0323,
)


os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES_0323
# 显式覆盖，确保使用配置中的 GPU 设置


def main() -> None:
    ensure_output_dirs_0323()
    # 创建 runs、reports 等输出目录，避免 Trainer 保存时报错

    # ---------- 加载 tokenizer 与 processor ----------
    print("Loading tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(BASE_MODEL_PATH_0323),
        use_fast=False,
        # use_fast=False：使用 Python 实现的 tokenizer，与 Qwen 特殊 token 兼容更好
        trust_remote_code=True,
        # Qwen 模型含自定义代码，需信任才能加载
    )
    processor = AutoProcessor.from_pretrained(
        str(BASE_MODEL_PATH_0323),
        trust_remote_code=True,
        use_fast=False,  # 与评测一致，避免 fast image processor 可能的数值问题
    )
    # processor：整合 tokenizer + 图像预处理，处理多模态输入

    # ---------- 加载基座模型 ----------
    print("Loading 32B Qwen2.5-VL base model in bfloat16...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(BASE_MODEL_PATH_0323),
        torch_dtype=torch.bfloat16,
        # bfloat16：省显存、加速，且数值范围比 fp16 大，训练更稳定
        trust_remote_code=True,
    )

    # ---------- 配置 LoRA ----------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        # 因果语言模型任务
        target_modules=LORA_TARGET_MODULES_0323,
        # 注入 LoRA 的层名，需与模型结构一致
        inference_mode=False,
        # 训练模式，会启用 dropout 等
        r=LORA_R_0323,
        # 秩，控制 LoRA 参数量
        lora_alpha=LORA_ALPHA_0323,
        # 缩放因子，输出乘以 alpha/r
        lora_dropout=LORA_DROPOUT_0323,
        # LoRA 层 dropout
        bias="none",
        # 不训练 bias，仅训练 LoRA 矩阵
    )

    print("Applying LoRA adapters...")
    model = get_peft_model(model, lora_config)
    # 将原层替换为 原层+LoRA，冻结原参数，仅训练 LoRA
    model.print_trainable_parameters()
    # 打印可训练参数量，用于确认 LoRA 生效

    # ---------- 构建训练集 ----------
    print(f"Loading training data from: {TRAIN_JSON_PATH_0323}")
    train_dataset = SupervisedDataset0323(
        json_path=str(TRAIN_JSON_PATH_0323),
        processor=processor,
        tokenizer=tokenizer,
    )
    print(f"Train dataset size: {len(train_dataset)}")

    model.config.use_cache = False
    # 训练时关闭 KV cache，否则 gradient_checkpointing 会冲突；可省显存

    # ---------- 训练参数 ----------
    training_args = TrainingArguments(
        output_dir=lora_output_dir_0323(),
        # 输出目录，checkpoint、日志等保存于此

        per_device_train_batch_size=1,
        # 每卡 batch size，32B 模型显存紧张，设为 1

        gradient_accumulation_steps=8,
        # 梯度累积步数，有效 batch = 1 * 8 = 8

        bf16=True,
        # 使用 bfloat16 混合精度训练

        logging_steps=10,
        # 每 10 步打印一次 loss 等

        logging_first_step=True,
        # 第一步也打印，便于确认训练已启动

        num_train_epochs=TRAIN_EPOCHS_0323,
        # 训练轮数，此处为 1

        save_strategy="epoch",
        # 每个 epoch 结束后保存

        learning_rate=1e-5,
        # 学习率，LoRA 通常用 1e-5~2e-5

        save_total_limit=2,
        # 最多保留 2 个 checkpoint，节省磁盘

        save_on_each_node=True,
        # 多机时每节点保存，单机无影响

        gradient_checkpointing=True,
        # 梯度检查点：用时间换显存，前向不存中间激活，反向时重算

        report_to="none",
        # 不上报 wandb/tensorboard 等

        dataloader_num_workers=4,
        # 数据加载线程数，加速 IO

        remove_unused_columns=False,
        # 保留 pixel_values、image_grid_thw 等，Trainer 默认会删掉"未用"列导致报错
    )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    # 梯度检查点下，须显式启用 input 的梯度，否则 LoRA 无法反向传播

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSupervisedDataset0323(
            pad_token_id=processor.tokenizer.pad_token_id
        ),
        # pad_token_id 用于 pad input_ids
    )

    print("Starting full 1-epoch LoRA SFT run...")
    trainer.train()

    print("Saving final adapter, tokenizer, and processor...")
    trainer.save_model(lora_output_dir_0323())
    # 保存 LoRA adapter（adapter_config.json、adapter_model.safetensors 等）
    tokenizer.save_pretrained(lora_output_dir_0323())
    processor.save_pretrained(lora_output_dir_0323())
    # 推理时需与基座配合使用，一并保存
    print(f"Training finished. Outputs saved to: {lora_output_dir_0323()}")


if __name__ == "__main__":
    main()
