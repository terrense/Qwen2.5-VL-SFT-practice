# Qwen2.5-VL-32B SFT 实验报告 (0323)

> **总纲领与环境配置**：请先阅读 `总纲领_0323.md` 与 `ENV_SETUP_0323.md`。

## 1. 实验概览

本实验在 Qwen2.5-VL-32B-Instruct 基座模型上，使用 LoRA 进行 1 个 epoch 的监督微调（SFT），面向面部症状识别任务。所有脚本与配置均带 `_0323` 后缀，路径统一为绝对路径。

## 2. 目录结构

```
/data/sx_files/qwen25vl_sft_chain_0323/
├── experiment_config_0323.py   # 实验配置（路径、LoRA 参数等）
├── sft_dataset_0323.py         # 数据集与 DataCollator
├── train_lora_full_0323.py     # LoRA 训练脚本
├── eval_base_0323.py           # 基座模型评测
├── eval_lora_0323.py           # LoRA 模型评测
├── inference_lora_0323.py      # 单图推理（基座或 LoRA）
├── run_train_0323.sh           # 训练启动脚本
├── run_eval_base_0323.sh       # 基座评测启动脚本
├── run_eval_lora_0323.sh       # LoRA 评测启动脚本
├── run_inference_0323.sh       # 推理启动脚本
├── requirements_0323.txt       # Python 依赖
├── SFT_experiment_report_0323.md  # 本报告
├── runs/                       # 训练输出
│   └── qwen25vl32b_full_1epoch_0323/
└── reports/                    # 评测与推理输出
    ├── base_eval_0323/         # 基座评测结果
    ├── lora_eval_0323/         # LoRA 评测结果
    └── inference_0323/         # 推理输出（如有）
```

## 3. 路径与配置

| 项目 | 路径或值 |
|------|----------|
| 基座模型 | `/data/bian/model/Qwen/Qwen2.5-VL-32B-Instruct` |
| 训练数据 | `/data/bian/Finetune_WHU/data_vl_all_face_train_Full_260310.json` |
| 验证数据 | `/data/bian/Finetune_WHU/data_vl_all_face_val_nopain.json` |
| 数据根目录 | `/data/bian/Finetune_WHU` |
| LoRA 输出 | `/data/sx_files/qwen25vl_sft_chain_0323/runs/qwen25vl32b_full_1epoch_0323` |
| 基座评测输出 | `/data/sx_files/qwen25vl_sft_chain_0323/reports/base_eval_0323/` |
| LoRA 评测输出 | `/data/sx_files/qwen25vl_sft_chain_0323/reports/lora_eval_0323/` |
| 示例图片 | `/data/bian/Finetune_WHU/qwenvl_dataset/gender/train/man/identity_1991_none_06_a831c5ecca72efa198fc_img_00197.jpg` |

### 图片路径解析

验证集中图片路径可能是绝对路径或相对路径（如 `./data_file/...`）。`resolve_image_path_0323()` 会将相对路径解析为基于 `/data/bian/Finetune_WHU` 的绝对路径。

## 4. 任务与字段

输出 JSON 包含四个字段：

- `pain_status`: `no_pain` / `pain`
- `facial_asymmetry`: `facial_asymmetry` / `none`
- `gender`: `man` / `woman`
- `levine_sign`: `levine` / `levine_none`

## 5. LoRA 与训练配置

| 参数 | 值 |
|------|-----|
| r | 64 |
| alpha | 16 |
| dropout | 0.05 |
| target_modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| epochs | 1 |
| batch_size | 1 |
| gradient_accumulation_steps | 8 |
| bf16 | True |
| gradient_checkpointing | True |
| save_strategy | epoch |
| save_total_limit | 2 |

## 6. 执行步骤

### 6.1 环境准备

**详见 `ENV_SETUP_0323.md`**。简要步骤：

```bash
conda create -n qwen25vl_sft_0323 python=3.10 -y
conda activate qwen25vl_sft_0323
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # 按 CUDA 版本选择
cd /data/sx_files/qwen25vl_sft_chain_0323
pip install -r requirements_0323.txt
# 验证 qwen_vl_utils: python -c "from qwen_vl_utils import process_vision_info; print('OK')"
```

### 6.2 基座模型评测（可选，用于对比）

```bash
chmod +x run_eval_base_0323.sh
./run_eval_base_0323.sh
```

结果写入 `reports/base_eval_0323/base_predictions_0323.json` 和 `base_metrics_0323.json`。

### 6.3 LoRA 训练

```bash
chmod +x run_train_0323.sh
./run_train_0323.sh
```

如需指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 ./run_train_0323.sh
```

### 6.4 LoRA 模型评测

训练完成后运行：

```bash
./run_eval_lora_0323.sh
```

结果写入 `reports/lora_eval_0323/lora_predictions_0323.json` 和 `lora_metrics_0323.json`。

### 6.5 单图推理

```bash
./run_inference_0323.sh
```

默认使用 `experiment_config_0323.py` 中的 `SAMPLE_IMAGE_PATH_0323`。若存在 LoRA checkpoint，将加载 LoRA 推理；否则仅使用基座模型。

## 7. 输出文件说明

- `base_predictions_0323.json` / `lora_predictions_0323.json`：每条样本的模型输出
- `base_metrics_0323.json` / `lora_metrics_0323.json`：各字段准确率、exact_match、混淆矩阵等
- `base_detailed_metrics_0323.json`：基座模型的 per-field 详细指标（仅基座评测生成）

## 8. 依赖

见 `requirements_0323.txt`。**重要**：`qwen-vl-utils` 提供 `process_vision_info`，安装命令 `pip install qwen-vl-utils`。若 IDE 出现波浪线，请检查解释器是否选中正确 conda 环境，详见 `ENV_SETUP_0323.md`。
