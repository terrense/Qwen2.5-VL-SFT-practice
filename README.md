# Qwen2.5-VL-32B Full Pipeline (0323)

这是一个围绕 `Qwen2.5-VL-32B-Instruct` 构建的完整实验与工程链路，任务目标是根据图像输出固定结构的四字段 JSON：`pain_status`、`facial_asymmetry`、`gender`、`levine_sign`。

当前目录已经包含一整套 `_0323` 脚本，不仅有 SFT / LoRA 训练与评测，也补齐了后处理与部署链路：`merge -> prune -> pruned eval -> quantize -> TensorRT export/build -> API deploy`。

## 1. Pipeline Overview

```text
Baseline eval
  -> LoRA SFT train
  -> LoRA eval
  -> Single-image inference
  -> Merge LoRA into base model
  -> 2:4 structured pruning
  -> Pruned / merged model eval
  -> Quantization export (int4_awq / fp8)
  -> TensorRT-LLM step1 ONNX export
  -> TensorRT-LLM step2 engine build
  -> OpenAI-compatible multimodal API deploy
```

## 2. Repository Layout

```text
qwen25vl_sft_chain_0323/
├── README.md
├── experiment_config_0323.py
├── sft_dataset_0323.py
├── train_lora_full_0323.py
├── eval_base_0323.py
├── eval_lora_0323.py
├── inference_lora_0323.py
├── run_train_0323.sh
├── run_eval_base_0323.sh
├── run_eval_lora_0323.sh
├── run_inference_0323.sh
├── requirements_0323.txt
├── ENV_SETUP_0323.md
├── 总纲领_0323.md
├── SFT_experiment_report_0323.md
├── post_sft/
│   ├── merge_lora_0323.py
│   ├── prune_2to4_0323.py
│   └── eval_pruned_0323.py
├── quantization/
│   └── quantize_0323.py
├── trt/
│   ├── step1_export_0323.py
│   └── step2_build_0323.py
├── deploy/
│   └── deploy_api_0323.py
├── scripts/
│   ├── run_merge_lora_0323.sh
│   ├── run_prune_0323.sh
│   ├── run_eval_pruned_0323.sh
│   ├── run_quantize_0323.sh
│   ├── run_export_onnx_0323.sh
│   ├── run_build_engine_0323.sh
│   └── run_deploy_api_0323.sh
├── runs/
└── reports/
```

## 3. Fixed Paths In This Version

当前 `_0323` 版本默认使用以下绝对路径，这些路径统一由 `experiment_config_0323.py` 管理：

- Base model: `/data/bian/model/Qwen/Qwen2.5-VL-32B-Instruct`
- Train set: `/data/bian/Finetune_WHU/data_vl_all_face_train_Full_260310.json`
- Val set: `/data/bian/Finetune_WHU/data_vl_all_face_val_nopain.json`
- Experiment root: `/data/sx_files/qwen25vl_sft_chain_0323`

## 4. Environment

已验证能通过 `Qwen2.5-VL generate()` 的关键组合：

- `transformers==4.57.5`
- `torch==2.9.0+cu128`
- `torchvision==0.24.0+cu128`
- `torchaudio==2.9.0+cu128`
- `triton==3.5.0`
- `peft==0.18.1`
- `accelerate==1.13.0`

安装顺序：

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements_0323.txt
```

`qwen-vl-utils` 的 pip 名称和 import 名称不同：

- pip: `qwen-vl-utils`
- import: `qwen_vl_utils`

## 5. Quick Start

先给脚本执行权限：

```bash
cd /data/sx_files/qwen25vl_sft_chain_0323
chmod +x run_*.sh scripts/*.sh
```

### 5.1 Research Loop

```bash
./run_eval_base_0323.sh
./run_train_0323.sh
./run_eval_lora_0323.sh
./run_inference_0323.sh
```

### 5.2 Post-SFT Engineering Loop

```bash
./scripts/run_merge_lora_0323.sh
./scripts/run_prune_0323.sh
./scripts/run_eval_pruned_0323.sh
```

### 5.3 Deployment-Oriented Loop

```bash
./scripts/run_quantize_0323.sh
./scripts/run_export_onnx_0323.sh [TensorRT-LLM builder args...]
./scripts/run_build_engine_0323.sh --plan-path /path/to/build_plan_0323.json
./scripts/run_deploy_api_0323.sh --mode merged
```

## 6. Stage-by-Stage Details

### A. Baseline evaluation

- Script: `eval_base_0323.py`
- Launcher: `run_eval_base_0323.sh`
- Output: `reports/base_eval_0323/`

### B. LoRA SFT

- Script: `train_lora_full_0323.py`
- Data module: `sft_dataset_0323.py`
- Launcher: `run_train_0323.sh`
- Output: `runs/qwen25vl32b_full_1epoch_0323/`

默认 LoRA 配置：

- `r = 64`
- `alpha = 16`
- `dropout = 0.05`
- `target_modules = q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`

### C. LoRA evaluation

- Script: `eval_lora_0323.py`
- Launcher: `run_eval_lora_0323.sh`
- Output: `reports/lora_eval_0323/`

### D. Single-image inference

- Script: `inference_lora_0323.py`
- Launcher: `run_inference_0323.sh`

### E. Merge LoRA

- Script: `post_sft/merge_lora_0323.py`
- Launcher: `scripts/run_merge_lora_0323.sh`
- Output: `post_sft/merged/qwen25vl32b_full_1epoch_merged_0323/`

也支持手动指定路径：

```bash
python post_sft/merge_lora_0323.py   --base-model /data/bian/model/Qwen/Qwen2.5-VL-32B-Instruct   --lora-checkpoint /data/sx_files/qwen25vl_sft_chain_0323/runs/qwen25vl32b_full_1epoch_0323   --output-dir /data/sx_files/qwen25vl_sft_chain_0323/post_sft/merged/custom_merged   --dtype bfloat16   --device-map auto
```

### F. Structured pruning (2:4)

- Script: `post_sft/prune_2to4_0323.py`
- Launcher: `scripts/run_prune_0323.sh`
- Output: `post_sft/pruned/qwen25vl32b_full_1epoch_merged_pruned_0323/`

默认会对这些线性层应用 2:4 稀疏：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

### G. Pruned / merged model evaluation

- Script: `post_sft/eval_pruned_0323.py`
- Launcher: `scripts/run_eval_pruned_0323.sh`
- Output: `reports/pruned_eval_0323/`

手动评测示例：

```bash
python post_sft/eval_pruned_0323.py   --model-path /data/sx_files/qwen25vl_sft_chain_0323/post_sft/merged/qwen25vl32b_full_1epoch_merged_0323   --tag merged_eval_0323
```

### H. Quantization

- Script: `quantization/quantize_0323.py`
- Launcher: `scripts/run_quantize_0323.sh`
- Output root: `quantization/exports_0323/`

支持：

- `int4_awq`
- `fp8`

示例：

```bash
python quantization/quantize_0323.py   --model-dir /data/sx_files/qwen25vl_sft_chain_0323/post_sft/merged/qwen25vl32b_full_1epoch_merged_0323   --qformat int4_awq   --calib-size 32
```

说明：这一段依赖 TensorRT-LLM / `tensorrt_llm` 环境，不属于纯 HuggingFace 运行时。

### I. TensorRT-LLM export / build

- Step1 script: `trt/step1_export_0323.py`
- Step2 script: `trt/step2_build_0323.py`
- Launchers:
  - `scripts/run_export_onnx_0323.sh`
  - `scripts/run_build_engine_0323.sh`

推荐顺序：

```text
merged model -> optional quantization -> step1_export_0323.py -> step2_build_0323.py
```

Step1 会导出 ONNX，并在 engine 目录记录 `build_plan_0323.json`；Step2 读取该计划文件，在独立进程里真正执行 TensorRT build。

### J. API deployment

- Script: `deploy/deploy_api_0323.py`
- Launcher: `scripts/run_deploy_api_0323.sh`
- Endpoint: `/v1/chat/completions`
- Default port: `9001`

默认加载 merge 后完整模型：

```bash
./scripts/run_deploy_api_0323.sh --mode merged
```

如果你想直接加载 base + LoRA：

```bash
./scripts/run_deploy_api_0323.sh --mode lora
```

## 7. Output Artifacts

你最终会得到三类产物：

### Experiment artifacts

- `reports/base_eval_0323/*`
- `reports/lora_eval_0323/*`
- `reports/pruned_eval_0323/*`
- `SFT_experiment_report_0323.md`

### Model artifacts

- `runs/qwen25vl32b_full_1epoch_0323/`
- `post_sft/merged/...`
- `post_sft/pruned/...`
- `quantization/exports_0323/*.pt`
- `trt/engines_0323/`

### Service artifacts

- `deploy/deploy_api_0323.py`
- OpenAI-compatible multimodal API endpoint

## 8. Notes For GitHub Publishing

如果你要把这个目录上传到自己的 GitHub，建议：

1. 把绝对路径迁移到环境变量或单独配置文件，不要把 `/data/...` 写死在公开仓库。
2. 不要上传大模型权重、训练数据、TensorRT engine；只保留脚本、配置和说明。
3. 对 `runs/`、`post_sft/merged/`、`post_sft/pruned/`、`quantization/exports_0323/`、`trt/engines_0323/` 做 `.gitignore`。
4. 在公开 README 里明确说明 TensorRT-LLM / TensorRT 属于可选部署依赖。

## 9. Related Docs

- `总纲领_0323.md`
- `ENV_SETUP_0323.md`
- `SFT_experiment_report_0323.md`
- `requirements_0323.txt`
