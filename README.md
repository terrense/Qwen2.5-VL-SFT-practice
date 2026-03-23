<<<<<<< HEAD
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

=======
# Qwen2.5-VL-32B Full Pipeline for Facial Symptom Recognition

本仓库整理了一套围绕 **Qwen2.5-VL-32B-Instruct** 的完整实验与工程链路，目标任务为 **面部症状识别 / 结构化 JSON 输出**。  
链路不仅包含 **SFT / LoRA 训练与评测**，还覆盖后续的 **LoRA 合并、结构化剪枝、量化、TensorRT 引擎构建、推理服务部署**。

这个 README 面向两类使用场景：

1. 你要直接复现当前 `_0323` 版本的 SFT 实验链路。
2. 你要把这套实验继续扩展到工程化部署，包括剪枝、量化、服务化和引擎化。

---

## 1. Pipeline Overview

完整链路如下：

```text
数据准备
  -> 基座模型评测
  -> LoRA SFT 训练
  -> LoRA 模型评测
  -> 单图推理验证
  -> 合并 LoRA 到基座模型
  -> 2:4 结构化剪枝
  -> 剪枝模型评测
  -> PTQ / AWQ / FP8 量化
  -> TensorRT-LLM / ONNX / 引擎构建
  -> API 部署 / OpenAI-compatible 服务
```

如果你只关心训练实验，最小闭环是：

```text
基座评测 -> LoRA SFT -> LoRA评测 -> 单图推理
```

如果你要做工程落地，建议继续扩展到：

```text
LoRA 合并 -> 剪枝/量化 -> TensorRT 引擎 -> 在线部署
```

---

## 2. Repository Layout

当前 `_0323` 目录中的核心文件：

>>>>>>> 2b4eb096774ec97dc5eeaaad3cb5811cd8291a17
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
<<<<<<< HEAD
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
=======
>>>>>>> 2b4eb096774ec97dc5eeaaad3cb5811cd8291a17
├── runs/
└── reports/
```

<<<<<<< HEAD
## 3. Fixed Paths In This Version

当前 `_0323` 版本默认使用以下绝对路径，这些路径统一由 `experiment_config_0323.py` 管理：

- Base model: `/data/bian/model/Qwen/Qwen2.5-VL-32B-Instruct`
- Train set: `/data/bian/Finetune_WHU/data_vl_all_face_train_Full_260310.json`
- Val set: `/data/bian/Finetune_WHU/data_vl_all_face_val_nopain.json`
- Experiment root: `/data/sx_files/qwen25vl_sft_chain_0323`

## 4. Environment

已验证能通过 `Qwen2.5-VL generate()` 的关键组合：
=======
同时，本链路还参考并继承了原始目录 `/data/bian/Finetune_WHU` 中的后处理与部署脚本，尤其包括：

- `merge_lora.py`
- `prune_2to4.py`
- `eval_pruned.py`
- `step1_export.py`
- `step2_build.py`
- `build_qwen25vl_mm.py`
- `qwen_tensorrt_llm/quantize.py`
- `vllm_service_multimodal.py`

---

## 3. Task Definition

模型输入为：

- 一张人脸或上半身图像
- 一段任务说明 prompt

模型输出为固定结构的 JSON：

```json
{
  "pain_status": "no_pain | pain",
  "facial_asymmetry": "facial_asymmetry | none",
  "gender": "man | woman",
  "levine_sign": "levine | levine_none"
}
```

当前 `_0323` 版本统一按以上四字段进行训练与评测。

---

## 4. Data / Model / Output Paths

### 4.1 Base model

```text
/data/bian/model/Qwen/Qwen2.5-VL-32B-Instruct
```

### 4.2 Training set

```text
/data/bian/Finetune_WHU/data_vl_all_face_train_Full_260310.json
```

### 4.3 Validation set

```text
/data/bian/Finetune_WHU/data_vl_all_face_val_nopain.json
```

### 4.4 Current experiment output root

```text
/data/sx_files/qwen25vl_sft_chain_0323
```

---

## 5. Environment

### 5.1 Verified working environment

本项目在当前机器上验证通过 `Qwen2.5-VL generate()` 的关键版本组合如下：
>>>>>>> 2b4eb096774ec97dc5eeaaad3cb5811cd8291a17

- `transformers==4.57.5`
- `torch==2.9.0+cu128`
- `torchvision==0.24.0+cu128`
- `torchaudio==2.9.0+cu128`
- `triton==3.5.0`
- `peft==0.18.1`
- `accelerate==1.13.0`

<<<<<<< HEAD
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
=======
### 5.2 Installation

先安装 PyTorch：

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
```

再安装项目依赖：

```bash
pip install -r requirements_0323.txt
```

### 5.3 Important note

`qwen-vl-utils` 的 pip 包名和 import 名不同：

- pip 安装名：`qwen-vl-utils`
- Python import：`qwen_vl_utils`

验证方式：

```bash
python -c "from qwen_vl_utils import process_vision_info; print('OK')"
```

更详细的环境说明见：

- `ENV_SETUP_0323.md`
- `总纲领_0323.md`

---

## 6. Stage A: Baseline Evaluation

基座模型评测用于建立 baseline，方便与 LoRA 或后续模型对比。

### 6.1 Script

- `eval_base_0323.py`
- `run_eval_base_0323.sh`

### 6.2 Command

```bash
cd /data/sx_files/qwen25vl_sft_chain_0323
./run_eval_base_0323.sh
```

### 6.3 Output

```text
reports/base_eval_0323/base_predictions_0323.json
reports/base_eval_0323/base_metrics_0323.json
reports/base_eval_0323/base_detailed_metrics_0323.json
```

---

## 7. Stage B: LoRA SFT Training

### 7.1 Goal

在 Qwen2.5-VL-32B 基座上进行 1 epoch 的 LoRA 监督微调。

### 7.2 LoRA config
>>>>>>> 2b4eb096774ec97dc5eeaaad3cb5811cd8291a17

- `r = 64`
- `alpha = 16`
- `dropout = 0.05`
<<<<<<< HEAD
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
=======
- `target_modules = q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

### 7.3 Training config

- `num_train_epochs = 1`
- `per_device_train_batch_size = 1`
- `gradient_accumulation_steps = 8`
- `bf16 = True`
- `gradient_checkpointing = True`
- `save_strategy = epoch`

### 7.4 Scripts

- `train_lora_full_0323.py`
- `sft_dataset_0323.py`
- `run_train_0323.sh`

### 7.5 Command

```bash
cd /data/sx_files/qwen25vl_sft_chain_0323
./run_train_0323.sh
```

### 7.6 Output

```text
runs/qwen25vl32b_full_1epoch_0323/
```

通常包含：

- adapter 权重
- tokenizer / processor 配置
- 训练 checkpoint

---

## 8. Stage C: LoRA Evaluation

训练完成后，在验证集上评测 LoRA adapter 的效果。

### 8.1 Scripts

- `eval_lora_0323.py`
- `run_eval_lora_0323.sh`

### 8.2 Command

```bash
cd /data/sx_files/qwen25vl_sft_chain_0323
./run_eval_lora_0323.sh
```

### 8.3 Output

```text
reports/lora_eval_0323/lora_predictions_0323.json
reports/lora_eval_0323/lora_metrics_0323.json
```

---

## 9. Stage D: Single-image Inference

用于快速验证模型是否能在单样本上输出合法 JSON。

### 9.1 Scripts

- `inference_lora_0323.py`
- `run_inference_0323.sh`

### 9.2 Command

```bash
cd /data/sx_files/qwen25vl_sft_chain_0323
./run_inference_0323.sh
```

若 LoRA checkpoint 存在，则会自动加载 LoRA；否则仅使用基座。

---

## 10. Stage E: Merge LoRA into Base Model

当你要做后续剪枝、量化、TensorRT 或部署时，通常需要先把 LoRA 合并进基座模型，得到独立可加载的完整权重。

### 10.1 Reference script

原始脚本：

```text
/data/bian/Finetune_WHU/merge_lora.py
```

### 10.2 Function

该脚本会：

1. 加载基座模型
2. 加载 LoRA adapter
3. 执行 `merge_and_unload()`
4. 保存 merged bf16 / fp16 / fp32 模型
5. 同时保存 tokenizer 和 processor

### 10.3 Example

```bash
python /data/bian/Finetune_WHU/merge_lora.py \
  /data/bian/model/Qwen/Qwen2.5-VL-32B-Instruct \
  /data/sx_files/qwen25vl_sft_chain_0323/runs/qwen25vl32b_full_1epoch_0323 \
  /data/sx_files/qwen25vl_sft_chain_0323/runs/qwen25vl32b_full_1epoch_0323_merged \
  --dtype bfloat16 \
  --device-map auto
```

### 10.4 Output

```text
runs/qwen25vl32b_full_1epoch_0323_merged/
```

---

## 11. Stage F: Structured Pruning (2:4)

对于 NVIDIA 稀疏加速链路，可以在 merged 模型上做 **2:4 structured pruning**。

### 11.1 Reference scripts

- `/data/bian/Finetune_WHU/prune_2to4.py`
- `/data/bian/Finetune_WHU/eval_pruned.py`

### 11.2 What it does

`prune_2to4.py` 会在指定线性层上强制执行 **每 4 个权重保留 2 个** 的结构化稀疏模式。默认作用于：
>>>>>>> 2b4eb096774ec97dc5eeaaad3cb5811cd8291a17

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

<<<<<<< HEAD
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
=======
### 11.3 Example

```bash
python /data/bian/Finetune_WHU/prune_2to4.py \
  /data/sx_files/qwen25vl_sft_chain_0323/runs/qwen25vl32b_full_1epoch_0323_merged \
  /data/sx_files/qwen25vl_sft_chain_0323/runs/qwen25vl32b_full_1epoch_0323_merged_pruned \
  --dtype bf16 \
  --device auto
```

### 11.4 Evaluate pruned model

`eval_pruned.py` 用于对剪枝模型重新评测。

该脚本支持：

- 直接评测 pruned model
- 对比未剪枝 checkpoint

---

## 12. Stage G: Quantization

量化部分主要来自原始 TensorRT-LLM 工程目录：

```text
/data/bian/Finetune_WHU/qwen_tensorrt_llm/quantize.py
```

### 12.1 Supported modes

从脚本定义看，当前量化路径主要支持：
>>>>>>> 2b4eb096774ec97dc5eeaaad3cb5811cd8291a17

- `int4_awq`
- `fp8`

<<<<<<< HEAD
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
=======
同时脚本注释指出：

- 若做 **int8 smoothquant**，应使用 `smoothquant.py`

### 12.2 What quantize.py does

量化脚本会：

1. 加载 HF 模型
2. 构造校准数据
3. 调用 `quantize_and_export`
4. 导出量化结果供后续 TensorRT / TRT-LLM 使用

### 12.3 Example

```bash
python /data/bian/Finetune_WHU/qwen_tensorrt_llm/quantize.py \
  --model_dir /data/sx_files/qwen25vl_sft_chain_0323/runs/qwen25vl32b_full_1epoch_0323_merged \
  --dtype float16 \
  --qformat int4_awq \
  --calib_size 32 \
  --export_path /data/sx_files/qwen25vl_sft_chain_0323/deploy/int4_awq_export.pt
```

### 12.4 Notes

这条量化链路更偏 **TensorRT-LLM / 引擎部署准备**，不是单纯为了 HuggingFace 原生推理。

---

## 13. Stage H: ONNX / TensorRT-LLM Build

如果你要把模型进一步部署到 TensorRT-LLM，可使用以下链路。

### 13.1 One-shot build

原始脚本：

```text
/data/bian/Finetune_WHU/build_qwen25vl_mm.py
```

它直接调用 `MultimodalEngineBuilder(args).build()`。

### 13.2 Two-step build

更稳妥的方式是两步走：

1. `step1_export.py`
2. `step2_build.py`

#### Step 1: export ONNX and save build plan

```bash
python /data/bian/Finetune_WHU/step1_export.py [TensorRT-LLM multimodal builder args...]
```

该脚本会：

- 只执行 PyTorch -> ONNX 导出
- 暂不直接 build TensorRT
- 在 engine 目录写入 `build_plan.json`

#### Step 2: build TensorRT engines

```bash
python /data/bian/Finetune_WHU/step2_build.py --plan_path /path/to/build_plan.json
```

该脚本会：

- 读取 `build_plan.json`
- 在新的进程里真正执行 TensorRT build
- 生成 `.engine`

### 13.3 Recommended usage

如果你的目标是：

- 把 LoRA 模型变成高性能推理引擎
- 减少 Python/HF 推理开销
- 为生产部署准备

推荐顺序为：

```text
LoRA merge -> 量化(可选) -> step1_export -> step2_build
```

---

## 14. Stage I: Deployment

### 14.1 OpenAI-compatible API service

参考脚本：

```text
/data/bian/Finetune_WHU/vllm_service_multimodal.py
```

这个脚本实际上是一个 **FastAPI + HuggingFace/PEFT** 的多模态服务，不是真正意义上的 vLLM 推理内核，但它提供了：

- `/v1/chat/completions`
- OpenAI 风格请求结构
- 图像 + 文本输入
- base64 或本地图片路径输入

### 14.2 What it does

服务端流程为：

1. 加载基座模型
2. 加载 LoRA adapter
3. 解析 OpenAI 风格请求
4. 从 `messages` 中提取图像和文本
5. 调用 `processor + process_vision_info + model.generate`
6. 返回 OpenAI 风格响应

### 14.3 Start service

```bash
python /data/bian/Finetune_WHU/vllm_service_multimodal.py
```

默认端口：

```text
0.0.0.0:9001
```

### 14.4 Suitable scenarios

适合：

- 内网验证
- Demo 服务
- 统一前后端协议
- 给标注系统或业务系统提供 OpenAI-compatible 接口

若你要更高吞吐、更低延迟，建议继续走 TensorRT-LLM 引擎部署。

---

## 15. Recommended End-to-end Workflow

### 15.1 Research / experiment workflow

```text
1. 基座评测
2. LoRA SFT
3. LoRA评测
4. 单图推理
5. 记录实验报告
```

### 15.2 Production-oriented workflow

```text
1. LoRA训练完成
2. merge LoRA -> merged model
3. 剪枝（可选）
4. 剪枝评测
5. 量化（AWQ / FP8 / SmoothQuant）
6. TensorRT-LLM / ONNX / engine build
7. API 或引擎部署
```

---

## 16. Minimal Commands Cheat Sheet

### Baseline eval

```bash
./run_eval_base_0323.sh
```

### LoRA train

```bash
./run_train_0323.sh
```

### LoRA eval

```bash
./run_eval_lora_0323.sh
```

### Single-image inference

```bash
./run_inference_0323.sh
```

### Merge LoRA

```bash
python /data/bian/Finetune_WHU/merge_lora.py BASE_MODEL LORA_CKPT OUTPUT_DIR --dtype bfloat16 --device-map auto
```

### 2:4 pruning

```bash
python /data/bian/Finetune_WHU/prune_2to4.py MERGED_MODEL OUTPUT_DIR --dtype bf16 --device auto
```

### Quantization

```bash
python /data/bian/Finetune_WHU/qwen_tensorrt_llm/quantize.py --model_dir MERGED_MODEL --qformat int4_awq
```

### TensorRT export/build

```bash
python /data/bian/Finetune_WHU/step1_export.py ...
python /data/bian/Finetune_WHU/step2_build.py --plan_path /path/to/build_plan.json
```

### API deploy

```bash
python /data/bian/Finetune_WHU/vllm_service_multimodal.py
```

---

## 17. Output Artifacts

你最终可能会得到几类产物：

### Experiment artifacts

- baseline predictions / metrics
- LoRA predictions / metrics
- inference outputs
- logs / reports

### Model artifacts

- LoRA adapter checkpoint
- merged full model
- pruned model
- quantized export
- TensorRT engine

### Service artifacts

- OpenAI-compatible API service
- engine-based deployment output

---

## 18. Notes for GitHub Publishing

如果你准备把这套链路上传到自己的 GitHub，建议：

1. 把路径参数全部改为配置项或环境变量，不要把 `/data/...` 写死在公开版 README 和脚本里。
2. 在 README 中明确：
   - 哪些是当前仓库已经落地的 `_0323` 脚本
   - 哪些是从原始工程迁移/参考过来的后处理与部署链路
3. 对大模型权重、数据集、TensorRT 引擎等大文件使用：
   - Git LFS
   - 或只保留下载说明，不直接上传
4. 把模型、数据、部署相关内容拆成：
   - `configs/`
   - `scripts/`
   - `deploy/`
   - `docs/`

---

## 19. Related Docs in This Directory
>>>>>>> 2b4eb096774ec97dc5eeaaad3cb5811cd8291a17

- `总纲领_0323.md`
- `ENV_SETUP_0323.md`
- `SFT_experiment_report_0323.md`
- `requirements_0323.txt`
<<<<<<< HEAD
=======

这些文件分别负责：

- 总体方案
- 环境配置
- `_0323` 实验细节
- 依赖安装

---

## 20. Current Focus of This Repository

当前最完整、最直接可运行的是 `_0323` 这条实验链：

```text
baseline eval -> LoRA SFT -> LoRA eval -> inference
```

而 merge / pruning / quantization / TensorRT / deployment 则已经有现成参考脚本，可以继续在此基础上收拢成完整工程版仓库。
>>>>>>> 2b4eb096774ec97dc5eeaaad3cb5811cd8291a17
