# FULL_CHAIN_RUNBOOK_0323

这份文档面向一个非常具体的使用场景：

- 你只拿到了目录 `qwen25vl_sft_chain_0323/`
- 你要从零开始把这条 chain 尽量完整地跑一遍
- 你希望知道每一步该做什么、输入是什么、会产出什么、什么算跑通

本文默认你当前目录为：

```text
/data/sx_files/qwen25vl_sft_chain_0323
```

---

## 1. 先理解这条链是什么

当前目录中的完整链路分成三层：

### 1.1 实验层

```text
baseline eval
-> LoRA SFT train
-> LoRA eval
-> single-image inference
```

这部分的目标是确认：

- 数据能读
- 模型能正常 generate
- LoRA 能训练
- 训练后指标能回收

### 1.2 后处理层

```text
merge
-> prune
-> pruned / merged eval
```

这部分的目标是把 LoRA adapter 变成一个可独立加载的完整模型，并验证后处理没有把模型弄坏。

### 1.3 部署层

```text
quantize
-> TensorRT step1 export
-> TensorRT step2 build
-> API deploy
```

这部分的目标是为高性能部署或服务化准备产物。

---

## 2. 哪些步骤一定要跑，哪些可以选跑

### 2.1 最小实验闭环

如果你只是想确认 SFT 实验完整可复现，最少跑这些：

1. `run_eval_base_0323.sh`
2. `run_train_0323.sh`
3. `run_eval_lora_0323.sh`
4. `run_inference_0323.sh`

### 2.2 完整工程闭环

如果你想把整个 chain 都走完，按这个顺序：

1. `run_eval_base_0323.sh`
2. `run_train_0323.sh`
3. `run_eval_lora_0323.sh`
4. `run_inference_0323.sh`
5. `scripts/run_merge_lora_0323.sh`
6. `scripts/run_prune_0323.sh`
7. `scripts/run_eval_pruned_0323.sh`
8. `scripts/run_quantize_0323.sh`
9. `scripts/run_export_onnx_0323.sh ...`
10. `scripts/run_build_engine_0323.sh --plan-path ...`
11. `scripts/run_deploy_api_0323.sh --mode merged`

注意：

- 第 1 到 7 步是当前目录的 HuggingFace / LoRA 主链。
- 第 8 到 11 步依赖 TensorRT-LLM / TensorRT 部署环境，不是普通 Python 环境就能跑。

---

## 3. 拿到目录后的第一次检查

先进入目录：

```bash
cd /data/sx_files/qwen25vl_sft_chain_0323
```

建议先确认这些文件存在：

```text
experiment_config_0323.py
requirements_0323.txt
train_lora_full_0323.py
eval_base_0323.py
eval_lora_0323.py
inference_lora_0323.py
post_sft/merge_lora_0323.py
post_sft/prune_2to4_0323.py
post_sft/eval_pruned_0323.py
quantization/quantize_0323.py
trt/step1_export_0323.py
trt/step2_build_0323.py
deploy/deploy_api_0323.py
scripts/run_merge_lora_0323.sh
scripts/run_prune_0323.sh
scripts/run_eval_pruned_0323.sh
scripts/run_quantize_0323.sh
scripts/run_export_onnx_0323.sh
scripts/run_build_engine_0323.sh
scripts/run_deploy_api_0323.sh
```

然后给脚本执行权限：

```bash
chmod +x run_*.sh scripts/*.sh
```

---

## 4. 环境准备

### 4.1 已验证可用的关键版本

当前项目已经验证过能通过 `Qwen2.5-VL generate()` 的核心组合：

- `transformers==4.57.5`
- `torch==2.9.0+cu128`
- `torchvision==0.24.0+cu128`
- `torchaudio==2.9.0+cu128`
- `triton==3.5.0`
- `peft==0.18.1`
- `accelerate==1.13.0`

如果你后面又遇到 `Floating point exception`，优先先看是不是这两个版本变了：

- `transformers`
- `torch`

### 4.2 基础 Python 环境安装

推荐使用独立环境。

示例：

```bash
conda create -n qwen25vl_sft_0323 python=3.10 -y
conda activate qwen25vl_sft_0323
```

先安装 PyTorch：

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
```

再安装项目依赖：

```bash
pip install -r requirements_0323.txt
```

### 4.3 核心依赖验证

验证 `qwen_vl_utils`：

```bash
python -c "from qwen_vl_utils import process_vision_info; print('OK')"
```

验证 `transformers + peft + torch`：

```bash
python -c "import torch; import transformers; import peft; print(torch.__version__, transformers.__version__, peft.__version__)"
```

如果这里都过不了，不要继续跑主链。

---

## 5. 固定路径说明

这个 `_0323` 项目不是通用模板版，而是已经绑定了当前机器上的绝对路径。

默认关键路径如下：

- Base model: `/data/bian/model/Qwen/Qwen2.5-VL-32B-Instruct`
- Train JSON: `/data/bian/Finetune_WHU/data_vl_all_face_train_Full_260310.json`
- Val JSON: `/data/bian/Finetune_WHU/data_vl_all_face_val_nopain.json`
- Experiment root: `/data/sx_files/qwen25vl_sft_chain_0323`

这些都集中定义在：

- `experiment_config_0323.py`

如果你换机器、换目录、换模型、换数据，优先改这个文件。

---

## 6. 全流程执行手册

下面是“拿到这个目录以后，完整走一遍”的推荐顺序。

---

## 6.1 Step 0: 进入目录

```bash
cd /data/sx_files/qwen25vl_sft_chain_0323
```

成功标志：

- 你当前就在项目根目录
- `run_*.sh` 和 `scripts/*.sh` 都能看见

---

## 6.2 Step 1: 基座模型评测

### 作用

建立 baseline，确认基座模型本身能在验证集上跑通。

### 命令

```bash
./run_eval_base_0323.sh
```

### 输入

- `BASE_MODEL_PATH_0323`
- `VAL_JSON_PATH_0323`

### 输出

```text
reports/base_eval_0323/base_predictions_0323.json
reports/base_eval_0323/base_metrics_0323.json
reports/base_eval_0323/base_detailed_metrics_0323.json
```

### 成功标志

- 脚本跑完无崩溃
- `reports/base_eval_0323/` 下出现 3 个 JSON
- `base_metrics_0323.json` 中能看到 `exact_match`、`json_error_rate` 等字段

### 失败先查什么

- `qwen_vl_utils` 是否安装
- `transformers` / `torch` 是否回退
- 数据路径是否存在

---

## 6.3 Step 2: LoRA 训练

### 作用

对 `Qwen2.5-VL-32B-Instruct` 做 1 epoch SFT。

### 命令

```bash
./run_train_0323.sh
```

如果想显式指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 ./run_train_0323.sh
```

### 输入

- `TRAIN_JSON_PATH_0323`
- `BASE_MODEL_PATH_0323`
- `experiment_config_0323.py` 中的 LoRA 和训练超参

### 输出

```text
runs/qwen25vl32b_full_1epoch_0323/
```

通常会包含：

- `adapter_config.json`
- adapter 权重
- checkpoint 子目录
- tokenizer / processor 文件

### 成功标志

- 训练完整跑完
- `runs/qwen25vl32b_full_1epoch_0323/` 目录存在
- 至少有 `adapter_config.json` 和权重文件

### 失败先查什么

- 显存是否足够
- 训练数据 JSON 是否能正确读取
- 当前环境里的 `torch` / `transformers` 是否是已验证版本

---

## 6.4 Step 3: LoRA 模型评测

### 作用

在验证集上评估 LoRA adapter 的效果。

### 命令

```bash
./run_eval_lora_0323.sh
```

### 前置条件

- Step 2 已完成
- `runs/qwen25vl32b_full_1epoch_0323/` 中存在 LoRA 产物

### 输出

```text
reports/lora_eval_0323/lora_predictions_0323.json
reports/lora_eval_0323/lora_metrics_0323.json
reports/lora_eval_0323/lora_detailed_metrics_0323.json
```

### 成功标志

- `reports/lora_eval_0323/` 下出现评测结果
- 指标文件可打开并看到四字段结果

### 你应该做的比较

把这一阶段输出和 Step 1 的 baseline 指标做比较，重点看：

- `exact_match`
- `json_error_rate`
- 四个字段的分类指标

---

## 6.5 Step 4: 单图推理验证

### 作用

用单张图片快速检查 LoRA 模型有没有最基本的可用性。

### 命令

```bash
./run_inference_0323.sh
```

### 输入

- `SAMPLE_IMAGE_PATH_0323`
- 默认 prompt

### 成功标志

- 能输出一段合法 JSON 或接近合法 JSON 的文本
- 没有在 `generate()` 阶段直接崩掉

### 如果没跑通

先不要进入 merge / prune / quantize，先把前面的基础链修通。

---

## 6.6 Step 5: Merge LoRA

### 作用

把 LoRA adapter 合并进基座模型，导出一个独立完整模型，后面剪枝、量化、部署都基于它。

### 命令

```bash
./scripts/run_merge_lora_0323.sh
```

### 等价手动命令

```bash
python post_sft/merge_lora_0323.py \
  --base-model /data/bian/model/Qwen/Qwen2.5-VL-32B-Instruct \
  --lora-checkpoint /data/sx_files/qwen25vl_sft_chain_0323/runs/qwen25vl32b_full_1epoch_0323 \
  --output-dir /data/sx_files/qwen25vl_sft_chain_0323/post_sft/merged/qwen25vl32b_full_1epoch_merged_0323 \
  --dtype bfloat16 \
  --device-map auto
```

### 输出

```text
post_sft/merged/qwen25vl32b_full_1epoch_merged_0323/
```

### 成功标志

- merge 后目录存在
- 目录里既有模型权重，也有 tokenizer / processor 文件

### 失败先查什么

- LoRA 目录是否存在
- LoRA 是否是和当前 base model 匹配的 adapter

---

## 6.7 Step 6: 2:4 结构化剪枝

### 作用

在 merged 模型上执行 2:4 structured sparsity。

### 命令

```bash
./scripts/run_prune_0323.sh
```

### 默认输出

```text
post_sft/pruned/qwen25vl32b_full_1epoch_merged_pruned_0323/
```

### 成功标志

- 脚本输出中能看到目标 Linear 层被处理
- pruned 模型目录生成成功

### 注意

这是“结构化稀疏化”步骤，不是普通 unstructured pruning。

---

## 6.8 Step 7: 剪枝后模型评测

### 作用

检查 prune 之后模型还能不能工作，以及指标下降多少。

### 命令

```bash
./scripts/run_eval_pruned_0323.sh
```

### 默认输出

```text
reports/pruned_eval_0323/pruned_model_0323/
```

### 成功标志

- 生成预测文件
- 生成 metrics 文件
- 能和 Step 3 的 LoRA 评测结果做对比

### 推荐比较对象

- `LoRA eval` vs `merged eval`
- `merged eval` vs `pruned eval`

如果 prune 后指标直接崩掉，先不要继续量化和 TensorRT。

---

## 6.9 Step 8: 量化导出

### 作用

把 merged 模型导出成后续 TensorRT-LLM 可用的量化结果。

### 命令

```bash
./scripts/run_quantize_0323.sh
```

### 支持格式

- `int4_awq`
- `fp8`

### 默认输出

```text
quantization/exports_0323/
```

### 成功标志

- 导出 `.pt` 结果文件
- 日志中没有 `tensorrt_llm` 相关 import 报错

### 这一步的前置条件

这一段不是普通训练环境，通常要求：

- 已安装 `tensorrt_llm`
- 已安装 TensorRT
- CUDA / TensorRT / TensorRT-LLM 版本彼此匹配

如果你没有单独部署环境，这一步可以暂时跳过。

---

## 6.10 Step 9: TensorRT Step1 导出 ONNX

### 作用

执行 PyTorch -> ONNX 导出，并记录后续 build 所需计划文件。

### 命令形式

```bash
./scripts/run_export_onnx_0323.sh [TensorRT-LLM builder args...]
```

### 关键说明

这一步不是固定一个命令，因为 `MultimodalEngineBuilder` 本身需要你传模型类型、输入尺寸、输出目录等参数。

### 成功标志

- ONNX 导出成功
- 在 engine 目录写出 `build_plan_0323.json`

### 失败先查什么

- TensorRT-LLM 环境是否完整
- builder 参数是否给全
- ONNX 导出目录是否可写

---

## 6.11 Step 10: TensorRT Step2 构建 engine

### 作用

读取 Step 9 的 `build_plan_0323.json`，在独立进程里真正构建 TensorRT engine。

### 命令

```bash
./scripts/run_build_engine_0323.sh --plan-path /path/to/build_plan_0323.json
```

### 成功标志

- engine 目录里生成 `.engine`
- 构建日志正常结束

### 注意

这一步通常很耗时，也非常依赖显卡、CUDA、TensorRT 环境。

---

## 6.12 Step 11: 启动 API 服务

### 作用

把当前模型作为一个 OpenAI-compatible 多模态服务暴露出来。

### 默认命令

```bash
./scripts/run_deploy_api_0323.sh --mode merged
```

如果想直接加载 base + LoRA：

```bash
./scripts/run_deploy_api_0323.sh --mode lora
```

### 默认接口

```text
POST /v1/chat/completions
```

### 默认端口

```text
0.0.0.0:9001
```

### 成功标志

- 服务成功启动
- 控制台无模型加载报错
- 能通过接口发图文请求拿到回复

---

## 7. 一张总表：每一步的输入、输出、成功标志

| Step | 命令 | 主要输入 | 主要输出 | 成功标志 |
|------|------|----------|----------|----------|
| 1 | `./run_eval_base_0323.sh` | base model + val json | `reports/base_eval_0323/` | 生成 base metrics |
| 2 | `./run_train_0323.sh` | base model + train json | `runs/qwen25vl32b_full_1epoch_0323/` | 出现 LoRA adapter 与 checkpoint |
| 3 | `./run_eval_lora_0323.sh` | base model + LoRA + val json | `reports/lora_eval_0323/` | 生成 LoRA metrics |
| 4 | `./run_inference_0323.sh` | sample image + prompt | 控制台输出 | 输出合法或接近合法 JSON |
| 5 | `./scripts/run_merge_lora_0323.sh` | base model + LoRA | `post_sft/merged/...` | merged 目录存在 |
| 6 | `./scripts/run_prune_0323.sh` | merged model | `post_sft/pruned/...` | pruned 目录存在 |
| 7 | `./scripts/run_eval_pruned_0323.sh` | pruned 或 merged model + val json | `reports/pruned_eval_0323/` | 生成 pruned metrics |
| 8 | `./scripts/run_quantize_0323.sh` | merged model | `quantization/exports_0323/` | 导出量化 pt |
| 9 | `./scripts/run_export_onnx_0323.sh ...` | TensorRT builder args | ONNX + `build_plan_0323.json` | 计划文件生成 |
| 10 | `./scripts/run_build_engine_0323.sh --plan-path ...` | build plan | TensorRT engine | `.engine` 生成 |
| 11 | `./scripts/run_deploy_api_0323.sh --mode merged` | merged model | 本地 API 服务 | `/v1/chat/completions` 可访问 |

---

## 8. 推荐的实际执行策略

### 8.1 如果你先求稳

先只跑到这里：

```text
Step 1 -> Step 2 -> Step 3 -> Step 4 -> Step 5
```

原因：

- 这几步完全属于当前已经验证过的 HuggingFace / LoRA 主链
- 能先验证训练和 merge 逻辑都没问题

### 8.2 如果你要继续后处理

接着跑：

```text
Step 6 -> Step 7
```

先确认 prune 没把模型剪坏。

### 8.3 如果你要走部署

最后再跑：

```text
Step 8 -> Step 9 -> Step 10 -> Step 11
```

这几步最依赖外部部署环境，不建议和基础实验混在一起排障。

---

## 9. 常见失败点

### 9.1 `Floating point exception`

优先检查：

- `transformers==4.57.5`
- `torch==2.9.0+cu128`

### 9.2 `ModuleNotFoundError: qwen_vl_utils`

执行：

```bash
pip install qwen-vl-utils
```

### 9.3 LoRA 评测找不到 checkpoint

说明 Step 2 训练还没真正产出完整 adapter。

### 9.4 quantize / trt 脚本 import 报错

说明你现在是“训练环境”，不是“部署环境”。这不是代码没准备好，而是部署依赖还没装齐。

---

## 10. 最后结论

如果你拿到这个目录，想完整走一遍 chain，建议按下面的现实顺序：

```text
先跑通实验主链：
baseline -> train -> lora eval -> inference

再跑通模型后处理：
merge -> prune -> pruned eval

最后再进部署链：
quantize -> step1 export -> step2 build -> api deploy
```

这样分层推进，最不容易在排障时把问题混在一起。
