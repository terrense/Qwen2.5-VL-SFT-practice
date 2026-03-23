# 0323 实验环境配置指南

本文档详细说明如何配置 0323 实验的 Python 环境，重点解决 `qwen_vl_utils` 等依赖的安装与验证。

---

## 1. 前置条件

- Linux 系统
- Python ≥ 3.8（推荐 3.10）
- CUDA 11.x 或 12.x
- 显存 ≥ 40GB（32B 模型 + LoRA）

---

## 2. 创建 Conda 环境（推荐）

```bash
conda create -n qwen25vl_sft_0323 python=3.10 -y
conda activate qwen25vl_sft_0323
```

---

## 3. 安装 PyTorch

按本机 CUDA 版本选择：

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

验证：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 4. 安装项目依赖

```bash
cd /data/sx_files/qwen25vl_sft_chain_0323
pip install -r requirements_0323.txt
```

---

## 5. 关键依赖：qwen-vl-utils

### 5.1 作用

`qwen-vl-utils` 提供 `process_vision_info`，用于处理 Qwen-VL 系列模型的图文输入（图片路径、URL、PIL 等），被 `sft_dataset_0323.py`、`eval_base_0323.py`、`eval_lora_0323.py`、`inference_lora_0323.py` 调用。

### 5.2 安装

```bash
pip install qwen-vl-utils
```

- **PyPI 包名**：`qwen-vl-utils`（带连字符）
- **导入名**：`qwen_vl_utils`（下划线）

### 5.3 验证

```bash
python -c "from qwen_vl_utils import process_vision_info; print('qwen_vl_utils OK')"
```

### 5.4 IDE 波浪线问题

若 VSCode/Cursor 中 `from qwen_vl_utils import ...` 出现红色波浪线，通常是因为：

1. **解释器未选对**：确保 IDE 使用的 Python 解释器与 `conda activate qwen25vl_sft_0323` 后的环境一致
2. **包未安装到当前环境**：在项目终端中执行 `pip install qwen-vl-utils`，并重启 IDE 或重新选择解释器

---

## 6. 完整环境验证

运行以下脚本，确认所有依赖可用：

```bash
cd /data/sx_files/qwen25vl_sft_chain_0323
python -c "
errors = []
try:
    from qwen_vl_utils import process_vision_info
    print('[OK] qwen_vl_utils')
except Exception as e:
    errors.append(('qwen_vl_utils', str(e)))

try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    print('[OK] transformers (Qwen2.5-VL)')
except Exception as e:
    errors.append(('transformers', str(e)))

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    print('[OK] peft')
except Exception as e:
    errors.append(('peft', str(e)))

try:
    import torch
    print('[OK] torch, CUDA:', torch.cuda.is_available())
except Exception as e:
    errors.append(('torch', str(e)))

try:
    from modelscope import AutoTokenizer
    print('[OK] modelscope')
except Exception as e:
    errors.append(('modelscope', str(e)))

if errors:
    print('\\n[FAIL]', errors)
else:
    print('\\nAll dependencies OK.')
"
```

---

## 7. 依赖列表概览

| 包名 | 用途 |
|------|------|
| torch, torchvision, torchaudio | 深度学习框架 |
| transformers ≥ 4.49 | Qwen2.5-VL 模型加载 |
| peft ≥ 0.7 | LoRA 微调 |
| qwen-vl-utils | 图文输入预处理 |
| modelscope | Tokenizer 加载（国内镜像友好） |
| accelerate | 分布式训练支持 |
| scikit-learn | 评测指标 |
| tqdm | 进度条 |
| pillow | 图像处理 |

---

## 8. 若使用国内镜像

```bash
pip install -r requirements_0323.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

若 `qwen-vl-utils` 在镜像中不可用，可单独从 PyPI 安装：

```bash
pip install qwen-vl-utils
```
