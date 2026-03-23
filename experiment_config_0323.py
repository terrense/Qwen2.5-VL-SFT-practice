"""
0323 实验配置文件
统一管理路径、LoRA 参数、训练超参数等，供训练、评测、推理脚本共用。
"""

from pathlib import Path  # 用于跨平台路径操作，支持 / 拼接、exists()、mkdir() 等


# ========== 路径配置 ==========
EXPERIMENT_ROOT_0323 = Path("/data/sx_files/qwen25vl_sft_chain_0323")
# 实验根目录，所有输出子目录（runs、reports）的父路径

SOURCE_ROOT_0323 = Path("/data/bian/Finetune_WHU")
# 数据根目录：训练/验证 JSON、图片相对路径均基于此解析

BASE_MODEL_PATH_0323 = Path("/data/bian/model/Qwen/Qwen2.5-VL-32B-Instruct")
# 基座模型路径，训练/评测/推理均从此加载 Qwen2.5-VL-32B-Instruct

# 训练与验证数据
TRAIN_JSON_PATH_0323 = SOURCE_ROOT_0323 / "data_vl_all_face_train_Full_260310.json"
# 训练集 JSON：每行为一条对话样本，含 conversations[0]（prompt+图片路径）和 conversations[1]（标签 JSON）

VAL_JSON_PATH_0323 = SOURCE_ROOT_0323 / "data_vl_all_face_val_nopain.json"
# 验证集 JSON，格式同训练集，用于评测基座与 LoRA 模型

# 推理用示例图片
SAMPLE_IMAGE_PATH_0323 = (
    SOURCE_ROOT_0323
    / "qwenvl_dataset/gender/train/man/identity_1991_none_06_a831c5ecca72efa198fc_img_00197.jpg"
)
# inference_lora_0323.py 默认使用的单图路径，可在此修改

# 输出目录
RUN_ROOT_0323 = EXPERIMENT_ROOT_0323 / "runs" / "qwen25vl32b_full_1epoch_0323"
# LoRA 训练输出目录：adapter、tokenizer、processor、checkpoint 均保存在此

BASE_EVAL_ROOT_0323 = EXPERIMENT_ROOT_0323 / "reports" / "base_eval_0323"
# 基座模型评测结果目录：base_predictions_0323.json、base_metrics_0323.json 等

LORA_EVAL_ROOT_0323 = EXPERIMENT_ROOT_0323 / "reports" / "lora_eval_0323"
# LoRA 模型评测结果目录：lora_predictions_0323.json、lora_metrics_0323.json

INFERENCE_ROOT_0323 = EXPERIMENT_ROOT_0323 / "reports" / "inference_0323"
# 推理输出目录（当前脚本未写入，预留扩展）

# ========== 后续工程链路目录 ==========
POST_SFT_ROOT_0323 = EXPERIMENT_ROOT_0323 / "post_sft"
# SFT 之后的模型处理链路根目录：merge、prune、pruned eval 等

MERGED_MODEL_ROOT_0323 = POST_SFT_ROOT_0323 / "merged" / "qwen25vl32b_full_1epoch_merged_0323"
# merge_lora_0323.py 的默认输出目录：基座 + LoRA 合并后的完整模型

PRUNED_MODEL_ROOT_0323 = POST_SFT_ROOT_0323 / "pruned" / "qwen25vl32b_full_1epoch_merged_pruned_0323"
# 2:4 结构化剪枝后的模型目录

PRUNED_EVAL_ROOT_0323 = EXPERIMENT_ROOT_0323 / "reports" / "pruned_eval_0323"
# 剪枝后模型的评测结果目录

QUANTIZATION_ROOT_0323 = EXPERIMENT_ROOT_0323 / "quantization"
# 量化链路根目录：导出的 AWQ / FP8 / PTQ 产物统一放这里

QUANT_EXPORT_ROOT_0323 = QUANTIZATION_ROOT_0323 / "exports_0323"
# quantize_0323.py 的默认导出目录

AWQ_EXPORT_PATH_0323 = QUANT_EXPORT_ROOT_0323 / "qwen25vl32b_merged_int4_awq_0323.pt"
# 默认 int4_awq 导出文件

FP8_EXPORT_PATH_0323 = QUANT_EXPORT_ROOT_0323 / "qwen25vl32b_merged_fp8_0323.pt"
# 默认 fp8 导出文件

TRT_ROOT_0323 = EXPERIMENT_ROOT_0323 / "trt"
# TensorRT / TensorRT-LLM 构建目录

TRT_ONNX_ROOT_0323 = TRT_ROOT_0323 / "onnx_0323"
# Step1 导出的 ONNX 与 build_plan.json 相关中间产物目录

TRT_ENGINE_ROOT_0323 = TRT_ROOT_0323 / "engines_0323"
# Step2 构建出的 TensorRT engine 目录

DEPLOY_ROOT_0323 = EXPERIMENT_ROOT_0323 / "deploy"
# API 服务部署相关目录

DEPLOY_SERVICE_ROOT_0323 = DEPLOY_ROOT_0323 / "service_0323"
# FastAPI / OpenAI-compatible 服务默认工作目录

SCRIPTS_ROOT_0323 = EXPERIMENT_ROOT_0323 / "scripts"
# 新增运行脚本统一放在 scripts/ 下，避免继续堆在根目录

# Conda 环境路径（可选，用于脚本激活）
CONDA_ENV_PATH_0323 = Path("/data/sx_files/.envs/qwen25vl_sft_0323")

# ========== 训练与推理 ==========
CUDA_VISIBLE_DEVICES_0323 = "0"
# 指定使用哪块 GPU，如 "0,1" 表示用 0 和 1 号卡；影响 os.environ["CUDA_VISIBLE_DEVICES"]

# LoRA 目标模块（Qwen2.5-VL 的 MLP 与注意力投影层）
LORA_TARGET_MODULES_0323 = [
    "q_proj",      # 注意力 Query 投影，LoRA 注入可微调检索表示
    "k_proj",      # 注意力 Key 投影
    "v_proj",      # 注意力 Value 投影
    "o_proj",      # 注意力输出投影
    "gate_proj",   # MLP gate 投影（SwiGLU 结构）
    "up_proj",     # MLP 上投影
    "down_proj",   # MLP 下投影
]
# 这些层会被替换为 原层 + LoRA(原层)，仅训练 LoRA 参数，大幅降低显存与计算

LORA_R_0323 = 64
# LoRA 秩（rank）：低秩矩阵的维度，越大表达能力越强但参数量与显存也越大；64 为常用中等值

LORA_ALPHA_0323 = 16
# LoRA 缩放因子：输出会乘以 alpha/r，用于控制 LoRA 对原层的贡献强度；r=64 时 16/64=0.25

LORA_DROPOUT_0323 = 0.05
# LoRA 层 dropout，训练时随机丢弃 5% 以减轻过拟合

# 序列与生成
MAX_LENGTH_0323 = 8192
# 训练时 input_ids 最大长度，超长则截断；Qwen2.5-VL 支持长序列

MAX_NEW_TOKENS_0323 = 128
# 推理时生成的最大 token 数，JSON 输出通常 50~100 token 足够

TRAIN_EPOCHS_0323 = 1
# 训练轮数，1 表示全量数据过一遍


def ensure_output_dirs_0323() -> None:
    """
    创建所有输出目录，避免运行时路径不存在。
    parents=True：自动创建父目录
    exist_ok=True：目录已存在则不报错
    """
    for path in [
        EXPERIMENT_ROOT_0323,
        RUN_ROOT_0323,
        BASE_EVAL_ROOT_0323,
        LORA_EVAL_ROOT_0323,
        INFERENCE_ROOT_0323,
        POST_SFT_ROOT_0323,
        MERGED_MODEL_ROOT_0323.parent,
        PRUNED_MODEL_ROOT_0323.parent,
        PRUNED_EVAL_ROOT_0323,
        QUANTIZATION_ROOT_0323,
        QUANT_EXPORT_ROOT_0323,
        TRT_ROOT_0323,
        TRT_ONNX_ROOT_0323,
        TRT_ENGINE_ROOT_0323,
        DEPLOY_ROOT_0323,
        DEPLOY_SERVICE_ROOT_0323,
        SCRIPTS_ROOT_0323,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_image_path_0323(raw_path: str) -> str:
    """
    将数据中的图片路径解析为绝对路径。
    支持三种形式：
    1. 绝对路径：直接返回
    2. ./ 开头的相对路径：基于 SOURCE_ROOT_0323 解析
    3. 其它相对路径：同样基于 SOURCE_ROOT_0323
    返回 str 以便传入 processor / process_vision_info
    """
    raw_path = raw_path.strip()  # 去掉首尾空白
    candidate = Path(raw_path)
    if candidate.is_absolute():
        # 已是绝对路径，无需处理
        return str(candidate)

    if raw_path.startswith("./"):
        # ./data_file/xxx.jpg -> SOURCE_ROOT / data_file/xxx.jpg
        candidate = SOURCE_ROOT_0323 / raw_path[2:]
        return str(candidate.resolve())

    # 如 data_file/xxx.jpg
    candidate = SOURCE_ROOT_0323 / raw_path
    return str(candidate.resolve())


def lora_output_dir_0323() -> str:
    """
    返回 LoRA 训练输出目录的字符串路径。
    Trainer 的 output_dir 需 str 类型。
    """
    return str(RUN_ROOT_0323)


def merged_model_dir_0323() -> str:
    """返回 merge 后完整模型的字符串路径。"""
    return str(MERGED_MODEL_ROOT_0323)


def pruned_model_dir_0323() -> str:
    """返回结构化剪枝后模型的字符串路径。"""
    return str(PRUNED_MODEL_ROOT_0323)
