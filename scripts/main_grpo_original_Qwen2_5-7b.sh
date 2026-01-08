#!/bin/bash

# --- 1. 定义可传入的参数 ---
MODEL_PATH="Qwen2.5-7B-Instruct/"
LABEL_PATH="data/original/labels.txt"
TRAIN_PATH="data/original/train.tsv"
TEST_PATH="data/original/test_small.tsv"
# -----------------------------

# --- 指定 accelerate 可执行文件的路径 ---
# accelerate 可执行文件通常在 anaconda3/envs/ssl/bin/accelerate
ACCELERATE_BIN="anaconda3/envs/ssl/bin/accelerate"
# ----------------------------------------


export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TOKENIZERS_PARALLELISM=false
# export TORCH_USE_DDP_STATIC_GRAPH=1 
export WANDB_MODE=disabled

SCRIPT_NAME="main_grpo.py"

# --- 2. 构造新的 OUTPUT_DIR 和 LOG_DIR ---
BASE_OUTPUT_DIR="grpo_output"

# 提取模型名称
# 使用 readlink -f 获取绝对路径并处理斜杠，然后用 basename 提取
MODEL_NAME=$(basename $(readlink -f "$MODEL_PATH" | sed 's/\/$//'))

# 提取数据组名 (假设它是 $LABEL_PATH 中 data/ 后面的目录名)
DATA_GROUP=$(basename $(dirname "$LABEL_PATH"))

# 构造最终的 OUTPUT_DIR (与 Python 脚本中的逻辑保持一致)
OUTPUT_DIR="$BASE_OUTPUT_DIR/$MODEL_NAME/$DATA_GROUP"

# LOG_DIR 放在最终的 OUTPUT_DIR 下
LOG_DIR="$OUTPUT_DIR/logs" 
LOG_FILE="$LOG_DIR/training_log.log"

# ---------------------------------------------

mkdir -p "$LOG_DIR"

echo "----------------------------------------"
echo "Starting GRPO training using accelerate from specified environment"
echo "Accelerate Path: $ACCELERATE_BIN"
echo "Script: $SCRIPT_NAME"
echo "Final Model Output Directory: $OUTPUT_DIR"
echo "Logging to: $LOG_FILE"
echo "----------------------------------------"

# 使用绝对路径运行 accelerate launch
"$ACCELERATE_BIN" launch \
    --multi_gpu \
    --num_machines 1 \
    --num_processes 8 \
    "$SCRIPT_NAME" \
    --model_path "$MODEL_PATH" \
    --label_path "$LABEL_PATH" \
    --train_data_path "$TRAIN_PATH" \
    --test_data_path "$TEST_PATH" \
    2>&1 | tee "$LOG_FILE"

echo "----------------------------------------"
echo "Training finished."
echo "----------------------------------------"