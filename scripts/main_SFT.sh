#!/bin/bash

# --- 1. Environment Variables and Paths ---
MODEL_PATH="Qwen2.5-7B-Instruct/"
LABEL_PATH="data/original/labels.txt"
TRAIN_DATA_PATH="data/original/train.tsv"
TEST_DATA_PATH="data/original/test_small.tsv"

TRAINING_SCRIPT="main_SFT.py"


accelerate launch \
    --num_processes 8 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --main_process_port 29500 \
    "$TRAINING_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --label_path "$LABEL_PATH" \
    --train_data_path "$TRAIN_DATA_PATH" \
    --test_data_path "$TEST_DATA_PATH" \
    --max_seq_length 1024

echo "✅ Training launch command executed."