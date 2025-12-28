import os
import re
import torch
import pandas as pd
import argparse
from typing import List, Dict, Union
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    # 移除 TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
# 导入 trl 库的核心组件
from trl import SFTTrainer, SFTConfig 
import json 
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="SFT training script for GoEmotions classification.")
    parser.add_argument(
        "--model_path",
        type=str,
        default='/inspire/hdd/project/exploration-topic/public/downloaded_ckpts/Qwen2.5-7B-Instruct/',
        help="Path to the pretrained CausalLM model."
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default="/inspire/hdd/project/exploration-topic/public/lzjjin/course/Goemotions/data/group/labels.txt",
        help="Path to the labels.txt file."
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="/inspire/hdd/project/exploration-topic/public/lzjjin/course/Goemotions/data/group/train.tsv",
        help="Path to the training data TSV file."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="/inspire/hdd/project/exploration-topic/public/lzjjin/course/Goemotions/data/group/test_small.tsv",
        help="Path to the testing/evaluation data TSV file."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenization."
    )
    return parser.parse_args()

# --- 2. 数据加载和处理 ---

def load_labels(label_path: str) -> List[str]:
    """加载情感标签列表"""
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found at {label_path}")
    with open(label_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# --- 核心格式化函数 (Mapper) ---
def format_goemotions_example(
    example: Dict[str, Union[str, int]], 
    labels: List[str], 
    system_prompt: str
) -> Union[Dict, None]: 
    """
    将 GoEmotions 的单个记录格式化为您要求的 prompt/completion 结构。
    """
    
    # 1. 解析标签 ID
    try:
        ids = [int(x) for x in str(example['label_ids']).split(',') if x.strip()]
    except (ValueError, AttributeError):
        return None

    if not ids: 
        return None 
        
    true_label_names = ", ".join([labels[i] for i in ids])
    target_response = f"<answer>{true_label_names}</answer>"
    
    # 2. 构造 prompt 
    prompt = [ 
        {'role': 'system', 'content': system_prompt}, 
        {'role': 'user', 'content': f"Text: \"{example['text']}\""}, 
    ] 
    
    # 3. 构造 completion 
    completion = [ 
        {'role': 'assistant', 'content': target_response} 
    ] 
    
    return { 
        'prompt': prompt, 
        'completion': completion,
    }

# --- 主要 Dataset 返回函数 ---
def prepare_dataset(
    data_path: str, 
    label_path: str, 
) -> Dataset:

    
    labels = load_labels(label_path)
    label_str = ", ".join(labels)
    
    SYSTEM_PROMPT_SFT = (
        f"You are an emotion classification expert. Identify one or more emotions in the text from the following list:\n"
        f"[{label_str}]\n\n"
        f"Format your final answer strictly as: <answer>emotion1, emotion2</answer>\n"
        f"If there is only one emotion: <answer>emotion1</answer>"
    )

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
        
    df = pd.read_csv(data_path, sep='\t', header=None, names=['text', 'label_ids', 'comment_id'])
    dataset = Dataset.from_pandas(df)

    from functools import partial
    mapper_func = partial(
        format_goemotions_example, 
        labels=labels, 
        system_prompt=SYSTEM_PROMPT_SFT
    )
    
    dataset = dataset.map(
        mapper_func, 
        remove_columns=df.columns.tolist(),
        batched=False
    )
    
    dataset = dataset.filter(lambda x: x is not None)
    return dataset

# --- 3. 评估指标计算 ---
def compute_metrics(eval_pred, labels: List[str]):
    """
    SFT 占位评估函数，因为 Trainer 默认只计算 Token 级损失，不自动计算 F1。
    """
    return {"placeholder_metric": 0.0}

# --- 4. 主函数 ---

def main():
    args = parse_args()
    
    DEFAULT_MODEL_PATH = args.model_path
    DEFAULT_LABEL_PATH = args.label_path
    TRAIN_DATA_PATH = args.train_data_path
    TEST_DATA_PATH = args.test_data_path
    MAX_SEQ_LENGTH = args.max_seq_length

    # 路径配置
    model_name = os.path.basename(os.path.normpath(DEFAULT_MODEL_PATH))

    try:
        path_parts = os.path.normpath(DEFAULT_LABEL_PATH).split(os.sep)
        data_index = path_parts.index('data')
        data_group = path_parts[data_index + 1]
    except (ValueError, IndexError):
        data_group = "default_data_type"

    OUTPUT_DIR = os.path.join(
        "/inspire/hdd/project/exploration-topic/public/lzjjin/course/Goemotions/sft_output",
        model_name,
        data_group
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Set OUTPUT_DIR to: {OUTPUT_DIR}")     
    
    labels = load_labels(DEFAULT_LABEL_PATH)

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    
    print("Preparing train dataset...")
    train_dataset = prepare_dataset(TRAIN_DATA_PATH, DEFAULT_LABEL_PATH)
    
    print("Preparing eval dataset...")
    eval_dataset = prepare_dataset(TEST_DATA_PATH, DEFAULT_LABEL_PATH)
    
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    
    # 4. 配置 PEFT (LoRA)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
    )
    
    model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        adam_beta1=0.9,       
        adam_beta2=0.99,      
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10, 
        bf16=True, 
        per_device_train_batch_size=32, 
        gradient_accumulation_steps=2, 
        per_device_eval_batch_size=64,
        num_train_epochs=2.0,
        save_steps=100,
        eval_steps=100, 
        max_grad_norm=0.1,
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
        save_total_limit=3,
        greater_is_better=False,
        max_length=MAX_SEQ_LENGTH, 
        completion_only_loss=True, 
        remove_unused_columns=False, 
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, labels)
    )

    print("Starting SFT training...")
    trainer.train()
    

    final_output_dir = os.path.join(OUTPUT_DIR, "final_checkpoint")

    os.makedirs(final_output_dir, exist_ok=True)
    
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"Model saved to: {final_output_dir}")

if __name__ == "__main__":
    main()