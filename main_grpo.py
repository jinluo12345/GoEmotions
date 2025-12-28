import os
import re
import torch
import pandas as pd
import argparse
from typing import List, Dict
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
import json 

# --- 1. 定义命令行参数解析 ---
def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training script for GoEmotions classification.")
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
    return parser.parse_args()

# --- 2. 核心逻辑修改：使用传入的参数 ---

def load_labels(label_path: str) -> List[str]:
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"{label_path}")
    with open(label_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def prepare_dataset(data_path: str, labels: List[str]):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path}")
    
    df = pd.read_csv(data_path, sep='\t', header=None, names=['text', 'label_ids', 'comment_id'])
    
    label_str = ", ".join(labels)
    system_prompt = (
        f"You are an emotion classification expert. Identify one or more emotions in the text from the following list:\n"
        f"[{label_str}]\n\n"
        f"First, analyze the text step-by-step to determine the underlying emotions.\n"
        f"Put your thinking process inside <think> ...</think> tags.\n" 
        f"Then, format your final answer strictly as: <answer>emotion1, emotion2</answer>\n"
        f"If there is only one emotion: <answer>emotion1</answer>"
    )

    data_records = []
    for _, row in df.iterrows():
        ids = [int(x) for x in str(row['label_ids']).split(',')]
        if not ids: 
             continue
        true_label_names = [labels[i] for i in ids]
        
        full_prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Text: \"{row['text']}\""}
        ]
        
        data_records.append({
            "prompt": full_prompt,
            "ground_truth": true_label_names
        })
        
    return Dataset.from_list(data_records)



def accuracy_reward_func(prompts, completions, ground_truth, **kwargs) -> List[float]:
    rewards = []
    responses = [c[0]['content'] for c in completions]
    
    for response, true_labels in zip(responses, ground_truth):
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if not match:
            rewards.append(0.0)
            continue
            
        content = match.group(1)
        pred_labels = set([x.strip().lower() for x in content.split(',') if x.strip()])
        true_labels_set = set([x.lower() for x in true_labels])
        
        if not pred_labels and not true_labels_set:
            rewards.append(1.0)
            continue
        elif not pred_labels:
            rewards.append(0.0)
            continue

        intersection = pred_labels.intersection(true_labels_set)
        union = pred_labels.union(true_labels_set)
        
        score = len(intersection) / len(union) if union else 0.0
        rewards.append(score)
        
    return rewards

def len_reward_func(completions, **kwargs) -> List[float]:
    responses = [c[0]['content'] for c in completions]
    return [-len(r) * 0.001 for r in responses]

def main():
    args = parse_args()
    
    DEFAULT_MODEL_PATH = args.model_path
    DEFAULT_LABEL_PATH = args.label_path
    TRAIN_DATA_PATH = args.train_data_path
    TEST_DATA_PATH = args.test_data_path

    model_name = os.path.basename(os.path.normpath(DEFAULT_MODEL_PATH))

    try:
        path_parts = os.path.normpath(DEFAULT_LABEL_PATH).split(os.sep)
        data_index = path_parts.index('data')
        data_group = path_parts[data_index + 1]
    except (ValueError, IndexError):
        data_group = "default_data_type"

    OUTPUT_DIR = os.path.join(
        "/inspire/hdd/project/exploration-topic/public/lzjjin/course/Goemotions/grpo_output",
        model_name,
        data_group
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Set OUTPUT_DIR to: {OUTPUT_DIR}") # 添加打印输出以确认路径

    
    labels = load_labels(DEFAULT_LABEL_PATH)

    print("Preparing train dataset...")
    train_dataset = prepare_dataset(TRAIN_DATA_PATH, labels)
    
    print("Preparing eval dataset...")
    eval_dataset = prepare_dataset(TEST_DATA_PATH, labels)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_PATH,
        torch_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
    )

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True, 
        per_device_train_batch_size=32, 
        gradient_accumulation_steps=2,
        num_generations=4, 
        max_prompt_length=512, 
        max_completion_length=1024,
        num_train_epochs=2,
        save_steps=100,
        eval_steps=100, 
        eval_strategy="steps", 
        max_grad_norm=0.1,
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
        log_completions=True,
        num_completions_to_print=5,
        logging_dir=os.path.join(OUTPUT_DIR,'runs/')
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[accuracy_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer.train()
    
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()