import argparse
import os
import json
import torch
import numpy as np
import re
from typing import List, Union, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm

# 默认配置
DEFAULT_MODEL_PATH = '/inspire/hdd/project/exploration-topic/public/downloaded_ckpts/Qwen3-8B/'
DEFAULT_PREDICTION_FILE = "predictions.json"
model_name_for_default = "Qwen3-8B"
DEFAULT_EVAL_OUTPUT_DIR = f"/inspire/hdd/project/exploration-topic/public/lzjjin/course/Goemotions/evaluation_results/{model_name_for_default}"

DEFAULT_N_TOP_TOKENS = 8
DEFAULT_NUM_SAMPLES = 100

# --- 辅助函数 ---
def load_prediction_data(file_path: str) -> List[dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prediction file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"Prediction file content must be a list, got {type(data)}")

    valid_data = [
        item for item in data 
        if item.get('prompt') is not None and item.get('generated_completion') is not None and item.get('text') is not None
    ]
    return valid_data

def save_json(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"\nSaved analysis results to {file_path}")

def get_predicted_emotion_tokens(
    tokenizer: AutoTokenizer, 
    prompt_text: str, 
    completion_text: str
) -> Tuple[List[int], List[str], List[str]]:
    
    match = re.search(r"<answer>(.*?)</answer>", completion_text, re.IGNORECASE)
    if not match:
        return [], [], [] 

    labels_str = match.group(1).strip()
    all_predicted_labels = [label.strip() for label in labels_str.split(',') if label.strip()]
    if not all_predicted_labels:
        return [], [], []
    
    full_text = prompt_text + completion_text
    inputs = tokenizer(full_text, truncation=True, max_length=1024, add_special_tokens=False)
    input_ids = inputs['input_ids']
    
    prompt_inputs = tokenizer(prompt_text, truncation=True, max_length=1024, add_special_tokens=False)
    prompt_len = len(prompt_inputs['input_ids'])
    
    query_indices = []
    completion_ids = input_ids[prompt_len:]
    completion_start_idx = prompt_len
    
    for label in all_predicted_labels:
        for current_label in [label, label.lstrip()]:
            label_tokens = tokenizer.encode(current_label, add_special_tokens=False)
            if not label_tokens: continue
            
            for i in range(len(completion_ids) - len(label_tokens) + 1):
                if completion_ids[i:i + len(label_tokens)] == label_tokens:
                    current_indices = [completion_start_idx + i + j for j in range(len(label_tokens))]
                    query_indices.extend(current_indices)
                    break
        
    return sorted(list(set(query_indices))), [], all_predicted_labels

def aggregate_token_to_word_saliency_simple(
    tokenizer: AutoTokenizer, 
    input_text: str,
    prompt_tokens: List[str],
    saliency_weights: np.ndarray
) -> List[Dict[str, float]]:
    
    words = input_text.split()
    word_saliency = []
    token_idx_in_prompt = 0
    
    for word in words:
        current_word_weights = []
        if token_idx_in_prompt >= len(prompt_tokens):
            word_saliency.append({"word": word, "attention_score": 0.0})
            continue
        
        accumulated_text = ""
        start_idx = token_idx_in_prompt
        
        while token_idx_in_prompt < len(prompt_tokens):
            next_token_text = tokenizer.convert_tokens_to_string([prompt_tokens[token_idx_in_prompt]]).lstrip()
            temp_text = (accumulated_text + next_token_text).strip()
            
            if word.startswith(temp_text) and len(word) >= len(temp_text):
                 accumulated_text = temp_text
                 current_word_weights.append(saliency_weights[token_idx_in_prompt])
                 token_idx_in_prompt += 1
                 if accumulated_text == word:
                     break
            elif word == accumulated_text: 
                break
            else: 
                if start_idx == token_idx_in_prompt:
                     current_word_weights.append(saliency_weights[token_idx_in_prompt])
                     token_idx_in_prompt += 1
                break
        
        if current_word_weights:
            avg_weight = np.mean(current_word_weights)
            word_saliency.append({"word": word, "attention_score": float(avg_weight)})
        else:
            word_saliency.append({"word": word, "attention_score": 0.0})
            
    return word_saliency


# --- 主分析函数 ---

def analyze_token_saliency(
    model_path: str, 
    data: List[dict], 
    layer_spec: Union[int, str], 
    n_top_tokens: int,
    num_samples: int
) -> List[Dict[str, Any]]:
    
    print(f"Loading model and tokenizer from {model_path}...")
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.output_attentions = True 
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        config=config,
        trust_remote_code=True, 
        device_map="auto", 
        torch_dtype=torch.float16
    ).eval()
    
    all_results = []
    
    if num_samples > 0:
        data = data[:num_samples]

    for i, item in enumerate(tqdm(data, desc=f"Analyzing token saliency ({len(data)} samples)")):
        prompt_text = item['prompt']
        completion_text = item['generated_completion']
        raw_text = item['text'] 
        full_text = prompt_text + completion_text
        
        query_indices, _, all_predicted_labels = get_predicted_emotion_tokens(
            tokenizer, prompt_text, completion_text
        )
        
        if not query_indices:
            all_results.append({**item, "analysis_status": "Skipped: Could not find predicted label tokens."})
            continue

        # 1. 在 full_text 上分词并获取 tokens (返回 pt 张量用于模型输入)
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024, add_special_tokens=False).to(model.device)
        input_ids = inputs['input_ids'][0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids) 
        
        # 修正: 使用 len() 获取列表长度
        prompt_len = len(tokenizer(prompt_text, truncation=True, max_length=1024, add_special_tokens=False)['input_ids'])
        
        if prompt_len >= len(input_ids):
             all_results.append({**item, "analysis_status": "Skipped: Prompt length >= full sequence length."})
             continue
             
        # 2. 通过字符串匹配定位 raw_text 的 Token 范围
        match = re.search(re.escape(raw_text), prompt_text)
        if not match:
             all_results.append({**item, "analysis_status": "Skipped: Could not locate raw text string in prompt string."})
             continue
             
        start_char_index = match.start()
        end_char_index = match.end()
        
        raw_text_start_token_idx = -1
        raw_text_end_token_idx = -1
        current_char_index = 0
        
        for token_idx in range(prompt_len):
            token_text = tokenizer.convert_tokens_to_string([tokens[token_idx]])
            
            if raw_text_start_token_idx == -1:
                if current_char_index <= start_char_index < current_char_index + len(token_text.lstrip()):
                     raw_text_start_token_idx = token_idx
                elif current_char_index == start_char_index and start_char_index == 0:
                     raw_text_start_token_idx = token_idx 
            
            if raw_text_start_token_idx != -1:
                 if current_char_index < end_char_index <= current_char_index + len(token_text.lstrip()):
                     raw_text_end_token_idx = token_idx + 1
                     break
                 
            current_char_index += len(token_text)

        if raw_text_start_token_idx == -1 or raw_text_end_token_idx == -1 or raw_text_start_token_idx >= raw_text_end_token_idx:
             all_results.append({**item, "analysis_status": "Skipped: Could not map raw text string match to prompt token indices."})
             continue
        
        # 3. 前向传播并获取注意力权重
        try:
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True) 
                attentions = outputs.attentions
        except Exception as e:
             all_results.append({**item, "analysis_status": f"Skipped: Inference Error - {e}"})
             continue
        
        if not attentions:
            all_results.append({**item, "analysis_status": "Skipped: Model did not return attention weights."})
            continue
            
        # 4. 提取或平均注意力权重
        atts_stacked = torch.stack([att.squeeze(0) for att in attentions]).cpu()
        
        if layer_spec == 'avg':
            attention_matrix = atts_stacked.mean(dim=(0, 1)).numpy() 
            layer_name = "Average Attention (All Layers & Heads)"
        else:
            try:
                layer_idx = int(layer_spec)
                actual_idx = len(attentions) + layer_idx if layer_idx < 0 else layer_idx
                attention_matrix = atts_stacked[actual_idx].mean(dim=0).numpy() 
                layer_name = f"Layer {actual_idx} Mean Attention (All Heads)"
            except Exception as e:
                all_results.append({**item, "analysis_status": f"Skipped: Layer index error - {e}"})
                continue

        # 5. 提取 Query Token 对 raw_text Token 的权重 (Key Tokens)
        saliency_block = attention_matrix[
            np.array(query_indices), 
            raw_text_start_token_idx:raw_text_end_token_idx 
        ] 
        
        saliency_weights_token_level = saliency_block.mean(axis=0) if saliency_block.ndim > 1 else saliency_block
        raw_text_tokens_text = tokens[raw_text_start_token_idx:raw_text_end_token_idx]
        
        if len(raw_text_tokens_text) != len(saliency_weights_token_level):
             all_results.append({**item, "analysis_status": "Skipped: Token length mismatch after slicing."})
             continue
        
        # 6. 词语级别聚合、排序和提取 Top N
        word_saliency_list = aggregate_token_to_word_saliency_simple(
            tokenizer, 
            raw_text, 
            raw_text_tokens_text, 
            saliency_weights_token_level
        )
        
        word_saliency_list.sort(key=lambda x: x['attention_score'], reverse=True)
        top_important_words = word_saliency_list[:n_top_tokens]
            
        # 7. 整理结果
        result_record = {
            "id": item.get('id', i),
            "text": item.get('text'), 
            "predicted_labels": all_predicted_labels, 
            "attention_aggregation_method": "Attention of predicted emotion tokens (Query) on raw_text tokens (Key), averaged and aggregated to words.",
            "attention_layer_info": layer_name,
            "top_important_words": top_important_words, 
            "analysis_status": "Success"
        }
        all_results.append(result_record)
        
    return all_results

# --- Main 块 ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and extract top attention words from LLM predictions.")
    
    # DEFAULT_PRED_PATH 拼接逻辑已简化，因为 DEFAULT_EVAL_OUTPUT_DIR 已包含模型名
    DEFAULT_PRED_PATH = os.path.join(DEFAULT_EVAL_OUTPUT_DIR, DEFAULT_PREDICTION_FILE)

    parser.add_argument("--prediction_file", type=str, default=DEFAULT_PRED_PATH, help="Path to the input prediction JSON file.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the LLM directory to load.")
    parser.add_argument("--n_top_tokens", type=int, default=DEFAULT_N_TOP_TOKENS, help="Number of top attention words to extract.")
    parser.add_argument("--layer", type=str, default='avg', choices=['avg', '-1', '-2', '0', '1', '2', '3', '4', '5'], help="Attention layer to analyze.")
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Number of samples to process from the prediction file. Use -1 to process all.")
    
    args = parser.parse_args()
    
    # --- 1. 参数处理 ---
    layer_spec_input = args.layer
    if layer_spec_input != 'avg':
        try:
            layer_spec_input = int(layer_spec_input)
        except ValueError:
            print("Error: Layer must be an integer or 'avg'.")
            exit(1)
            
    # --- 2. 加载数据 ---
    prediction_file_path = os.path.abspath(args.prediction_file)
    try:
        data_to_analyze = load_prediction_data(prediction_file_path)
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"Failed to load data: {e}")
        exit(1)
    
    if not data_to_analyze:
        print("No valid data found for analysis.")
    else:
        print(f"Loaded {len(data_to_analyze)} samples. Processing {args.num_samples if args.num_samples > 0 else 'all'} samples.")
        
        # --- 3. 执行分析 ---
        analysis_results = analyze_token_saliency(
            args.model_path, 
            data_to_analyze, 
            layer_spec_input, 
            args.n_top_tokens,
            args.num_samples
        )
        
        # --- 4. 保存结果 ---
        output_dir = os.path.dirname(prediction_file_path)
        # 注意: 保存文件名更改为 "important_words_raw_text.json"，以便与先前的逻辑保持一致，并清晰地反映结果内容
        save_path = os.path.join(output_dir, "important_words.json") 
        save_json(save_path, analysis_results)