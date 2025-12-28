import argparse
import os
import json
import torch
import numpy as np
import re
from typing import List, Union, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm

# --- 保留原始默认配置 ---
DEFAULT_MODEL_PATH = '/inspire/hdd/project/exploration-topic/public/lzjjin/course/Goemotions/sft_output/Qwen2.5-7B-Instruct/group/Qwen2.5-7b-sft'
DEFAULT_PREDICTION_FILE = "predictions.json"
model_name_for_default = "Qwen2.5-7B-sft"

BASE_EVAL_OUTPUT_DIR = f"/inspire/hdd/project/exploration-topic/public/lzjjin/course/Goemotions/evaluation_results/{model_name_for_default}/group"

DEFAULT_NUM_SAMPLES = 100

# --- 辅助函数 ---
def load_prediction_data(file_path: str) -> List[dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prediction file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"Prediction file content must be a list, got {type(data)}")

    # 只要有 prompt 和 completion 即可，不再强制要求 text
    valid_data = [
        item for item in data 
        if item.get('prompt') is not None and item.get('generated_completion') is not None
    ]
    return valid_data

def save_json(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"\nSaved analysis results to {file_path}")

def get_answer_indices(tokenizer, full_token_ids, completion_text, prompt_len):
    """
    在 completion 部分定位 <answer> 内容对应的 token 索引
    """
    match = re.search(r"<answer>(.*?)</answer>", completion_text, re.IGNORECASE)
    if not match:
        return []

    answer_content = match.group(1).strip()
    if not answer_content:
        return []

    # 编码 answer 内容
    answer_ids = tokenizer.encode(answer_content, add_special_tokens=False)
    
    # 只在 completion 对应的位点寻找 (即 prompt_len 之后)
    completion_ids = full_token_ids[prompt_len:]
    n = len(answer_ids)
    
    for i in range(len(completion_ids) - n + 1):
        if list(completion_ids[i : i + n]) == answer_ids:
            # 返回相对于全序列的索引
            return [prompt_len + i + j for j in range(n)]
    return []

# --- 主分析函数 ---

def analyze_token_saliency(
    model_path: str, 
    data: List[dict], 
    layer_spec: Union[int, str], 
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

    # 统计变量
    total_nan_found = 0

    for i, item in enumerate(tqdm(data, desc=f"Analyzing {len(data)} samples")):
        prompt_text = item['prompt']
        completion_text = item['generated_completion']
        full_text = prompt_text + completion_text
        
        # 1. 对全文本进行分词
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096, add_special_tokens=False).to(model.device)
        input_ids = inputs['input_ids'][0]
        full_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        # 2. 获取 Prompt 的长度，用于区分问题和答案
        prompt_inputs = tokenizer(prompt_text, truncation=True, max_length=4096, add_special_tokens=False)
        prompt_len = len(prompt_inputs['input_ids'])
        
        # 3. 定位 <answer> 标签内的 Token 索引 (作为 Query)
        query_indices = get_answer_indices(tokenizer, input_ids.cpu().numpy(), completion_text, prompt_len)
        
        if not query_indices:
            all_results.append({**item, "analysis_status": "Skipped: Could not find <answer> tokens."})
            continue

        # 4. 前向传播获取注意力
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            attentions = outputs.attentions

        # 使用 float32 堆叠，防止半精度计算导致的 NaN
        atts_stacked = torch.stack([att.squeeze(0) for att in attentions]).cpu().float()
        
        if layer_spec == 'avg':
            attention_matrix = atts_stacked.mean(dim=(0, 1)).numpy()
            layer_name = "Average Attention"
        else:
            layer_idx = int(layer_spec)
            actual_idx = len(attentions) + layer_idx if layer_idx < 0 else layer_idx
            attention_matrix = atts_stacked[actual_idx].mean(dim=0).numpy()
            layer_name = f"Layer {actual_idx}"

        # 检查并修复 NaN
        if np.isnan(attention_matrix).any():
            attention_matrix = np.nan_to_num(attention_matrix, nan=0.0)
            total_nan_found += 1

        prompt_scores = attention_matrix[query_indices, :prompt_len]
        if prompt_scores.ndim > 1:
            final_scores = prompt_scores.mean(axis=0)
        else:
            final_scores = prompt_scores

        token_analysis = []
        for idx in range(prompt_len):
            # 过滤 'Ġ' 等特殊分词符号，还原为正常文本/空格
            raw_token = full_tokens[idx]
            clean_token = tokenizer.convert_tokens_to_string([raw_token])
            
            token_analysis.append({
                "index": idx,
                "token": clean_token,
                "score": float(final_scores[idx])
            })
            
        result_record = {
            "id": item.get('id', i),
            "prompt": prompt_text,
            "completion": completion_text,
            "answer_token_indices": query_indices,
            "attention_layer": layer_name,
            "prompt_tokens_score": token_analysis, # 记录所有 token 及索引
            "analysis_status": "Success"
        }
        all_results.append(result_record)
    
    # 输出最终的统计报告
    print(f"\n" + "="*30)
    print(f"分析完成！")
    print(f"共处理样本: {len(data)}")
    print(f"检测到并修复 NaN 的样本数: {total_nan_found}")
    print("="*30)
        
    return all_results

# --- Main 块 ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze token-level attention scores.")
    
    parser.add_argument("--base_dir", type=str, default=BASE_EVAL_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--layer", type=str, default='avg', help="Layer to analyze or 'avg'")
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES)
    
    args = parser.parse_args()
    
    # 参数转换
    layer_spec_input = args.layer
    if layer_spec_input != 'avg':
        layer_spec_input = int(layer_spec_input)
    
    # 核心判断逻辑
    dirs_to_process = []

    # 1. 检查输入的路径本身是否包含关键文件夹名（original 或 group）
    # 或者检查该目录下是否直接存在 predictions.json
    current_has_pred = os.path.exists(os.path.join(args.base_dir, DEFAULT_PREDICTION_FILE))
    is_sub_folder = any(sub in args.base_dir for sub in ["original", "group"])

    if current_has_pred and is_sub_folder:
        # 如果输入的路径就是子目录本身，则只处理这一个
        dirs_to_process = [args.base_dir]
    else:
        # 否则，尝试寻找旗下的子目录
        for sub in ["original", "group"]:
            sub_path = os.path.join(args.base_dir, sub)
            if os.path.isdir(sub_path):
                dirs_to_process.append(sub_path)
        
        # 如果旗下也没有子目录，但当前目录有文件，也保底处理一下
        if not dirs_to_process and current_has_pred:
            dirs_to_process = [args.base_dir]

    if not dirs_to_process:
        print(f"错误：在 {args.base_dir} 及其预设子目录中未发现 {DEFAULT_PREDICTION_FILE}")
    else:
        for target_dir in dirs_to_process:
            prediction_file_path = os.path.join(target_dir, DEFAULT_PREDICTION_FILE)
            
            print(f"\n>>> 目标目录: {target_dir}")
            
            # 加载数据
            data_to_analyze = load_prediction_data(prediction_file_path)
            
            if not data_to_analyze:
                print(f"跳过：在 {target_dir} 中未找到有效数据。")
                continue
                
            # 执行分析
            analysis_results = analyze_token_saliency(
                args.model_path, 
                data_to_analyze, 
                layer_spec_input, 
                args.num_samples
            )
            
            # 保存结果
            save_path = os.path.join(target_dir, "prompt_token_scores.json") 
            save_json(save_path, analysis_results)

    print("\n所有任务执行完毕！")