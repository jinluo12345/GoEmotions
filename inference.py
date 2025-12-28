import argparse
import os
import re
import json
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Union
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_MODEL_PATH = '/inspire/hdd/project/exploration-topic/public/lzjjin/course/Goemotions/grpo_output/Qwen2.5-7B-Instruct/original/Qwen2.5-7b-grpo'
DEFAULT_LABEL_PATH = "/inspire/hdd/project/exploration-topic/public/lzjjin/course/Goemotions/data/original/labels.txt"
DEFAULT_TEST_PATH = "/inspire/hdd/project/exploration-topic/public/lzjjin/course/Goemotions/data/original/test.tsv"
DEFAULT_OUTPUT_DIR = "/inspire/hdd/project/exploration-topic/public/lzjjin/course/Goemotions/evaluation_results"

def load_labels(label_path: str) -> List[str]:
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"{label_path}")
    with open(label_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def load_test_data(test_path: str):
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{test_path}")
    
    df = pd.read_csv(test_path, sep='\t', header=None, names=['text', 'label_ids', 'comment_id'])
    texts = df['text'].tolist()
    
    ground_truth = []
    for label_str in df['label_ids'].astype(str):
        ids = [int(x) for x in label_str.split(',')]
        ground_truth.append(ids)
        
    return texts, ground_truth

def create_prompts(texts: List[str], labels: List[str], use_cot: bool = False) -> List[str]:
    label_str = ", ".join(labels)
    prompts = []
    for text in texts:
        if use_cot:
            # 启用思维链 (CoT) 的 Prompt
            prompt = (
                f"You are an emotion classification expert. Identify one or more emotions in the text from the following list:\n"
                f"[{label_str}]\n\n"
                f"First, analyze the text step-by-step to determine the underlying emotions.\n"
                f"Put your thinking process inside <think> ...</think> tags.\n" 
                f"Then, format your final answer strictly as: <answer>emotion1, emotion2</answer>\n"
                f"If there is only one emotion: <answer>emotion1</answer>"
                f"Text: \"{text}\"\n\n"
            )
        else:
            # 原始 Prompt (直接输出结果)
            prompt = (
                f"You are an emotion classification expert. Identify one or more emotions"
                f"in the text from the following list:\n"
                f"[{label_str}]\n\n"
                f"Format your output strictly as: <answer>emotion1, emotion2</answer>\n"
                f"If there is only one emotion: <answer>emotion1</answer>\n"
                f"Text: \"{text}\"\n\n"
                f"Answer:"
            )
        prompts.append(prompt)
    return prompts

def parse_prediction(output_text: str, label_to_id: Dict[str, int]) -> List[int]:
    match = re.search(r"<answer>(.*?)(?:</answer>|$)", output_text, re.DOTALL)
    if not match:
        return []
    
    content = match.group(1)
    pred_labels = [x.strip().lower() for x in content.split(',')]
    
    pred_ids = []
    for p in pred_labels:
        clean_p = p.strip().strip('.').strip()
        if clean_p in label_to_id:
            pred_ids.append(label_to_id[clean_p])
    
    return list(set(pred_ids))

# --- Inference Backends ---

def run_vllm_inference(model_path: str, prompts: List[str], use_cot: bool = False):
    from vllm import LLM, SamplingParams
    
    max_new_tokens = 2048 if use_cot else 512
    
    print(f"Loading vLLM model from {model_path}...")
    llm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=0.9)
    # stop 参数确保遇到 </answer> 就停止，但我们仍然需要记录完整的生成文本。
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens, stop=["</answer>"])
    
    print(f"Generating with vLLM (CoT={use_cot}, max_tokens={max_new_tokens})...")
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for output in outputs:
        # generated_text 是模型生成的完整序列，包括 CoT/答案，但不含提示词
        generated_text = output.outputs[0].text
        # 手动补全结束符以便后续解析，并确保保存的是模型的完整输出
        if not generated_text.strip().endswith("</answer>"):
             generated_text += "</answer>"
        results.append(generated_text)
    return results

def run_hf_inference(model_path: str, prompts: List[str], use_cot: bool = False):
    from transformers import AutoModelForCausalLM
    
    # 默认值已经调整以适应 CoT
    max_new_tokens = 512 if use_cot else 128

    print(f"Loading HF Causal model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    results = []
    batch_size = 4
    
    print(f"Generating with HF Transformers (CoT={use_cot}, max_tokens={max_new_tokens})...")
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        input_lengths = [len(x) for x in inputs['input_ids']]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                temperature=0.01, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        for j, output_ids in enumerate(outputs):
            # 只解码新生成的部分，这部分就是模型的完整 'completion'
            generated_ids = output_ids[input_lengths[j]:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append(text)
            
    return results

def run_bert_inference(model_path: str, texts: List[str], target_label_to_id: Dict[str, int], threshold: float = 0.3) -> List[List[int]]:
    """
    BERT 模型是判别式模型，不涉及 Prompt 或 CoT 生成，逻辑保持不变。
    """
    print(f"Loading BERT model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map="auto")

    model_id2label = model.config.id2label
    
    predictions = []
    batch_size = 32
    
    print("Generating with BERT...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            
        for prob_row in probs:
            model_pred_indices = np.where(prob_row > threshold)[0]
            
            mapped_ids = []
            for idx in model_pred_indices:
                label_name = model_id2label[idx].lower() 
                if label_name in target_label_to_id:
                    mapped_ids.append(target_label_to_id[label_name])
            
            predictions.append(mapped_ids)
            
    return predictions

# --- Utils ---

def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved to {file_path}")

def main(args):
    # 1. 准备数据
    labels = load_labels(args.label_path)
    label_to_id = {label.lower(): idx for idx, label in enumerate(labels)}
    texts, true_labels = load_test_data(args.test_path)
    
    predictions = []
    # generated_completions 用于存储模型生成的完整文本，BERT 模式下为 None
    generated_completions = [None] * len(texts)
    prompts = [None] * len(texts)
    
    # 2. 根据 Backend 选择推理方式
    if args.backend == 'vllm':
        prompts = create_prompts(texts, labels, use_cot=args.use_cot)
        generated_texts = run_vllm_inference(args.model_path, prompts, use_cot=args.use_cot)
        generated_completions = generated_texts
        for gen_text in generated_texts:
            predictions.append(parse_prediction(gen_text, label_to_id))
            
    elif args.backend == 'hf':
        prompts = create_prompts(texts, labels, use_cot=args.use_cot)
        generated_texts = run_hf_inference(args.model_path, prompts, use_cot=args.use_cot)
        # **关键：保存完整的生成文本**
        generated_completions = generated_texts
        for gen_text in generated_texts:
            predictions.append(parse_prediction(gen_text, label_to_id))
            
    elif args.backend == 'bert':
        if args.use_cot:
            print("Warning: BERT backend does not support Chain-of-Thought (use_cot). Ignoring flag.")
        predictions = run_bert_inference(args.model_path, texts, label_to_id, threshold=0.3)
        # generated_completions 保持为 [None]，符合 BERT 的输出特性。
        
    else:
        raise ValueError("Backend must be 'vllm', 'hf', or 'bert'")
    
    # 3. 整理保存数据
    saved_records = []
    for i in range(len(texts)):
        saved_records.append({
            "id": i,
            "text": texts[i],
            # prompt 只有在 VLLM/HF 模式下才被创建
            "prompt": prompts[i],
            "ground_truth_ids": true_labels[i],
            "ground_truth_labels": [labels[idx] for idx in true_labels[i]],
            # generated_completion 包含 VLLM/HF 的完整输出 (包括 CoT 或直接答案)
            "generated_completion": generated_completions[i],
            "predicted_ids": predictions[i],
            "predicted_labels": [labels[idx] for idx in predictions[i]]
        })

    # 4. 计算指标
    mlb = MultiLabelBinarizer(classes=range(len(labels)))
    y_true = mlb.fit_transform(true_labels)
    y_pred = mlb.transform(predictions)
    
    acc = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, target_names=labels, zero_division=0, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    
    print(f"\nExact Match Accuracy: {acc:.4f}")
    print(report_str)
    
    # 5. 保存结果
    model_name = os.path.basename(os.path.normpath(args.model_path))
    
    # --- 关键修改：提取数据类型 ---
    try:
        # 假设路径格式为 .../data/{data_type}/...
        path_parts = os.path.normpath(args.test_path).split(os.sep)
        data_index = path_parts.index('data')
        data_type = path_parts[data_index + 1] # 提取 'data' 后面的文件夹名，例如 'original' 或 'group'
    except (ValueError, IndexError):
        data_type = "default_data"
        
    # --- 构建最终的模型名称 (模型名 + 数据类型 + CoT 标志) ---
    final_model_name = f"{model_name}"
    if args.use_cot:
        final_model_name += "_cot"
        
    output_dir = os.path.join(args.output_dir, final_model_name)
    output_dir = os.path.join(output_dir, data_type)
    os.makedirs(output_dir, exist_ok=True)
    
    save_json(os.path.join(output_dir, "predictions.json"), saved_records)
    
    metrics_data = {
        "model_path": args.model_path,
        "backend": args.backend,
        "use_cot": args.use_cot,
        "exact_match_accuracy": acc,
        "classification_report": report_dict
    }
    save_json(os.path.join(output_dir, "metrics.json"), metrics_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--label_path", type=str, default=DEFAULT_LABEL_PATH)
    parser.add_argument("--test_path", type=str, default=DEFAULT_TEST_PATH)
    
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "hf", "bert"], help="Inference backend: vllm, hf or bert")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save output json")

    parser.add_argument("--use_cot", action="store_true", help="Enable Chain-of-Thought prompting to encourage reasoning before answering.")
    
    args = parser.parse_args()
    main(args)