import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel
import os

def merge_latest_model(base_model_path, training_output_dir, final_output_path):
    """
    直接使用最新的LoRA权重合并模型
    """
    print("正在加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu"  # 先加载到CPU避免显存不足
    )
    
    print("正在加载最新的LoRA适配器...")
    # 直接从训练输出目录加载LoRA适配器（最新的权重）
    model = PeftModel.from_pretrained(
        base_model, 
        training_output_dir,  # 这里就是你的训练输出目录
        # dtype=torch.float16
    )
    
    print("正在合并权重...")
    # 合并LoRA权重到基础模型
    merged_model = model.merge_and_unload()
    
    print("正在保存完整模型...")
    # 创建输出目录
    os.makedirs(final_output_path, exist_ok=True)
    
    # 保存合并后的完整模型
    merged_model.save_pretrained(
        final_output_path,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    
    # 保存tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(final_output_path)
    
    print(f"完整模型已保存到: {final_output_path}")
    print("现在你可以使用 AutoModel.from_pretrained() 直接加载了")

if __name__ == "__main__":

    base_model_path = "Qwen2.5-7B-Instruct"
    training_output_dir = "grpo_output/Qwen2.5-7B-Instruct/original/checkpoint-678"
    final_output_path = "grpo_output/Qwen2.5-7B-Instruct/original/Qwen2.5-7b-grpo"
    merge_latest_model(base_model_path, training_output_dir, final_output_path)
        
    print("\n使用方法:")
    print(f"""
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("{final_output_path}", trust_remote_code=True)
model = AutoModel.from_pretrained("{final_output_path}", trust_remote_code=True, torch_dtype=torch.bfloat16)
    """)
        