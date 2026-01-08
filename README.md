# GoEmotions Emotion Classification Project

本项目基于 **Qwen2.5-7B-Instruct** 模型，针对 **GoEmotions** 数据集进行情感分类任务的微调。项目包含 **SFT (Supervised Fine-Tuning)** 和 **GRPO (Group Relative Policy Optimization)** 两种训练阶段，支持 Chain-of-Thought (CoT) 推理以及 Token 级别的注意力机制分析。

## 📁 目录结构

建议的项目文件结构如下：`markdown
# GoEmotions Emotion Classification Project

本项目基于 **Qwen2.5-7B-Instruct** 模型，针对 **GoEmotions** 数据集进行情感分类任务的微调。项目包含 **SFT (Supervised Fine-Tuning)** 和 **GRPO (Group Relative Policy Optimization)** 两种训练阶段，支持 Chain-of-Thought (CoT) 推理以及 Token 级别的注意力机制分析。

## 🛠️ 环境依赖

请确保安装了以下核心库：

```bash
pip install torch pandas numpy datasets transformers peft trl accelerate scikit-learn vllm
```

## 🚀 快速开始

### 1. 模型下载

使用 `download.`download.py` 下载 Qwen2.5 基础模型：

```bash
python download.py
```

*注意：请*注意：请在脚本中修改 `target_dir` 为你的实际路径。*

### 2. SFT 训练 (Supervised Fine-Tuning)

使用 `main_SFT.py` 进行监督微调。该脚本使用 `trl` 库的 `SFTTrainer` 和 LoRA 技术。

```bash
python main_SFT.py \
    --model_path "/path/to/Qwen2.5-7B-Instruct" \
    --train_data_path "/path/to/data/group/train.tsv" \
    --label_path "/path/to/data/group/labels.txt" \
    --test_data_path "/path/to/data/group/test_small.tsv"
```

### 3. GRPO 训练 (Reinforcement Learning)

GRPO (Group Relative Policy Optimization) 用于进一步优化模型，鼓励模型生成准确的标签格式并进行推理。

**启动方式：**
使用 `scripts/` 文件夹下的 Shell 脚本进行多 GPU 分布式训练：

```bash
bash scripts/main_grpo_original_Qwen2_5-7b.sh
```

**GRPO 奖励函数说明：**

* **Accuracy**Accuracy Reward:** 预测标签与真实标签的 Jaccard 相似系数（交并比）。公式为：
  [ R_{acc} = \frac{|P \cap G|}{|P \cup G|} ]
  其中 ( P ) 为预测标签集合，( G ) 为真实标签集合。
* **Length Reward:** 惩罚过长的回复，防止模型输出冗余信息。

### 4. 模型合并

训练完成后，使用 `merge_`merge_lora.py` 将 LoRA 权重合并回基础模型，以便进行推理或部署。

```bash
python merge_lora.py
```

*请在脚本内的 `if **name** == "**main**":*请在脚本内的 `if __name__ == "__main__":` 部分修改 `base_model_path` 和 `training_output_dir`。*

---

## ⚡ 推理与评估 (Inference)

使用 `inference.py` 对测试集进行评估。支持多种后端 (`vllm`, `hf`, `bert`) 和思维链 (CoT) 模式。

### 参数说明

* `--backend`: 推理后端，推荐使用 `vllm` 以获得更快的速度。
