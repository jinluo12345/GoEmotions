from huggingface_hub import snapshot_download

model_id = "monologg/bert-base-cased-goemotions-group"

# 你想要的最终目录（文件直接放在这里）
target_dir = "/inspire/hdd/project/exploration-topic/public/downloaded_ckpts/bert-base-cased-goemotions-group"

snapshot_download(
    repo_id=model_id,
    local_dir=target_dir,          # 指定下载目录
    local_dir_use_symlinks=False,  # 不用软链接，直接复制文件
    # ignore_patterns=["*.pt", "*.bin"],  # 需要的话可以过滤某些大文件
)

print("模型已下载到：", target_dir)
