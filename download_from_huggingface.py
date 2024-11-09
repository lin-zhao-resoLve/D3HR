from huggingface_hub import snapshot_download

# 指定模型仓库 ID 和文件名（如模型权重或配置文件）
# repo_id = "google/vit-base-patch16-224-in21k"
repo_id = "Qwen/Qwen2-0.5B"
filename = "pytorch_model.bin"  # 你要下载的模型文件

# 指定下载路径
local_dir = "/scratch/zhao.lin1/Qwen2-0.5B"

# 下载模型文件到指定路径

snapshot_download(repo_id=repo_id, local_dir=local_dir)
