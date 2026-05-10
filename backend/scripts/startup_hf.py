import os
import sys
from huggingface_hub import hf_hub_download, list_repo_files

def startup_sync():
    repo_id = os.getenv("HF_DATASET_REPO")
    token = os.getenv("HF_TOKEN_READ") or os.getenv("HF_TOKEN_WRITE")
    
    if not repo_id:
        print("ℹ️ No HF_DATASET_REPO set, skipping startup sync.")
        return

    print(f"🔄 Starting startup sync from {repo_id}...")
    
    # 1. Xác định thư mục data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    try:
        # Liệt kê các file trong repo để tải về
        files = list_repo_files(repo_id, repo_type="dataset", token=token)
        
        for file_path in files:
            # Chỉ tải các file trong thư mục users/ (của người dùng) và metadata.db
            if file_path.endswith(".db") or "users/" in file_path or "tq_index" in file_path:
                print(f"📥 Downloading {file_path}...")
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    repo_type="dataset",
                    token=token,
                    local_dir=data_dir,
                    local_dir_use_symlinks=False
                )
        print("✅ Startup sync completed successfully.")
    except Exception as e:
        print(f"⚠️ Startup sync failed: {e}")

if __name__ == "__main__":
    startup_sync()
