import os
import asyncio
import shutil
from huggingface_hub import HfApi, snapshot_download
from concurrent.futures import ThreadPoolExecutor

class SyncService:
    def __init__(self):
        self.api = HfApi()
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.repo_id = os.getenv("HF_DATASET_REPO")
        # Dùng HF_TOKEN_WRITE để có quyền ghi vào Dataset
        self.token = os.getenv("HF_TOKEN_WRITE") or os.getenv("HF_TOKEN")

    def download_from_hub(self):
        """Tải toàn bộ dữ liệu từ HF Dataset về khi khởi động Space"""
        if not self.repo_id or not self.token:
            print("WARNING: Download skipped: HF_DATASET_REPO or HF_TOKEN not set.")
            return

        try:
            print(f"[*] Downloading data from HF Dataset: {self.repo_id}...")
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")
            
            # Tải snapshot của dataset về thư mục data
            # allow_patterns giúp ta chỉ lấy những file cần thiết
            snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                local_dir=data_dir,
                token=self.token,
                allow_patterns=["metadata.db", "user_indexes/**/*", "uploads/**/*"]
            )
            print("SUCCESS: Data restored from Cloud.")
        except Exception as e:
            print(f"WARNING: Could not download data from Hub (First run?): {e}")

    async def sync_to_hub(self, user_id: str = "all"):
        """Đẩy dữ liệu lên Hugging Face Dataset ngầm (Background Task)"""
        if not self.repo_id or not self.token:
            return

        async with self.lock:
            try:
                loop = asyncio.get_event_loop()
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                data_dir = os.path.join(base_dir, "data")

                # 1. Đồng bộ Metadata DB
                db_path = os.path.join(data_dir, "metadata.db")
                if os.path.exists(db_path):
                    await loop.run_in_executor(
                        self.executor,
                        self._upload_file,
                        db_path,
                        "metadata.db"
                    )

                # 2. Đồng bộ Index của User (Toàn bộ thư mục user_indexes)
                user_idx_dir = os.path.join(data_dir, "user_indexes")
                if os.path.exists(user_idx_dir):
                    await loop.run_in_executor(
                        self.executor,
                        self._upload_folder,
                        user_idx_dir,
                        "user_indexes"
                    )

                # 3. Đồng bộ File gốc của User (Toàn bộ thư mục uploads)
                uploads_dir = os.path.join(data_dir, "uploads")
                if os.path.exists(uploads_dir):
                    await loop.run_in_executor(
                        self.executor,
                        self._upload_folder,
                        uploads_dir,
                        "uploads"
                    )

                print(f"SUCCESS: Cloud Sync complete.")
            except Exception as e:
                print(f"ERROR: Sync failed: {e}")

    def _upload_file(self, local_path, path_in_repo):
        self.api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type="dataset",
            token=self.token
        )

    def _upload_folder(self, local_dir, path_in_repo):
        self.api.upload_folder(
            folder_path=local_dir,
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type="dataset",
            token=self.token,
            delete_patterns=None # Không xóa file cũ trên Hub
        )

# Global instance
sync_service = SyncService()
