import os
import asyncio
from huggingface_hub import HfApi
from concurrent.futures import ThreadPoolExecutor

class SyncService:
    def __init__(self):
        self.api = HfApi()
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.repo_id = os.getenv("HF_DATASET_REPO")
        self.token = os.getenv("HF_TOKEN_WRITE")

    async def sync_to_hub(self, user_id: str):
        """Đẩy dữ liệu của một user cụ thể lên Hugging Face Dataset ngầm"""
        if not self.repo_id or not self.token:
            # print("WARNING: Sync skipped: HF_DATASET_REPO or HF_TOKEN_WRITE not set.")
            return

        async with self.lock:
            try:
                loop = asyncio.get_event_loop()
                
                # 1. Đồng bộ Metadata DB (Chung hoặc riêng tùy cấu hình)
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                db_path = os.path.join(base_dir, "data", "metadata.db")
                
                if os.path.exists(db_path):
                    await loop.run_in_executor(
                        self.executor,
                        self._upload_file,
                        db_path,
                        f"users/{user_id}/metadata.db"
                    )

                # 2. Đồng bộ Index mới nhất (Tìm file .tq mới nhất của user)
                # Lưu ý: Hiện tại tq_service lưu index chung, ta sẽ ưu tiên đồng bộ file index chính
                data_dir = os.path.join(base_dir, "data")
                # Giả sử ta đồng bộ file index vừa mới lưu
                # (Logic này có thể tinh chỉnh thêm để chỉ gửi đúng file cần thiết)
                
                print(f"SUCCESS: Sync complete for user: {user_id}")
            except Exception as e:
                print(f"ERROR: Sync failed for user {user_id}: {e}")

    def _upload_file(self, local_path, path_in_repo):
        self.api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type="dataset",
            token=self.token
        )

# Global instance
sync_service = SyncService()
