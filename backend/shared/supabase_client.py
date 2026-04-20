"""
supabase_client.py
------------------
SupabaseManager: kết nối Supabase để lưu/tải model weights (centroids.pkl)
và cache lịch sử truy vấn thay vì lưu file nặng trong Git.

Được dùng bởi:
- rag_pq/quantizer.py  (tải/lưu PQ codebook)
- arq_rag/reranker.py  (tải QJL projection matrix)
- query_analyzer.py    (cache query complexity)
"""

import os
import io
import pickle
import logging
from typing import Any, Optional

from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class SupabaseManager:
    """
    Manager tập trung cho mọi thao tác với Supabase.

    - Storage bucket: lưu file model weights (.pkl, .npy)
    - PostgreSQL table: cache query_complexity, lịch sử chat
    """

    def __init__(self):
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")
        self.bucket = os.getenv("SUPABASE_BUCKET", "arq-rag-models")

        if not url or not key:
            raise EnvironmentError(
                "SUPABASE_URL và SUPABASE_KEY phải được khai báo trong .env"
            )

        self.client: Client = create_client(url, key)
        logger.info(f"SupabaseManager khởi tạo. Bucket: {self.bucket}")

    # ------------------------------------------------------------------
    # Storage: lưu/tải binary objects (pickle)
    # ------------------------------------------------------------------

    def upload_pickle(self, obj: Any, remote_path: str) -> bool:
        """
        Serialize object bất kỳ bằng pickle và upload lên Supabase Storage.

        Args:
            obj: Object cần lưu (ProductQuantizer, np.ndarray, dict, ...)
            remote_path: Đường dẫn trong bucket, vd: "pq/centroids.pkl"

        Returns:
            True nếu upload thành công.
        """
        try:
            buffer = io.BytesIO()
            pickle.dump(obj, buffer)
            buffer.seek(0)

            self.client.storage.from_(self.bucket).upload(
                path=remote_path,
                file=buffer.read(),
                file_options={"content-type": "application/octet-stream", "upsert": "true"},
            )
            logger.info(f"Đã upload {remote_path} lên bucket '{self.bucket}'")
            return True
        except Exception as e:
            logger.error(f"Lỗi upload_pickle({remote_path}): {e}")
            return False

    def download_pickle(self, remote_path: str) -> Optional[Any]:
        """
        Tải file từ Supabase Storage và deserialize bằng pickle.

        Args:
            remote_path: Đường dẫn trong bucket, vd: "pq/centroids.pkl"

        Returns:
            Object đã deserialize, hoặc None nếu không tìm thấy.
        """
        try:
            data = self.client.storage.from_(self.bucket).download(remote_path)
            obj = pickle.loads(data)
            logger.info(f"Đã tải {remote_path} từ bucket '{self.bucket}'")
            return obj
        except Exception as e:
            logger.warning(f"Không tải được {remote_path}: {e}")
            return None

    # ------------------------------------------------------------------
    # PostgreSQL: cache query complexity
    # ------------------------------------------------------------------

    def get_cached_complexity(self, query_hash: str) -> Optional[str]:
        """
        Tra cache độ phức tạp câu hỏi từ bảng query_cache.
        Tránh gọi LLM lại cho những câu hỏi đã phân tích.

        Args:
            query_hash: MD5 hash của câu hỏi (lowercase, stripped)

        Returns:
            "SIMPLE" | "COMPLEX" | None nếu chưa có trong cache
        """
        try:
            result = (
                self.client.table("query_cache")
                .select("complexity")
                .eq("query_hash", query_hash)
                .limit(1)
                .execute()
            )
            if result.data:
                return result.data[0]["complexity"]
            return None
        except Exception as e:
            logger.warning(f"get_cached_complexity error: {e}")
            return None

    def set_cached_complexity(self, query_hash: str, query_text: str, complexity: str) -> bool:
        """
        Lưu kết quả phân tích độ phức tạp vào bảng query_cache.

        Args:
            query_hash: MD5 hash của câu hỏi
            query_text: Câu hỏi gốc (để debug)
            complexity: "SIMPLE" hoặc "COMPLEX"
        """
        try:
            self.client.table("query_cache").upsert({
                "query_hash": query_hash,
                "query_text": query_text[:500],  # Giới hạn độ dài lưu
                "complexity": complexity,
            }).execute()
            return True
        except Exception as e:
            logger.warning(f"set_cached_complexity error: {e}")
            return False

    # ------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------

    def save_chat(
        self,
        session_id: str,
        model_name: str,
        question: str,
        answer: str,
        latency_ms: float,
    ) -> bool:
        """Lưu lịch sử truy vấn vào bảng chat_history."""
        try:
            self.client.table("chat_history").insert({
                "session_id": session_id,
                "model_name": model_name,
                "question": question[:2000],
                "answer": answer[:10000],
                "latency_ms": latency_ms,
            }).execute()
            return True
        except Exception as e:
            logger.warning(f"save_chat error: {e}")
            return False
