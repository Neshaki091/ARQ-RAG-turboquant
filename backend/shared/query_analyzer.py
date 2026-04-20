"""
query_analyzer.py
-----------------
QueryAnalyzer: phân loại câu hỏi thành SIMPLE hoặc COMPLEX
bằng LLM (Groq), kèm cache Supabase để tránh gọi LLM lại.

Kết quả phân loại ảnh hưởng trực tiếp đến:
  - SIMPLE  → limit=20, top_k=5   (nhanh, tiết kiệm)
  - COMPLEX → limit=80, top_k=20  (sâu, ngữ cảnh rộng hơn)

Cơ chế cache:
  Query → MD5 hash → Tra bảng query_cache (Supabase)
  Nếu cache hit → trả về ngay (0ms LLM latency)
  Nếu cache miss → gọi Groq → lưu cache → trả về
"""

import hashlib
import logging
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from shared.supabase_client import SupabaseManager

load_dotenv()
logger = logging.getLogger(__name__)

# Cấu hình tham số retrieval theo độ phức tạp
RETRIEVAL_CONFIG = {
    "SIMPLE": {"limit": 20, "top_k": 5},
    "COMPLEX": {"limit": 80, "top_k": 20},
}

COMPLEXITY_SYSTEM_PROMPT = """Bạn là bộ phân loại câu hỏi cho hệ thống RAG khoa học.
Nhiệm vụ DUY NHẤT: phân loại câu hỏi là SIMPLE hoặc COMPLEX.

Quy tắc phân loại:
- SIMPLE: Câu hỏi định nghĩa, khái niệm đơn lẻ, tra cứu trực tiếp.
  Ví dụ: "PQ là gì?", "nomic-embed-text có bao nhiêu chiều?"
- COMPLEX: Câu hỏi so sánh, phân tích đa chiều, yêu cầu tổng hợp nhiều nguồn.
  Ví dụ: "So sánh ADC và SDC trong Product Quantization về độ chính xác và tốc độ"

Chỉ trả về đúng 1 từ: SIMPLE hoặc COMPLEX. Không giải thích."""


class QueryAnalyzer:
    """
    Phân tích độ phức tạp câu hỏi để điều chỉnh tham số retrieval.

    Tích hợp 2 tầng:
    1. Cache Supabase (ưu tiên) — tránh gọi LLM cho câu hỏi cũ
    2. Groq LLM (fallback) — phân tích câu hỏi mới
    """

    def __init__(self, supabase_manager: SupabaseManager):
        self.db = supabase_manager
        self.llm = ChatGroq(
            model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            temperature=0,
            max_tokens=10,  # Chỉ cần 1 từ: SIMPLE/COMPLEX
            api_key=os.getenv("GROQ_API_KEY", ""),
        )

    def _hash_query(self, query: str) -> str:
        """MD5 hash của query (chuẩn hóa lowercase + strip)."""
        normalized = query.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()

    def classify(self, query: str) -> Literal["SIMPLE", "COMPLEX"]:
        """
        Phân loại câu hỏi — ưu tiên cache trước khi gọi LLM.

        Args:
            query: Câu hỏi từ người dùng

        Returns:
            "SIMPLE" hoặc "COMPLEX"
        """
        query_hash = self._hash_query(query)

        # Tầng 1: Tra cache Supabase
        cached = self.db.get_cached_complexity(query_hash)
        if cached in ("SIMPLE", "COMPLEX"):
            logger.debug(f"Cache hit: query_hash={query_hash[:8]}... → {cached}")
            return cached

        # Tầng 2: Gọi LLM Groq
        try:
            messages = [
                SystemMessage(content=COMPLEXITY_SYSTEM_PROMPT),
                HumanMessage(content=f"Câu hỏi: {query}"),
            ]
            response = self.llm.invoke(messages)
            raw = response.content.strip().upper()

            # Parse kết quả — chỉ chấp nhận SIMPLE/COMPLEX
            complexity: Literal["SIMPLE", "COMPLEX"] = (
                "COMPLEX" if "COMPLEX" in raw else "SIMPLE"
            )

            # Lưu vào cache Supabase
            self.db.set_cached_complexity(query_hash, query, complexity)
            logger.info(f"LLM classify: '{query[:50]}...' → {complexity}")
            return complexity

        except Exception as e:
            logger.warning(f"Lỗi classify LLM: {e}. Fallback về COMPLEX.")
            return "COMPLEX"  # Fallback an toàn: lấy nhiều context hơn

    def get_retrieval_params(self, query: str) -> dict:
        """
        Trả về dict tham số retrieval dựa trên độ phức tạp câu hỏi.

        Returns:
            {
                "complexity": "SIMPLE" | "COMPLEX",
                "limit": int,  # số ứng viên lấy từ Qdrant
                "top_k": int,  # số kết quả sau reranking
            }
        """
        complexity = self.classify(query)
        params = RETRIEVAL_CONFIG[complexity].copy()
        params["complexity"] = complexity
        return params
