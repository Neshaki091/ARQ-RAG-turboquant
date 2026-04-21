"""
chat_service.py
===============
ChatDispatcher — Bộ điều phối tập trung cho mọi yêu cầu chat.

Routing theo model name:
  "raw"      → RAGRawHandler
  "pq"       → RAGPQHandler      (default)
  "sq8"      → RAGSq8Handler
  "arq"      → ARQRAGHandler
  "adaptive" → RAGAdaptiveHandler

Khởi tạo singleton cho mỗi handler (lazy, chỉ init khi được dùng lần đầu).
Chia sẻ các shared instances: QdrantManager, SupabaseManager, OllamaEmbedder.
"""

import logging
import os
from typing import Any, Dict, Iterator, Optional

from shared.embed import OllamaEmbedder
from shared.supabase_client import SupabaseManager
from shared.vector_store import QdrantManager

from models.rag_raw.handler import RAGRawHandler
from models.rag_pq.handler import RAGPQHandler
from models.rag_sq8.handler import RAGSq8Handler
from models.arq_rag.handler import ARQRAGHandler
from models.rag_adaptive.handler import RAGAdaptiveHandler

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {"raw", "pq", "sq8", "arq", "adaptive"}
DEFAULT_MODEL = "pq"


class ChatDispatcher:
    """
    Bộ điều phối truy vấn đến đúng model handler.

    Shared resources (khởi tạo 1 lần, dùng bởi tất cả handlers):
      - QdrantManager: connection pool đến Qdrant
      - SupabaseManager: connection đến Supabase
      - OllamaEmbedder: HTTP client đến Ollama

    Handlers được khởi tạo lazy (khi được gọi lần đầu).
    """

    def __init__(self):
        logger.info("Khởi tạo ChatDispatcher...")

        # Shared infrastructure
        self.qdrant = QdrantManager()
        self.supabase = SupabaseManager()
        self.embedder = OllamaEmbedder()

        # Đảm bảo tất cả Qdrant collections tồn tại khi startup
        self.qdrant.ensure_all_collections()

        # Handler registry (lazy init)
        self._handlers: Dict[str, Any] = {}
        logger.info(f"ChatDispatcher ready. Supported models: {SUPPORTED_MODELS}")

    def _get_handler(self, model: str) -> Any:
        """Lấy handler cho model, khởi tạo nếu chưa có."""
        if model not in self._handlers:
            logger.info(f"Khởi tạo handler: {model}")
            kwargs = dict(
                qdrant=self.qdrant,
                supabase=self.supabase,
                embedder=self.embedder,
            )
            if model == "raw":
                self._handlers[model] = RAGRawHandler(**kwargs)
            elif model == "pq":
                self._handlers[model] = RAGPQHandler(**kwargs)
            elif model == "sq8":
                self._handlers[model] = RAGSq8Handler(**kwargs)
            elif model == "arq":
                self._handlers[model] = ARQRAGHandler(**kwargs)
            elif model == "adaptive":
                self._handlers[model] = RAGAdaptiveHandler(**kwargs)
            else:
                raise ValueError(f"Model không hỗ trợ: {model}. Chọn: {SUPPORTED_MODELS}")

        return self._handlers[model]

    def chat(
        self,
        query: str,
        model: str = DEFAULT_MODEL,
        top_k: Optional[int] = None,
        limit: Optional[int] = None,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Xử lý truy vấn và trả về kết quả đầy đủ.

        Args:
            query: Câu hỏi của người dùng
            model: Tên model ("raw", "pq", "sq8", "arq", "adaptive")
            top_k: Số kết quả sau reranking (None = dùng default của model)
            limit: Số candidates từ Qdrant (None = dùng default)
            session_id: ID phiên chat
            filters: Bộ lọc metadata Qdrant

        Returns:
            Dict với keys: answer, model, metrics, sources, session_id
        """
        if not query or not query.strip():
            return {
                "answer": "Vui lòng nhập câu hỏi.",
                "model": model,
                "metrics": {},
                "sources": [],
            }

        model = model.lower().strip()
        if model not in SUPPORTED_MODELS:
            logger.warning(f"Model không hợp lệ '{model}', dùng default '{DEFAULT_MODEL}'")
            model = DEFAULT_MODEL

        handler = self._get_handler(model)
        kwargs = {"query_text": query, "session_id": session_id}
        if top_k is not None:
            kwargs["top_k"] = top_k
        if limit is not None:
            kwargs["limit"] = limit
        if filters is not None:
            kwargs["filters"] = filters

        return handler.query(**kwargs)

    def chat_stream(
        self,
        query: str,
        model: str = DEFAULT_MODEL,
        top_k: Optional[int] = None,
        limit: Optional[int] = None,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        """
        Streaming version — yield từng token LLM.
        Dùng cho SSE endpoint.
        """
        if not query or not query.strip():
            yield "Vui lòng nhập câu hỏi."
            return

        model = model.lower().strip()
        if model not in SUPPORTED_MODELS:
            model = DEFAULT_MODEL

        handler = self._get_handler(model)
        kwargs = {"query_text": query, "session_id": session_id}
        if top_k is not None:
            kwargs["top_k"] = top_k
        if limit is not None:
            kwargs["limit"] = limit
        if filters is not None:
            kwargs["filters"] = filters

        yield from handler.query_stream(**kwargs)

    def get_collections_status(self) -> Dict[str, Any]:
        """Trả về trạng thái tất cả Qdrant collections."""
        return {"collections": self.qdrant.get_all_collections_info()}
