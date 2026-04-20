"""
handler.py — RAG Raw (Float32 Baseline)
========================================
Đây là baseline chuẩn: lưu và tìm kiếm vector float32 gốc,
KHÔNG nén, KHÔNG reranking — dùng trực tiếp Qdrant cosine score.

Mục đích:
  - Chuẩn baseline để so sánh với PQ, SQ8, ARQ
  - Độ chính xác cao nhất (không mất thông tin do nén)
  - Tốc độ chậm hơn PQ vì không có ADC table lookup

Collection: vector_raw
"""

import logging
import os
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from shared.embed import OllamaEmbedder
from shared.supabase_client import SupabaseManager
from shared.vector_store import QdrantManager, COLLECTION_NAMES

load_dotenv()
logger = logging.getLogger(__name__)

COLLECTION_NAME = COLLECTION_NAMES["raw"]  # "vector_raw"
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "120000"))

SYSTEM_PROMPT = """[ARQ-RAG | Float32 Baseline]
Bạn là trợ lý nghiên cứu khoa học. Trả lời DỨT KHOÁT, không chào hỏi.
Sử dụng LaTeX ($...$) cho công thức toán học.
Trích dẫn nguồn [source] nếu có."""


class RAGRawHandler:
    """
    Handler baseline Float32 — cosine similarity thuần tuý qua Qdrant.

    Không có bước reranking: Qdrant score = kết quả cuối.
    Dùng làm ground truth để đánh giá chất lượng của PQ và SQ8.
    """

    def __init__(
        self,
        qdrant: QdrantManager,
        supabase: SupabaseManager,
        embedder: OllamaEmbedder,
    ):
        self.qdrant = qdrant
        self.supabase = supabase
        self.embedder = embedder
        self.llm = ChatGroq(
            model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.1,
            max_tokens=2048,
            api_key=os.getenv("GROQ_API_KEY", ""),
            streaming=True,
        )

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        limit: int = 40,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pipeline RAG chuẩn Float32: embed → search → LLM."""
        t_start = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())

        # Embed query
        t_embed = time.perf_counter()
        query_vec = self.embedder.embed_text(query_text)
        embed_ms = (time.perf_counter() - t_embed) * 1000

        # Search Qdrant (float32 cosine)
        t_search = time.perf_counter()
        results = self.qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k,  # Raw không cần candidate pool lớn
        )
        search_ms = (time.perf_counter() - t_search) * 1000

        if not results:
            return self._empty_response(t_start)

        # Build context
        context_parts = [
            f"[{pt.payload.get('source','?')}]\n{pt.payload.get('text','')}"
            for pt in results if pt.payload.get("text")
        ]
        context = "\n\n---\n\n".join(context_parts)[:MAX_CONTEXT_CHARS]

        # LLM Generate
        t_llm = time.perf_counter()
        prompt = f"Context:\n{'='*60}\n{context}\n{'='*60}\n\nCâu hỏi: {query_text}\n\nTrả lời:"
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        answer = self.llm.invoke(messages).content
        llm_ms = (time.perf_counter() - t_llm) * 1000

        total_ms = (time.perf_counter() - t_start) * 1000

        try:
            self.supabase.save_chat(session_id, "raw", query_text, answer, total_ms)
        except Exception:
            pass

        return {
            "answer": answer,
            "model": "rag_raw",
            "session_id": session_id,
            "metrics": {
                "total_latency_ms": round(total_ms, 2),
                "embed_latency_ms": round(embed_ms, 2),
                "retrieve_latency_ms": round(search_ms, 2),
                "llm_latency_ms": round(llm_ms, 2),
                "retrieval_count": len(results),
                "rerank_count": 0,
                "note": "Float32 baseline — no compression, no reranking",
            },
            "sources": [pt.payload.get("source", "?") for pt in results],
        }

    def query_stream(
        self,
        query_text: str,
        top_k: int = 10,
        limit: int = 40,
        session_id: Optional[str] = None,
    ) -> Iterator[str]:
        """Streaming version của query()."""
        t_start = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())

        query_vec = self.embedder.embed_text(query_text)
        results = self.qdrant.search(COLLECTION_NAME, query_vec, limit=top_k)

        if not results:
            yield "[RAG-Raw] Không tìm thấy kết quả."
            return

        context_parts = [
            f"[{pt.payload.get('source','?')}]\n{pt.payload.get('text','')}"
            for pt in results if pt.payload.get("text")
        ]
        context = "\n\n---\n\n".join(context_parts)[:MAX_CONTEXT_CHARS]
        prompt = f"Context:\n{'='*60}\n{context}\n{'='*60}\n\nCâu hỏi: {query_text}\n\nTrả lời:"
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]

        full_answer = ""
        for chunk in self.llm.stream(messages):
            if chunk.content:
                full_answer += chunk.content
                yield chunk.content

        total_ms = (time.perf_counter() - t_start) * 1000
        try:
            self.supabase.save_chat(session_id, "raw", query_text, full_answer, total_ms)
        except Exception:
            pass

    def _empty_response(self, t_start: float) -> Dict[str, Any]:
        total_ms = (time.perf_counter() - t_start) * 1000
        return {
            "answer": "[RAG-Raw] Không tìm thấy tài liệu phù hợp.",
            "model": "rag_raw",
            "metrics": {"total_latency_ms": round(total_ms, 2), "retrieval_count": 0},
            "sources": [],
        }
