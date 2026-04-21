"""
handler.py — RAG-SQ8 (Scalar Quantization 8-bit)
==================================================
RAGSq8Handler: pipeline RAG với Scalar Quantization.

Khác biệt với RAG-PQ:
  - Reranking dùng decoded dot product thay vì ADC table lookup
  - Codebook nhỏ hơn (chỉ cần min/max toàn cục)
  - Nén ít hơn (4x) nhưng không cần training phức tạp

Collection: vector_sq8
"""

import logging
import os
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from models.rag_sq8.quantizer import ScalarQuantizer
from shared.embed import OllamaEmbedder
from shared.supabase_client import SupabaseManager
from shared.vector_store import QdrantManager, COLLECTION_NAMES

load_dotenv()
logger = logging.getLogger(__name__)

COLLECTION_NAME = COLLECTION_NAMES["sq8"]  # "vector_sq8"
SQ8_PARAMS_PATH = "sq8/params.pkl"
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "120000"))

SYSTEM_PROMPT = """[ARQ-RAG | Scalar Quantization 8-bit]
Bạn là trợ lý nghiên cứu khoa học. Trả lời DỨT KHOÁT, không chào hỏi.
Sử dụng LaTeX ($...$) cho công thức toán học.
Trích dẫn nguồn [source] nếu có."""


class RAGSq8Handler:
    """
    Handler RAG với Scalar Quantization 8-bit.

    Pipeline query:
    1. Embed query (float32)
    2. Qdrant search (limit=40)
    3. Reranking: decode SQ8 codes → dot product với query
    4. Top-K → context → Groq LLM
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
        self._sq: Optional[ScalarQuantizer] = None

        self.llm = ChatGroq(
            model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.1,
            max_tokens=2048,
            api_key=os.getenv("GROQ_API_KEY", ""),
            streaming=True,
        )

    def _get_sq(self) -> ScalarQuantizer:
        if self._sq is not None:
            return self._sq
        data = self.supabase.download_pickle(SQ8_PARAMS_PATH)
        if data is None:
            raise RuntimeError("Chưa có SQ8 params trong Supabase. Chạy ingest trước.")
        self._sq = ScalarQuantizer.from_dict(data)
        logger.info(f"ScalarQuantizer loaded: min={self._sq.min_val:.4f}, max={self._sq.max_val:.4f}")
        return self._sq

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        limit: int = 40,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())

        # Embed
        query_vec = self.embedder.embed_text(query_text)

        # Retrieval
        t_retrieve = time.perf_counter()
        candidates = self.qdrant.search(COLLECTION_NAME, query_vec, limit=limit, payload_filter=filters)
        retrieve_ms = (time.perf_counter() - t_retrieve) * 1000

        if not candidates:
            return self._empty_response(t_start)

        # Reranking bằng SQ8 decode + dot product
        t_rerank = time.perf_counter()
        sq = self._get_sq()
        sq8_codes_list, valid = [], []
        for pt in candidates:
            codes = pt.payload.get("sq8_codes")
            if codes is not None:
                sq8_codes_list.append(codes)
                valid.append(pt)

        if sq8_codes_list:
            codes_matrix = np.array(sq8_codes_list, dtype=np.uint8)  # (N, D)
            scores = sq.encode_and_score(query_vec, codes_matrix)     # (N,)
            top_idx = np.argsort(scores)[::-1][:top_k]
            reranked = [valid[i] for i in top_idx]
        else:
            reranked = candidates[:top_k]

        rerank_ms = (time.perf_counter() - t_rerank) * 1000

        # Build context
        context_parts = [
            f"[{pt.payload.get('source','?')}]\n{pt.payload.get('text','')}"
            for pt in reranked if pt.payload.get("text")
        ]
        context = "\n\n---\n\n".join(context_parts)[:MAX_CONTEXT_CHARS]

        # LLM
        t_llm = time.perf_counter()
        prompt = f"Context:\n{'='*60}\n{context}\n{'='*60}\n\nCâu hỏi: {query_text}\n\nTrả lời:"
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        answer = self.llm.invoke(messages).content
        llm_ms = (time.perf_counter() - t_llm) * 1000

        total_ms = (time.perf_counter() - t_start) * 1000
        try:
            self.supabase.save_chat(session_id, "sq8", query_text, answer, total_ms)
        except Exception:
            pass

        return {
            "answer": answer,
            "model": "rag_sq8",
            "session_id": session_id,
            "metrics": {
                "total_latency_ms": round(total_ms, 2),
                "retrieve_latency_ms": round(retrieve_ms, 2),
                "rerank_latency_ms": round(rerank_ms, 2),
                "llm_latency_ms": round(llm_ms, 2),
                "retrieval_count": len(candidates),
                "rerank_count": len(reranked),
                "compression": "4x (SQ8)",
            },
            "sources": [pt.payload.get("source", "?") for pt in reranked],
        }

    def query_stream(
        self,
        query_text: str,
        top_k: int = 10,
        limit: int = 40,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        t_start = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())

        query_vec = self.embedder.embed_text(query_text)
        candidates = self.qdrant.search(COLLECTION_NAME, query_vec, limit=limit, payload_filter=filters)
        if not candidates:
            yield "[RAG-SQ8] Không tìm thấy kết quả."
            return

        sq = self._get_sq()
        sq8_codes_list, valid = [], []
        for pt in candidates:
            codes = pt.payload.get("sq8_codes")
            if codes is not None:
                sq8_codes_list.append(codes)
                valid.append(pt)

        if sq8_codes_list:
            codes_matrix = np.array(sq8_codes_list, dtype=np.uint8)
            scores = sq.encode_and_score(query_vec, codes_matrix)
            top_idx = np.argsort(scores)[::-1][:top_k]
            reranked = [valid[i] for i in top_idx]
        else:
            reranked = candidates[:top_k]

        context_parts = [
            f"[{pt.payload.get('source','?')}]\n{pt.payload.get('text','')}"
            for pt in reranked if pt.payload.get("text")
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
            self.supabase.save_chat(session_id, "sq8", query_text, full_answer, total_ms)
        except Exception:
            pass

    def _empty_response(self, t_start: float) -> Dict[str, Any]:
        total_ms = (time.perf_counter() - t_start) * 1000
        return {
            "answer": "[RAG-SQ8] Không tìm thấy tài liệu phù hợp.",
            "model": "rag_sq8",
            "metrics": {"total_latency_ms": round(total_ms, 2), "retrieval_count": 0},
            "sources": [],
        }
