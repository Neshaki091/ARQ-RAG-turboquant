"""
handler.py — RAG Adaptive (Dynamic Top-K)
==========================================
RAGAdaptiveHandler: tự động điều chỉnh tham số retrieval
dựa trên độ phức tạp của câu hỏi được phát hiện bởi LLM.

Chiến lược điều chỉnh:
  SIMPLE  → limit=20, top_k=5   (nhanh, tiết kiệm token LLM)
  COMPLEX → limit=80, top_k=20  (sâu, bao phủ ngữ cảnh rộng hơn)

Thuật toán bên dưới: dùng RAG-PQ (vì kết hợp tốc độ ADC + độ chính xác)
Collection: vector_pq (reuse)

Điểm khác biệt:
  - Bước phân tích query TRƯỚC khi retrieval
  - Cache kết quả phân tích vào Supabase (bỏ qua LLM để phân tích lần sau)
  - Metrics bao gồm cả query_complexity và retrievl params đã dùng
"""

import logging
import os
import time
import uuid
from typing import Any, Dict, Iterator, Optional

import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from models.rag_pq.quantizer import ProductQuantizer
from shared.embed import OllamaEmbedder
from shared.query_analyzer import QueryAnalyzer
from shared.supabase_client import SupabaseManager
from shared.vector_store import QdrantManager, COLLECTION_NAMES

load_dotenv()
logger = logging.getLogger(__name__)

COLLECTION_NAME = COLLECTION_NAMES["pq"]   # Dùng vector_pq
PQ_CODEBOOK_PATH = "pq/centroids.pkl"
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "120000"))

SYSTEM_PROMPT = """[ARQ-RAG | Adaptive RAG]
Bạn là trợ lý nghiên cứu khoa học. Trả lời DỨT KHOÁT, không chào hỏi.
Sử dụng LaTeX ($...$) cho công thức toán học.
Trích dẫn nguồn [source] nếu có."""


class RAGAdaptiveHandler:
    """
    Adaptive RAG: tự điều chỉnh top_k/limit dựa theo query complexity.

    Query SIMPLE  → nhẹ nhàng (limit=20, top_k=5)
    Query COMPLEX → đào sâu (limit=80, top_k=20)
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
        self.analyzer = QueryAnalyzer(supabase_manager=supabase)
        self._pq: Optional[ProductQuantizer] = None

        self.llm = ChatGroq(
            model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.1,
            max_tokens=2048,
            api_key=os.getenv("GROQ_API_KEY", ""),
            streaming=True,
        )

    def _get_pq(self) -> ProductQuantizer:
        if self._pq is not None:
            return self._pq
        data = self.supabase.download_pickle(PQ_CODEBOOK_PATH)
        if data is None:
            raise RuntimeError("Chưa có PQ codebook trong Supabase.")
        self._pq = ProductQuantizer.from_dict(data)
        return self._pq

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        limit: Optional[int] = None,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Pipeline Adaptive RAG:
          1. Phân tích complexity (cache-first)
          2. Điều chỉnh limit/top_k theo complexity
          3. PQ RAG pipeline (embed → ADC → retrieve → rerank → LLM)
        """
        t_start = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())

        # ── Bước 1: Phân tích query complexity ────────────────────────
        t_analyze = time.perf_counter()
        params = self.analyzer.get_retrieval_params(query_text)
        analyze_ms = (time.perf_counter() - t_analyze) * 1000

        # Override nếu caller cung cấp giá trị cụ thể
        effective_limit = limit if limit is not None else params["limit"]
        effective_top_k = top_k if top_k is not None else params["top_k"]
        complexity = params["complexity"]

        logger.info(
            f"[Adaptive] complexity={complexity}, "
            f"limit={effective_limit}, top_k={effective_top_k}"
        )

        # ── Bước 2: Embed query ────────────────────────────────────────
        query_vec = self.embedder.embed_text(query_text)

        # ── Bước 3: ADC table + Retrieval ─────────────────────────────
        pq = self._get_pq()
        adc_table = pq.compute_adc_table(query_vec)

        t_retrieve = time.perf_counter()
        candidates = self.qdrant.search(COLLECTION_NAME, query_vec, limit=effective_limit, payload_filter=filters)
        retrieve_ms = (time.perf_counter() - t_retrieve) * 1000

        if not candidates:
            return self._empty_response(t_start, complexity)

        # ── Bước 4: ADC Reranking ──────────────────────────────────────
        t_rerank = time.perf_counter()
        pq_codes_list, valid = [], []
        for pt in candidates:
            codes = pt.payload.get("pq_codes")
            if codes is not None:
                pq_codes_list.append(codes)
                valid.append(pt)

        if pq_codes_list:
            codes_matrix = np.array(pq_codes_list, dtype=np.uint8)
            adc_scores = pq.adc_score_batch(adc_table, codes_matrix)
            top_indices = np.argsort(adc_scores)[::-1][:effective_top_k]
            reranked = [valid[i] for i in top_indices]
        else:
            reranked = candidates[:effective_top_k]

        rerank_ms = (time.perf_counter() - t_rerank) * 1000

        # ── Bước 5: Build context ──────────────────────────────────────
        context_parts = [
            f"[{pt.payload.get('source','?')}]\n{pt.payload.get('text','')}"
            for pt in reranked if pt.payload.get("text")
        ]
        context = "\n\n---\n\n".join(context_parts)[:MAX_CONTEXT_CHARS]

        # ── Bước 6: LLM Generation ─────────────────────────────────────
        t_llm = time.perf_counter()
        prompt = f"Context:\n{'='*60}\n{context}\n{'='*60}\n\nCâu hỏi: {query_text}\n\nTrả lời:"
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        answer = self.llm.invoke(messages).content
        llm_ms = (time.perf_counter() - t_llm) * 1000

        total_ms = (time.perf_counter() - t_start) * 1000
        try:
            self.supabase.save_chat(session_id, "adaptive", query_text, answer, total_ms)
        except Exception:
            pass

        return {
            "answer": answer,
            "model": "rag_adaptive",
            "session_id": session_id,
            "metrics": {
                "total_latency_ms": round(total_ms, 2),
                "analyze_latency_ms": round(analyze_ms, 2),
                "retrieve_latency_ms": round(retrieve_ms, 2),
                "rerank_latency_ms": round(rerank_ms, 2),
                "llm_latency_ms": round(llm_ms, 2),
                "query_complexity": complexity,
                "effective_limit": effective_limit,
                "effective_top_k": effective_top_k,
                "retrieval_count": len(candidates),
                "rerank_count": len(reranked),
            },
            "sources": [pt.payload.get("source", "?") for pt in reranked],
        }

    def query_stream(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        limit: Optional[int] = None,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        """Streaming version của Adaptive RAG."""
        t_start = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())

        params = self.analyzer.get_retrieval_params(query_text)
        effective_limit = limit or params["limit"]
        effective_top_k = top_k or params["top_k"]

        query_vec = self.embedder.embed_text(query_text)
        pq = self._get_pq()
        adc_table = pq.compute_adc_table(query_vec)
        candidates = self.qdrant.search(COLLECTION_NAME, query_vec, limit=effective_limit, payload_filter=filters)

        if not candidates:
            yield "[Adaptive] Không tìm thấy kết quả."
            return

        pq_codes_list, valid = [], []
        for pt in candidates:
            codes = pt.payload.get("pq_codes")
            if codes is not None:
                pq_codes_list.append(codes)
                valid.append(pt)

        if pq_codes_list:
            codes_matrix = np.array(pq_codes_list, dtype=np.uint8)
            adc_scores = pq.adc_score_batch(adc_table, codes_matrix)
            top_idx = np.argsort(adc_scores)[::-1][:effective_top_k]
            reranked = [valid[i] for i in top_idx]
        else:
            reranked = candidates[:effective_top_k]

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
            self.supabase.save_chat(session_id, "adaptive", query_text, full_answer, total_ms)
        except Exception:
            pass

    def _empty_response(self, t_start: float, complexity: str) -> Dict[str, Any]:
        total_ms = (time.perf_counter() - t_start) * 1000
        return {
            "answer": "[Adaptive] Không tìm thấy tài liệu phù hợp.",
            "model": "rag_adaptive",
            "metrics": {
                "total_latency_ms": round(total_ms, 2),
                "query_complexity": complexity,
                "retrieval_count": 0,
            },
            "sources": [],
        }
