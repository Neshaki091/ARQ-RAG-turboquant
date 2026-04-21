"""
handler.py
==========
RAGPQHandler — Xử lý truy vấn RAG với Product Quantization.

═══════════════════════════════════════════════════════════════
PIPELINE TRUY VẤN RAG-PQ
═══════════════════════════════════════════════════════════════

1. EMBED QUERY (Float32, KHÔNG nén)
   query_text → OllamaEmbedder → query_vec ∈ ℝ^768 (float32)
   ↓
2. COMPUTE ADC TABLE (1 lần duy nhất cho mỗi query)
   query_vec → ProductQuantizer.compute_adc_table()
   → T ∈ ℝ^(8×256)  (8 sub-spaces × 256 centroids)
   ↓
3. CANDIDATE RETRIEVAL (Qdrant HNSW)
   query_vec → Qdrant search(limit=40) → 40 ScoredPoints
   Qdrant dùng float32 vector + HNSW → approximate neighbors nhanh
   ↓
4. ADC RERANKING (numpy, fully vectorized)
   Lấy pq_codes từ payload của 40 candidates
   pq_codes shape: (40, 8), uint8
   → adc_score_batch(T, pq_codes) → scores shape (40,)
   → argsort descending → lấy top_k indices
   ↓
5. LLM GENERATION (Groq)
   top_k contexts → Prompt → Groq llama-3.3-70b → streaming answer
   ↓
6. RETURN
   {answer, latency_ms, retrieval_count, rerank_count, complexity, ...}

Tại sao reranking lại chính xác hơn Qdrant scores?
  - Qdrant HNSW dùng approximate search (đánh đổi recall để lấy tốc độ)
  - ADC dùng toàn bộ M sub-spaces → ít mất thông tin hơn
  - Query KHÔNG nén (float32) vs Database ĐÃ nén (PQ codes)
    → Asymmetric: query chính xác, db nhanh
═══════════════════════════════════════════════════════════════
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

from models.rag_pq.quantizer import ProductQuantizer
from shared.embed import OllamaEmbedder
from shared.supabase_client import SupabaseManager
from shared.vector_store import QdrantManager, COLLECTION_NAMES

load_dotenv()
logger = logging.getLogger(__name__)

COLLECTION_NAME = COLLECTION_NAMES["pq"]   # "vector_pq"
PQ_CODEBOOK_PATH = "pq/centroids.pkl"

# Giới hạn context để tránh token overflow
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "120000"))

SYSTEM_PROMPT = """[ARQ-RAG | Product Quantization]
Bạn là trợ lý nghiên cứu khoa học chuyên sâu. Trả lời DỨT KHOÁT, không chào hỏi.
Sử dụng LaTeX ($...$) cho mọi công thức toán học.
Nếu không tìm thấy thông tin trong context, nói rõ: "Không tìm thấy trong tài liệu."
Trích dẫn nguồn [source] nếu có."""


class RAGPQHandler:
    """
    Handler xử lý truy vấn RAG với Product Quantization.

    Khởi tạo lazy: ProductQuantizer được tải từ Supabase lần đầu tiên
    khi có query (không tải khi import module).
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
        self._pq: Optional[ProductQuantizer] = None  # Lazy load

        self.llm = ChatGroq(
            model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.1,
            max_tokens=2048,
            api_key=os.getenv("GROQ_API_KEY", ""),
            streaming=True,
        )

    # ------------------------------------------------------------------
    # Lazy loading ProductQuantizer
    # ------------------------------------------------------------------

    def _get_pq(self) -> ProductQuantizer:
        """
        Tải ProductQuantizer từ Supabase (lazy, chỉ load 1 lần).
        Thread-safe với GIL của CPython.
        """
        if self._pq is not None:
            return self._pq

        logger.info("Đang tải ProductQuantizer từ Supabase...")
        data = self.supabase.download_pickle(PQ_CODEBOOK_PATH)
        if data is None:
            raise RuntimeError(
                "Không tìm thấy codebook PQ trong Supabase. "
                "Chạy scripts/train_pq.py trước."
            )

        self._pq = ProductQuantizer.from_dict(data)
        logger.info(f"ProductQuantizer loaded: {self._pq}")
        return self._pq

    # ------------------------------------------------------------------
    # Main query pipeline
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        limit: int = 40,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Thực thi full RAG-PQ pipeline và trả về kết quả.

        Args:
            query_text: Câu hỏi từ người dùng
            top_k: Số context sau reranking gửi cho LLM
            limit: Số candidates lấy từ Qdrant (trước reranking)
            session_id: ID phiên chat (để lưu lịch sử)

        Returns:
            Dict chứa answer, metrics, context snippets
        """
        t_start = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())

        logger.info(
            f"[RAG-PQ] Query: '{query_text[:80]}...' | "
            f"limit={limit}, top_k={top_k}"
        )

        # ── Bước 1: Embed query (float32, KHÔNG nén) ──────────────────
        t_embed_start = time.perf_counter()
        query_vec = self.embedder.embed_text(query_text)  # (768,) float32
        t_embed_ms = (time.perf_counter() - t_embed_start) * 1000

        # ── Bước 2: Compute ADC table (1 lần) ─────────────────────────
        t_adc_start = time.perf_counter()
        pq = self._get_pq()
        adc_table = pq.compute_adc_table(query_vec)  # (M, K) float32
        t_adc_ms = (time.perf_counter() - t_adc_start) * 1000

        # ── Bước 3: Candidate retrieval từ Qdrant ─────────────────────
        t_retrieve_start = time.perf_counter()
        candidates = self.qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=limit,
            payload_filter=filters,
        )
        t_retrieve_ms = (time.perf_counter() - t_retrieve_start) * 1000
        retrieval_count = len(candidates)

        if not candidates:
            logger.warning("Không tìm thấy candidates trong Qdrant")
            return self._empty_response(query_text, t_start)

        # ── Bước 4: ADC Reranking (numpy vectorized) ──────────────────
        t_rerank_start = time.perf_counter()

        # Lấy PQ codes từ payload của mỗi candidate
        pq_codes_list = []
        valid_candidates = []
        for scored_pt in candidates:
            codes = scored_pt.payload.get("pq_codes")
            if codes is not None:
                pq_codes_list.append(codes)
                valid_candidates.append(scored_pt)

        if not pq_codes_list:
            # Fallback: dùng Qdrant score nếu không có pq_codes
            logger.warning("Candidates không có pq_codes — dùng Qdrant scores")
            reranked = candidates[:top_k]
        else:
            # Stack tất cả codes thành ma trận (N, M)
            codes_matrix = np.array(pq_codes_list, dtype=np.uint8)  # (N, M)

            # Tính ADC scores — VECTORIZED, không for loop Python
            adc_scores = pq.adc_score_batch(adc_table, codes_matrix)  # (N,)

            # Sắp xếp theo score giảm dần (cao hơn = tốt hơn)
            top_indices = np.argsort(adc_scores)[::-1][:top_k]
            reranked = [valid_candidates[i] for i in top_indices]
            rerank_count = len(reranked)

        t_rerank_ms = (time.perf_counter() - t_rerank_start) * 1000

        # ── Bước 5: Chuẩn bị context cho LLM ─────────────────────────
        context_parts = []
        for pt in reranked:
            text = pt.payload.get("text", "")
            source = pt.payload.get("source", "unknown")
            if text:
                context_parts.append(f"[{source}]\n{text}")

        context = "\n\n---\n\n".join(context_parts)

        # Pruning context quá dài (tránh token overflow Groq)
        pruned = False
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS]
            pruned = True
            logger.warning(
                f"Context bị pruning: {len(context)}/{MAX_CONTEXT_CHARS} chars"
            )

        # ── Bước 6: LLM Generation (Groq Streaming) ───────────────────
        t_llm_start = time.perf_counter()
        prompt = self._build_prompt(query_text, context, pruned)
        answer = self._generate_answer(prompt)
        t_llm_ms = (time.perf_counter() - t_llm_start) * 1000

        # ── Tổng hợp metrics ───────────────────────────────────────────
        total_ms = (time.perf_counter() - t_start) * 1000

        # Lưu lịch sử (không blocking)
        try:
            self.supabase.save_chat(
                session_id=session_id,
                model_name="pq",
                question=query_text,
                answer=answer,
                latency_ms=total_ms,
            )
        except Exception:
            pass

        return {
            "answer": answer,
            "model": "rag_pq",
            "session_id": session_id,
            "metrics": {
                "total_latency_ms": round(total_ms, 2),
                "embed_latency_ms": round(t_embed_ms, 2),
                "adc_table_ms": round(t_adc_ms, 2),
                "retrieve_latency_ms": round(t_retrieve_ms, 2),
                "rerank_latency_ms": round(t_rerank_ms, 2),
                "llm_latency_ms": round(t_llm_ms, 2),
                "retrieval_count": retrieval_count,
                "rerank_count": rerank_count if pq_codes_list else top_k,
                "context_chars": len(context),
                "context_pruned": pruned,
                "pq_config": {
                    "M": pq.M,
                    "K": pq.K,
                    "compression_ratio": f"{pq.compression_ratio:.0f}x",
                },
            },
            "sources": [
                pt.payload.get("source", "unknown") for pt in reranked
            ],
        }

    def query_stream(
        self,
        query_text: str,
        top_k: int = 10,
        limit: int = 40,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        """
        Streaming version của query() — yield từng token LLM ngay khi có.
        Dùng cho SSE endpoint /chat/stream.
        """
        t_start = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())

        # Embed + ADC + Retrieval + Reranking (giống query())
        query_vec = self.embedder.embed_text(query_text)
        pq = self._get_pq()
        adc_table = pq.compute_adc_table(query_vec)

        candidates = self.qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=limit,
            payload_filter=filters,
        )

        if not candidates:
            yield "[RAG-PQ] Không tìm thấy kết quả phù hợp."
            return

        pq_codes_list = []
        valid_candidates = []
        for pt in candidates:
            codes = pt.payload.get("pq_codes")
            if codes is not None:
                pq_codes_list.append(codes)
                valid_candidates.append(pt)

        if pq_codes_list:
            codes_matrix = np.array(pq_codes_list, dtype=np.uint8)
            adc_scores = pq.adc_score_batch(adc_table, codes_matrix)
            top_indices = np.argsort(adc_scores)[::-1][:top_k]
            reranked = [valid_candidates[i] for i in top_indices]
        else:
            reranked = candidates[:top_k]

        context_parts = [
            f"[{pt.payload.get('source', '?')}]\n{pt.payload.get('text', '')}"
            for pt in reranked
            if pt.payload.get("text")
        ]
        context = "\n\n---\n\n".join(context_parts)[:MAX_CONTEXT_CHARS]

        prompt = self._build_prompt(query_text, context, False)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        full_answer = ""
        for chunk in self.llm.stream(messages):
            token = chunk.content
            if token:
                full_answer += token
                yield token

        # Lưu lịch sử sau khi stream xong
        total_ms = (time.perf_counter() - t_start) * 1000
        try:
            self.supabase.save_chat(
                session_id=session_id,
                model_name="pq",
                question=query_text,
                answer=full_answer,
                latency_ms=total_ms,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, context: str, pruned: bool) -> str:
        """Tạo prompt đầy đủ gửi cho LLM."""
        pruned_note = "\n⚠️ [Context đã bị cắt do giới hạn token]" if pruned else ""
        return (
            f"Context tài liệu:{pruned_note}\n"
            f"{'='*60}\n"
            f"{context}\n"
            f"{'='*60}\n\n"
            f"Câu hỏi: {query}\n\n"
            f"Trả lời:"
        )

    def _generate_answer(self, prompt: str) -> str:
        """Gọi Groq LLM non-streaming, trả về full answer."""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return response.content

    def _empty_response(self, query: str, t_start: float) -> Dict[str, Any]:
        """Trả về response rỗng khi không có candidates."""
        total_ms = (time.perf_counter() - t_start) * 1000
        return {
            "answer": "[RAG-PQ] Không tìm thấy tài liệu phù hợp trong database.",
            "model": "rag_pq",
            "metrics": {
                "total_latency_ms": round(total_ms, 2),
                "retrieval_count": 0,
                "rerank_count": 0,
            },
            "sources": [],
        }
