"""
handler.py — ARQ-RAG Handler (TurboQuant)
==========================================
Pipeline nâng cao nhất: PQ + ADC + QJL reranking.

So với RAGPQHandler:
  - Thêm QJL sketch scores để bù trừ lỗi PQ
  - Lấy qjl_sketch từ payload mỗi candidate
  - Combined score = 0.7*ADC + 0.3*QJL

Collection: vector_arq (= vector_raw nhưng payload có pq_codes + qjl_sketch)
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

from models.arq_rag.reranker import ARQReranker
from models.rag_pq.quantizer import ProductQuantizer
from shared.embed import OllamaEmbedder
from shared.supabase_client import SupabaseManager
from shared.vector_store import QdrantManager, COLLECTION_NAMES

load_dotenv()
logger = logging.getLogger(__name__)

COLLECTION_NAME = COLLECTION_NAMES["arq"]  # "vector_arq"
PQ_CODEBOOK_PATH = "pq/centroids.pkl"
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "120000"))

SYSTEM_PROMPT = """[ARQ-RAG | TurboQuant ADC+QJL]
Bạn là trợ lý nghiên cứu khoa học chuyên sâu. Trả lời DỨT KHOÁT, không chào hỏi.
Sử dụng LaTeX ($...$) cho mọi công thức toán học.
Trích dẫn nguồn [source] nếu có."""


class ARQRAGHandler:
    """
    Handler TurboQuant = PQ encoding + ADC + QJL combined reranking.

    Candidate retrieval: limit=40 từ Qdrant
    Reranking: ARQReranker.compute_combined_scores() → top_k
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
        self._pq: Optional[ProductQuantizer] = None
        self.reranker = ARQReranker()

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
        top_k: int = 10,
        limit: int = 40,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())

        # Embed query (float32)
        query_vec = self.embedder.embed_text(query_text)

        # ADC table
        pq = self._get_pq()
        adc_table = pq.compute_adc_table(query_vec)

        # Retrieval
        t_retrieve = time.perf_counter()
        candidates = self.qdrant.search(COLLECTION_NAME, query_vec, limit=limit)
        retrieve_ms = (time.perf_counter() - t_retrieve) * 1000

        if not candidates:
            return self._empty_response(t_start)

        # Collect PQ codes + QJL sketches từ payload
        t_rerank = time.perf_counter()
        pq_codes_list, qjl_sketches_list, valid = [], [], []
        for pt in candidates:
            pq_codes = pt.payload.get("pq_codes")
            qjl_sketch = pt.payload.get("qjl_sketch")
            if pq_codes is not None:
                pq_codes_list.append(pq_codes)
                qjl_sketches_list.append(qjl_sketch if qjl_sketch else None)
                valid.append(pt)

        if not pq_codes_list:
            reranked = candidates[:top_k]
        else:
            codes_matrix = np.array(pq_codes_list, dtype=np.uint8)  # (N, M)

            # QJL sketches (có thể thiếu một số)
            qjl_matrix = None
            if all(s is not None for s in qjl_sketches_list):
                qjl_matrix = np.array(qjl_sketches_list, dtype=np.int8)  # (N, r)

            # TurboQuant combined scores
            combined = self.reranker.compute_combined_scores(
                query_vec=query_vec,
                adc_table=adc_table,
                pq_codes_batch=codes_matrix,
                qjl_sketches_batch=qjl_matrix,
            )

            top_idx = np.argsort(combined)[::-1][:top_k]
            reranked = [valid[i] for i in top_idx]

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
            self.supabase.save_chat(session_id, "arq", query_text, answer, total_ms)
        except Exception:
            pass

        return {
            "answer": answer,
            "model": "arq_rag",
            "session_id": session_id,
            "metrics": {
                "total_latency_ms": round(total_ms, 2),
                "retrieve_latency_ms": round(retrieve_ms, 2),
                "rerank_latency_ms": round(rerank_ms, 2),
                "llm_latency_ms": round(llm_ms, 2),
                "retrieval_count": len(candidates),
                "rerank_count": len(reranked),
                "rerank_method": f"ADC(α={self.reranker.alpha}) + QJL(r={self.reranker.r})",
            },
            "sources": [pt.payload.get("source", "?") for pt in reranked],
        }

    def query_stream(
        self,
        query_text: str,
        top_k: int = 10,
        limit: int = 40,
        session_id: Optional[str] = None,
    ) -> Iterator[str]:
        t_start = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())

        query_vec = self.embedder.embed_text(query_text)
        pq = self._get_pq()
        adc_table = pq.compute_adc_table(query_vec)
        candidates = self.qdrant.search(COLLECTION_NAME, query_vec, limit=limit)

        if not candidates:
            yield "[ARQ-RAG] Không tìm thấy kết quả."
            return

        pq_codes_list, qjl_sketches_list, valid = [], [], []
        for pt in candidates:
            pq_codes = pt.payload.get("pq_codes")
            qjl_sketch = pt.payload.get("qjl_sketch")
            if pq_codes is not None:
                pq_codes_list.append(pq_codes)
                qjl_sketches_list.append(qjl_sketch)
                valid.append(pt)

        if pq_codes_list:
            codes_matrix = np.array(pq_codes_list, dtype=np.uint8)
            qjl_matrix = (
                np.array(qjl_sketches_list, dtype=np.int8)
                if all(s is not None for s in qjl_sketches_list)
                else None
            )
            combined = self.reranker.compute_combined_scores(
                query_vec, adc_table, codes_matrix, qjl_matrix
            )
            top_idx = np.argsort(combined)[::-1][:top_k]
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
            self.supabase.save_chat(session_id, "arq", query_text, full_answer, total_ms)
        except Exception:
            pass

    def _empty_response(self, t_start: float) -> Dict[str, Any]:
        total_ms = (time.perf_counter() - t_start) * 1000
        return {
            "answer": "[ARQ-RAG] Không tìm thấy tài liệu phù hợp.",
            "model": "arq_rag",
            "metrics": {"total_latency_ms": round(total_ms, 2), "retrieval_count": 0},
            "sources": [],
        }
