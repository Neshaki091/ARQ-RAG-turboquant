import os
import time
import logging
import numpy as np
import pickle
from .quantization import TurboQuantProd
from shared.vector_store import VectorStoreManager
from langchain_core.messages import HumanMessage, SystemMessage
from shared.context_filter import filter_relevant_contexts

logger = logging.getLogger("ARQ-RAG")

class ModelHandler:
    def __init__(self, chat_service):
        self.cs = chat_service
        self.vm = VectorStoreManager()
        # Initializing TurboQuant directly in the handler for "ownership"
        self.tq = TurboQuantProd(d=768, b=4)
        
        # 1. Load Global Weights (New Unified Method)
        weights_path = "backend/data/model_weights.pkl"
        if os.path.exists(weights_path):
            try:
                with open(weights_path, "rb") as f:
                    weights = pickle.load(f)
                
                # Load ARQ weights
                arq_weights = weights.get("arq", {})
                if arq_weights:
                    self.tq.tq_mse.Pi = arq_weights.get("Pi", self.tq.tq_mse.Pi)
                    self.tq.S = arq_weights.get("S", self.tq.S)
                    self.tq.tq_mse.centroids = arq_weights.get("centroids", self.tq.tq_mse.centroids)
                    self.tq.alpha = arq_weights.get("alpha", self.tq.alpha)
                    logger.info(f"[ARQ-RAG] Đã load GLOBAL weights từ {weights_path}")
            except Exception as e:
                logger.error(f"[ARQ-RAG] Lỗi khi load global weights: {e}")
        
        # 2. Legacy: Load centroids if available (Fallthrough)
        elif os.path.exists("backend/data/centroids.npy"):
            centroids_path = "backend/data/centroids.npy"
            self.tq.tq_mse.centroids = np.load(centroids_path)
            logger.info(f"[ARQ-RAG] Đã load legacy centroids từ {centroids_path}")

    async def handle(self, query, model_name, limit, top_k):
        logger.info("=" * 60)
        logger.info("[ARQ-RAG] BẮT ĐẦU TÌM KIẾM CHUNK (TurboQuant + ADC Reranking)")
        logger.info(f"  Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        logger.info(f"  Params: limit={limit}, top_k={top_k}, model={model_name}")
        logger.info(f"  TurboQuant Config: d={self.tq.d}, b={self.tq.b}, "
                     f"num_centroids={self.tq.tq_mse.num_centroids}")
        logger.info("-" * 60)

        # 1. Embedding query
        t0 = time.time()
        query_vector = np.array(self.cs.embed_manager.get_embedding(query))
        embed_time = time.time() - t0
        logger.info(f"  [Bước 1] Embedding query -> vector dim={query_vector.shape[0]}, "
                     f"norm={np.linalg.norm(query_vector):.4f}, thời gian={embed_time:.3f}s")

        # 2. Vector search (Qdrant - initial candidate retrieval)
        t1 = time.time()
        search_results = self.vm.search("vector_arq", query_vector, limit=limit)
        search_time = time.time() - t1
        logger.info(f"  [Bước 2] Qdrant search collection='vector_arq' -> "
                     f"trả về {len(search_results)} candidates, thời gian={search_time:.3f}s")

        if search_results:
            qdrant_scores = [getattr(hit, 'score', None) for hit in search_results]
            logger.info(f"    Qdrant scores (top-5): {qdrant_scores[:5]}")
            logger.info(f"    Qdrant scores (min/max): min={min(s for s in qdrant_scores if s is not None):.4f}, "
                         f"max={max(s for s in qdrant_scores if s is not None):.4f}"
                         if any(s is not None for s in qdrant_scores) else "    Qdrant scores: N/A")

        # 3. ARQ Reranking (ADC Direct Scoring)
        t2 = time.time()
        idx_batch = np.array([hit.payload["idx"] for hit in search_results])
        qjl_batch = np.array([hit.payload["qjl"] for hit in search_results])
        gamma_batch = np.array([hit.payload["gamma"] for hit in search_results])
        orig_norms = np.array([hit.payload.get("orig_norm", 1.0) for hit in search_results])

        logger.info(f"  [Bước 3] ADC Reranking: idx_batch shape={idx_batch.shape}, "
                     f"qjl_batch shape={qjl_batch.shape}, "
                     f"gamma range=[{gamma_batch.min():.4f}, {gamma_batch.max():.4f}]")

        scores = self.tq.compute_score_batch(query_vector, idx_batch, qjl_batch, gamma_batch, orig_norms=orig_norms)
        rerank_time = time.time() - t2

        logger.info(f"    ADC scores (top-5): {sorted(scores, reverse=True)[:5]}")
        logger.info(f"    ADC scores (min/max): min={scores.min():.4f}, max={scores.max():.4f}")
        logger.info(f"    ADC reranking thời gian={rerank_time:.4f}s")

        refined_results = []
        for i, score in enumerate(scores):
            refined_results.append((score, search_results[i]))
            
        refined_results.sort(key=lambda x: x[0], reverse=True)
        top_hits = [x[1] for x in refined_results[:top_k]]
        top_scores = [x[0] for x in refined_results[:top_k]]
        
        raw_contexts = [hit.payload["content"] for hit in top_hits]
        
        # [MỚI] Khử nhiễu ngữ cảnh - Lấy Top theo dynamic top_k
        final_contexts = filter_relevant_contexts(query, raw_contexts, top_n=top_k)
        
        logger.info(f"  [Bước 3] Lọc ngữ cảnh: {len(raw_contexts)} -> {len(final_contexts)} chunk chất lượng cao")

        logger.info(f"  [Bước 3] Sau rerank: chọn top_k={top_k} chunk")
        for i, (hit, adc_score) in enumerate(zip(top_hits, top_scores)):
            qdrant_score = getattr(hit, 'score', 'N/A')
            content_preview = hit.payload["content"][:80].replace('\n', ' ')
            logger.info(f"    #{i+1} | ADC={adc_score:.4f} | Qdrant={qdrant_score} | "
                         f"file={hit.payload.get('file', 'N/A')} | preview: {content_preview}...")

        # 4. Generation (Sử dụng Gemini 3.1 Flash Lite - Hỗ trợ Long Context)
        MAX_CONTEXT_CHARS = 120000 
        context_text = "\n\n".join(final_contexts)
        
        if len(context_text) > MAX_CONTEXT_CHARS:
            logger.warning(f"  [Tối ưu] Context quá lớn ({len(context_text)} ký tự). Đang cắt tỉa...")
            context_text = context_text[:MAX_CONTEXT_CHARS] + "\n\n[...Cắt tỉa...]"

        system_instructions = rf"""Bạn là một chuyên gia RAG. Đọc <NGỮ CẢNH> bên dưới và TRẢ LỜI CÂU HỎI một cách NGẮN GỌN, TRỰC TIẾP.

QUY TẮC:
1. NGẮN GỌN: Chỉ trả lời ý chính, không chào hỏi, không kết luận rườm rà.
2. LATEX: Dùng Markdown LaTeX ($...$ hoặc $$...$$).
3. NGUỒN: Bắt đầu câu trả lời bằng [ARQ-RAG].

<NGỮ CẢNH>:
{context_text}

CÂU HỎI: "{query}" """

        llm = self.cs.get_llm(model_name)
        messages = [HumanMessage(content=system_instructions)]
        
        logger.info(f"  [Bước 4] Đang gọi LLM ({model_name}) với prompt siêu tập trung...")
        start_time = time.time()
        response = llm.invoke(messages)
        answer = self.cs._extract_text(response.content)
        
        if not answer.startswith("[ARQ-RAG]"):
            answer = f"[ARQ-RAG] {answer}"
            
        latency = time.time() - start_time
        logger.info(f"  [Bước 4] LLM trả lời xong, latency={latency:.2f}s, ")

        total_time = time.time() - t0
        logger.info(f"[ARQ-RAG] HOÀN THÀNH | Tổng thời gian={total_time:.2f}s")
        logger.info("=" * 60)

        return {
            "answer": answer,
            "sources": [{"file": h.payload["file"], "content": h.payload["content"]} for h in top_hits],
            "contexts": final_contexts, # Trả về context để ChatService có thể chấm điểm Ragas sau
            "latency": round(latency, 2)
        }
