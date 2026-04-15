import os
import time
import logging
import numpy as np
from shared.vector_store import VectorStoreManager
from langchain_core.messages import HumanMessage, SystemMessage
from shared.context_filter import filter_relevant_contexts

logger = logging.getLogger("SharedRAG")

class StandardRAGHandler:
    def __init__(self, chat_service, collection_name, label):
        self.cs = chat_service
        self.vm = VectorStoreManager()
        self.collection_name = collection_name
        self.label = label

    async def handle(self, query, model_name, limit, top_k):
        logger.info("=" * 60)
        logger.info(f"[{self.label}] BẮT ĐẦU TÌM KIẾM CHUNK (Standard Handler)")
        logger.info(f"  Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        logger.info(f"  Params: limit={limit}, top_k={top_k}, model={model_name}, "
                     f"collection={self.collection_name}")
        logger.info("-" * 60)

        # 1. Embedding query
        t0 = time.time()
        query_vector = np.array(self.cs.embed_manager.get_embedding(query))
        embed_time = time.time() - t0
        logger.info(f"  [Bước 1] Embedding query -> vector dim={query_vector.shape[0]}, "
                     f"norm={np.linalg.norm(query_vector):.4f}, thời gian={embed_time:.3f}s")

        # 2. Vector search (Qdrant)
        t1 = time.time()
        search_results = self.vm.search(self.collection_name, query_vector, limit=limit)
        search_time = time.time() - t1
        logger.info(f"  [Bước 2] Qdrant search collection='{self.collection_name}' -> "
                     f"trả về {len(search_results)} kết quả, thời gian={search_time:.3f}s")

        if search_results:
            scores = [getattr(hit, 'score', None) for hit in search_results]
            logger.info(f"    Qdrant scores (top-5): {scores[:5]}")
            logger.info(f"    Qdrant scores (min/max): min={min(s for s in scores if s is not None):.4f}, "
                         f"max={max(s for s in scores if s is not None):.4f}"
                         if any(s is not None for s in scores) else "    Qdrant scores: N/A")

        # 3. Select Contexts (Standard takes top_k directly)
        top_hits = search_results[:top_k]
        raw_contexts = [hit.payload["content"] for hit in top_hits]
        
        # [MỚI] Khử nhiễu và Tối ưu Payload
        final_contexts = filter_relevant_contexts(query, raw_contexts, top_n=3)
        
        logger.info(f"  [Bước 3] Lọc ngữ cảnh: {len(raw_contexts)} -> {len(final_contexts)} chunk chất lượng")
        for i, hit in enumerate(top_hits):
            score = getattr(hit, 'score', 'N/A')
            content_preview = hit.payload["content"][:80].replace('\n', ' ')
            logger.info(f"    #{i+1} | score={score} | file={hit.payload.get('file', 'N/A')} | "
                         f"preview: {content_preview}...")

        # 4. Generation (Payload Optimization to avoid 413 error & TPM limit)
        MAX_CONTEXT_CHARS = 24000
        context_text = "\n\n".join(final_contexts)
        
        if len(context_text) > MAX_CONTEXT_CHARS:
            logger.warning(f"  [Tối ưu] Context quá lớn ({len(context_text)} ký tự). Đang cắt tỉa về {MAX_CONTEXT_CHARS}...")
            context_text = context_text[:MAX_CONTEXT_CHARS] + "\n\n[...Nội dung bị cắt tỉa để tránh lỗi giới hạn API...]"

        system_instructions = rf"""Bạn là chuyên gia RAG. Đọc <NGỮ CẢNH> bên dưới và TRẢ LỜI CÂU HỎI.

QUY TẮC:
1. TRỰC DIỆN: Không chào hỏi, không kết luận thừa.
2. LATEX: Dùng Markdown LaTeX ($...$ hoặc $$...$$).
3. NGUỒN: Bắt đầu bằng [{self.label}].

<NGỮ CẢNH>:
{context_text}

CÂU HỎI: "{query}" """

        llm = self.cs.get_llm(model_name)
        messages = [HumanMessage(content=system_instructions)]
        
        logger.info(f"  [Bước 4] Đang gọi LLM ({model_name}) với prompt siêu tập trung...")
        start_time = time.time()
        response = llm.invoke(messages)
        answer = self.cs._extract_text(response.content)
        
        if not answer.strip().startswith(self.label):
            answer = f"{self.label} {answer}"
            
        latency = time.time() - start_time
        logger.info(f"  [Bước 4] LLM trả lời xong, latency={latency:.2f}s, "
                     f"answer_len={len(answer)} ký tự")

        total_time = time.time() - t0
        logger.info(f"[{self.label}] HOÀN THÀNH | Tổng thời gian={total_time:.2f}s")
        logger.info("=" * 60)

        return {
            "answer": answer,
            "sources": [{"file": h.payload["file"], "content": h.payload["content"]} for h in top_hits],
            "contexts": final_contexts,
            "latency": round(latency, 2)
        }
