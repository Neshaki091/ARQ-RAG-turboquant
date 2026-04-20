import os
import time
import logging
import numpy as np
from shared.native_engine import NativeEngine
from langchain_core.messages import HumanMessage

logger = logging.getLogger("RAG-Adaptive")

class ModelHandler:
    def __init__(self, chat_service):
        self.cs = chat_service
        self.engine = NativeEngine()

    async def handle(self, query, model_name, limit, top_k, language: str = "en"):
        t0 = time.time()
        # 1. Embedding
        query_vector = np.array(self.cs.embed_manager.get_embedding(query))
        
        # 2. Native Search (Search Wide with 'limit' e.g. 20)
        logger.info(f"  [Adaptive] Searching wide with limit={limit}...")
        results, search_time_ms, load_time = self.engine.search("vector_raw", query_vector, limit)
        
        # 3. Focus Narrow (Filter to top_k e.g. 5)
        # Vì cùng dùng Cosine, ta chỉ cần lấy top_k kết quả đầu tiên từ kết quả search rộng
        logger.info(f"  [Adaptive] Focusing narrow from {len(results)} to top_k={top_k}...")
        results = results[:top_k]

        # 4. Contexts
        contexts = [res['payload'].get('content', "") for res in results]
        context_text = "\n\n".join(contexts)

        # 4. Generation
        lang_instruction = "Trả lời bằng Tiếng Việt." if language == "vi" else "Respond in English."
        system_instructions = rf"""You are a RAG expert. Read the <CONTEXT> below and answer CONCISELY.
RULES: Start your answer with [Adaptive-RAG]. Language: {lang_instruction}
<CONTEXT>:
{context_text}
QUESTION: "{query}" """

        llm = self.cs.get_llm(model_name)
        messages = [HumanMessage(content=system_instructions)]
        
        start_gen = time.time()
        response = llm.invoke(messages)
        answer = self.cs._extract_text(response.content)
        
        if not answer.startswith("[Adaptive-RAG]"):
            answer = f"[Adaptive-RAG] {answer}"
            
        gen_latency = time.time() - start_gen

        return {
            "answer": answer,
            "sources": [{"file": res['payload'].get("file", "unknown"), "content": res['payload'].get("content", "")} for res in results],
            "contexts": contexts,
            "latency": round(gen_latency, 2),
            "retrieval_latency_ms": round(search_time_ms, 2),
        }
