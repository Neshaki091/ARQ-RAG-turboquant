import os
import time
import logging
import numpy as np
from shared.native_engine import NativeEngine
from langchain_core.messages import HumanMessage

logger = logging.getLogger("ARQ-RAG")

class ModelHandler:
    def __init__(self, chat_service):
        self.cs = chat_service
        self.engine = NativeEngine()

    async def handle(self, query: str, model_name: str, limit: int = 50, top_k: int = 10, language: str = "en"):
        t0 = time.time()
        # 1. Embedding
        query_vector = np.array(self.cs.embed_manager.get_embedding(query))
        
        # 2. Native Engine Retrieval (ARQ Scoring in RAM)
        results, search_time_ms, load_time = self.engine.search("vector_arq", query_vector, top_k)
        
        if load_time:
            logger.info(f"  [ARQ-RAG] Cache miss. Loaded ARQ data from Cloud in {load_time:.2f}s")
            
        # 3. Contexts
        contexts = [res['payload'].get('content', "") for res in results]
        context_text = "\n\n".join(contexts)

        # 4. Generation
        lang_instruction = "Trả lời bằng Tiếng Việt." if language == "vi" else "Respond in English."
        system_instructions = rf"""You are a RAG expert. Answer CONCISELY.
RULES: Start with [ARQ-RAG]. {lang_instruction}
<CONTEXT>: {context_text}
QUESTION: {query}"""

        llm = self.cs.get_llm(model_name)
        start_gen = time.time()
        response = llm.invoke([HumanMessage(content=system_instructions)])
        answer = self.cs._extract_text(response.content)
        if not answer.startswith("[ARQ-RAG]"):
            answer = f"[ARQ-RAG] {answer}"

        return {
            "answer": answer,
            "sources": [{"file": res['payload'].get("file", "unknown"), "content": res['payload'].get("content", "")} for res in results],
            "contexts": contexts,
            "latency": round(time.time() - start_gen, 2),
            "retrieval_latency_ms": round(search_time_ms, 2),
        }
