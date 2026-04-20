import os
import time
import logging
import numpy as np
from shared.vector_store import VectorStoreManager
from langchain_core.messages import HumanMessage, SystemMessage
from shared.context_filter import filter_relevant_contexts

logger = logging.getLogger("RAG-RAW")

from shared.native_engine import NativeEngine

logger = logging.getLogger("RAG-RAW")

class ModelHandler:
    def __init__(self, chat_service):
        self.cs = chat_service
        self.engine = NativeEngine()

    async def handle(self, query, model_name, limit, top_k, language: str = "en"):
        logger.info("=" * 60)
        logger.info("[RAG-RAW] BẮT ĐẦU TÌM KIẾM CHUNK")
        logger.info("-" * 60)

        # 1. Embedding query
        t0 = time.time()
        query_vector = np.array(self.cs.embed_manager.get_embedding(query))
        
        # 2. Native Engine Search
        results, search_time_ms, load_time = self.engine.search("vector_raw", query_vector, top_k)
        
        if load_time:
            logger.info(f"  [Bước 2] Cache miss. Loaded RAW data in {load_time:.2f}s")
        
        # 3. Prepare contexts
        contexts = [res['payload'].get('content', "") for res in results]
        context_text = "\n\n".join(contexts)
        
        # 4. Generation
        lang_instruction = "Trả lời bằng Tiếng Việt." if language == "vi" else "Respond in English."
        system_instructions = rf"""You are a RAG expert. Read the <CONTEXT> below and answer CONCISELY.
RULES: Start your answer with [RAG-RAW]. Language: {lang_instruction}
<CONTEXT>:
{context_text}
QUESTION: "{query}" """

        llm = self.cs.get_llm(model_name)
        messages = [HumanMessage(content=system_instructions)]
        
        start_gen = time.time()
        response = llm.invoke(messages)
        answer = self.cs._extract_text(response.content)
        
        if not answer.startswith("[RAG-RAW]"):
            answer = f"[RAG-RAW] {answer}"
            
        gen_latency = time.time() - start_gen

        return {
            "answer": answer,
            "sources": [{"file": res['payload'].get("file", "unknown"), "content": res['payload'].get("content", "")} for res in results],
            "contexts": contexts,
            "latency": round(gen_latency, 2),
            "retrieval_latency_ms": round(search_time_ms, 2),
        }

