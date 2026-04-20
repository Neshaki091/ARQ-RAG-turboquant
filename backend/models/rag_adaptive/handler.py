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
        
        # 2. Native Search (On 'vector_adaptive' collection)
        results, search_time_ms, load_time = self.engine.search("vector_adaptive", query_vector, top_k)

        # 3. Contexts
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
