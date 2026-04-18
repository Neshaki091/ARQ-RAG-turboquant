import os
import json
import time
import logging
import numpy as np
from typing import List, Dict

logger = logging.getLogger("ChatService")

# Shared Imports
from shared.embed import EmbeddingManager
from shared.vector_store import VectorStoreManager
from shared.ragas_eval import RagasEvaluator
from shared.query_analyzer import QueryAnalyzer

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Import Model Handlers
from models.arq_rag.handler import ModelHandler as ARQRAGHandler
from models.rag_raw.handler import ModelHandler as RawHandler
from models.rag_pq.handler import ModelHandler as PQHandler
from models.rag_sq8.handler import ModelHandler as SQ8Handler
from models.rag_adaptive.handler import ModelHandler as AdaptiveHandler

class ChatService:
    def __init__(self):
        self.embed_manager = EmbeddingManager()
        self.vector_manager = VectorStoreManager()
        self.ragas_evaluator = RagasEvaluator()
        self.query_analyzer = QueryAnalyzer()
        
        # Initialize specialized Model Handlers (one per researcher/model)
        self.handlers = {
            "vector_raw": RawHandler(self),
            "vector_pq": PQHandler(self),
            "vector_sq8": SQ8Handler(self),
            "vector_adaptive": AdaptiveHandler(self),
            "vector_arq": ARQRAGHandler(self)
        }
        
    def get_llm(self, model_name: str = None):
        """Hệ thống Routing đa nền tảng cho ARQ-RAG."""
        # Ưu tiên Gemini 3.1 Flash Lite Preview làm Generator mặc định (Tận dụng 1M Context)
        if model_name is None or model_name.lower() in ["groq", "gemma-4-31b"]:
            model_name = "gemini-3.1-flash-lite-preview"
            
        # 1. Routing tới Google Generative AI (Gemini / Gemma)
        if "gemini" in model_name.lower() or "gemma" in model_name.lower():
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                max_output_tokens=2048,
                request_timeout=60
            )
        
        # 2. Routing tới Groq (Llama / Qwen)
        return ChatGroq(
            model_name=model_name,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
            max_tokens=2048,
            max_retries=3,
            request_timeout=60
        )

    def _extract_text(self, content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
                else:
                    text_parts.append(str(part))
            return "".join(text_parts)
        return str(content)

    async def chat_stream(self, query: str, model_name: str, collection_name: str):
        """Hàm điều phối Luồng Chat Modular."""
        logger.info("=" * 70)
        logger.info(f"[ChatService] NHẬN YÊU CẦU CHAT MỚI")
        logger.info(f"  Query: {query[:120]}{'...' if len(query) > 120 else ''}")
        logger.info(f"  Model: {model_name} | Collection: {collection_name}")
        
        yield json.dumps({"type": "status", "message": "🔍 Đang trích xuất đặc trưng câu hỏi..."}) + "\n"
        
        # Adaptive logic (shared across relevant models)
        limit = 40
        top_k = 15
        if collection_name in ["vector_adaptive", "vector_arq"]:
            yield json.dumps({"type": "status", "message": "🧠 Phân tích độ phức tạp (Adaptive Mode)..."}) + "\n"
            analysis = self.query_analyzer.analyze(query)
            logger.info(f"  [Adaptive] Kết quả phân tích: complexity={analysis['complexity']}, "
                         f"limit={analysis['limit']}, top_k={analysis['top_k']}")
            yield json.dumps({"type": "status", "message": f"📌 {analysis['label']}"}) + "\n"
            limit = analysis["limit"]
            top_k = analysis["top_k"]
        else:
            logger.info(f"  [Standard] Sử dụng tham số cố định: limit={limit}, top_k={top_k}")
            yield json.dumps({"type": "status", "message": "🛡️ Chế độ xử lý: STANDARD (Cố định)"}) + "\n"

        # Delegate to the specific Model Handler
        handler = self.handlers.get(collection_name)
        if not handler:
            logger.error(f"  Không tìm thấy handler cho collection: {collection_name}")
            yield json.dumps({"type": "error", "message": "Không tìm thấy mô hình tương ứng."}) + "\n"
            return

        logger.info(f"  Dispatch -> {handler.__class__.__name__} (collection={collection_name})")
        yield json.dumps({"type": "status", "message": f"⚡ Đang chạy quy trình RAG của mô hình {collection_name}..."}) + "\n"
        
        try:
            # 1. Process request via specific handler (Generation Phase) - with timeout
            import asyncio
            try:
                result = await asyncio.wait_for(handler.handle(query, model_name, limit, top_k), timeout=90)
            except asyncio.TimeoutError:
                yield json.dumps({"type": "error", "message": "⏳ Timeout: LLM không phản hồi trong 90 giây. Hãy thử lại hoặc chọn mô hình Ollama Local."}) + "\n"
                return
            
            # 2. Yield final text response immediately to the UI
            # Tích hợp đánh giá RAGAS cho Chat mode (Cần thiết cho Nghiên cứu/Luận văn)
            yield json.dumps({"type": "status", "message": f"📊 Đang chấm điểm chất lượng (RAGAS + {self.ragas_evaluator.model_name})..."}) + "\n"
            
            scores = {"faithfulness": 0.0, "answer_relevancy": 0.0, "answer_relevance": 0.0}
            if "contexts" in result and result["answer"]:
                try:
                    scores = self.ragas_evaluator.evaluate(query, result["contexts"], result["answer"])
                except Exception as eval_err:
                    logger.error(f"  [RAGAS] Lỗi chấm điểm: {eval_err}")

            final_result = {
                "type": "final",
                **result,
                "method": collection_name,
                "scores": scores
            }
            yield json.dumps(final_result) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "message": f"Lỗi Handler: {str(e)}"}) + "\n"
