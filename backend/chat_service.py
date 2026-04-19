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
from shared.query_analyzer import QueryAnalyzer
import psutil

# LangChain Imports
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
        self.query_analyzer = QueryAnalyzer()
        self.process = psutil.Process() # Theo dõi tiến trình hiện tại
        
        # Initialize specialized Model Handlers (one per researcher/model)
        self.handlers = {
            "vector_raw": RawHandler(self),
            "vector_pq": PQHandler(self),
            "vector_sq8": SQ8Handler(self),
            "vector_adaptive": AdaptiveHandler(self),
            "vector_arq": ARQRAGHandler(self)
        }
        
    def get_llm(self, model_name: str = None):
        """Routing LLM cho sinh câu trả lời (Generation).
        
        Phân tách API Key theo mục đích:
        - GOOGLE_API_KEY_2: Dành cho generation (endpoint này)
        - GOOGLE_API_KEY  : Dành cho evaluation/scoring (benchmark.py, RAGAS)
        """
        # Mặc định: dùng gemini-3.1-flash-lite-preview với key 2
        GENERATION_MODEL    = "gemini-3.1-flash-lite-preview"
        GENERATION_API_KEY  = os.getenv("GOOGLE_API_KEY_2")

        # Tất cả generation đều dùng Gemini flash-lite (GOOGLE_API_KEY_2)
        # Groq chỉ dùng trong query_analyzer.py cho phân loại query, không dùng ở đây
        return ChatGoogleGenerativeAI(
            model=GENERATION_MODEL,
            google_api_key=GENERATION_API_KEY,
            temperature=0,
            max_output_tokens=512,
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
        
        # 1. Unified Query Analysis (Chạy cho tất cả các mô hình)
        yield json.dumps({"type": "status", "message": "🧠 Phân tích độ phức tạp câu hỏi..."}) + "\n"
        analysis = self.query_analyzer.analyze(query)
        
        limit = analysis["limit"]
        top_k = analysis["top_k"]
        complexity = analysis["complexity"]
        language = analysis["language"]   # "vi" hoặc "en"
        
        # 2. Phân tách tham số: Baseline (Brute-force) vs Research (Optimized)
        is_baseline = collection_name in ["vector_raw", "vector_pq", "vector_sq8"]
        
        if is_baseline:
            # Đối với Baseline: Dùng toàn bộ những gì tìm được để làm Upper Bound chính xác nhất
            top_k = limit
            logger.info(f"  [Baseline Mode] {collection_name} | complexity={complexity} | lang={language} | limit=top_k={limit}")
            yield json.dumps({"type": "status", "message": f"🛡️ Chế độ: BASELINE ({complexity}) | Full Context={limit}"}) + "\n"
        else:
            # Đối với ARQ/Adaptive: Dùng cơ chế lọc tinh túy (Efficiency)
            logger.info(f"  [Research Mode] {collection_name} | complexity={complexity} | lang={language} | limit={limit}, top_k={top_k}")
            yield json.dumps({"type": "status", "message": f"⚡ Chế độ: {collection_name.upper()} ({complexity}) | Search={limit}, Focus={top_k}"}) + "\n"

        # Delegate to the specific Model Handler
        handler = self.handlers.get(collection_name)
        if not handler:
            logger.error(f"  Không tìm thấy handler cho collection: {collection_name}")
            yield json.dumps({"type": "error", "message": "Không tìm thấy mô hình tương ứng."}) + "\n"
            return

        logger.info(f"  Dispatch -> {handler.__class__.__name__} (collection={collection_name})")
        yield json.dumps({"type": "status", "message": f"⚡ Đang chạy quy trình RAG của mô hình {collection_name}..."}) + "\n"
        
        try:
            # 1. Bắt đầu đo hiệu năng
            start_time = time.time()
            start_mem = self.process.memory_info().rss / (1024 * 1024) # MB

            # 2. Xử lý yêu cầu qua handler (truyền thêm language)
            import asyncio
            try:
                result = await asyncio.wait_for(handler.handle(query, model_name, limit, top_k, language=language), timeout=120)
            except asyncio.TimeoutError:
                yield json.dumps({"type": "error", "message": "⏳ Timeout: LLM không phản hồi trong 120 giây."}) + "\n"
                return
            
            # 3. Kết thúc đo hiệu năng
            end_time = time.time()
            end_mem = self.process.memory_info().rss / (1024 * 1024) # MB
            
            latency_ms = int((end_time - start_time) * 1000)
            peak_ram_mb = round(max(0, end_mem - start_mem), 2)
            
            logger.info(f"  [Performance] Latency: {latency_ms}ms | RAM Info: {round(end_mem, 2)}MB")

            final_result = {
                "type": "final",
                **result,
                "method": collection_name,
                "metrics": {
                    "latency_ms": latency_ms,
                    "peak_ram_mb": peak_ram_mb,
                    "total_ram_mb": round(end_mem, 2)
                }
            }
            # Yield final chunk — frontend và benchmark pipeline đều dùng type "final"
            yield json.dumps(final_result) + "\n"

        except Exception as e:
            logger.error(f"  [ChatService] Lỗi Handler: {str(e)}")
            yield json.dumps({"type": "error", "message": f"Lỗi Handler: {str(e)}"}) + "\n"
