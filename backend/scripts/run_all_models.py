import os
import json
import asyncio
import time
import logging
import psutil
from typing import List, Dict
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Đảm bảo import được các module từ backend
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from chat_service import ChatService
from shared.supabase_client import SupabaseManager

load_dotenv()

# Cấu hình logging chuyên dụng cho Benchmark
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SuperBenchmark")

class KeyRotator:
    """Xoay tua nhiều API Key để vượt giới hạn RPD và TPM"""
    def __init__(self, keys: List[str]):
        self.keys = [k for k in keys if k]
        self.index = 0
        if not self.keys:
            raise ValueError("❌ Không tìm thấy API Key nào trong cấu hình!")
        logger.info(f"🔑 Đã khởi tạo bộ xoay tua với {len(self.keys)} API Keys.")

    def get_next_key(self):
        key = self.keys[self.index]
        self.index = (self.index + 1) % len(self.keys)
        return key

class SuperBenchmarkRunner:
    def __init__(self):
        self.sm = SupabaseManager()
        self.cs = ChatService()
        self.process = psutil.Process()
        
        # Lấy các Keys từ môi trường và lọc bỏ các giá trị None/rỗng
        raw_keys = [
            os.getenv("GOOGLE_API_KEY"),
            os.getenv("GOOGLE_API_KEY_2")
        ]
        valid_keys = [str(k) for k in raw_keys if k and isinstance(k, str)]
        self.rotator = KeyRotator(valid_keys)
        
        # Danh sách mô hình thực nghiệm (Collection names)
        self.models_to_test = [
            "vector_raw",
            "vector_pq",
            "vector_sq8",
            "vector_adaptive",
            "vector_arq"
        ]
        
        # Sử dụng mô hình Gemma 4 26B theo yêu cầu của Researcher
        self.target_model = "gemma-4-26b-it" 

    async def run_single_test(self, model_label: str, question_data: Dict):
        """Gửi yêu cầu thực nghiệm tới Backend Docker via API"""
        query = question_data["question"]
        ground_truth = question_data.get("ground_truth", "")
        topic = question_data.get("topic", "General")
        question_id = question_data.get("id")
        
        # 1. Chọn Key xoay tua cho lượt này
        current_key = self.rotator.get_next_key()
        
        payload = {
            "query": query,
            "model": self.target_model,
            "collection": model_label,
            "google_api_key": current_key
        }
        
        try:
            import httpx
            async with httpx.AsyncClient(timeout=120.0) as client:
                # 2. Gọi API của Docker Backend
                api_url = "http://localhost:8000/api/benchmark/query"
                response = await client.post(api_url, json=payload)
                response.raise_for_status()
                result = response.json()
                
                # 3. Lưu kết quả vào bảng 'benchmarks'
                self.sm.supabase.table("benchmarks").insert({
                    "model_name": model_label,
                    "question": query,
                    "answer": result["answer"],
                    "contexts": result["contexts"],
                    "ground_truth": ground_truth,
                    "latency_ms": result["latency_ms"],
                    "peak_ram_mb": result["peak_ram_mb"],
                    "total_ram_mb": result["total_ram_mb"],
                    "cpu_percent": result.get("cpu_percent", 0),
                    "topic": topic
                }).execute()
                
                logger.info(f"  ✅ [{model_label}] OK | RAM: {result['peak_ram_mb']}MB | Latency: {result['latency_ms']}ms")
                return True

        except Exception as e:
            logger.error(f"  ❌ Lỗi khi gọi API cho mô hình {model_label}: {e}")
            return False

    async def start_all(self):
        """Khởi động toàn bộ cuộc thi"""
        logger.info("🎬 Bắt đầu SIÊU THỰC NGHIỆM ARQ-RAG")
        
        # 1. Lấy đề thi (483 câu)
        queries = self.sm.get_benchmark_queries()
        if not queries:
            logger.error("❌ Không tìm thấy câu hỏi nào trong bảng benchmark_queries!")
            return
            
        logger.info(f"📊 Đã tải {len(queries)} câu hỏi chuẩn từ Database.")
        
        total_runs = len(self.models_to_test) * len(queries)
        current_run = 0
        
        # 2. Vòng lặp Mô hình -> Câu hỏi
        for model in self.models_to_test:
            logger.info(f"\n🚀 Đang chạy mô hình: {model.upper()}")
            
            for i, q in enumerate(queries):
                current_run += 1
                progress = (current_run / total_runs) * 100
                
                logger.info(f"  [{progress:.1f}%] {model} | Câu {i+1}/{len(queries)}: {q['question'][:50]}...")
                
                success = await self.run_single_test(model, q)
                
                # Giãn cách nhẹ để bảo vệ API và RAM
                await asyncio.sleep(1.5 if len(self.rotator.keys) > 1 else 3)
                
        logger.info("\n🎉 SIÊU THỰC NGHIỆM HOÀN TẤT! Toàn bộ kết quả đã nằm trong bảng 'benchmarks'.")

if __name__ == "__main__":
    runner = SuperBenchmarkRunner()
    asyncio.run(runner.start_all())
