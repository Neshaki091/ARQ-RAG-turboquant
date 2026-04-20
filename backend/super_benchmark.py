import os
import json
import asyncio
import time
import logging
import psutil
from typing import List, Dict
from dotenv import load_dotenv

# Khi chạy trong Docker Backend, các file này nằm cùng thư mục /app
from chat_service import ChatService
from shared.supabase_client import SupabaseManager

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SuperBenchmark")

class KeyRotator:
    def __init__(self, keys: List[str]):
        self.keys = [k for k in keys if k]
        self.index = 0
        if not self.keys:
            raise ValueError("❌ Không tìm thấy API Key nào!")
        logger.info(f"🔑 Bộ xoay tua: {len(self.keys)} Keys.")

    def get_next_key(self):
        key = self.keys[self.index]
        self.index = (self.index + 1) % len(self.keys)
        return key

def clean_text(obj):
    """Lọc bỏ ký tự NULL (\u0000) gây lỗi Postgres"""
    if isinstance(obj, str):
        return obj.replace("\u0000", "").replace("\x00", "")
    if isinstance(obj, list):
        return [clean_text(i) for i in obj]
    if isinstance(obj, dict):
        return {k: clean_text(v) for k, v in obj.items()}
    return obj

class SuperBenchmarkRunner:
    def __init__(self):
        self.sm = SupabaseManager()
        self.cs = ChatService()
        self.progress_file = "/app/data/benchmark_progress.json"
        
        # Tạo file tiến độ nếu chưa có
        if not os.path.exists(self.progress_file):
            with open(self.progress_file, "w") as f:
                json.dump({}, f)

        raw_keys = [
            os.getenv("GOOGLE_API_KEY"),
            os.getenv("GOOGLE_API_KEY_2")
        ]
        valid_keys = [str(k) for k in raw_keys if k and isinstance(k, str)]
        self.rotator = KeyRotator(valid_keys)
        
        self.models_to_test = ["vector_raw", "vector_pq", "vector_sq8", "vector_adaptive", "vector_arq"]
        self.target_model = "gemma-4-26b-it" 

    def load_progress(self):
        try:
            with open(self.progress_file, "r") as f:
                return json.load(f)
        except:
            return {}

    def save_progress(self, model_label, question_id):
        progress = self.load_progress()
        if model_label not in progress:
            progress[model_label] = []
        if question_id not in progress[model_label]:
            progress[model_label].append(question_id)
        with open(self.progress_file, "w") as f:
            json.dump(progress, f, indent=2)

    async def run_single_test(self, model_label: str, question_data: Dict):
        query = question_data["question"]
        q_id = str(question_data.get("_id") or question_data.get("id"))
        ground_truth = question_data.get("ground_truth", "")
        topic = question_data.get("topic", "General")
        
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
                api_url = "http://127.0.0.1:8000/api/benchmark/query"
                response = await client.post(api_url, json=payload)
                response.raise_for_status()
                result = response.json()
                
                # --- LÀM SẠCH DỮ LIỆU TRƯỚC KHI LƯU ---
                safe_answer = clean_text(result["answer"])
                safe_contexts = clean_text(result["contexts"])
                
                self.sm.supabase.table("benchmarks").insert({
                    "model_name": model_label,
                    "question": clean_text(query),
                    "answer": safe_answer,
                    "contexts": safe_contexts,
                    "ground_truth": clean_text(ground_truth),
                    "latency_ms": result["latency_ms"],
                    "peak_ram_mb": result.get("peak_ram_mb", 0),
                    "topic": topic
                }).execute()
                
                # Lưu tiến độ sau khi insert thành công
                self.save_progress(model_label, q_id)
                logger.info(f"  ✅ [{model_label}] OK | Q_ID: {q_id} | RAM: {result['peak_ram_mb']}MB")
                return True
        except Exception as e:
            logger.error(f"  ❌ Lỗi API {model_label} (Q_ID: {q_id}): {e}")
            return False

    async def start_all(self, count=None, model_name=None):
        if count is None:
            count = 400
            
        logger.info(f"🎬 Bắt đầu SIÊU THỰC NGHIỆM TRONG DOCKER (Bản Resume) | Limit={count}")
        
        # Nếu người dùng chỉ định 1 model, chỉ chạy model đó
        if model_name:
            if model_name not in self.models_to_test:
                logger.error(f"❌ Model '{model_name}' không hợp lệ. Danh sách: {self.models_to_test}")
                return
            models_to_run = [model_name]
        else:
            models_to_run = self.models_to_test

        queries = self.sm.get_benchmark_queries()
        if not queries: 
            logger.error("❌ Không tải được câu hỏi từ Supabase.")
            return
        
        # Nếu có giới hạn số lượng câu hỏi
        if count:
            queries = queries[:count]
            logger.info(f"📊 Đã giới hạn thực nghiệm: Chỉ chạy {len(queries)} câu hỏi đầu tiên.")
        
        total_runs = len(models_to_run) * len(queries)
        current_run = 0
        
        for model in models_to_run:
            logger.info(f"\n🚀 Mô hình: {model.upper()}")
            progress = self.load_progress()
            completed_ids = [str(cid) for cid in progress.get(model, [])]
            
            for i, q in enumerate(queries):
                current_run += 1
                q_id = str(q.get("_id") or q.get("id"))
                
                if q_id in completed_ids:
                    logger.info(f"  [SKIPPED] {model} | Câu {i+1} (ID: {q_id}) đã hoàn thành.")
                    continue

                logger.info(f"  [{current_run}/{total_runs}] {model} | Câu {i+1} (ID: {q_id})")
                await self.run_single_test(model, q)
                await asyncio.sleep(1.5)
        logger.info("\n🎉 HOÀN TẤT!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ARQ-RAG Super Benchmark Runner")
    parser.add_argument("--count", "-n", type=int, default=None, help="Số lượng câu hỏi muốn chạy (mặc định: tất cả)")
    parser.add_argument("--model", "-m", type=str, default=None, help="Tên mô hình muốn chạy (vd: vector_arq)")
    args = parser.parse_args()

    runner = SuperBenchmarkRunner()
    asyncio.run(runner.start_all(count=args.count, model_name=args.model))
