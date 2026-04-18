import os
import time
import psutil
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from shared.query_analyzer import QueryAnalyzer
from shared.supabase_client import SupabaseManager

logger = logging.getLogger("Benchmark")

class BenchmarkManager:
    def __init__(self, embeddings, chunks):
        self.embeddings = embeddings
        self.chunks = chunks
        self.dimension = embeddings.shape[1] if embeddings is not None else 768
        self.query_analyzer = QueryAnalyzer()
        self.results_dir = "results"
        self.cumulative_file = os.path.join(self.results_dir, "cumulative_results.xlsx")
        os.makedirs(self.results_dir, exist_ok=True)

    def get_current_ram(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def load_queries(self, file_path="backend/data/benchmark_queries.json"):
        """Nạp danh sách câu hỏi từ Supabase Database (Ưu tiên), hoặc từ file JSON cục bộ."""
        # 1. Thử lấy từ Supabase trước
        try:
            sm = SupabaseManager()
            db_queries = sm.get_benchmark_queries()
            if db_queries and len(db_queries) > 0:
                print(f"📦 Đã tải {len(db_queries)} câu hỏi Ground Truth từ Supabase Database!")
                return db_queries
        except Exception as e:
            print(f"Không nạp được từ Supabase, chuyển sang đọc file local: {e}")

        # 2. Đọc file local nếu Supabase lỗi hoặc trống
        if os.path.exists(file_path):
            print(f"📂 Đang nạp từ file local: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Chuyển đổi về dạng chuẩn: [{"question": "...", "ground_truth": "..."}]
                queries = []
                for item in data:
                    if isinstance(item, str):
                        queries.append({"question": item, "ground_truth": None})
                    elif isinstance(item, dict):
                        queries.append({
                            "question": item.get("question", ""),
                            "ground_truth": item.get("ground_truth")
                        })
                return queries
        
        # Mặc định
        return [{"question": "What are the key findings?", "ground_truth": None}] * 10

    async def run_batch(self, chat_service, start_idx=0, end_idx=10, model_targets=None):
        """
        Chạy benchmark theo đợt (Batch).
        Gọi trực tiếp ModelHandlers từ ChatService để đảm bảo tính nhất quán.
        """
        if model_targets is None:
            model_targets = ["vector_raw", "vector_adaptive", "vector_pq", "vector_sq8", "vector_arq"]
        
        all_queries = self.load_queries()
        test_queries = all_queries[start_idx:end_idx]
        
        batch_results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = os.path.join(self.results_dir, f"batch_{start_idx}_{end_idx}_{timestamp}.xlsx")

        print(f"🚀 Bắt đầu Benchmark Batch: {start_idx} -> {end_idx}")

        for collection_name in model_targets:
            handler = chat_service.handlers.get(collection_name)
            if not handler: continue
            
            print(f"📊 Đang chạy Model: {collection_name}...")
            
            for i, item in enumerate(test_queries):
                actual_idx = start_idx + i
                query_text = item["question"]
                ground_truth = item["ground_truth"]
                
                # Giả định tham số mặc định cho benchmark
                limit = 40
                top_k = 15
                
                # Logic Adaptive cho các model có hỗ trợ
                if collection_name in ["vector_adaptive", "vector_arq"]:
                    analysis = self.query_analyzer.analyze(query_text)
                    limit = analysis["limit"]
                    top_k = analysis["top_k"]

                start_ram = self.get_current_ram()
                
                # Thực hiện gọi Handler
                res = await handler.handle(query_text, model_name="groq", limit=limit, top_k=top_k)
                
                end_ram = self.get_current_ram()

                # Call RAGAS Evaluation với Ground Truth (nếu có)
                logger.info(f"   [RAGAS] Đang chấm điểm nội bộ cho Q_{actual_idx} ...")
                scores = {m: 0.0 for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_similarity", "answer_correctness"]}
                if "contexts" in res and res["answer"]:
                    scores = chat_service.ragas_evaluator.evaluate(query_text, res["contexts"], res["answer"], ground_truth=ground_truth)
                
                # Ghi nhận kết quả
                entry = {
                    "Model": collection_name,
                    "QueryID": actual_idx,
                    "Query": query_text,
                    "Latency": res["latency"],
                    "RAM_Usage": round(end_ram - start_ram, 2),
                    "Faithfulness": scores.get("faithfulness", 0),
                    "Answer_Relevance": scores.get("answer_relevance", 0) or scores.get("answer_relevancy", 0),
                    "Context_Precision": scores.get("context_precision", 0),
                    "Context_Recall": scores.get("context_recall", 0),
                    "Answer_Similarity": scores.get("answer_similarity", 0),
                    "Answer_Correctness": scores.get("answer_correctness", 0),
                    "Timestamp": datetime.now().isoformat()
                }
                batch_results.append(entry)
                print(f"   [OK] Q_{actual_idx} | Latency: {res['latency']}s | Scores: {scores}")
                
                # [MỚI] Delay 30s giữa các câu hỏi để tuân thủ 15 RPM của Google/Groq
                if i < len(test_queries) - 1:
                    print(f"   ⏳ Đang chờ 30 giây trước câu hỏi tiếp theo (RPM Safety Mode)...")
                    import asyncio
                    await asyncio.sleep(30)

        # 1. Lưu file Batch riêng
        df_batch = pd.DataFrame(batch_results)
        df_batch.to_excel(batch_file, index=False)
        
        # 2. Gộp vào file Cumulative (Tích lũy)
        if os.path.exists(self.cumulative_file):
            df_old = pd.read_excel(self.cumulative_file)
            df_combined = pd.concat([df_old, df_batch], ignore_index=True)
            df_combined.to_excel(self.cumulative_file, index=False)
        else:
            df_batch.to_excel(self.cumulative_file, index=False)

        return batch_file, self.cumulative_file
if __name__ == "__main__":
    import asyncio
    from chat_service import ChatService
    
    async def main():
        # Setup paths
        data_dir = "backend/data"
        chunks_path = os.path.join(data_dir, "chunks.json")
        embeddings_path = os.path.join(data_dir, "embeddings.npy")
        
        if not os.path.exists(chunks_path) or not os.path.exists(embeddings_path):
            print("Error: Missing chunks.json or embeddings.npy in backend/data")
            return
            
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        embeddings = np.load(embeddings_path)
        
        cs = ChatService()
        bm = BenchmarkManager(embeddings, chunks)
        
        # Run a small validation batch (first 3 questions)
        batch_file, _ = await bm.run_batch(cs, start_idx=0, end_idx=3)
        print(f"Validation Benchmark Completed. Results: {batch_file}")

    asyncio.run(main())
