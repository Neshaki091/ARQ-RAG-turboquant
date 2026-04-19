import os
import json
import logging
import time
from typing import List, Dict
from dotenv import load_dotenv
from google import genai
from google.genai import types
from shared.supabase_client import SupabaseManager

load_dotenv()

# Cấu hình Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BatchEval")

class BatchEvaluator:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.sm = SupabaseManager()
        self.model_name = "gemini-1.5-flash"

    def _build_prompt(self, cases: List[Dict]) -> str:
        prompt = """Bạn là một chuyên gia đánh giá hệ thống RAG. Nhiệm vụ của bạn là chấm điểm các cặp (Câu hỏi, Ngữ cảnh, Câu trả lời) dựa trên tiêu chí:
1. Faithfulness (0.0 - 1.0): Câu trả lời có trung thực và bám sát ngữ cảnh không?
2. Answer Relevancy (0.0 - 1.0): Câu trả lời có đúng trọng tâm câu hỏi không?

Dữ liệu đầu vào là danh sách các Case. Hãy trả về kết quả dưới dạng JSON ARRAY.
Mỗi phần tử trong Array phải có định dạng: 
{"case_index": int, "faithfulness": float, "answer_relevancy": float, "reasoning": "giải thích ngắn bằng tiếng Việt"}

DANH SÁCH CÁC CASE CẦN CHẤM:
"""
        for i, case in enumerate(cases):
            prompt += f"\n--- CASE {i} ---\n"
            prompt += f"CÂU HỎI: {case['question']}\n"
            prompt += f"NGỮ CẢNH: {case['contexts']}\n"
            prompt += f"CÂU TRẢ LỜI: {case['answer']}\n"
        
        return prompt

    def evaluate_batch(self, cases: List[Dict]) -> List[Dict]:
        """Gửi 1 lô câu hỏi sang Gemini và nhận kết quả JSON"""
        if not cases:
            return []

        prompt = self._build_prompt(cases)
        
        try:
            logger.info(f"🚀 Đang gửi Batch ({len(cases)} cases) sang {self.model_name}...")
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0
                )
            )
            
            # Parse kết quả JSON
            results = json.loads(response.text)
            
            # Nếu kết quả là một object chứa list, hãy bóc tách
            if isinstance(results, dict) and "results" in results:
                results = results["results"]
            
            logger.info(f"✅ Đã nhận kết quả cho {len(results)} cases.")
            return results
        except Exception as e:
            logger.error(f"❌ Lỗi khi chấm điểm Batch: {e}")
            return []

    def run_benchmark_eval(self, limit=500, batch_size=25):
        """Quét Supabase và chấm điểm cho các dòng chưa có điểm"""
        # 1. Lấy dữ liệu chưa có điểm từ Supabase
        # Giả định bảng là 'benchmarks' và query lọc các dòng có faithfulness IS NULL
        try:
            # Sửa từ self.sm.client thành self.sm.supabase
            query = self.sm.supabase.table("benchmarks").select("*").is_("faithfulness", "null").limit(limit)
            records = query.execute().data
            
            if not records:
                logger.info("🎉 Không còn dòng nào cần chấm điểm.")
                return

            logger.info(f"📊 Tìm thấy {len(records)} dòng cần đánh giá. Bắt đầu chia Batch...")

            # 2. Chia Batch và chấm điểm
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                cases = []
                for r in batch:
                    # Đảm bảo lấy đúng tên cột (query/question, context)
                    q_text = r.get("query") or r.get("question")
                    c_text = r.get("context") or ""
                    a_text = r.get("answer") or ""
                    
                    cases.append({
                        "question": q_text,
                        "contexts": str(c_text)[:8000], # Tối ưu context
                        "answer": a_text
                    })
                
                scores = self.evaluate_batch(cases)
                
                # 3. Cập nhật kết quả vào Supabase
                if scores:
                    for j, score in enumerate(scores):
                        if j < len(batch):
                            record_id = batch[j]["id"]
                            self.sm.supabase.table("benchmarks").update({
                                "faithfulness": score.get("faithfulness", 0.0),
                                "answer_relevance": score.get("answer_relevancy", 0.0),
                            }).eq("id", record_id).execute()
                
                logger.info(f"✨ Đã hoàn thành và cập nhật Batch {i//batch_size + 1}")
                time.sleep(1) # Nghỉ ngắn

        except Exception as e:
            logger.error(f"❌ Lỗi trong quy trình Benchmark: {e}")

if __name__ == "__main__":
    evaluator = BatchEvaluator()
    # evaluator.run_benchmark_eval(limit=50, batch_size=10)
    print("Sẵn sàng chạy Batch Evaluator. Hãy gọi run_benchmark_eval() để bắt đầu.")
