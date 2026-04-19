import os
import json
import logging
import time
import asyncio
from typing import List, Dict
from dotenv import load_dotenv

# DeepEval & TruLens Imports
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.gemini_model import GeminiModel

from shared.supabase_client import SupabaseManager

load_dotenv()

# Cấu hình Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdvancedBatchEval")

class KeyRotator:
    def __init__(self, keys: List[str]):
        self.keys = [k for k in keys if k]
        self.index = 0
    def get_next_key(self):
        if not self.keys: return None
        key = self.keys[self.index]
        self.index = (self.index + 1) % len(self.keys)
        return key

class BatchEvaluator:
    def __init__(self):
        # Lấy các Keys từ môi trường và lọc bỏ các giá trị None/rỗng
        raw_keys = [os.getenv("GOOGLE_API_KEY"), os.getenv("GOOGLE_API_KEY_2")]
        valid_keys = [str(k) for k in raw_keys if k and isinstance(k, str)]
        
        self.rotator = KeyRotator(valid_keys)
        self.sm = SupabaseManager()
        # Nâng cấp lên Gemini 3.1 Flash Lite làm Judge chuẩn
        self.model_name = "gemini-3.1-flash-lite-preview"
        
        # Khởi tạo Judge LLM cho DeepEval
        os.environ["GOOGLE_API_KEY"] = self.rotator.get_next_key() # Key khởi tạo
        self.judge_llm = GeminiModel(model_name=self.model_name)
        
        # Thiết lập các Metrics
        self.faithfulness_metric = FaithfulnessMetric(threshold=0.5, model=self.judge_llm)
        self.relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=self.judge_llm)

    def evaluate_single_case(self, question: str, answer: str, contexts: List[str]) -> Dict:
        """Chấm điểm 1 câu duy nhất bằng DeepEval Metrics chuyên sâu"""
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=contexts
        )
        
        # Xoay tua Key trước mỗi lượt chấm để tăng độ ổn định
        os.environ["GOOGLE_API_KEY"] = self.rotator.get_next_key()
        
        try:
            self.faithfulness_metric.measure(test_case)
            self.relevancy_metric.measure(test_case)
            
            return {
                "faithfulness": self.faithfulness_metric.score,
                "answer_relevance": self.relevancy_metric.score,
                "reasoning": f"F: {self.faithfulness_metric.reason} | R: {self.relevancy_metric.reason}"
            }
        except Exception as e:
            logger.error(f"❌ Lỗi chấm điểm DeepEval: {e}")
            return {"faithfulness": 0.0, "answer_relevance": 0.0, "reasoning": str(e)}

    def run_benchmark_eval(self, limit=100):
        """Quét bảng 'benchmarks' và chấm điểm cho các dòng còn thiếu"""
        try:
            # 1. Lấy dữ liệu chưa có điểm
            query = self.sm.supabase.table("benchmarks").select("*").is_("faithfulness", "null").limit(limit)
            records = query.execute().data
            
            if not records:
                logger.info("🎉 Không còn dữ liệu nào cần chấm điểm.")
                return

            logger.info(f"📊 Đang bắt đầu chấm điểm Nâng cao (DeepEval) cho {len(records)} dòng...")

            for i, r in enumerate(records):
                # Chuẩn bị dữ liệu
                q_text = r.get("question") or ""
                a_text = r.get("answer") or ""
                c_list = r.get("contexts") or []
                
                logger.info(f"  [{i+1}/{len(records)}] Đang chấm: {q_text[:30]}...")
                
                # Chấm điểm
                scores = self.evaluate_single_case(q_text, a_text, c_list)
                
                # Cập nhật kết quả vào Supabase
                self.sm.supabase.table("benchmarks").update({
                    "faithfulness": scores["faithfulness"],
                    "answer_relevance": scores["answer_relevance"]
                }).eq("id", r["id"]).execute()
                
                # Sleep nhẹ để tránh rate limit
                time.sleep(1)

            logger.info(f"✅ Đã hoàn thành chấm điểm DeepEval cho {len(records)} dòng.")

        except Exception as e:
            logger.error(f"❌ Lỗi trong quy trình Benchmark: {e}")

if __name__ == "__main__":
    evaluator = BatchEvaluator()
    # evaluator.run_benchmark_eval(limit=10) # Để chạy thử
