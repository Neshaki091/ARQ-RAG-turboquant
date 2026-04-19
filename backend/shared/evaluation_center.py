import os
import logging
import asyncio
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv

try:
    # DeepEval imports
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase
    from deepeval.models.base_model import DeepEvalBaseLLM
    HAS_DEEPEVAL = True
except ImportError:
    HAS_DEEPEVAL = False
    # Mock classes for linting and safety
    class LLMTestCase: pass
    class DeepEvalBaseLLM: pass
    class FaithfulnessMetric: pass
    class AnswerRelevancyMetric: pass

try:
    # TruLens imports
    from trulens_eval import Tru, Feedback, Select
    from trulens_eval.feedback.provider.ollama import Ollama as OllamaProvider
    HAS_TRULENS = True
except ImportError:
    HAS_TRULENS = False
    # Mock classes for linting and safety
    class Tru: pass
    class Feedback: pass
    class Select: pass
    class OllamaProvider: pass

import numpy as np

load_dotenv()

# Tắt log thừa
logging.getLogger("deepeval").setLevel(logging.WARNING)
logging.getLogger("trulens_eval").setLevel(logging.WARNING)

class OllamaDeepEvalLLM(DeepEvalBaseLLM):
    """Cầu nối giữa DeepEval và Ollama"""
    def __init__(self, model_name="gemma:9b", base_url="http://arq_rag_ollama:11434"):
        self.model_name = model_name
        self.base_url = base_url
        import ollama
        self.client = ollama.Client(host=base_url)

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        response = self.client.chat(model=self.model_name, messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']

    async def a_generate(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.generate, prompt)
        return response

class AdvancedEvaluator:
    def __init__(self, use_cloud=False):
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://arq_rag_ollama:11434")
        self.use_cloud = use_cloud
        
        # 1. Khởi tạo Judge LLM cho DeepEval
        if use_cloud:
            from deepeval.models.gemini_model import GeminiModel
            self.judge_llm = GeminiModel(model_name="gemini-1.5-flash")
            print("[Eval] Sử dụng Gemini 1.5 Flash làm giám khảo.")
        else:
            self.judge_llm = OllamaDeepEvalLLM(model_name="gemma:9b", base_url=self.ollama_url)
            print(f"[Eval] Sử dụng Ollama ({self.judge_llm.model_name}) làm giám khảo.")

        # 2. Thiết lập DeepEval Metrics
        self.faithfulness_metric = FaithfulnessMetric(threshold=0.5, model=self.judge_llm)
        self.relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=self.judge_llm)

        # 3. Thiết lập TruLens
        self.tru = Tru()
        # Mặc định dùng Ollama làm provider cho feedback của TruLens
        self.tru_provider = OllamaProvider(base_url=self.ollama_url, model_engine="gemma:9b")
        
        # Định nghĩa các hàm feedback cho TruLens (RAG Triad)
        self.f_groundedness = (
            Feedback(self.tru_provider.groundedness_measure_with_cot_reasons, name="Groundedness")
            .on(Select.RecordCalls.retrieve.rets.collect()) # Context
            .on_output() # Answer
        )
        self.f_qa_relevance = (
            Feedback(self.tru_provider.relevance_with_cot_reasons, name="Answer Relevance")
            .on_input()
            .on_output()
        )
        self.f_qs_relevance = (
            Feedback(self.tru_provider.relevance_with_cot_reasons, name="Context Relevance")
            .on_input()
            .on(Select.RecordCalls.retrieve.rets.collect())
        )

    def evaluate(self, query: str, contexts: List[str], answer: str) -> Dict:
        """Thực hiện chấm điểm bằng cả DeepEval và TruLens"""
        print(f"[Eval] Đang chấm điểm cho câu hỏi: {query[:50]}...")
        
        # A. Chấm bằng DeepEval
        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=contexts
        )
        
        try:
            self.faithfulness_metric.measure(test_case)
            self.relevancy_metric.measure(test_case)
            
            deepeval_scores = {
                "faithfulness": self.faithfulness_metric.score,
                "answer_relevancy": self.relevancy_metric.score,
                "deepeval_reasoning": {
                    "faithfulness": self.faithfulness_metric.reason,
                    "relevancy": self.relevancy_metric.reason
                }
            }
        except Exception as e:
            print(f"[Eval] Lỗi DeepEval: {e}")
            deepeval_scores = {"faithfulness": 0.0, "answer_relevancy": 0.0}

        # B. Trả về kết quả tổng hợp
        # Lưu ý: TruLens thường chạy theo cơ chế Recorder, ở đây ta lấy điểm DeepEval làm core
        # nhưng vẫn giữ format cũ để không làm gãy Frontend
        return {
            "faithfulness": deepeval_scores["faithfulness"],
            "answer_relevancy": deepeval_scores["answer_relevancy"],
            "answer_relevance": deepeval_scores["answer_relevancy"], # Alias
            "context_precision": 0.0, # Sẽ bổ sung sau nếu cần
            "context_recall": 0.0,
            "answer_similarity": 0.0,
            "answer_correctness": 0.0,
            "details": deepeval_scores.get("deepeval_reasoning", {})
        }

if __name__ == "__main__":
    # Test nhanh
    evaluator = AdvancedEvaluator(use_cloud=False)
    res = evaluator.evaluate(
        query="Mô hình ARQ là gì?",
        contexts=["ARQ là một thuật toán nén vector thích nghi giúp giảm dung lượng lưu trữ."],
        answer="ARQ là thuật toán nén vector."
    )
    print(f"Kết quả test: {res}")
