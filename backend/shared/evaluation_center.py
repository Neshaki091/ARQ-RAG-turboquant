import logging
from typing import List, Dict

# Dummy Evaluator - Đã loại bỏ hoàn toàn DeepEval và TruLens để tối ưu hiệu năng
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EvaluatorDummy")

class AdvancedEvaluator:
    def __init__(self, use_cloud=True):
        logger.info("ℹ️ Hệ thống đánh giá (Scoring) đã được vô hiệu hóa theo yêu cầu.")

    def evaluate(self, query: str, contexts: List[str], answer: str) -> Dict:
        """Trả về kết quả trống để duy trì tính tương thích với hệ thống hiện tại"""
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "answer_relevance": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "answer_similarity": 0.0,
            "answer_correctness": 0.0,
            "details": {"info": "Scoring disabled"}
        }

if __name__ == "__main__":
    evaluator = AdvancedEvaluator()
    print(evaluator.evaluate("test", ["test"], "test"))
