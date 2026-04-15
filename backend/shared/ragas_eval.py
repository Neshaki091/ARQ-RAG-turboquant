import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_precision, 
    context_recall,
    answer_similarity,
    answer_correctness
)
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

class RagasEvaluator:
    def __init__(self, ollama_url=None):
        if ollama_url is None:
            # Ưu tiên lấy từ ENV, nếu không có thì dùng mặc định là service name 'ollama'
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

        # Khởi tạo engine chấm điểm qua API
        api_key = os.getenv("GROQ_API_KEY")
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        if not api_key:
            print("WARNING: GROQ_API_KEY không tồn tại trong .env. Vui lòng bổ sung!")
        
        print(f"[RAGAS] Đang sử dụng LLM: {self.model_name}")
        self.llm = ChatGroq(
            model_name=self.model_name,
            api_key=api_key,
            temperature=0
        )
        
        # Vẫn dùng Ollama local cho Embedding vì model nhẹ và nhanh
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text", 
            base_url=ollama_url
        )

    def evaluate(self, query, context_list, answer, ground_truth=None):
        """
        Thực hiện đánh giá với RAGAS.
        Nếu có ground_truth, sẽ chấm đầy đủ 6 chỉ số.
        Nếu không có, chỉ chấm 2 chỉ số cơ bản (Faithfulness, Relevancy).
        """
        data = {
            "question": [query],
            "contexts": [context_list],
            "answer": [str(answer)],
        }
        
        # Chỉ số mặc định (không cần Ground Truth)
        metrics = [faithfulness, answer_relevancy]
        
        # Thêm các chỉ số nâng cao nếu có Ground Truth
        if ground_truth:
            data["ground_truth"] = [str(ground_truth)]
            metrics.extend([context_precision, context_recall, answer_similarity, answer_correctness])
            print(f"[RAGAS] Tìm thấy Ground Truth. Đang chấm Full Metrics (6 chỉ số)...")
        else:
            print(f"[RAGAS] Không có Ground Truth. Chỉ chấm Faithfulness & Relevancy.")
        
        dataset = Dataset.from_dict(data)
        
        try:
            print(f"[RAGAS] Đang đánh giá bằng mô hình {self.model_name}...")
            result = evaluate(
                dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )
            scores = result.to_pandas().to_dict('records')[0]
            
            # Đảm bảo trả về đầy đủ các key để tránh lỗi Frontend/Benchmark
            all_metric_keys = [
                "faithfulness", "answer_relevancy", "context_precision", 
                "context_recall", "answer_similarity", "answer_correctness"
            ]
            for m in all_metric_keys:
                if m not in scores:
                    scores[m] = 0.0
            
            # Alias cho tương thích ngược
            scores["answer_relevance"] = scores["answer_relevancy"]
            
            print(f"[RAGAS] Kết quả chấm điểm: {scores}")
            return scores
            
        except Exception as e:
            print(f"[RAGAS] Lỗi khi chạy đánh giá: {e}")
            return {m: 0.0 for m in [
                "faithfulness", "answer_relevancy", "answer_relevance", 
                "context_precision", "context_recall", "answer_similarity", "answer_correctness"
            ]}
