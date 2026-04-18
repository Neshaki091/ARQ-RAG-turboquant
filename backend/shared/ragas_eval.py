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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

class RagasEvaluator:
    def __init__(self, ollama_url=None):
        if ollama_url is None:
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

        # Khởi tạo engine chấm điểm qua Google API (Tránh lỗi TPM 12K của Groq)
        # Sử dụng Gemma 4 31B (Mô hình Giám khảo chính)
        self.model_name = "gemma-4-31b-it"
        
        print(f"[RAGAS] Đang sử dụng LLM giám khảo: {self.model_name}")
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
            max_output_tokens=2048,
            request_timeout=120 # Tăng timeout cho tác vụ chấm điểm phức tạp
        )
        
        # Vẫn dùng Ollama local cho Embedding theo yêu cầu (Nomic Embed)
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
        # CẤU HÌNH CHO GÓI FREE (12K TPM): Giới hạn TỔNG context ở mức 28000 ký tự (~7000 tokens)
        # Điều này để dành room cho Prompt chấm điểm và Output mà không bị lỗi 429
        total_chars = 0
        truncated_contexts = []
        MAX_TOTAL_CHARS = 28000 
        total_chars = 0
        truncated_contexts = []
        
        # Lấy tối đa 5 đoạn chất lượng nhất nếu có thể, nhưng không quá 100000 ký tự
        for ctx in context_list:
            ctx_str = str(ctx)
            if total_chars + len(ctx_str) > MAX_TOTAL_CHARS:
                remaining = MAX_TOTAL_CHARS - total_chars
                if remaining > 100:
                    truncated_contexts.append(ctx_str[:remaining] + "...")
                break
            truncated_contexts.append(ctx_str)
            total_chars += len(ctx_str)

        data = {
            "question": [query],
            "contexts": [truncated_contexts],
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
