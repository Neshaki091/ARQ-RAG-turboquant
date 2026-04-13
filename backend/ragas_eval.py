import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

class RagasEvaluator:
    def __init__(self, ollama_url=None):
        if ollama_url is None:
            # Ưu tiên lấy từ ENV, nếu không có thì dùng mặc định là service name 'ollama'
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

        # Sử dụng Gemini 3.1 Flash Lite làm engine chấm điểm qua API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("WARNING: GOOGLE_API_KEY không tồn tại trong .env. Vui lòng bổ sung!")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite", 
            google_api_key=api_key,
            temperature=0
        )
        
        # Vẫn dùng Ollama local cho Embedding vì model nhẹ và nhanh
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text", 
            base_url=ollama_url
        )

    def evaluate(self, query, context_list, answer):
        """
        Thực hiện đánh giá với RAGAS thật và Gemini API.
        """
        data = {
            "question": [query],
            "contexts": [context_list],
            "answer": [str(answer)],
        }
        
        dataset = Dataset.from_dict(data)
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        
        try:
            result = evaluate(
                dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )
            return result.to_pandas().to_dict('records')[0]
        except Exception as e:
            print(f"Lỗi khi chạy RAGAS evaluation (Gemini API): {e}")
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0
            }
