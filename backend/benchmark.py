import os
import time
import psutil
import numpy as np
import httpx
from quantization import QuantizationManager
from ragas_eval import RagasEvaluator

class BenchmarkManager:
    def __init__(self, embeddings, chunks, ollama_url="http://host.docker.internal:11434"):
        self.embeddings = embeddings
        self.chunks = chunks
        self.ollama_url = f"{ollama_url}/api/generate"
        self.qm = QuantizationManager(dimension=embeddings.shape[1] if embeddings is not None else 768)
        self.evaluator = RagasEvaluator(ollama_url=ollama_url)
        self.progress = 0
        self.results = []
        self.llm_model = "qwen2.5:3b"

    def get_current_ram(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def generate_answer(self, query, contexts):
        """Sử dụng Qwen 2.5 3B để tạo câu trả lời từ context."""
        context_text = "\n\n".join(contexts)
        prompt = f"""
        Hệ thống của bạn là một trợ lý nghiên cứu khoa học. 
        Hãy trả lời câu hỏi dựa trên thông tin hỗ trợ được cung cấp dưới đây.
        Nếu không có thông tin trong tài liệu, hãy nói rằng bạn không biết, đừng tự ý bịa đặt.

        Tài liệu hỗ trợ:
        {context_text}

        Câu hỏi: {query}
        Trả lời:"""

        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(self.ollama_url, json=payload)
                return response.json()["response"]
        except Exception as e:
            print(f"Lỗi khi gọi Qwen (Ollama): {e}")
            return "Không thể tạo câu trả lời."

    def run_benchmark(self, num_test_sets=2, queries_per_set=5):
        models = ["RAG-RAW", "RAG-Adaptive", "RAG-PQ", "RAG-SQ8", "ARQ-RAG"]
        
        # Câu hỏi nòng cốt cho demo (thực tế nên lấy từ dataset)
        sample_queries = [
            "What is the main contribution of this paper?",
            "Explain the architecture of the proposed system.",
            "Compare the performance with previous baselines.",
            "What dataset was used for evaluation?",
            "Describe the quantization method used."
        ]
        
        total_steps = len(models) * num_test_sets * queries_per_set
        current_step = 0
        
        raw_db = self.qm.build_raw(self.embeddings)
        pq_db = self.qm.build_pq(self.embeddings)
        sq8_db = self.qm.build_sq8(self.embeddings)
        arq_db = self.qm.build_arq(self.embeddings)

        for model in models:
            for ts_idx in range(num_test_sets):
                for q_idx in range(queries_per_set):
                    # 1. Start timer & RAM
                    start_time = time.time()
                    query_text = sample_queries[q_idx % len(sample_queries)]
                    
                    # 2. Embedding query (mô phỏng hoặc thực tế nhúng)
                    # Ở đây dùng vector ngẫu nhiên cho benchmark tốc độ search, 
                    # nhưng query_text dùng cho generation.
                    q_vec = np.random.rand(self.qm.dimension).astype('float32')

                    # 3. Retrieval
                    if model == "RAG-RAW":
                        scores = np.dot(raw_db, q_vec)
                        top_indices = np.argsort(scores)[-5:][::-1]
                    elif model == "RAG-Adaptive":
                        k = np.random.randint(5, 10)
                        scores = np.dot(raw_db, q_vec)
                        top_indices = np.argsort(scores)[-k:][::-1]
                    elif model == "RAG-PQ":
                        _, top_indices = pq_db.search(q_vec.reshape(1, -1), 5)
                        top_indices = top_indices[0]
                    elif model == "RAG-SQ8":
                        _, top_indices = sq8_db.search(q_vec.reshape(1, -1), 5)
                        top_indices = top_indices[0]
                    elif model == "ARQ-RAG":
                        scores = self.qm.tq_prod.compute_score_batch(q_vec, arq_db["idx"], arq_db["qjl"], arq_db["gamma"])
                        top_indices = np.argsort(scores)[-5:][::-1]

                    # 4. Generation (Sử dụng Qwen 2.5 3B)
                    relevant_contexts = [self.chunks[i]["content"] for i in top_indices]
                    answer_text = self.generate_answer(query_text, relevant_contexts)
                    
                    # 5. End timer & measure RAM
                    latency = (time.time() - start_time) * 1000
                    ram_usage = self.get_current_ram()
                    
                    # 6. RAGAS Evaluation (Sử dụng Gemini API)
                    metrics = self.evaluator.evaluate(query_text, relevant_contexts, answer_text)
                    
                    self.results.append({
                        "Model": model,
                        "TestSet": f"Set_{ts_idx+1}",
                        "QueryID": f"Q_{q_idx+1}",
                        "Latency": latency,
                        "RAM": ram_usage,
                        "Faithfulness": metrics.get("faithfulness", 0),
                        "Answer Relevance": metrics.get("answer_relevance", 0),
                        "Context Precision": metrics.get("context_precision", 0),
                        "Context Recall": metrics.get("context_recall", 0)
                    })
                    
                    current_step += 1
                    self.progress = int((current_step / total_steps) * 100)

        return self.results
