import time
import numpy as np
from sentence_transformers import CrossEncoder

class RerankService:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self.model = None
        self.device = None

    def _lazy_load(self):
        if self.model is not None:
            return
        import torch
        print(f"MODEL: Loading Cross-Encoder Reranker ({self.model_name})...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CrossEncoder(self.model_name, device=self.device)
        print(f"Reranker loaded on {self.device}")

    def rerank(self, query: str, search_results: list, top_k: int):
        """Xếp hạng lại danh sách kết quả bằng Cross-Encoder"""
        if not search_results:
            return []
            
        self._lazy_load()
        
        # Chuẩn bị cặp (Câu hỏi, Đoạn văn) - search_results là list các dict có field 'text'
        pairs = [[query, r.get('text', '')] for r in search_results]
        
        if not pairs:
            return search_results[:top_k]
            
        start_time = time.time()
        # Dự đoán điểm số tương đồng
        scores = self.model.predict(pairs)
        
        # Gán điểm mới và sắp xếp lại
        for i, r in enumerate(search_results):
            r['score'] = float(scores[i])
            r['reranked'] = True
            
        # Sắp xếp giảm dần theo điểm của Cross-Encoder
        ranked_results = sorted(search_results, key=lambda x: x['score'], reverse=True)
        
        return ranked_results[:top_k]

# Global instance
rerank_service = RerankService()
