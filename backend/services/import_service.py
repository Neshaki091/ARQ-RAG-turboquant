import os
import numpy as np
import json
from .tq_service import tq_service
from .metadata_service import metadata_service

class ImportService:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    def import_precomputed(self, user_id: int = -1):
        # Sử dụng đường dẫn tương đối dựa trên data_dir
        vector_path = os.path.join(self.data_dir, "Vector", "nomic_768_raw.npy")
        payload_path = os.path.join(self.data_dir, "Vector", "nomic_768_raw_payload.json")
        
        # Nếu là system import từ wiki_benchmark
        if user_id == -1:
            wiki_dir = os.path.join("f:\\IT project\\DoAn\\DEMO_ARQ_RAG", "Benchmark", "data", "wiki_benchmark")
            if os.path.exists(wiki_dir):
                # Bạn có thể trỏ đến file wiki thật nếu cần
                # Ở đây giả định dùng file trong data/Vector nếu có, hoặc trỏ sang wiki
                pass

        if not os.path.exists(vector_path) or not os.path.exists(payload_path):
            raise FileNotFoundError(f"Missing precomputed files at {vector_path} or {payload_path}")

        # 1. Load Vectors
        vectors = np.load(vector_path)
        
        # 2. Load Corresponding Payloads
        with open(payload_path, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)
            
        payloads = []
        for i in range(len(vectors)):
            if isinstance(metadata_list, list):
                meta = metadata_list[i] if i < len(metadata_list) else {}
            else:
                meta = metadata_list.get(str(i), {})
                
            payloads.append({
                "text": meta.get("content", meta.get("text", "")),
                "source": meta.get("file", meta.get("source", "Unknown")),
                "page": meta.get("page", 0)
            })
        
        count = min(len(vectors), len(payloads))
        vectors = vectors[:count]
        payloads = payloads[:count]

        # 3. Index in TurboQuant
        current_offset = metadata_service.get_count()
        if current_offset == 0:
            tq_service.index_vectors(vectors)
        else:
            tq_service.add_vectors(vectors, current_offset)
        tq_service.save()
        
        # 4. Save to Local Metadata with user_id
        metadata_service.add_chunks(current_offset, payloads, user_id=user_id)
        
        return count

# Global instance
import_service = ImportService(data_dir="data")

# Global instance
import_service = ImportService(data_dir="data")
