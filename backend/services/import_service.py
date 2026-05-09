import os
import numpy as np
import json
from .tq_service import tq_service
from .metadata_service import metadata_service

class ImportService:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    def import_precomputed(self):
        # Sử dụng đường dẫn tương đối dựa trên data_dir
        vector_path = os.path.join(self.data_dir, "Vector", "nomic_768_raw.npy")
        payload_path = os.path.join(self.data_dir, "Vector", "nomic_768_raw_payload.json")
        
        if not os.path.exists(vector_path) or not os.path.exists(payload_path):
            raise FileNotFoundError(f"Missing precomputed files at {vector_path} or {payload_path}")

        # 1. Load Vectors
        vectors = np.load(vector_path)
        
        # 2. Load Corresponding Payloads (from Vector folder to ensure alignment)
        metadata_path = os.path.join(self.data_dir, "Vector", "nomic_768_raw_payload.json")
        if not os.path.exists(metadata_path):
             # Thử đường dẫn thay thế nếu không thấy
             metadata_path = os.path.join(self.data_dir, "metadata.json")
             
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)
            
        print(f"📖 Loaded {len(metadata_list)} payloads from internal folder")
        
        # 3. Prepare Payloads for SQLite
        payloads = []
        for i in range(len(vectors)):
            # Handle both List and Dict formats for robustness
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
        if count == 0:
            raise ValueError("Bộ dữ liệu import đang bị trống (0 vectors hoặc 0 payloads). Vui lòng kiểm tra lại file nguồn.")
            
        vectors = vectors[:count]
        payloads = payloads[:count]

        # 3. Index in TurboQuant
        tq_service.index_vectors(vectors)
        tq_service.save()
        
        # 4. Save to Local Metadata
        metadata_service.metadata = {} 
        metadata_service.add_chunks(0, payloads)
        
        return count

# Global instance
import_service = ImportService(data_dir="data")
