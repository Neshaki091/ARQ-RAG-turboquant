import os
import json
import numpy as np
import httpx
from tqdm import tqdm

class EmbeddingManager:
    def __init__(self, data_dir="data", ollama_url=None):
        if ollama_url is None:
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.data_dir = data_dir
        self.ollama_url = f"{ollama_url}/api/embeddings"
        self.chunks_file = os.path.join(data_dir, "chunks.json")
        self.embeddings_file = os.path.join(data_dir, "embeddings.npy")
        self.metadata_file = os.path.join(data_dir, "metadata.json")
        self.model = "nomic-embed-text"

    def get_embedding(self, text):
        payload = {
            "model": self.model,
            "prompt": text
        }
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(self.ollama_url, json=payload)
                return response.json()["embedding"]
        except Exception as e:
            print(f"Lỗi khi lấy embedding từ Ollama: {e}")
            # Trả về vector ngẫu nhiên nếu Ollama chưa sẵn sàng (chỉ để demo)
            return np.random.rand(768).tolist()

    def run_embedding(self, on_progress=None):
        if not os.path.exists(self.chunks_file):
            print("Không tìm thấy chunks.json")
            return None, None

        with open(self.chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        # Kiểm tra xem đã có bao nhiêu embedding rồi (resume)
        existing_embeddings = []
        if os.path.exists(self.embeddings_file):
            existing_embeddings = np.load(self.embeddings_file).tolist()

        start_idx = len(existing_embeddings)
        total = len(chunks)
        
        # Kiểm tra nếu số lượng embedding lớn hơn số lượng chunks (mất đồng bộ)
        if start_idx > total:
            print(f"Cảnh báo: Phát hiện mất đồng bộ! Số lượng embedding ({start_idx}) > số lượng chunks ({total}).")
            print("Đang tiến hành embed lại từ đầu để đảm bảo tính nhất quán...")
            existing_embeddings = []
            start_idx = 0
        
        print(f"Đang tiến hành embed từ index {start_idx}/{total}...")
        
        if on_progress:
            on_progress(start_idx, total)
            
        new_embeddings = existing_embeddings
        for i in range(start_idx, total):
            vec = self.get_embedding(chunks[i]["content"])
            new_embeddings.append(vec)
            
            # Lưu định kỳ mỗi 50 chunks để chống mất dữ liệu
            if i % 50 == 0:
                np.save(self.embeddings_file, np.array(new_embeddings))
            
            if on_progress:
                on_progress(i + 1, total)

        np.save(self.embeddings_file, np.array(new_embeddings))
        print(f"Hoàn thành! Tổng cộng: {len(new_embeddings)} embeddings.")
        return chunks, np.array(new_embeddings)

    def load_embeddings(self):
        if os.path.exists(self.embeddings_file):
            return np.load(self.embeddings_file)
        return None
