import os
import json
import fitz  # PyMuPDF
import numpy as np
from shared.supabase_client import SupabaseManager
from shared.vector_store import VectorStoreManager

# Import Builders for modular indexing
from models.arq_rag.builder import ARQBuilder
from models.rag_pq.builder import PQBuilder
from models.rag_sq8.builder import SQ8Builder
from models.rag_raw.builder import RawBuilder
from models.rag_adaptive.builder import AdaptiveBuilder

class IngestionManager:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.chunks_file = os.path.join(data_dir, "chunks.json")
        self.metadata_file = os.path.join(data_dir, "metadata.json")
        os.makedirs(data_dir, exist_ok=True)
        self.supabase = SupabaseManager()
        self.vector_manager = VectorStoreManager()
        
        # Registry of models to ingest
        self.models = {
            "vector_raw": RawBuilder(),
            "vector_adaptive": AdaptiveBuilder(),
            "vector_pq": PQBuilder(),
            "vector_sq8": SQ8Builder(),
            "vector_arq": ARQBuilder()
        }

    def load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"processed_files": [], "total_chunks": 0}

    def save_metadata(self, metadata):
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

    def extract_text(self, pdf_stream):
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def chunk_text(self, text, chunk_size=800, overlap=100):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def process_n_files(self, n=5, on_progress=None):
        metadata = self.load_metadata()
        processed_files = metadata.get("processed_files", [])
        all_remote_files = self.supabase.list_files("papers")
        pending_files = [f for f in all_remote_files if f not in processed_files]
        
        to_process = pending_files[:n]
        total = len(to_process)
        
        if on_progress:
            on_progress(0, total)
        
        new_chunks = []
        if os.path.exists(self.chunks_file):
            with open(self.chunks_file, "r", encoding="utf-8") as f:
                new_chunks = json.load(f)

        for i, filename in enumerate(to_process):
            try:
                pdf_content = self.supabase.get_file_content("papers", filename)
                text = self.extract_text(pdf_content)
                chunks = self.chunk_text(text)
                
                for idx, content in enumerate(chunks):
                    new_chunks.append({
                        "file": filename,
                        "chunk_id": f"{filename}_{idx}",
                        "content": content
                    })
                processed_files.append(filename)
                if on_progress:
                    on_progress(i + 1, total)
            except Exception as e:
                print(f"Lỗi khi xử lý {filename}: {e}")

        with open(self.chunks_file, "w", encoding="utf-8") as f:
            json.dump(new_chunks, f, indent=4, ensure_ascii=False)
            
        metadata["processed_files"] = processed_files
        metadata["total_chunks"] = len(new_chunks)
        self.save_metadata(metadata)
        return len(to_process)

    def sync_to_qdrant(self, chunks, embeddings):
        """Đồng bộ hóa toàn bộ mô hình vào Qdrant."""
        emb_array = np.array(embeddings)
        
        # 1. Huấn luyện Centroids (chỉ dành cho ARQ - đặc thù)
        if "vector_arq" in self.models:
            print("Đang huấn luyện Centroids cho ARQ...")
            self.models["vector_arq"].train_centroids(emb_array)

        # 2. Xử lý từng mô hình
        for name, builder in self.models.items():
            print(f"🔄 Đang xử lý Indexing cho mô hình: {name}")
            
            # Khởi tạo collection với cấu hình của mô hình đó
            self.vector_manager.create_collection_modular(name, builder.get_storage_config())
            
            # Xây dựng dữ liệu bổ sung (extra payloads)
            index_data = builder.build_index(emb_array)
            
            # Chuyển đổi index_data thành list of dicts phù hợp với payload
            extra_payloads = None
            if index_data is not None:
                if name == "vector_arq":
                    extra_payloads = []
                    for i in range(len(embeddings)):
                        extra_payloads.append({
                            "idx": index_data["idx"][i].tolist(),
                            "qjl": index_data["qjl"][i].tolist(),
                            "gamma": float(index_data["gamma"][i]),
                            "orig_norm": float(index_data["orig_norm"][i])
                        })
                elif name in ["vector_pq", "vector_sq8"]:
                    # Với PQ/SQ8 thủ công, ta lưu mã nén vào payload nếu muốn so sánh
                    extra_payloads = [{"codes": index_data[i].tolist()} for i in range(len(embeddings))]

            # Upsert
            self.vector_manager.upsert_collection(name, chunks, embeddings, extra_payloads)
        
        return True
