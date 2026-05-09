import fitz # PyMuPDF
import ollama
import numpy as np
import time
from .tq_service import tq_service
from .metadata_service import metadata_service

class IngestionService:
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
        
    def get_embeddings(self, texts):
        import requests
        if not texts: return np.array([])
        
        start_time = time.time()
        batch_size = 64 # Tăng lên 64 để tận dụng GPU tốt hơn
        all_embeddings = []
        session = requests.Session() # Dùng session để giữ kết nối
        
        print(f"📡 Processing {len(texts)} chunks in batches of {batch_size}...")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_start = time.time()
            print(f"   [Batch {i//batch_size + 1}] Embedding {len(batch)} queries...", end="", flush=True)
            try:
                response = session.post(
                    "http://localhost:11434/api/embed",
                    json={"model": "nomic-embed-text", "input": batch},
                    timeout=120
                )
                data = response.json()
                if "embeddings" in data:
                    all_embeddings.extend(data["embeddings"])
                    print(f" Done ({time.time() - batch_start:.2f}s)")
                else:
                    print(f" Error: {data}")
                    raise Exception(f"Unexpected response: {data}")
            except Exception as e:
                print(f" ❌ FAILED: {e}")
                raise e
                
        print(f"✅ Embedding complete. Total time: {time.time() - start_time:.2f}s")
        return np.array(all_embeddings, dtype='float32')

    def process_pdf(self, file_path: str, filename: str):
        # 0. Get current offset from local metadata
        current_offset = metadata_service.get_count()
            
        # 1. Parse PDF and Word-based Chunking (Logic from cloud_ingest.py)
        doc = fitz.open(file_path)
        chunks = []
        chunk_size = 400
        overlap = 50
        
        for page in doc:
            text = page.get_text()
            words = text.split()
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                if len(chunk_words) < 50: continue # Bỏ qua các đoạn quá ngắn
                
                content = " ".join(chunk_words)
                chunks.append({
                    "text": content,
                    "page": page.number + 1,
                    "source": filename
                })
        
        # 2. Embed
        texts = [c['text'] for c in chunks]
        embeddings = self.get_embeddings(texts)
        
        # 3. TurboQuant Indexing
        if current_offset == 0:
            tq_service.index_vectors(embeddings)
        else:
            tq_service.add_vectors(embeddings, current_offset)

        # 4. Local Metadata Storage (Replacing Qdrant)
        metadata_service.add_chunks(current_offset, chunks)
        
        return len(chunks)

    def delete_document(self, filename: str):
        # 1. Delete from local metadata and get affected IDs
        deleted_ids = metadata_service.delete_by_filename(filename)
            
        # 2. Soft delete from TQ engine
        for vid in deleted_ids:
            tq_service.delete_vector(vid)

# Global instance
ingestion_service = IngestionService()
