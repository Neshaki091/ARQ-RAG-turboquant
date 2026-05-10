import fitz # PyMuPDF
import ollama
import numpy as np
import os
import time
from .tq_service import tq_service
from .metadata_service import metadata_service

class IngestionService:
    def __init__(self, model_name: str = "facebook/dpr-question_encoder-multiset-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None
        
    def _lazy_load_model(self):
        if self.model is not None:
            return
            
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        # Lấy HF Token từ môi trường
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        
        print(f"MODEL: Loading DPR Question Encoder ({self.model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token)
        self.model = AutoModel.from_pretrained(self.model_name, token=hf_token)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"DEVICE: E5 Model running on -> {self.device.upper()}")
        print(f"SUCCESS: E5 Model loaded on {self.device}")

    def get_embeddings(self, texts, is_query: bool = True):
        if not texts: return np.array([])
        
        import torch
        import numpy as np
        
        # Load model một lần duy nhất
        self._lazy_load_model()
            
        start_time = time.time()
        all_embeddings = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                outputs = self.model(**inputs)
                
                # DPR Model trả về pooler_output trực tiếp cho câu hỏi
                if hasattr(outputs, 'pooler_output'):
                    embeddings = outputs.pooler_output
                else:
                    # Fallback cho các model khác
                    mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                
                # Normalize L2
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.extend(embeddings.cpu().numpy())
                
        return np.array(all_embeddings, dtype='float32')

    def process_pdf(self, file_path: str, filename: str, user_id: int, session_id: str = "default"):
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
        if not chunks:
            print(f"WARNING: No valid chunks found in {filename}. Skipping indexing.")
            return 0
            
        # 2. Embed
        texts = [c['text'] for c in chunks]
        embeddings = self.get_embeddings(texts, is_query=False) # <--- Dùng prefix 'passage: '
        
        # 3. TurboQuant Indexing (Tự động handle tạo mới hoặc thêm vào theo user_id)
        tq_service.add_vectors(embeddings, current_offset, user_id=user_id)

        # 4. Local Metadata Storage (With user_id & session_id)
        metadata_service.add_chunks(current_offset, chunks, user_id=user_id, session_id=session_id)
        
        return len(chunks)

    def delete_document(self, filename: str, user_id: int):
        # 1. Delete from local metadata and get affected IDs
        deleted_ids = metadata_service.delete_by_filename(filename, user_id)
            
        # 2. Soft delete from TQ engine
        for vid in deleted_ids:
            tq_service.delete_vector(vid, user_id=user_id)

# Global instance
ingestion_service = IngestionService()
