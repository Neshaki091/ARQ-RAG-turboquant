import fitz # PyMuPDF

import numpy as np
import os
import time
from .tq_service import tq_service
from .metadata_service import metadata_service

class IngestionService:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None
        self.is_onnx = False

    def _check_avx2_support(self):
        """Kiểm tra xem CPU có hỗ trợ AVX2 không (Cần cho ONNX Runtime hiện đại)"""
        import platform
        try:
            if platform.system() == "Windows":
                # Trên Windows, cách nhanh nhất là kiểm tra qua wmic hoặc đơn giản là dùng try-except khi load
                # Để an toàn cho i3-3217U, chúng ta sẽ mặc định trả về False nếu không chắc chắn
                # hoặc cho phép ghi đè qua .env
                if os.getenv("FORCE_ONNX") == "1": return True
                if os.getenv("DISABLE_ONNX") == "1": return False
                
                # Check tên CPU (i3-3xxx là đời 3, không có AVX2)
                import subprocess
                cmd = "wmic cpu get name"
                output = subprocess.check_output(cmd, shell=True).decode()
                if "i3-3" in output or "i5-3" in output or "i7-3" in output:
                    return False
                return True # Các đời cao hơn thường có AVX2
            else:
                # Trên Linux
                with open('/proc/cpuinfo', 'r') as f:
                    return 'avx2' in f.read().lower()
        except:
            return False
        
    def _lazy_load_model(self):
        if self.model is not None:
            return
            
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token)
        self.device = 'cpu'

        # KIỂM TRA AVX2 ĐỂ QUYẾT ĐỊNH DÙNG ONNX HAY TORCH
        use_onnx = self._check_avx2_support() and (os.getenv("DISABLE_ONNX") != "1")
        
        if use_onnx:
            try:
                from optimum.onnxruntime import ORTModelForFeatureExtraction
                print(f"MODEL: CPU supports AVX2. Trying ONNX Optimized Mode...")
                
                model_cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "onnx_model")
                if os.path.exists(model_cache_dir):
                    self.model = ORTModelForFeatureExtraction.from_pretrained(
                        model_cache_dir, provider="CPUExecutionProvider"
                    )
                else:
                    self.model = ORTModelForFeatureExtraction.from_pretrained(
                        self.model_name, token=hf_token, export=True, provider="CPUExecutionProvider"
                    )
                    os.makedirs(model_cache_dir, exist_ok=True)
                    self.model.save_pretrained(model_cache_dir)
                    self.tokenizer.save_pretrained(model_cache_dir)
                
                self.is_onnx = True
                self.model.model.request_inter_ops_threads = 1
                self.model.model.request_intra_ops_threads = os.cpu_count()
                print(f"SUCCESS: ONNX E5 Model ready.")
                return
            except Exception as e:
                print(f"WARNING: ONNX load failed ({e}). Falling back to standard Torch.")
        
        # FALLBACK: DÙNG TORCH NGUYÊN BẢN (An toàn cho i3 đời cũ)
        print(f"MODEL: Using Standard PyTorch Mode (Safe for older CPUs)...")
        self.model = AutoModel.from_pretrained(self.model_name, token=hf_token)
        self.model.eval()
        self.is_onnx = False
        print(f"SUCCESS: Standard PyTorch E5 Model ready.")

    def get_embeddings(self, texts, is_query: bool = True):
        if not texts: return np.array([])
        
        import torch
        import numpy as np
        
        # Load model một lần duy nhất
        self._lazy_load_model()
            
        # E5 Model yêu cầu prefix: 'query: ' cho câu hỏi và 'passage: ' cho văn bản
        prefix = "query: " if is_query else "passage: "
        prefixed_texts = [f"{prefix}{t}" for t in texts]
            
        # Tối ưu hóa CPU Threads cho Torch
        if torch.get_num_threads() < os.cpu_count():
            torch.set_num_threads(os.cpu_count())
            
        all_embeddings = []
        # Giảm batch_size xuống 8 để CPU xử lý từng đợt nhỏ, tránh nghẽn luồng quá lâu
        batch_size = 8
        
        with torch.no_grad():
            for i in range(0, len(prefixed_texts), batch_size):
                batch = prefixed_texts[i:i + batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                outputs = self.model(**inputs)
                
                # Trích xuất last_hidden_state tùy theo loại model
                if self.is_onnx:
                    token_embeddings = outputs.last_hidden_state
                else:
                    token_embeddings = outputs[0] # PyTorch AutoModel return tuple/dict
                
                # Mean Pooling cho E5
                mask = inputs['attention_mask']
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
