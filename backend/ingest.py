import os
import json
import fitz  # PyMuPDF
from supabase_client import SupabaseManager

class IngestionManager:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.chunks_file = os.path.join(data_dir, "chunks.json")
        self.metadata_file = os.path.join(data_dir, "metadata.json")
        os.makedirs(data_dir, exist_ok=True)
        self.supabase = SupabaseManager()

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
        # Basic chunking logic
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
        print(f"Bắt đầu xử lý {total} file mới (Yêu cầu: {n})")
        
        if on_progress:
            on_progress(0, total)
        
        new_chunks = []
        if os.path.exists(self.chunks_file):
            with open(self.chunks_file, "r", encoding="utf-8") as f:
                try:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, list):
                        new_chunks = loaded_data
                    else:
                        print(f"Cảnh báo: {self.chunks_file} không phải là list. Đang reset...")
                except json.JSONDecodeError:
                    print(f"Lỗi: {self.chunks_file} bị hỏng. Đang reset...")

        for i, filename in enumerate(to_process):
            print(f"Đang tải và xử lý: {filename}...")
            
            try:
                pdf_content = self.supabase.get_file_content("papers", filename)
                if pdf_content is None:
                    print(f"Lỗi: Không thể tải nội dung {filename}")
                    continue
                    
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

        # Lưu chunks
        with open(self.chunks_file, "w", encoding="utf-8") as f:
            json.dump(new_chunks, f, indent=4, ensure_ascii=False)
            
        metadata["processed_files"] = processed_files
        metadata["total_chunks"] = len(new_chunks)
        self.save_metadata(metadata)
        
        return len(to_process)
