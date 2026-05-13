import os
import sqlite3
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def find_db_path():
    """Tự động tìm file .db trong thư mục input của Kaggle"""
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            if filename.endswith('.db'):
                path = os.path.join(dirname, filename)
                print(f"[*] Found Database at: {path}")
                return path
    return None

def embed_on_kaggle():
    # 1. Tự động tìm đường dẫn DB
    db_path = find_db_path()
    if not db_path:
        print("[ERROR] No .db file found in /kaggle/input. Please 'Add Data' to your notebook.")
        return

    output_dir = "/kaggle/working/raw_vectors"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Cấu hình Model
    device = "cuda"
    model_name = "intfloat/multilingual-e5-base"
    print(f"[*] Loading model {model_name} on {device.upper()}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    # Ép kiểu sang FP16 để tối ưu bộ nhớ
    model = model.half()
    model.eval()

    # 3. Kết nối Database ở chế độ Read-Only (Rất quan trọng trên Kaggle)
    # Sử dụng URI mode=ro để tránh lỗi 'unable to open database file'
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM chunks")
        total_chunks = cursor.fetchone()[0]
        print(f"[*] Total chunks to process: {total_chunks:,}")
    except Exception as e:
        print(f"[ERROR] Could not open database: {e}")
        return

    # 4. Thực hiện nhúng (Embedding)
    batch_size = 1024 
    cursor.execute("SELECT text FROM chunks ORDER BY id ASC")
    
    pbar = tqdm(total=total_chunks)
    all_embeddings = []
    count = 0
    file_idx = 0
    
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
            
        texts = [f"passage: {row[0]}" for row in rows]
        
        with torch.no_grad():
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            outputs = model(**inputs)
            
            mask = inputs.attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).to(outputs.last_hidden_state.dtype)
            embeddings = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        count += len(rows)
        pbar.update(len(rows))
        
        # Lưu block mỗi 500k vector
        if count >= 500000:
            raw_file = os.path.join(output_dir, f"system_raw_{file_idx}.npy")
            np.save(raw_file, np.vstack(all_embeddings))
            print(f"\n[+] Saved block {file_idx}")
            all_embeddings = []
            count = 0
            file_idx += 1

    if all_embeddings:
        raw_file = os.path.join(output_dir, f"system_raw_{file_idx}.npy")
        np.save(raw_file, np.vstack(all_embeddings))

    print(f"\n[SUCCESS] Embedding Complete! Files saved in /kaggle/working/raw_vectors")
    conn.close()

if __name__ == "__main__":
    embed_on_kaggle()
