import os
import random
import time
import numpy as np
import torch
from datasets import load_dataset
import google.generativeai as genai
from typing import List, Tuple, Sequence, Any, Dict, Optional
import json
from dotenv import load_dotenv

try:
    from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
except ImportError:
    DPRQuestionEncoder = None

# --- CONFIG ---
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=env_path)

class GeminiModelManager:
    def __init__(self):
        self.keys = []
        for i in range(1, 20):
            k = os.getenv(f"GOOGLE_API_KEY_{i}")
            if k: self.keys.append(k)
            else: break
            
        if not self.keys:
            k = os.getenv("GOOGLE_API_KEY")
            if k: self.keys.append(k)
            else:
                k = os.getenv("GEMINI_API_KEY")
                if k: self.keys.append(k)
                
        if not self.keys:
            print(f"❌ LỖI: Không tìm thấy bất kỳ GOOGLE_API_KEY nào trong file .env")
            print(f"   (Đang tìm tại: {env_path})")
            exit(1)
            
        self.current_idx = 0
        self.model_name = 'gemini-3.1-flash-lite-preview'
        self.model = None
        self._configure()
        print(f"🔑 Đã nạp {len(self.keys)} API Keys để sẵn sàng xoay tua.")
        
    def _configure(self):
        genai.configure(api_key=self.keys[self.current_idx])
        self.model = genai.GenerativeModel(self.model_name)
        
    def rotate(self):
        if len(self.keys) > 1:
            self.current_idx = (self.current_idx + 1) % len(self.keys)
            self._configure()
            print(f"   🔄 Đã xoay tua sang API Key thứ {self.current_idx + 1} / {len(self.keys)}")
        else:
            print(f"   ⚠️ Không có API Key khác để xoay tua!")
            
    def generate_content(self, prompt):
        return self.model.generate_content(prompt)

manager = GeminiModelManager()

def _load_wiki_dpr_streaming():
    """Logic tải vượt rào giống benchmark.py và giới hạn cột để tiết kiệm RAM"""
    dataset_name = "facebook/wiki_dpr"
    dataset_config = "psgs_w100.multiset" 
    split = "train"
    
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    # Chỉ lấy cột 'text' để tránh nạp 'embeddings' cực nặng gây MemoryError
    try:
        kwargs = {"split": split, "streaming": True}
        if hf_token:
            kwargs["token"] = hf_token
        ds = load_dataset(dataset_name, dataset_config, **kwargs)
        return ds.select_columns(['text'])
    except Exception as e:
        if "Dataset scripts are no longer supported" in str(e):
            parquet_glob = f"hf://datasets/{dataset_name}/data/psgs_w100/multiset/*.parquet"
            print(f"⚠️ Script bị chặn, đang tải trực tiếp Parquet: {parquet_glob}")
            parquet_kwargs = {"data_files": parquet_glob, "split": split, "streaming": True}
            if hf_token:
                parquet_kwargs["token"] = hf_token
            ds = load_dataset("parquet", **parquet_kwargs)
            # Một số parquet có thể đặt tên cột là 'content' thay vì 'text'
            cols = ds.column_names
            target_col = 'text' if 'text' in cols else ('content' if 'content' in cols else cols[0])
            print(f"   (Sử dụng cột: {target_col})")
            return ds.select_columns([target_col])
        raise e

def generate_questions_from_chunks(texts: List[str], n_questions: int = 4) -> List[str]:
    """Sử dụng Gemini để sinh nhiều câu hỏi từ 100 đoạn văn."""
    prompt = f"""
    Dưới đây là 100 đoạn văn bản từ Wikipedia.
    Hãy sinh ra CHÍNH XÁC {n_questions} câu hỏi thực tế, ngắn gọn mà câu trả lời có thể tìm thấy trong các đoạn văn này.
    YÊU CẦU ĐỊNH DẠNG:
    - Mỗi dòng đúng 1 câu hỏi.
    - Không đánh số thứ tự.
    - Không thêm bất kỳ văn bản giải thích nào.

    Nội dung văn bản:
    {chr(10).join(texts)}
    """
    
    try:
        response = manager.generate_content(prompt)
        lines = [ln.strip() for ln in response.text.splitlines() if ln.strip()]
        cleaned = []
        for ln in lines:
            q = ln.lstrip("- ").strip()
            if q and q[0].isdigit() and ". " in q[:5]:
                q = q.split(". ", 1)[1].strip()
            if q:
                cleaned.append(q)
        return cleaned[:n_questions]
    except Exception as e:
        error_msg = str(e).lower()
        print(f"   ⚠️ Lỗi khi gọi Gemini: {e}")
        if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
            manager.rotate()
        return []

def embed_questions(questions: List[str], output_pt: str):
    """Sử dụng DPR Question Encoder để biến text thành vector."""
    if DPRQuestionEncoder is None:
        print("⚠️ Không thể embed: Chưa cài 'transformers'. Hãy chạy 'pip install transformers'")
        return

    print(f"\n🧠 Đang nạp DPR Question Encoder để embed {len(questions)} câu hỏi...")
    model_name = "facebook/dpr-question_encoder-single-nq-base"
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
    model = DPRQuestionEncoder.from_pretrained(model_name)
    model.eval()

    queries = []
    with torch.no_grad():
        for i, q_text in enumerate(questions):
            inputs = tokenizer(q_text, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            v = outputs.pooler_output.cpu().numpy()[0]
            # Chuẩn hóa L2 để phù hợp với benchmark (Cosine Similarity)
            v = v / (np.linalg.norm(v) + 1e-10)
            queries.append(v)
            if (i+1) % 50 == 0:
                print(f"   Đã embed {i+1}/{len(questions)}...")

    queries_t = torch.from_numpy(np.array(queries, dtype=np.float32))
    torch.save(queries_t, output_pt)
    print(f"✅ Đã lưu file embedding tại: {output_pt}")

def main():
    import gc
    print("🚀 Đang khởi tạo luồng dữ liệu (DPR Wikipedia)...")
    try:
        ds = _load_wiki_dpr_streaming()
    except Exception as e:
        print(f"❌ Không thể tải dataset: {e}")
        return
    
    total_blocks = 100
    docs_per_block = 100
    questions_per_block = 4
    total_needed = total_blocks * questions_per_block
    wait_time = 3 
    total_corpus_size = 5000000 
    
    print(f"🎲 Đang chọn ngẫu nhiên {total_blocks} block trong {total_corpus_size:,} chunk...")
    random_starts = sorted([random.randint(0, total_corpus_size - docs_per_block) for _ in range(total_blocks)])
    
    output_dir = os.path.join(script_dir, "..", "data", "query")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "benchmark_queries_400.json")
    
    all_generated_questions = []
    print(f"🧠 Đang quét và sinh câu hỏi (Giãn cách {wait_time}s/block)...")
    start_time = time.time()
    
    def save_checkpoint():
        if all_generated_questions:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_generated_questions, f, ensure_ascii=False, indent=2)

    from itertools import islice
    iterator = iter(ds)
    current_ds_idx = 0
    
    for i, start_idx in enumerate(random_starts):
        try:
            skip_count = start_idx - current_ds_idx
            if skip_count > 0:
                # Tiêu thụ skip_count phần tử
                next(islice(iterator, skip_count - 1, skip_count), None)
                current_ds_idx = start_idx
                # Giải phóng RAM sau khi skip một lượng lớn dữ liệu
                if skip_count > 100000:
                    gc.collect()
                
            cluster_texts = []
            for _ in range(docs_per_block):
                row = next(iterator)
                # Lấy text từ bất kỳ cột nào khả dụng
                text = row.get('text', row.get('content', list(row.values())[0]))
                cluster_texts.append(str(text))
                current_ds_idx += 1
            
            print(f"   [Block {i+1}/{total_blocks}] Index {start_idx:,}...", end="", flush=True)
            
            questions: List[str] = []
            for attempt in range(3):
                questions = generate_questions_from_chunks(cluster_texts, n_questions=questions_per_block)
                if len(questions) >= questions_per_block:
                    break
                print(f"   💾 Lỗi hoặc thiếu câu hỏi. Đang lưu Checkpoint {len(all_generated_questions)} câu xuống JSON trước khi thử lại (Attempt {attempt+1}/3)...")
                save_checkpoint()
                time.sleep(3)

            if questions:
                all_generated_questions.extend(questions)
                print(f" ✅ +{len(questions)} câu.")
                save_checkpoint() # Lưu liên tục phòng hờ
            else:
                print(f" ❌ Gemini không phản hồi.")
                
            # Thêm thời gian nghỉ giữa các block để tránh Rate Limit
            time.sleep(wait_time)
                
            # Đảm bảo không giữ list text quá lâu
            del cluster_texts
            if i % 10 == 0:
                gc.collect()

        except StopIteration:
            print("🛑 Hết dữ liệu.")
            break
        except Exception as e:
            print(f" ❌ Lỗi: {repr(e)}")
            print("   🔄 Đang hồi phục luồng dữ liệu...")
            try:
                gc.collect()
                ds = _load_wiki_dpr_streaming()
                iterator = iter(ds)
                current_ds_idx = 0
                time.sleep(2)
            except: break

    # Ghi đè file json lần cuối cùng
    save_checkpoint()
    
    # --- THÊM BƯỚC EMBEDDING ---
    output_pt = output_file.replace(".json", ".pt")
    embed_questions(all_generated_questions, output_pt)
        
    total_duration = (time.time() - start_time) / 60
    print(f"\n✨ HOÀN THÀNH! Đã lưu {len(all_generated_questions)} câu hỏi vào: {output_file}")
    print(f"⏱️ Tổng thời gian: {total_duration:.2f} phút.")

if __name__ == "__main__":
    main()
