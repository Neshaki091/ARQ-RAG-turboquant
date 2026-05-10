import os
import sys
import json
from tqdm import tqdm

# Add backend to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from services.metadata_service import metadata_service
from datasets import load_dataset

def _load_wiki_dpr_streaming():
    """Logic tải vượt rào giống generate_queries.py"""
    dataset_name = "facebook/wiki_dpr"
    dataset_config = "psgs_w100.multiset" 
    split = "train"
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    try:
        kwargs = {"split": split, "streaming": True}
        if hf_token: kwargs["token"] = hf_token
        ds = load_dataset(dataset_name, dataset_config, **kwargs)
        return ds.select_columns(['text', 'title'])
    except Exception as e:
        if "Dataset scripts are no longer supported" in str(e):
            parquet_glob = f"hf://datasets/{dataset_name}/data/psgs_w100/multiset/*.parquet"
            print(f"⚠️ Script bị chặn, đang tải trực tiếp Parquet: {parquet_glob}")
            parquet_kwargs = {"data_files": parquet_glob, "split": split, "streaming": True}
            if hf_token: parquet_kwargs["token"] = hf_token
            ds = load_dataset("parquet", **parquet_kwargs)
            return ds
        raise e

def hydrate_system_memory(limit=5000000):
    # Khai báo đường dẫn gốc
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(script_dir)
    
    # Khởi tạo cấu trúc DB nếu chưa có
    metadata_service._init_db()
    
    print(f"🚀 Starting hydration of {limit:,} Wikipedia chunks into System Memory...")
    
    # 1. Kiểm tra Index 4096 của bạn
    index_path = os.path.join(backend_root, "data", "tq_index_4b_nl4096")
    if os.path.exists(index_path):
        print(f"⚡ Đã tìm thấy bộ Index 4096 tại: {index_path}")
    else:
        print(f"⚠️ Cảnh báo: Chưa tìm thấy Index 4096. AI sẽ không thể tìm kiếm nếu thiếu Index.")

    # 2. Kiểm tra tiến độ hiện tại
    current_count = metadata_service.get_count()
    print(f"📊 Database hiện tại đang có: {current_count:,} văn bản.")
    
    if current_count >= limit:
        print(f"✅ Đã nạp đủ {current_count:,} văn bản. Không cần chạy thêm.")
        return
    
    try:
        ds = _load_wiki_dpr_streaming()
        
        # Nhảy cóc qua những gì đã nạp (Cực nhanh)
        if current_count > 0:
            print(f"⏩ Jumping over {current_count:,} existing rows...")
            ds = ds.skip(current_count)
        
        batch = []
        batch_size = 5000 
        count = current_count
        
        # Thanh tiến trình bắt đầu từ current_count
        pbar = tqdm(total=limit, initial=current_count, desc="Hydrating Metadata")
        
        for i, row in enumerate(ds):
            if count >= limit:
                break
                
            batch.append({
                "text": row['text'],
                "source": row.get('title', 'Wikipedia'),
                "user_id": -1,
                "session_id": "system"
            })
            
            if len(batch) >= batch_size:
                metadata_service.add_chunks(count, batch, user_id=-1)
                count += len(batch)
                pbar.update(len(batch))
                batch = []
                
        # Final batch
        if batch:
            metadata_service.add_chunks(count, batch, user_id=-1)
            count += len(batch)
            pbar.update(len(batch))
            
        pbar.close()
        print(f"✅ HOÀN TẤT: Đã nạp {count:,} văn bản vào metadata.db.")
        
    except Exception as e:
        print(f"❌ ERROR during hydration: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100000, help="Number of chunks to import")
    args = parser.parse_args()
    
    hydrate_system_memory(limit=args.limit)
