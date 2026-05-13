import os
import sys
import torch
import numpy as np
import gc

# Thêm đường dẫn backend vào sys.path để load TQ_engine_lib
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from TQ_engine_lib.quantizer import TQEngine

def build_system_index():
    data_dir = os.path.join(backend_dir, "data")
    raw_dir = os.path.join(data_dir, "RAW")
    output_path = os.path.join(data_dir, "tq_index_4bit_np4096_system")
    
    if not os.path.exists(raw_dir):
        print(f"[ERROR] RAW directory not found at {raw_dir}")
        return

    # 1. Tìm các file RAW
    raw_files = sorted([f for f in os.listdir(raw_dir) if f.startswith("system_raw_") and f.endswith(".npy")],
                       key=lambda x: int(x.split("_")[-1].split(".")[0]))
    
    if not raw_files:
        print(f"[ERROR] No raw files found in {raw_dir}")
        return

    print(f"[*] Found {len(raw_files)} raw blocks. Creating a unified memmap...")

    # 2. Tạo một file memmap tổng hợp để engine.index xử lý hiệu quả
    total_vectors = 0
    dim = 0
    for f in raw_files:
        temp = np.load(os.path.join(raw_dir, f), mmap_mode='r')
        total_vectors += temp.shape[0]
        dim = temp.shape[1]
    
    combined_raw_path = os.path.join(data_dir, "combined_raw_temp.npy")
    from numpy.lib.format import open_memmap
    combined_mm = open_memmap(combined_raw_path, mode='w+', dtype=np.float32, shape=(total_vectors, dim))
    
    curr = 0
    for f in raw_files:
        print(f"  Merging {f} into combined memmap...")
        block = np.load(os.path.join(raw_dir, f))
        # Chuẩn hóa ngay khi merge để tiết kiệm bước sau
        norms = np.linalg.norm(block, axis=1, keepdims=True) + 1e-10
        combined_mm[curr:curr+len(block)] = block / norms
        curr += len(block)
        del block
        gc.collect()
    
    combined_mm.flush()
    print(f"[+] Combined memmap created at: {combined_raw_path} ({total_vectors:,} vectors)")

    # 3. Chạy Engine Index (Sử dụng chính file combined_mm)
    print(f"[*] Phase 2: Starting TurboQuant Indexing (4-bit, IVF 4096)...")
    engine = TQEngine(dim=dim, bits=4, use_ivf=True, ivf_nlist=4096)
    
    # Truyền memmap vào, engine.index sẽ xử lý theo từng chunk để không tràn RAM
    engine.index(combined_mm, save_path=output_path)
    
    # 4. Dọn dẹp
    del combined_mm
    gc.collect()
    if os.path.exists(combined_raw_path):
        try:
            os.remove(combined_raw_path)
        except:
            print(f"⚠️ Cảnh báo: Không thể xóa file tạm {combined_raw_path}, hãy xóa thủ công sau.")

    print("\n[SUCCESS] System Index built successfully!")
    print(f"Location: {output_path}")

if __name__ == "__main__":
    build_system_index()
