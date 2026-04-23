import os
import torch
import numpy as np
import gc
import ctypes
from tq_engine import TQEngine

def empty_windows_working_set():
    try:
        ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
    except:
        pass

def prepare_benchmark_data(name, total_tokens, dim, bits=3):
    # Dành cho bài test 5M, tất cả các bản bit đều vào chung folder stress_5m
    is_5m = "5m" in name
    base_folder = "stress_5m" if is_5m else name
    data_dir = f"data/{base_folder}"
    
    # Prefix cho file TQ để phân biệt các bản bit trong cùng 1 folder
    tq_subfolder = f"tq_data/{bits}bit" if is_5m else ""
    tq_dir = f"{data_dir}/{tq_subfolder}" if is_5m else data_dir
    os.makedirs(tq_dir, exist_ok=True)
    
    config_path = f"{tq_dir}/config.npz"
    
    # Kiểm tra xem dữ liệu bản bit này đã tồn tại chưa
    if os.path.exists(config_path) and os.path.exists(f"{tq_dir}/tq_packed_{dim}.npy"):
        print(f"⏩ {name} ({bits}-bit) already exists in {tq_dir}. Skipping...")
        return

    os.makedirs(data_dir, exist_ok=True)
    print(f"\n🚧 Preparing REAL data for {name} ({total_tokens} tokens, D={dim}, {bits}-bits)...")
    print(f"📦 Target TQ Folder: {tq_dir} | 🛡️ RAM Limit: 4GB Mode")
    
    # Init TQ
    tq = TQEngine(dim=dim, bits=bits)
    
    # Save Config riêng cho từng bản bit
    np.savez(config_path, 
             Pi=tq.mse_quantizer.Pi.cpu().numpy(),
             S=tq.S.cpu().numpy() if tq.S is not None else np.zeros((dim, dim)),
             centroids=tq.mse_quantizer.centroids.cpu().numpy(),
             qjl_scale=tq.qjl_scale)
    
    # Kiểm tra xem file chung (raw/sq8/pq) đã tồn tại chưa
    common_exists = os.path.exists(f"{data_dir}/raw_{dim}.npy")
    
    # Open files
    f_raw = open(f"{data_dir}/raw_{dim}.npy", "ab" if common_exists else "wb")
    f_sq8 = open(f"{data_dir}/sq8_{dim}.npy", "ab" if common_exists else "wb")
    f_pq = open(f"{data_dir}/pq_codes_{dim}.npy", "ab" if common_exists else "wb")
    
    # File TQ luôn sinh mới cho bản bit này (wb)
    f_packed = open(f"{tq_dir}/tq_packed_{dim}.npy", "wb")
    f_signs = open(f"{tq_dir}/tq_signs_{dim}.npy", "wb")
    f_norms = open(f"{tq_dir}/tq_norms_{dim}.npy", "wb")
    f_res_norms = open(f"{tq_dir}/tq_res_norms_{dim}.npy", "wb")

    # Shard 100k + MicroBatch 25k để TUYỆT ĐỐI không vượt 4GB RAM
    shard_size = 100_000
    micro_batch_val = 25_000
    
    for i in range(0, total_tokens, shard_size):
        curr_size = min(shard_size, total_tokens - i)
        print(f"  Quantizing chunk {i//1000}k...")
        
        shard_raw = torch.randn(curr_size, dim)
        
        # Chỉ ghi raw/sq8/pq nếu file này chưa tồn tại từ trước
        if not common_exists:
            shard_raw.numpy().astype(np.float32).tofile(f_raw)
            sq8_data = (torch.clamp(shard_raw * 127 + 128, 0, 255)).to(torch.uint8)
            sq8_data.numpy().tofile(f_sq8)
            np.random.randint(0, 256, (curr_size, 96), dtype='uint8').tofile(f_pq)
        
        with torch.no_grad():
            q_data = tq.quantize(shard_raw, micro_batch=micro_batch_val)
        
        q_data.mse_indices.cpu().numpy().astype(np.uint8).tofile(f_packed)
        q_data.qjl_signs.cpu().numpy().astype(np.uint8).tofile(f_signs)
        q_data.norms.cpu().numpy().astype(np.float32).tofile(f_norms)
        q_data.residual_norms.cpu().numpy().astype(np.float32).tofile(f_res_norms)
        
        del shard_raw, q_data
        gc.collect()
        empty_windows_working_set()

    for f in [f_raw, f_packed, f_signs, f_norms, f_res_norms, f_sq8, f_pq]: f.close()
    print(f"✨ Data for {name} saved successfully.")

if __name__ == "__main__":
    print("\n" + "="*45)
    print("🛠️  TURBOQUANT DATA PREPARATION TOOL (MAX 4GB RAM)")
    print("="*45)
    print("\nSelect test data to generate:")
    print("  1. Small Test (1M vectors, D=128) -> data/stress_1m")
    print("  2. Deep 768D (1M vectors, D=768)   -> data/stress_768d")
    print("  3. Massive 5M [ 3-bit ]            -> data/stress_5m")
    print("  4. Massive 5M [ 5-bit ]            -> data/stress_5m")
    print("  5. Massive 5M [ 9-bit ]            -> data/stress_5m")
    print("  6. Generate ALL 5M Variants        -> data/stress_5m")
    print("  7. GENERATE EVERYTHING")
    
    choice = input("\nEnter choice (1-7): ").strip()
    
    if choice in ["1", "7"]:
        prepare_benchmark_data("stress_1m", 1_000_000, 128, bits=3)
    if choice in ["2", "7"]:
        prepare_benchmark_data("stress_768d", 1_000_000, 768, bits=3)
    if choice in ["3", "6", "7"]:
        prepare_benchmark_data("stress_5m", 5_000_000, 768, bits=3)
    if choice in ["4", "6", "7"]:
        prepare_benchmark_data("stress_5m", 5_000_000, 768, bits=5)
    if choice in ["5", "6", "7"]:
        prepare_benchmark_data("stress_5m", 5_000_000, 768, bits=9)

    print("\n✅ Preparation complete. Run benchmarks to start analyzing.")
    print("="*45)
