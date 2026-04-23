import torch
import warnings
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
import time
import psutil
import os
import gc
import numpy as np
from tq_engine import TQEngine
try:
    import tq_native_lib
except ImportError:
    tq_native_lib = None

def get_current_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_768d_benchmark_persistence():
    # Cấu hình nạp dữ liệu
    DATA_NAME = "stress_768d"
    DATA_DIR = f"data/{DATA_NAME}"
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data not found in {DATA_DIR}. Please run eval/prepare_data.py first.")
        return

    # Mốc RAM cơ sở tuyệt đối khi chưa nạp gì
    GLOBAL_BASELINE = get_current_memory_mb()
    
    # Load config
    config = np.load(f"{DATA_DIR}/config.npz")
    Pi = torch.from_numpy(config['Pi'])
    S = torch.from_numpy(config['S'])
    centroids = config['centroids']
    qjl_scale = float(config['qjl_scale'])

    # Mmap data (D=768, 1M Tokens)
    total_tokens = 1_000_000
    dim = 768
    raw_mmap = np.memmap(f"{DATA_DIR}/raw_768.npy", dtype='float32', mode='r', shape=(total_tokens, dim))
    tq_packed_mmap = np.memmap(f"{DATA_DIR}/tq_packed_768.npy", dtype='uint8', mode='r', shape=(total_tokens, dim // 4))
    tq_signs_mmap = np.memmap(f"{DATA_DIR}/tq_signs_768.npy", dtype='uint8', mode='r', shape=(total_tokens, dim // 8))
    tq_norms_mmap = np.memmap(f"{DATA_DIR}/tq_norms_768.npy", dtype='float32', mode='r', shape=(total_tokens,))
    tq_res_norms_mmap = np.memmap(f"{DATA_DIR}/tq_res_norms_768.npy", dtype='float32', mode='r', shape=(total_tokens,))

    TOTAL_TOKENS = raw_mmap.shape[0]
    DIM = raw_mmap.shape[1]
    RAW_BATCH = 100_000   
    TQ_BATCH = 500_000  
    N_QUERIES = 5
    
    print(f"🚀 [768D Persistence Test] {TOTAL_TOKENS} Tokens | D={DIM}")
    print(f"📊 Mode: DISK STREAMING (mmap)")
    print(f"📊 Strategy: RAW Batch={RAW_BATCH} | TQ Batch={TQ_BATCH} | Queries={N_QUERIES}")
    print("-" * 60)

    # 1. RAW
    gc.collect()
    peak_raw_max = get_current_memory_mb()
    start_time_raw = time.perf_counter()
    
    try:
        for q_idx in range(N_QUERIES):
            query = torch.randn(1, DIM)
            for i in range(0, TOTAL_TOKENS, RAW_BATCH):
                shard = torch.from_numpy(raw_mmap[i:i+RAW_BATCH])
                _ = torch.einsum("qd,kd->qk", query, shard)
                peak_raw_max = max(peak_raw_max, get_current_memory_mb())
                del shard
                gc.collect() # Force immediate reclamation
            print(f"  Query {q_idx+1}/{N_QUERIES} finished.")
    except Exception as e:
        print(f"❌ RAW failed: {e}")
    time_raw = time.perf_counter() - start_time_raw

    # 2. TQ
    gc.collect()
    peak_tq_max = get_current_memory_mb()
    start_time_tq = time.perf_counter()
    
    if tq_native_lib is None:
        print("❌ tq_native_lib not found!")
        return

    try:
        for q_idx in range(N_QUERIES):
            query = torch.randn(1, DIM)
            q_rot = np.ascontiguousarray(torch.matmul(query.float(), Pi.t()).cpu().numpy().astype(np.float32))
            q_sketch = np.ascontiguousarray(torch.matmul(query.float(), S.t()).cpu().numpy().astype(np.float32))
            
            for i in range(0, TOTAL_TOKENS, TQ_BATCH):
                mse_packed = tq_packed_mmap[i:i+TQ_BATCH][np.newaxis, ...]
                qjl_signs = tq_signs_mmap[i:i+TQ_BATCH][np.newaxis, ...]
                norms = tq_norms_mmap[i:i+TQ_BATCH][np.newaxis, ...]
                res_norms = tq_res_norms_mmap[i:i+TQ_BATCH][np.newaxis, ...]

                _ = tq_native_lib.mse_score_simd(q_rot, mse_packed, norms, centroids, 2)
                _ = tq_native_lib.qjl_score_simd(q_sketch, qjl_signs, res_norms, qjl_scale)
                
                peak_tq_max = max(peak_tq_max, get_current_memory_mb())
                del mse_packed, qjl_signs, norms, res_norms
                gc.collect() # Force immediate reclamation
            print(f"  Query {q_idx+1}/{N_QUERIES} finished.")
        time_tq = time.perf_counter() - start_time_tq
    except Exception as e:
        print(f"❌ TQ failed: {e}")
        time_tq = 0
        
    avg_time_raw = time_raw / N_QUERIES
    avg_time_tq = time_tq / N_QUERIES
    
    # Delta so với lúc chưa chạy gì
    actual_peak_raw = peak_raw_max - GLOBAL_BASELINE
    actual_peak_tq = peak_tq_max - GLOBAL_BASELINE

    print("\n" + "="*40)
    print(f"🏆 FINAL 768D PERSISTENCE RESULTS")
    print("="*40)
    print(f"RAW (Batch {RAW_BATCH//1000}k) | Peak RAM: {actual_peak_raw:.2f} MB | Avg Time: {avg_time_raw:.4f}s")
    print(f"TQ  (Batch {TQ_BATCH//1000}k) | Peak RAM: {actual_peak_tq:.2f} MB | Avg Time: {avg_time_tq:.4f}s")
    print("-" * 40)
    ram_saving = actual_peak_raw / actual_peak_tq if actual_peak_tq > 0.1 else (actual_peak_raw / 0.1)
    print(f"🚀 Speedup: {avg_time_raw/avg_time_tq:.2f}x | 📉 RAM Saving: {ram_saving:.2f}x")
    print(f"📈 Throughput TQ: {(TOTAL_TOKENS / avg_time_tq) / 1e6:.2f} Million tokens/sec")

if __name__ == "__main__":
    run_768d_benchmark_persistence()
