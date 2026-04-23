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

def calculate_metrics(raw_scores, test_scores, name="TQ", ks=[1, 5, 10]):
    raw_scores = torch.from_numpy(raw_scores).flatten()
    test_scores = torch.from_numpy(test_scores).flatten()
    
    max_k = max(ks)
    _, raw_indices = torch.topk(raw_scores, max_k)
    raw_indices_set = set(raw_indices.cpu().numpy()[:max_k])
    
    results = {}
    for k in ks:
        _, tq_indices = torch.topk(test_scores, k)
        tq_indices_set = set(tq_indices.cpu().numpy())
        intersection = len(raw_indices_set & tq_indices_set)
        results[f"Recall@{k}"] = (intersection / k) * 100
        
    mse = torch.mean((raw_scores - test_scores)**2).item()
    results["MSE"] = mse
    return results

def run_thorough_debug_test():
    DATA_NAME = "stress_1m"
    DATA_DIR = f"data/{DATA_NAME}"
    DIM = 128
    
    if not os.path.exists(DATA_DIR):
        print("❌ Run prepare_data.py first!")
        return

    config = np.load(f"{DATA_DIR}/config.npz")
    tq = TQEngine(dim=DIM, bits=3) # Init to get constants
    
    # Mmap data
    total_tokens = 1_000_000
    raw_mmap = np.memmap(f"{DATA_DIR}/raw_128.npy", dtype='float32', mode='r', shape=(total_tokens, DIM))
    tq_packed_mmap = np.memmap(f"{DATA_DIR}/tq_packed_128.npy", dtype='uint8', mode='r', shape=(total_tokens, DIM // 4))
    tq_signs_mmap = np.memmap(f"{DATA_DIR}/tq_signs_128.npy", dtype='uint8', mode='r', shape=(total_tokens, DIM // 8))
    tq_norms_mmap = np.memmap(f"{DATA_DIR}/tq_norms_128.npy", dtype='float32', mode='r', shape=(total_tokens,))
    tq_res_norms_mmap = np.memmap(f"{DATA_DIR}/tq_res_norms_128.npy", dtype='float32', mode='r', shape=(total_tokens,))

    TEST_N = 1000 # Test nhỏ để soi lỗi
    query = torch.randn(1, DIM)
    q_norm = query / (query.norm(dim=-1, keepdim=True) + 1e-10)

    # 1. RAW Scores
    keys_raw = torch.from_numpy(raw_mmap[:TEST_N])
    raw_scores = torch.einsum("qd,kd->qk", q_norm, keys_raw).flatten().cpu().numpy()

    # 2. TQ-Python Scores (Dequantize manually for Gound Truth of TQ)
    # Chúng ta dùng chính logic nén/giải nén của TQEngine để biết "lẽ ra TQ phải ra bao nhiêu"
    from tq_engine.quantizer import ProdQuantized, MSEQuantized
    
    q_packed = ProdQuantized(
        mse_indices=torch.from_numpy(tq_packed_mmap[:TEST_N]).unsqueeze(0),
        qjl_signs=torch.from_numpy(tq_signs_mmap[:TEST_N]).unsqueeze(0),
        residual_norms=torch.from_numpy(tq_res_norms_mmap[:TEST_N]).unsqueeze(0),
        norms=torch.from_numpy(tq_norms_mmap[:TEST_N]).unsqueeze(0),
        mse_bits=2
    )
    # Logic nạp lại centroids và ma trận xoay từ config chuẩn
    tq.mse_quantizer.Pi = torch.from_numpy(config['Pi'])
    tq.S = torch.from_numpy(config['S'])
    tq.mse_quantizer.centroids = torch.from_numpy(config['centroids'])
    tq.qjl_scale = float(config['qjl_scale'])

    with torch.no_grad():
        py_scores = tq.attention_score(q_norm, q_packed).flatten().cpu().numpy()

    # 3. TQ-Rust Scores
    rust_scores = np.zeros(TEST_N)
    if tq_native_lib:
        q_rot = np.ascontiguousarray(torch.matmul(q_norm.float(), tq.mse_quantizer.Pi.t()).cpu().numpy().astype(np.float32))
        q_sketch = np.ascontiguousarray(torch.matmul(q_norm.float(), tq.S.t()).cpu().numpy().astype(np.float32))
        
        mse_p = tq_packed_mmap[:TEST_N][np.newaxis, ...]
        qjl_s = tq_signs_mmap[:TEST_N][np.newaxis, ...]
        norms = tq_norms_mmap[:TEST_N][np.newaxis, ...]
        res_n = tq_res_norms_mmap[:TEST_N][np.newaxis, ...]
        
        s1 = tq_native_lib.mse_score_simd(q_rot, mse_p, norms, config['centroids'], 2)
        s2 = tq_native_lib.qjl_score_simd(q_sketch, qjl_s, res_n, tq.qjl_scale)
        rust_scores = (s1 + s2).flatten()

    # 4. Compare Results
    m_py = calculate_metrics(raw_scores, py_scores, "TQ-Python")
    m_rust = calculate_metrics(raw_scores, rust_scores, "TQ-Rust")

    print(f"\n🔬 [Comparison Results for {TEST_N} tokens]")
    print(f"--- TQ-Python (Ideal TQ) ---")
    print(f"    Recall@10: {m_py['Recall@10']:.2f}% | MSE: {m_py['MSE']:.6f}")
    print(f"--- TQ-Rust (Actual Lib) ---")
    print(f"    Recall@10: {m_rust['Recall@10']:.2f}% | MSE: {m_rust['MSE']:.6f}")
    
    print("\n🔍 First 3 Scores Comparison:")
    print(f"    RAW:       {raw_scores[:3]}")
    print(f"    TQ-Python: {py_scores[:3]}")
    print(f"    TQ-Rust:   {rust_scores[:3]}")

if __name__ == "__main__":
    run_thorough_debug_test()
