import torch
import numpy as np
import time
from tq_engine import TQEngine
from tq_engine.rotation import rotate_forward
from tq_engine.quantizer import ProdQuantized
import tq_native_lib

def run_final_benchmark():
    DATA_PATH = 'f:/IT project/DoAn/Turboquant-rust demo/turboquant_v2/data'
    corpus_raw_path = f'{DATA_PATH}/qwen_768_raw.npy'
    corpus_tq_path = f'{DATA_PATH}/qwen_tq_768.npz'
    queries_path = f'{DATA_PATH}/qwen_768_queries.npy'
    
    print("--- Loading Data ---")
    corpus_raw = np.load(corpus_raw_path)
    queries = np.load(queries_path)[:100]  # Take only first 100 queries
    tq_data = np.load(corpus_tq_path)
    
    print(f"Corpus: {corpus_raw.shape} | Queries: {queries.shape}")
    
    # 1. Initialize TQ Engine with saved parameters
    dim = corpus_raw.shape[1]
    engine = TQEngine(dim=dim, bits=3)
    engine.mse_quantizer.Pi = torch.from_numpy(tq_data['Pi'])
    engine.S = torch.from_numpy(tq_data['S'])
    engine.mse_quantizer.centroids = torch.from_numpy(tq_data['centroids'])
    engine.qjl_scale = float(tq_data['qjl_scale'])
    mse_bits = int(tq_data.get('mse_bits', 2))

    # 2. RUN RAW SEARCH (Ground Truth)
    print("\n--- Running RAW Search (Baseline) ---")
    start_raw = time.time()
    # Matrix multiplication for RAW
    raw_scores = queries @ corpus_raw.T
    raw_top10 = np.argsort(-raw_scores, axis=1)[:, :10]
    raw_time = time.time() - start_raw
    print(f"RAW Search Time: {raw_time:.4f}s")

    # 3. RUN TQ SEARCH (Rust Implementation)
    print("\n--- Running TurboQuant Search (3-bit) ---")
    
    # Prepare data for Rust
    packed_indices = tq_data['packed_indices']
    qjl_signs = tq_data['signs']
    norms = tq_data['norms']
    res_norms = tq_data['res_norms']
    
    # Prep queries: rotate and project
    q_tensor = torch.from_numpy(queries).float()
    q_rotated = rotate_forward(q_tensor, engine.mse_quantizer.Pi)
    q_sketched = torch.matmul(q_tensor, engine.S.T)
    
    q_rotated_np = q_rotated.numpy()
    q_sketched_np = q_sketched.numpy()
    centroids_np = engine.mse_quantizer.centroids.numpy()

    # Call Rust SIMD kernels
    start_tq = time.time()
    
    # Reshape database arrays to 3D (1, N, D) for the batch-enabled Rust kernels
    packed_indices_3d = packed_indices[np.newaxis, ...]
    qjl_signs_3d = qjl_signs[np.newaxis, ...]
    norms_3d = norms[np.newaxis, ...]
    res_norms_3d = res_norms[np.newaxis, ...]

    all_tq_scores = []
    
    # Process queries one by one to avoid broadcasting issues in Rust kernels
    for i in range(len(queries)):
        q_rot_single = q_rotated_np[i:i+1, :]
        q_sketch_single = q_sketched_np[i:i+1, :]

        # 1. Compute MSE scores
        mse_scores = tq_native_lib.mse_score_simd(
            q_rot_single,
            packed_indices_3d,
            norms_3d,
            centroids_np,
            mse_bits
        )
        
        # 2. Compute QJL residual scores
        qjl_scores = tq_native_lib.qjl_score_simd(
            q_sketch_single,
            qjl_signs_3d,
            res_norms_3d,
            engine.qjl_scale
        )
        
        # 3. Final score = MSE + QJL
        all_tq_scores.append(mse_scores + qjl_scores)
    
    tq_scores = np.vstack(all_tq_scores)
    tq_top10 = np.argsort(-tq_scores, axis=1)[:, :10]
    tq_time = time.time() - start_tq
    print(f"TurboQuant Search Time: {tq_time:.4f}s")
    print(f"Speedup: {raw_time / tq_time:.2f}x")

    # 4. EVALUATE RECALL
    def calculate_recall(gt_indices, test_indices, k):
        recalls = []
        for i in range(len(gt_indices)):
            gt_set = set(gt_indices[i, :k])
            test_set = set(test_indices[i, :k])
            intersection = len(gt_set & test_set)
            recalls.append(intersection / k)
        return np.mean(recalls) * 100

    print("\n--- Accuracy Metrics ---")
    print(f"Recall@1:  {calculate_recall(raw_top10, tq_top10, 1):.2f}%")
    print(f"Recall@5:  {calculate_recall(raw_top10, tq_top10, 5):.2f}%")
    print(f"Recall@10: {calculate_recall(raw_top10, tq_top10, 10):.2f}%")
    
    # MSE between scores
    mse = np.mean((raw_scores - tq_scores)**2)
    print(f"Score MSE: {mse:.6f}")

if __name__ == "__main__":
    run_final_benchmark()
