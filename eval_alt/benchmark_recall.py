import torch
import numpy as np
import os
import sys
import time
import math
from sklearn.cluster import MiniBatchKMeans

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TQ_engine_lib.quantizer import TQEngine

# =============================================================================
# Metric Functions
# =============================================================================

def measure_accuracy(predicted_indices, ground_truth_indices, k_values):
    metrics = {"top1_in_k": {}, "set_recall": {}}
    gt_top1 = ground_truth_indices[0]
    
    for k in k_values:
        pred_set = set(predicted_indices[:k])
        # Top-1 in K: Is the actual best result in the predicted Top-K?
        metrics["top1_in_k"][k] = 1.0 if gt_top1 in pred_set else 0.0
        # Set Recall@K: How many of the actual Top-K are in the predicted Top-K?
        gt_set = set(ground_truth_indices[:k])
        metrics["set_recall"][k] = len(pred_set.intersection(gt_set)) / k if k > 0 else 0.0
        
    return metrics

# =============================================================================
# Evaluators
# =============================================================================

def evaluate_sq(vectors, queries, ground_truth, k_values, bits=8):
    print(f"  ---- SQ {bits}-bit...")
    n_levels = 2**bits - 1
    v_min, v_max = vectors.min(), vectors.max()
    scale = (v_max - v_min) / n_levels
    sq_data = ((vectors - v_min) / scale).to(torch.uint8).clamp(0, n_levels)
    sq_dequant = sq_data.float() * scale + v_min
    
    all_top1 = {k: [] for k in k_values}
    all_recall = {k: [] for k in k_values}
    start = time.perf_counter()
    for i in range(len(queries)):
        scores = torch.matmul(queries[i:i+1], sq_dequant.T).view(-1)
        _, topk = torch.topk(scores, max(k_values))
        m = measure_accuracy(topk.tolist(), ground_truth[i], k_values)
        for k in k_values:
            all_top1[k].append(m["top1_in_k"][k])
            all_recall[k].append(m["set_recall"][k])
            
    qps = len(queries) / (time.perf_counter() - start)
    m_top1 = {k: np.mean(all_top1[k]) * 100 for k in k_values}
    m_recall = {k: np.mean(all_recall[k]) * 100 for k in k_values}
    return m_top1, m_recall, qps

def evaluate_pq(vectors, queries, ground_truth, k_values, bits_per_dim=2):
    dim = vectors.shape[1]
    n_vectors = vectors.shape[0]
    M = int(dim * bits_per_dim / 8)
    sub_dim = dim // M
    TRAIN_SIZE = 10000
    print(f"  PQ {bits_per_dim}-bit/dim (M={M}, sub_dim={sub_dim} | Training on {TRAIN_SIZE} samples)...")
    
    vectors_np = vectors.cpu().numpy()
    reconstructed = np.zeros_like(vectors_np)
    
    # Training and Encoding
    for m in range(M):
        sub = vectors_np[:, m*sub_dim : (m+1)*sub_dim]
        sub_train = sub[:min(TRAIN_SIZE, n_vectors)]
        kmeans = MiniBatchKMeans(n_clusters=256, n_init=1, max_iter=20, batch_size=2048, random_state=42).fit(sub_train)
        indices = kmeans.predict(sub)
        reconstructed[:, m*sub_dim : (m+1)*sub_dim] = kmeans.cluster_centers_[indices]
        
    reconstructed_t = torch.from_numpy(reconstructed).to(vectors.device)
    all_top1 = {k: [] for k in k_values}
    all_recall = {k: [] for k in k_values}
    start = time.perf_counter()
    for i in range(len(queries)):
        scores = torch.matmul(queries[i:i+1], reconstructed_t.T).view(-1)
        _, topk = torch.topk(scores, max(k_values))
        m_acc = measure_accuracy(topk.tolist(), ground_truth[i], k_values)
        for k in k_values:
            all_top1[k].append(m_acc["top1_in_k"][k])
            all_recall[k].append(m_acc["set_recall"][k])
            
    qps = len(queries) / (time.perf_counter() - start)
    m_top1 = {k: np.mean(all_top1[k]) * 100 for k in k_values}
    m_recall = {k: np.mean(all_recall[k]) * 100 for k in k_values}
    return m_top1, m_recall, qps

def evaluate_tq_native(vectors, queries, ground_truth, k_values, bits=4):
    print(f"  ---- TQ {bits}-bit (SQ+QJL Native)...")
    dim = vectors.shape[1]
    engine = TQEngine(dim=dim, bits=bits, device="cpu")
    pq_data = engine.quantize(vectors, online_clustering=False)
    
    all_top1 = {k: [] for k in k_values}
    all_recall = {k: [] for k in k_values}
    start = time.perf_counter()
    max_k = max(k_values)
    
    for i in range(len(queries)):
        top_indices, _ = engine.native_cosine_search(queries[i], pq_data, top_k=max_k)
        m = measure_accuracy(top_indices.tolist(), ground_truth[i], k_values)
        for k in k_values:
            all_top1[k].append(m["top1_in_k"][k])
            all_recall[k].append(m["set_recall"][k])
            
    qps = len(queries) / (time.perf_counter() - start)
    m_top1 = {k: np.mean(all_top1[k]) * 100 for k in k_values}
    m_recall = {k: np.mean(all_recall[k]) * 100 for k in k_values}
    return m_top1, m_recall, qps

# =============================================================================
# MAIN
# =============================================================================

def run_accuracy_benchmark():
    print("\n" + "="*95)
    print("TURBOQUANT COMPREHENSIVE RECALL: SQ vs PQ vs TQ (via TQ_engine_lib)")
    print("===============================================================================================")
    print("Config: PQ centroids trained on 10,000 samples.")
    print("===============================================================================================")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data_recall@k", "vector_raw.npy")
    QUERY_PATH = os.path.join(BASE_DIR, "data_recall@k", "query.npy")
    
    K_VALUES = [1, 2, 4, 8, 16, 32, 64]
    NUM_QUERIES = 50

    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    vectors = torch.from_numpy(np.load(DATA_PATH)).float()
    vectors = torch.nn.functional.normalize(vectors, dim=-1)
    queries = torch.from_numpy(np.load(QUERY_PATH)).float()[:NUM_QUERIES]
    queries = torch.nn.functional.normalize(queries, dim=-1)
    
    N, DIM = vectors.shape
    print(f"Dataset: {N} vectors x {DIM}d | {NUM_QUERIES} queries")

    print("\nComputing Ground Truth...")
    ground_truth = []
    for i in range(NUM_QUERIES):
        scores = torch.matmul(queries[i:i+1], vectors.T).view(-1)
        _, topk = torch.topk(scores, max(K_VALUES))
        ground_truth.append(topk.cpu().tolist())

    results = []
    
    # Run all methods
    bit_modes = [2, 4]
    methods = ["SQ", "PQ", "TQ"]
    
    for bits in bit_modes:
        for method in methods:
            if method == "SQ":
                m_t, m_r, q = evaluate_sq(vectors, queries, ground_truth, K_VALUES, bits=bits)
            elif method == "PQ":
                m_t, m_r, q = evaluate_pq(vectors, queries, ground_truth, K_VALUES, bits_per_dim=bits)
            else:
                m_t, m_r, q = evaluate_tq_native(vectors, queries, ground_truth, K_VALUES, bits=bits)
            
            results.append({
                "label": f"{method} {bits}-bit",
                "top1": m_t,
                "recall": m_r,
                "qps": q
            })

    # --- TABLE 1: TOP-1 IN K ---
    print("\n" + "="*110)
    print("TABLE 1: TOP-1 IN K PROBABILITY (Is the true best result within predicted Top-K?)")
    print("="*110)
    hdr_cols = " | ".join([f"P@K={k:<2}" for k in K_VALUES])
    print(f"{'Method':<12} | {hdr_cols} | {'QPS':>8}")
    print("-" * 110)
    for res in results:
        top1_cols = " | ".join([f"{res['top1'][k]:5.1f}%" for k in K_VALUES])
        print(f"{res['label']:<12} | {top1_cols} | {res['qps']:>8.1f}")

    # --- TABLE 2: SET RECALL@K ---
    print("\n" + "="*110)
    print("TABLE 2: SET RECALL@K (Percentage of actual Top-K items found in predicted Top-K)")
    print("="*110)
    hdr_cols = " | ".join([f"R@K={k:<2}" for k in K_VALUES])
    print(f"{'Method':<12} | {hdr_cols} | {'QPS':>8}")
    print("-" * 110)
    for res in results:
        recall_cols = " | ".join([f"{res['recall'][k]:5.1f}%" for k in K_VALUES])
        print(f"{res['label']:<12} | {recall_cols} | {res['qps']:>8.1f}")
    
    print("\n" + "="*110)
    print("(*) PQ (Product Quantization) centroids trained on only 10,000 vectors for realistic scenario.")
    print("(*) PQ search is simulated via full reconstruction in this script.")

if __name__ == "__main__":
    run_accuracy_benchmark()
