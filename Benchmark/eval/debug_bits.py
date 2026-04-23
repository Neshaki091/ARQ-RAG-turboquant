import torch
import numpy as np
from tq_engine import TQEngine
import os

def debug_recall_1000():
    DATA_DIR = "data/stress_1m"
    DIM = 128
    
    config = np.load(f"{DATA_DIR}/config.npz")
    raw_mmap = np.memmap(f"{DATA_DIR}/raw_128.npy", dtype='float32', mode='r', shape=(1000000, DIM))
    tq_packed_mmap = np.memmap(f"{DATA_DIR}/tq_packed_128.npy", dtype='uint8', mode='r', shape=(1000000, DIM // 4))
    tq_signs_mmap = np.memmap(f"{DATA_DIR}/tq_signs_128.npy", dtype='uint8', mode='r', shape=(1000000, DIM // 8))
    tq_norms_mmap = np.memmap(f"{DATA_DIR}/tq_norms_128.npy", dtype='float32', mode='r', shape=(1000000,))
    tq_res_norms_mmap = np.memmap(f"{DATA_DIR}/tq_res_norms_128.npy", dtype='float32', mode='r', shape=(1000000,))

    # Lấy 1000 vector đầu làm Keys
    test_n = 1000
    keys_raw = torch.from_numpy(raw_mmap[:test_n])
    
    # query là chính Vector số 0 (đảm bảo Signal cực mạnh tại index 0)
    query = keys_raw[0].unsqueeze(0)
    q_norm = query / (query.norm() + 1e-10)

    # 1. RAW Scores
    raw_scores = torch.matmul(q_norm, keys_raw.t()).flatten().numpy()
    
    # 2. TQ-Python Scores (Manual Calculation)
    tq = TQEngine(dim=DIM, bits=3)
    tq.mse_quantizer.Pi = torch.from_numpy(config['Pi'])
    tq.S = torch.from_numpy(config['S'])
    tq.mse_quantizer.centroids = torch.from_numpy(config['centroids'])
    tq.qjl_scale = float(config['qjl_scale'])

    from tq_engine.quantizer import ProdQuantized
    q_packed = ProdQuantized(
        mse_indices=torch.from_numpy(tq_packed_mmap[:test_n]).unsqueeze(0),
        qjl_signs=torch.from_numpy(tq_signs_mmap[:test_n]).unsqueeze(0),
        residual_norms=torch.from_numpy(tq_res_norms_mmap[:test_n]).unsqueeze(0),
        norms=torch.from_numpy(tq_norms_mmap[:test_n]).unsqueeze(0),
        mse_bits=2
    )
    
    with torch.no_grad():
        tq_scores = tq.attention_score(q_norm, q_packed).flatten().numpy()

    # 3. Check Rank of Vector 0
    raw_rank = np.argsort(-raw_scores)
    tq_rank = np.argsort(-tq_scores)
    
    print(f"🥇 [Vector 0 Rank Status]")
    print(f"    Raw Top-1 Index: {raw_rank[0]} (Score: {raw_scores[raw_rank[0]]:.4f})")
    print(f"    TQ Top-1 Index:  {tq_rank[0]} (Score: {tq_scores[tq_rank[0]]:.4f})")
    
    # 4. Recall@10
    raw_top10 = set(raw_rank[:10])
    tq_top10 = set(tq_rank[:10])
    intersection = len(raw_top10 & tq_top10)
    print(f"\n📊 Recall@10: {intersection/10*100:.2f}%")

if __name__ == "__main__":
    debug_recall_1000()
