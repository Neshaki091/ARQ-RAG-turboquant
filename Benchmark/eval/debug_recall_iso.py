import torch
import numpy as np
from tq_engine import TQEngine
from tq_engine.quantizer import ProdQuantized, MSEQuantized

def run_isolated_recall_test():
    DATA_DIR = "data/stress_1m"
    DIM = 128
    config = np.load(f"{DATA_DIR}/config.npz")
    
    # Load raw and quantized data
    total_n = 1000000
    test_n = 1000
    raw_mmap = np.memmap(f"{DATA_DIR}/raw_128.npy", dtype='float32', mode='r', shape=(total_n, DIM))
    tq_packed_mmap = np.memmap(f"{DATA_DIR}/tq_packed_128.npy", dtype='uint8', mode='r', shape=(total_n, DIM // 4))
    tq_signs_mmap = np.memmap(f"{DATA_DIR}/tq_signs_128.npy", dtype='uint8', mode='r', shape=(total_n, DIM // 8))
    tq_norms_mmap = np.memmap(f"{DATA_DIR}/tq_norms_128.npy", dtype='float32', mode='r', shape=(total_n,))
    tq_res_norms_mmap = np.memmap(f"{DATA_DIR}/tq_res_norms_128.npy", dtype='float32', mode='r', shape=(total_n,))

    keys_raw = torch.from_numpy(raw_mmap[:test_n])
    # Query là vector số 0 để có Signal cực mạnh
    query = keys_raw[0].unsqueeze(0)
    q_norm = query / (query.norm() + 1e-10)

    # 1. Ground Truth (RAW)
    raw_scores = torch.matmul(q_norm, keys_raw.t()).flatten().numpy()

    # 2. Setup TQ Engine
    tq = TQEngine(dim=DIM, bits=3)
    tq.mse_quantizer.Pi = torch.from_numpy(config['Pi'])
    tq.S = torch.from_numpy(config['S'])
    tq.mse_quantizer.centroids = torch.from_numpy(config['centroids'])
    tq.qjl_scale = float(config['qjl_scale'])

    # 3. Test Components
    mse_packed = torch.from_numpy(tq_packed_mmap[:test_n]).unsqueeze(0)
    qjl_signs = torch.from_numpy(tq_signs_mmap[:test_n]).unsqueeze(0)
    res_norms = torch.from_numpy(tq_res_norms_mmap[:test_n]).unsqueeze(0)
    norms = torch.from_numpy(tq_norms_mmap[:test_n]).unsqueeze(0)

    with torch.no_grad():
        # A. MSE Only Score
        mse_q = MSEQuantized(indices=mse_packed, norms=norms, bits=2)
        k_mse = tq.mse_quantizer.dequantize(mse_q)
        scores_mse = torch.matmul(q_norm, k_mse.transpose(-2, -1)).flatten().cpu().numpy()

        # B. QJL Only Score (Residual part)
        q_sketched = torch.matmul(q_norm.float(), tq.S.t())
        signs = tq._unpack_qjl_signs(qjl_signs)
        scores_qjl = torch.matmul(q_sketched, signs.transpose(-2, -1))
        scores_qjl = (scores_qjl * (tq.qjl_scale * res_norms.unsqueeze(-2))).flatten().cpu().numpy()

        # C. Combined
        combined_scores = scores_mse + scores_qjl

    # Recall Function
    def get_recall(gt_scores, test_scores, k=10):
        gt_idx = set(np.argsort(-gt_scores)[:k])
        test_idx = set(np.argsort(-test_scores)[:k])
        return len(gt_idx & test_idx) / k * 100

    print(f"🔬 [Isolated Recall Analysis]")
    print(f"    MSE (Algorithm 1) Recall@10: {get_recall(raw_scores, scores_mse):.2f}%")
    print(f"    QJL (Algorithm 2) Recall@10: {get_recall(raw_scores, scores_qjl):.2f}%")
    print(f"    Combined Recall@10:         {get_recall(raw_scores, combined_scores):.2f}%")
    
    print("\n🔍 Score Ranges:")
    print(f"    MSE Score Range: [{scores_mse.min():.4f}, {scores_mse.max():.4f}]")
    print(f"    QJL Score Range: [{scores_qjl.min():.4f}, {scores_qjl.max():.4f}]")

if __name__ == "__main__":
    run_isolated_recall_test()
