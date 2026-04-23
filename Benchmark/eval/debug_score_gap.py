import torch
import numpy as np
from tq_engine import TQEngine
from tq_engine.rotation import rotate_forward
import tq_native_lib

def debug_score_gap():
    DATA_PATH = 'f:/IT project/DoAn/Turboquant-rust demo/turboquant_v2/data'
    corpus_raw = np.load(f'{DATA_PATH}/qwen_768_raw.npy')
    queries = np.load(f'{DATA_PATH}/qwen_768_queries.npy')
    tq_data = np.load(f'{DATA_PATH}/qwen_tq_768.npz')

    # Initialize Engine
    engine = TQEngine(dim=768, bits=3)
    engine.mse_quantizer.Pi = torch.from_numpy(tq_data['Pi'])
    engine.S = torch.from_numpy(tq_data['S'])
    engine.mse_quantizer.centroids = torch.from_numpy(tq_data['centroids'])
    engine.qjl_scale = float(tq_data['qjl_scale'])

    # Pick Query 0 and Vector 0
    q = queries[0:1].astype(np.float32)
    v = corpus_raw[0:1].astype(np.float32)
    
    # 1. RAW Score (Dot Product)
    raw_score = np.dot(q, v.T)[0, 0]
    print(f"RAW Score: {raw_score:.6f}")

    # 2. TQ Components
    q_tensor = torch.from_numpy(q).float()
    v_tensor = torch.from_numpy(v).float()
    
    # Check Rotation
    q_rotated = rotate_forward(q_tensor, engine.mse_quantizer.Pi)
    v_rotated = rotate_forward(v_tensor, engine.mse_quantizer.Pi)
    
    q_rot_np = q_rotated.numpy().astype(np.float32)
    v_rot_np = v_rotated.numpy().astype(np.float32)
    
    # Real rotated dot product (should match RAW score)
    rot_dot = np.dot(q_rot_np, v_rot_np.T)[0, 0]
    print(f"Rotated DOT (should be same as RAW): {rot_dot:.6f}")

    # MSE Quantized Score
    packed_indices = tq_data['packed_indices'][0:1].astype(np.uint8)
    norms = tq_data['norms'][0:1].astype(np.float32)
    centroids_np = engine.mse_quantizer.centroids.numpy().astype(np.float32)
    
    mse_score = tq_native_lib.mse_score_simd(
        q_rot_np, # 2D: (1, 768)
        packed_indices[np.newaxis, ...], # 3D: (1, 1, 192)
        norms[np.newaxis, ...], # 3D: (1, 1)
        centroids_np, # 1D
        2 # mse_bits
    )[0, 0]
    print(f"TQ MSE Score: {mse_score:.6f}")

    # QJL Score
    q_sketch = torch.matmul(q_tensor, engine.S.T).numpy().astype(np.float32)
    qjl_signs = tq_data['signs'][0:1].astype(np.uint8)
    res_norms = tq_data['res_norms'][0:1].astype(np.float32)
    
    q_sketch_np = q_sketch.astype(np.float32)
    qjl_score = tq_native_lib.qjl_score_simd(
        q_sketch_np, # 2D: (1, 96) - assuming S.T is for sketching
        qjl_signs[np.newaxis, ...], # 3D: (1, 1, 12)
        res_norms[np.newaxis, ...], # 3D: (1, 1)
        engine.qjl_scale
    )[0, 0]
    print(f"TQ QJL Score: {qjl_score:.6f}")
    
    total_tq = mse_score + qjl_score
    print(f"Total TQ Score: {total_tq:.6f}")
    print(f"Error: {abs(raw_score - total_tq):.6f}")

if __name__ == "__main__":
    debug_score_gap()
