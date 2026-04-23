import torch
import numpy as np
from tq_engine import TQEngine
import os

raw_path = 'f:/IT project/DoAn/Turboquant-rust demo/turboquant_v2/data/qwen_768_raw.npy'
output_path = 'f:/IT project/DoAn/Turboquant-rust demo/turboquant_v2/data/qwen_tq_768.npz'
DIM = 768

def quantize():
    print(f"Loading raw vectors from: {raw_path}")
    raw_data = np.load(raw_path)
    X = torch.from_numpy(raw_data).float()
    
    # Load Payloads (Context)
    import json
    payload_path = raw_path.replace('.npy', '_payload.json')
    if os.path.exists(payload_path):
        with open(payload_path, 'r', encoding='utf-8') as f:
            payloads = json.load(f)
        print(f"Loaded {len(payloads)} payloads for context.")
    else:
        payloads = []
        print("Warning: No payload file found.")
    
    # Initialize Engine (3-bit: 2-bit MSE + 1-bit QJL)
    print("Initializing TQEngine (3-bit)...")
    engine = TQEngine(dim=DIM, bits=3)
    
    # Run Quantization
    print(f"Quantizing {X.shape[0]} vectors...")
    # Tên hàm quantize trả về đối tượng chứa các mảng đã nén
    quantized_obj = engine.quantize(X)
    
    # Extract components for saving
    # indices: (N, DIM/4) uint8 - (2 bits per subvector)
    # signs: (N, DIM/8) uint8 - (1 bit per dimension)
    # norms: (N,) float32
    # res_norms: (N,) float32
    
    print("Saving quantized components...")
    np.savez_compressed(
        output_path,
        packed_indices=quantized_obj.mse_indices.numpy(),
        signs=quantized_obj.qjl_signs.numpy(),
        norms=quantized_obj.norms.numpy(),
        res_norms=quantized_obj.residual_norms.numpy(),
        # Save engine config for reconstruction
        Pi=engine.mse_quantizer.Pi.numpy(),
        S=engine.S.numpy(),
        centroids=engine.mse_quantizer.centroids.numpy(),
        qjl_scale=engine.qjl_scale,
        mse_bits=quantized_obj.mse_bits,
        payloads=np.array(payloads, dtype=object)
    )
    
    print(f"✅ Quantization complete! Saved to {output_path}")
    
    # Calculate Compression Ratio
    original_size = X.element_size() * X.nelement()
    compressed_size = (
        quantized_obj.mse_indices.element_size() * quantized_obj.mse_indices.nelement() +
        quantized_obj.qjl_signs.element_size() * quantized_obj.qjl_signs.nelement() +
        quantized_obj.norms.element_size() * quantized_obj.norms.nelement() +
        quantized_obj.residual_norms.element_size() * quantized_obj.residual_norms.nelement()
    )
    print(f"Original Size: {original_size / 1024 / 1024:.2f} MB")
    print(f"Compressed Size: {compressed_size / 1024 / 1024:.2f} MB")
    print(f"Compression Ratio: {original_size / compressed_size:.2f}x")

if __name__ == "__main__":
    quantize()
