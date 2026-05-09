import os
import sys
import numpy as np
import json
import torch

# Add root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.services.tq_service import tq_service
from backend.services.metadata_service import metadata_service

def import_precomputed():
    vector_path = "data/Vector/nomic_768_raw.npy"
    payload_path = "data/Vector/nomic_768_raw_payload.json"
    
    if not os.path.exists(vector_path) or not os.path.exists(payload_path):
        print(f"❌ Không tìm thấy file dữ liệu tại {vector_path} hoặc {payload_path}")
        return

    print("🚀 Đang tải dữ liệu pre-computed...")
    
    # 1. Load Vectors
    vectors = np.load(vector_path)
    print(f"✅ Đã tải {len(vectors)} vectors (dim={vectors.shape[1]})")
    
    # 2. Load Payload
    with open(payload_path, 'r', encoding='utf-8') as f:
        payloads = json.load(f)
    print(f"✅ Đã tải {len(payloads)} chunks nội dung")
    
    if len(vectors) != len(payloads):
        print("⚠️ Cảnh báo: Số lượng vector và payload không khớp nhau!")
        count = min(len(vectors), len(payloads))
        vectors = vectors[:count]
        payloads = payloads[:count]

    # 3. Index in TurboQuant
    print("⚡ Đang nén và tạo chỉ mục TurboQuant SIMD...")
    tq_service.index_vectors(vectors)
    tq_service.save()
    
    # 4. Save to Local Metadata
    print("💾 Đang lưu trữ metadata...")
    # Reset metadata cũ để tránh trùng lặp nếu cần, hoặc cộng dồn
    # Ở đây chúng ta cộng dồn từ ID 0
    metadata_service.metadata = {} # Reset cho sạch nếu import bộ lớn
    metadata_service.add_chunks(0, payloads)
    
    print(f"\n✨ HOÀN THÀNH! Đã nạp thành công {len(payloads)} chunk từ 1114 paper.")
    print(f"📊 Hệ thống hiện đã sẵn sàng để truy vấn siêu tốc.")

if __name__ == "__main__":
    import_all = input("Bạn có muốn xóa dữ liệu cũ và nạp bộ dữ liệu 1114 paper này không? (y/n): ")
    if import_all.lower() == 'y':
        import_precomputed()
