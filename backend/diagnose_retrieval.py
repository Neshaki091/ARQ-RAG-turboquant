import os
import sys
import numpy as np
import httpx
from qdrant_client import QdrantClient

# Thêm đường dẫn để import quantization
sys.path.append("/app")
from quantization import QuantizationManager

# Cấu hình kết nối Docker
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
OLLAMA_URL = "http://ollama:11434/api/embeddings"
COLLECTION = "vector_arq"

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
qm = QuantizationManager()

def get_embedding(text):
    payload = {"model": "nomic-embed-text", "prompt": text}
    try:
        res = httpx.post(OLLAMA_URL, json=payload, timeout=60.0)
        return np.array(res.json()["embedding"])
    except Exception as e:
        print(f"Lỗi kết nối Ollama: {e}")
        return None

query = "Tại sao giao thức Variable-Block-Length (VBL) ARQ lại được coi là tối ưu hơn so với Fixed-Block-Length (FBL) ARQ trong việc đạt được sự đánh đổi DMDT, và yếu tố nào quyết định giới hạn hiệu suất của toàn mạng?"

print(f"--- Đang kiểm chứng logic ADAPTIVE LIMIT (80) ---")
q_vec = get_embedding(query)

if q_vec is None:
    sys.exit(1)

# 1. Tìm kiếm thô với limit = 80
try:
    response = client.query_points(
        collection_name=COLLECTION,
        query=q_vec,
        limit=80,
        with_payload=True
    )
    search_results = response.points
except Exception as e:
    print(f"Lỗi Qdrant Query: {e}")
    sys.exit(1)

print(f"Lấy {len(search_results)} ứng viên ban đầu.")

# 2. Áp dụng BATCH Reranking
idx_batch = np.array([hit.payload["idx"] for hit in search_results])
qjl_batch = np.array([hit.payload["qjl"] for hit in search_results])
gamma_batch = np.array([hit.payload["gamma"] for hit in search_results])

scores = qm.tq_prod.compute_score_batch(q_vec, idx_batch, qjl_batch, gamma_batch)

refined_results = []
for i, score in enumerate(scores):
    refined_results.append((score, search_results[i]))

# Sắp xếp lại
refined_results.sort(key=lambda x: x[0], reverse=True)

print("\n--- KẾT QUẢ TOP 5 SAU KHI TỐI ƯU HÓA ---")
found_target = False
for i, (score, hit) in enumerate(refined_results[:5]):
    content = hit.payload["content"][:250].replace("\n", " ")
    print(f"{i+1}. Score: {score:.4f} | ID: {hit.id}")
    print(f"   Content: {content}...")
    if "variable" in content.lower() or "vbl" in content.lower():
        found_target = True

if found_target:
    print("\n✅ THÀNH CÔNG: Thông tin VBL/FBL đã xuất hiện trong Top 5!")
else:
    print("\n❌ THẤT BẠI: Thông tin mục tiêu vẫn chưa vào được Top 5.")

# Kiểm tra vị trí cũ
for i, hit in enumerate(search_results):
    if hit.id == 379: # ID target từ lần trước
        print(f"\nThông tin mục tiêu (ID 379) nằm ở Rank {i+1} trong Raw Search.")
        for r_idx, (r_score, r_hit) in enumerate(refined_results):
            if r_hit.id == 379:
                print(f"=> Sau khi Rerank với Alpha chuẩn: Nằm ở Rank {r_idx+1}")
                break
