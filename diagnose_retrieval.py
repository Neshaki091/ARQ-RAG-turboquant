import os
import numpy as np
import httpx
from qdrant_client import QdrantClient
from quantization import QuantizationManager

# Cấu hình kết nối (Chạy trực tiếp trên máy host qua localhost)
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
OLLAMA_URL = "http://localhost:11434/api/embeddings"
COLLECTION = "vector_arq"

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
qm = QuantizationManager()

def get_embedding(text):
    payload = {"model": "nomic-embed-text", "prompt": text}
    res = httpx.post(OLLAMA_URL, json=payload, timeout=60.0)
    return np.array(res.json()["embedding"])

query = "Tại sao giao thức Variable-Block-Length (VBL) ARQ lại được coi là tối ưu hơn so với Fixed-Block-Length (FBL) ARQ trong việc đạt được sự đánh đổi DMDT, và yếu tố nào quyết định giới hạn hiệu suất của toàn mạng?"

print(f"--- Đang chẩn đoán truy vấn: {query} ---")
q_vec = get_embedding(query)

# 1. Tìm kiếm thô từ Qdrant (lấy top 40 ứng viên)
search_results = client.search(
    collection_name=COLLECTION,
    query_vector=q_vec,
    limit=40,
    with_payload=True
)

print(f"Tìm thấy {len(search_results)} ứng viên ban đầu.")

# 2. Áp dụng Reranking TurboQuant
refined_results = []
for hit in search_results:
    if "idx" not in hit.payload:
        print(f"WARNING: Chunk {hit.id} thiếu dữ liệu TurboQuant. Bạn cần Ingest lại!")
        continue
        
    idx = np.array(hit.payload["idx"])
    qjl = np.array(hit.payload["qjl"])
    gamma = float(hit.payload["gamma"])
    
    score = qm.tq_prod.compute_score_direct(q_vec, idx, qjl, gamma)
    refined_results.append((score, hit))

# Sắp xếp lại
refined_results.sort(key=lambda x: x[0], reverse=True)

print("\n--- TOP 5 KẾT QUẢ SAU KHI RERANK ---")
for i, (score, hit) in enumerate(refined_results[:5]):
    content = hit.payload["content"][:200].replace("\n", " ")
    print(f"{i+1}. Score: {score:.4f} | ID: {hit.id} | File: {hit.payload.get('file')}")
    print(f"   Content: {content}...")

print("\n--- KIỂM TRA TỪ KHÓA 'VARIABLE' TRONG TOP 40 ---")
found_vbl = False
for i, hit in enumerate(search_results):
    if "variable" in hit.payload["content"].lower():
        print(f"Tìm thấy từ khóa 'variable' ở Rank {i+1} (Raw Search) | ID: {hit.id}")
        found_vbl = True

if not found_vbl:
    print("KHÔNG tìm thấy từ khóa 'variable' trong toàn bộ top 40 ứng viên gốc.")
