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
        print(f"Lỗi: {e}")
        return None

# Câu hỏi thực tế sau khi đã được Gemini dịch sang bản tiếng Anh chuyên ngành
query_en = "Why is Variable-Block-Length (VBL) ARQ considered optimal compared to Fixed-Block-Length (FBL) ARQ for achieving the DMDT tradeoff, and what factor determines the performance limit of the entire network?"

print(f"--- CHẨN ĐOÁN VỚI TRUY VẤN TIẾNG ANH (ENGLISH) ---")
print(f"Query: {query_en}")
q_vec = get_embedding(query_en)

if q_vec is None: sys.exit(1)

# Tìm kiếm thô với limit = 80 (như code thực tế hiện tại)
response = client.query_points(collection_name=COLLECTION, query=q_vec, limit=80, with_payload=True)
search_results = response.points

print(f"Lấy {len(search_results)} ứng viên ban đầu.")

# Batch Reranking
idx_batch = np.array([hit.payload["idx"] for hit in search_results])
qjl_batch = np.array([hit.payload["qjl"] for hit in search_results])
gamma_batch = np.array([hit.payload["gamma"] for hit in search_results])

scores = qm.tq_prod.compute_score_batch(q_vec, idx_batch, qjl_batch, gamma_batch)

refined_results = []
for i, score in enumerate(scores):
    # Tính score thô (cosine) để so sánh
    raw_score = search_results[i].score
    refined_results.append({
        "score": score,
        "raw_score": raw_score,
        "hit": search_results[i]
    })

# Sắp xếp lại
refined_results.sort(key=lambda x: x["score"], reverse=True)

print("\n--- TOP 5 KẾT QUẢ SAU KHI RERANK ---")
for i, res in enumerate(refined_results[:5]):
    hit = res["hit"]
    content = hit.payload["content"][:200].replace("\n", " ")
    print(f"{i+1}. TQ Score: {res['score']:.4f} | Raw Score: {res['raw_score']:.4f} | ID: {hit.id}")
    print(f"   Content: {content}...")

print("\n--- KIỂM TRA RANK CỦA TARGET CHUNK (ID 379) ---")
for i, hit in enumerate(search_results):
    if hit.id == 379:
        print(f"Target Chunk (ID 379) Rank trong Raw Search: {i+1} | Raw Score: {hit.score:.4f}")
        # Tìm vị trí sau khi rerank
        for r_idx, r_res in enumerate(refined_results):
            if r_res["hit"].id == 379:
                print(f"=> Rank sau khi Rerank (TurboQuant): {r_idx+1} | TQ Score: {r_res['score']:.4f}")
                break
