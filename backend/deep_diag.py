import sys
import numpy as np
import httpx
from qdrant_client import QdrantClient

sys.path.append("/app")
from quantization import QuantizationManager

# Cấu hình
OLLAMA_URL = "http://ollama:11434/api/embeddings"
COLLECTION = "vector_arq"
client = QdrantClient(host="qdrant", port=6333)
qm = QuantizationManager()

def get_embedding(text):
    res = httpx.post(OLLAMA_URL, json={"model": "nomic-embed-text", "prompt": text}, timeout=60.0)
    return np.array(res.json()["embedding"])

query_en = "Why is Variable-Block-Length (VBL) ARQ considered optimal compared to Fixed-Block-Length (FBL) ARQ for achieving the DMDT tradeoff, and what factor determines the performance limit of the entire network?"

print(f"--- ĐANG PHẪU THUẬT ĐIỂM SỐ (DEEP SCORE ANALYSIS) ---")
q_vec = get_embedding(query_en)

# Lấy top 80
response = client.query_points(collection_name=COLLECTION, query=q_vec, limit=80, with_payload=True)
hits = response.points

# Tìm target (ID 379)
target_hit = next((h for h in hits if h.id == 379), None)

if target_hit:
    print(f"Tìm thấy Target (ID 379) trong Top 80.")
    print(f"Raw Score (Qdrant): {target_hit.score:.4f}")
    
    # Tính điểm TurboQuant
    idx = np.array(target_hit.payload["idx"])
    qjl = np.array(target_hit.payload["qjl"])
    gamma = float(target_hit.payload["gamma"])
    
    # Simulating the score decomposition
    q_pi = np.dot(qm.tq_prod.tq_mse.Pi, q_vec)
    q_s = np.dot(qm.tq_prod.S, q_vec)
    
    mse_score = np.dot(qm.tq_prod.tq_mse.centroids[idx], q_pi)
    qjl_dot = np.dot(qjl.astype(float), q_s)
    qjl_score = qm.tq_prod.alpha * gamma * qjl_dot
    
    tq_total = mse_score + qjl_score
    
    print(f"--- CHI TIẾT ĐIỂM TURBOQUANT ---")
    print(f"MSE Part: {mse_score:.4f}")
    print(f"QJL Part: {qjl_score:.4f} (Alpha: {qm.tq_prod.alpha:.6f})")
    print(f"Tổng TQ Score: {tq_total:.4f}")
    
    # So sánh với Rank 1 (Top hiện tại của ARQ)
    # Re-calculate all scores
    all_scores = []
    for h in hits:
        i = np.array(h.payload["idx"])
        q = np.array(h.payload["qjl"])
        g = float(h.payload["gamma"])
        s = qm.tq_prod.compute_score_direct(q_vec, i, q, g)
        all_scores.append((s, h))
    
    all_scores.sort(key=lambda x: x[0], reverse=True)
    
    print(f"\n--- SO SÁNH THỨ HẠNG ---")
    print(f"Vị trí Raw (theo Qdrant): 1-80 (Target 379 đang là rank {next(i for i,h in enumerate(hits) if h.id==379)+1})")
    print(f"Vị trí sau Rerank (TurboQuant): {next(i for i,x in enumerate(all_scores) if x[1].id==379)+1}")
    
    print("\nTop 5 hiện tại của ARQ:")
    for i in range(5):
        s, h = all_scores[i]
        print(f"{i+1}. ID: {h.id} | TQ Score: {s:.4f} | Raw: {h.score:.4f}")
else:
    print("KHÔNG tìm thấy Target 379 trong Top 80 ứng viên!")
