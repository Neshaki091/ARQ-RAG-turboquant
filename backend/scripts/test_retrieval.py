import os
import sys
import torch

# Thêm đường dẫn để load các service
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from services.tq_service import tq_service
from services.ingestion_service import ingestion_service
from services.metadata_service import metadata_service

def quick_test():
    print("--- TESTING RETRIEVAL SYSTEM ---")
    
    # 1. Khởi tạo
    tq_service.load()
    
    # 2. Thử tìm kiếm
    query = "What is the capital of France?"
    print(f"QUERY: {query}")
    
    # Lấy embedding (DPR)
    emb = ingestion_service.get_embeddings([query], is_query=True)
    
    # Search
    print("\n[STEP 1] Calling TurboQuant Engine directly...")
    query_t = torch.from_numpy(emb).float()
    ids, scores = tq_service.system_engine.search(query_t, top_k=5, n_probe=256)
    
    # Chuyển sang numpy để loop an toàn
    if torch.is_tensor(ids): ids = ids.cpu().numpy()
    if torch.is_tensor(scores): scores = scores.cpu().numpy()
    
    # Flatten nếu là mảng 2D (batch size 1)
    if len(ids.shape) > 1: ids = ids[0]
    if len(scores.shape) > 1: scores = scores[0]

    print(f"RAW RESULTS FROM ENGINE:")
    for i, (idx, score) in enumerate(zip(ids, scores)):
        print(f"  [{i+1}] ID: {int(idx)}, Score: {float(score):.4f}")

    # 3. Thử mapping sang Metadata (Service call)
    print("\n[STEP 2] Mapping to Metadata via TQService...")
    results = tq_service.search(emb, top_k=5, scope="system")
    
    print(f"RESULTS WITH TEXT CONTENT: {len(results)}")
    for i, res in enumerate(results):
        payload = metadata_service.get_chunk_metadata(res['id'], user_id=-1)
        if payload:
            print(f"--- Result {i+1} (Score: {res['score']:.4f}) ---")
            print(f"Source: {payload.get('source', 'Unknown')}")
            print(f"Text: {payload.get('text', 'No text found')[:150]}...")
        else:
            print(f"--- Result {i+1} (Score: {res['score']:.4f}) - [METADATA MISSING] ---")

if __name__ == "__main__":
    quick_test()
