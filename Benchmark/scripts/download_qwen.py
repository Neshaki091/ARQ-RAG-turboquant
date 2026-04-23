from qdrant_client import QdrantClient
import numpy as np
import os

# Qdrant Cloud Credentials
URL = "https://652d2b34-7275-4e1b-965d-cb82ffc3c3b1.europe-west3-0.gcp.cloud.qdrant.io"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxODYyOTg1MzA3LCJzdWJqZWN0IjoiYXBpLWtleTpkN2JkZjhiMy00Yjk2LTQ4NWMtOGVhNS02Yjk4OGIyMDYzNGIifQ.6pLe8ToDR9F0syR8rNbXvYcQv1SMAQeX5NPkPlJs5_A"
COLLECTION_NAME = "vector_raw"
TARGET_COUNT = 28378
DIM = 768
OUTPUT_PATH = 'f:/IT project/DoAn/Turboquant-rust demo/turboquant_v2/data/qwen_768_raw.npy'

def download_data():
    client = QdrantClient(url=URL, api_key=API_KEY)
    
    all_vectors = []
    all_payloads = []
    next_page_offset = None
    count = 0
    
    print(f"Connecting to Qdrant Cloud: {URL}")
    print(f"Targeting collection: {COLLECTION_NAME}")
    
    while count < TARGET_COUNT:
        # Fetch vectors AND payload (text context)
        response, next_page_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=min(500, TARGET_COUNT - count),
            offset=next_page_offset,
            with_vectors=True,
            with_payload=True
        )
        
        for point in response:
            if point.vector:
                all_vectors.append(np.array(point.vector, dtype=np.float32))
                all_payloads.append(point.payload)
                count += 1
        
        print(f"Progress: {count}/{TARGET_COUNT} vectors and payloads downloaded...")
        
        if next_page_offset is None:
            break
            
    # Save Vectors
    corpus = np.stack(all_vectors)
    np.save(OUTPUT_PATH, corpus)
    
    # Save Payloads (Context)
    import json
    payload_path = OUTPUT_PATH.replace('.npy', '_payload.json')
    with open(payload_path, 'w', encoding='utf-8') as f:
        json.dump(all_payloads, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved vectors to {OUTPUT_PATH}")
    print(f"✅ Saved context payloads to {payload_path}")
    
    # Check for zero vectors
    norms = np.linalg.norm(corpus, axis=1)
    zeros = (norms < 1e-9).sum()
    print(f"Integrity Check: Found {zeros} zero-norm vectors.")

if __name__ == "__main__":
    download_data()
