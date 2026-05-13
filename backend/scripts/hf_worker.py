import os
import sys
import torch
import numpy as np
import time
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import HfApi

# Thêm đường dẫn để load các service
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from services.tq_service import tq_service
from services.metadata_service import metadata_service
from services.ingestion_service import ingestion_service

def run_worker(limit=100000, batch_size=128, push_every=10000):
    print(f"🚀 HF Worker Started: Target {limit} chunks")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Device: {device}")
    
    # 1. Load Dataset
    print("🌐 Loading Dataset from HF...")
    ds = load_dataset("facebook/wiki_dpr", "psgs_w100.multiset", split="train", streaming=True)
    
    count = 0
    batch_texts = []
    batch_metadata = []
    all_embeddings = []
    
    repo_id = os.getenv("HF_DATASET_REPO")
    token = os.getenv("HF_TOKEN")
    
    if not repo_id or not token:
        print("⚠️ Warning: HF_DATASET_REPO or HF_TOKEN not set. Results will only be saved locally.")

    pbar = tqdm(total=limit)
    
    for i, row in enumerate(ds):
        if i >= limit: break
        
        batch_texts.append(row['text'])
        batch_metadata.append({
            "text": row['text'],
            "source": row.get('title', 'Wikipedia'),
            "user_id": -1,
            "session_id": "system"
        })
        
        if len(batch_texts) >= batch_size:
            # Embedding
            emb = ingestion_service.get_embeddings(batch_texts, is_query=False)
            all_embeddings.append(emb)
            
            # Save Metadata
            metadata_service.add_chunks(count, batch_metadata, user_id=-1)
            
            count += len(batch_texts)
            pbar.update(len(batch_texts))
            
            # Checkpoint & Push
            if count % push_every == 0:
                print(f"\n💾 Saving checkpoint at {count}...")
                save_and_push(all_embeddings, repo_id, token)
            
            batch_texts = []
            batch_metadata = []

    # Final Save
    if batch_texts:
        emb = ingestion_service.get_embeddings(batch_texts, is_query=False)
        all_embeddings.append(emb)
        metadata_service.add_chunks(count, batch_metadata, user_id=-1)
    
    save_and_push(all_embeddings, repo_id, token)
    print("✅ WORKER COMPLETED!")

def save_and_push(all_embeddings, repo_id, token):
    if not all_embeddings: return
    
    vectors = np.vstack(all_embeddings)
    tq_service.system_engine.ivf_nlist = 4096
    tq_service.system_engine.index(vectors)
    
    save_path = os.path.join(tq_service.data_dir, "tq_index_4bit_np4096_system")
    tq_service.system_engine.save_index(save_path)
    
    if repo_id and token:
        api = HfApi(token=token)
        # Push Index
        for file in os.listdir(save_path):
            api.upload_file(
                path_or_fileobj=os.path.join(save_path, file),
                path_in_repo=f"tq_index_system_e5/{file}",
                repo_id=repo_id,
                repo_type="dataset"
            )
        # Push DB
        api.upload_file(
            path_or_fileobj=metadata_service.db_path,
            path_in_repo="metadata.db",
            repo_id=repo_id,
            repo_type="dataset"
        )
        print(f"📤 Pushed to {repo_id}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100000)
    args = parser.parse_args()
    run_worker(limit=args.limit)
