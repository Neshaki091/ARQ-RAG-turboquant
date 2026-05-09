import torch
import numpy as np
import os
import sys

# Ensure local TQ_engine_lib is prioritized
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TQ_engine_lib.quantizer import TQEngine

def run_demo():
    print("=== TurboQuant Real-time Engine Demo ===")
    dim = 768
    n_list = 128
    n_probe = 8
    
    # 1. Initialize Engine
    engine = TQEngine(dim=dim, bits=4, use_ivf=True, ivf_nlist=n_list, ivf_nprobe=n_probe)
    
    # 2. Generate initial data (10k vectors)
    print("\n[1] Indexing 10,000 initial vectors...")
    vectors = torch.nn.functional.normalize(torch.randn(10000, dim), dim=-1)
    engine.index(vectors)
    
    # 3. Perform a search
    query = torch.nn.functional.normalize(torch.randn(1, dim), dim=-1)
    ids, scores = engine.search(query, top_k=5)
    
    target_id = ids[0].item()
    print(f"Top result before deletion: ID={target_id}, Score={scores[0].item():.4f}")
    
    # 4. Soft Delete the top result
    print(f"\n[2] Deleting vector ID={target_id}...")
    engine.delete(target_id)
    
    # 5. Search again to verify deletion
    ids_after, scores_after = engine.search(query, top_k=5)
    print(f"Top result after deletion: ID={ids_after[0].item()}, Score={scores_after[0].item():.4f}")
    
    if target_id not in [id.item() for id in ids_after]:
        print("=> SUCCESS: ID has been removed from results!")
    else:
        print("=> FAILED: ID still exists!")
        
    # 6. Add a NEW vector (Incremental Add)
    special_id = 9999
    print(f"\n[3] Adding a brand new vector (ID={special_id})...")
    # We make the new vector the same as query to ensure it becomes Top-1
    new_vector = query.clone() 
    engine.add(new_vector, special_id)
    
    # 7. Search for the new vector
    ids_final, scores_final = engine.search(query, top_k=5)
    print(f"Top result after adding new vector: ID={ids_final[0].item()}, Score={scores_final[0].item():.4f}")
    
    if ids_final[0].item() == special_id:
        print(f"=> SUCCESS: New vector (ID={special_id}) was found immediately!")
    else:
        print("=> FAILED: New vector not found!")

if __name__ == "__main__":
    run_demo()
