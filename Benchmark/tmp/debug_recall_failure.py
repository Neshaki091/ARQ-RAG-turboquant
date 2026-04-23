import numpy as np
import pandas as pd
from tqdm import tqdm

def debug_retrieval():
    corpus_path = 'f:/IT project/DoAn/Turboquant-rust demo/turboquant_v2/data/qwen_768_raw.npy'
    queries_path = 'f:/IT project/DoAn/Turboquant-rust demo/turboquant_v2/data/qwen_768_queries.npy'
    
    print("--- 🔬 Deep Search Debug: Ollama vs Cloud Data ---")
    corpus = np.load(corpus_path)
    queries = np.load(queries_path)
    
    # Normalize corpus once for cosine similarity
    print("Normalizing corpus...")
    norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    corpus_norm = corpus / norms

    for i in range(5): # Kiểm tra 5 câu hỏi đầu tiên
        q = queries[i]
        q_norm = q / (np.linalg.norm(q) + 1e-9)
        
        # Brute-force Cosine Similarity
        scores = np.dot(corpus_norm, q_norm)
        
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        print(f"\n[Query {i}]")
        print(f"  Best Match Index in Corpus: {best_idx}")
        print(f"  Max Cosine Similarity: {best_score:.4f}")
        
        # In thêm một vài scores top đầu để xem phân phối
        top_scores = np.sort(scores)[-5:][::-1]
        print(f"  Top 5 scores: {top_scores}")

if __name__ == "__main__":
    debug_retrieval()
