import torch
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from tq_engine import TQEngine
from tq_engine.rotation import rotate_forward
import tq_native_lib
import os

def run_rag_demo(query_text):
    DATA_PATH = 'f:/IT project/DoAn/Turboquant-rust demo/turboquant_v2/data'
    corpus_tq_path = f'{DATA_PATH}/qwen_tq_768.npz'
    payload_path = f'{DATA_PATH}/qwen_768_raw_payload.json'
    MODEL_NAME = 'nomic-ai/nomic-embed-text-v1.5'

    print(f"--- RAG Demo: {query_text} ---")
    
    # 1. Load TQ index (with integrated payloads)
    if not os.path.exists(corpus_tq_path):
        print("Data files not ready yet. Please wait for the download and quantization to finish.")
        return

    tq_data = np.load(corpus_tq_path, allow_pickle=True)
    payloads = tq_data['payloads']
    
    # 2. Initialize Engine
    dim = 768
    engine = TQEngine(dim=dim, bits=3)
    engine.mse_quantizer.Pi = torch.from_numpy(tq_data['Pi'])
    engine.S = torch.from_numpy(tq_data['S'])
    engine.mse_quantizer.centroids = torch.from_numpy(tq_data['centroids'])
    engine.qjl_scale = float(tq_data['qjl_scale'])
    mse_bits = int(tq_data.get('mse_bits', 2))

    # 3. Embed Query with Ollama
    print(f"Embedding query with Ollama ({MODEL_NAME})...")
    import requests
    OLLAMA_URL = "http://localhost:11434/api/embeddings"
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "nomic-embed-text",
            "prompt": f"search_query: {query_text}"
        }
    )
    response.raise_for_status()
    q_vector = torch.tensor(response.json()['embedding']).unsqueeze(0).float()
    
    # 4. Search with TurboQuant (Rust SIMD)
    q_rotated = rotate_forward(q_vector, engine.mse_quantizer.Pi).cpu().numpy()
    q_sketched = torch.matmul(q_vector, engine.S.T).cpu().numpy()
    
    packed_indices = tq_data['packed_indices'][np.newaxis, ...]
    qjl_signs = tq_data['signs'][np.newaxis, ...]
    norms = tq_data['norms'][np.newaxis, ...]
    res_norms = tq_data['res_norms'][np.newaxis, ...]
    centroids_np = engine.mse_quantizer.centroids.numpy()

    print("Searching with TurboQuant (3-bit)...")
    mse_scores = tq_native_lib.mse_score_simd(q_rotated, packed_indices, norms, centroids_np, mse_bits)
    qjl_scores = tq_native_lib.qjl_score_simd(q_sketched, qjl_signs, res_norms, engine.qjl_scale)
    tq_scores = mse_scores + qjl_scores
    
    # Get Top 3
    top_indices = np.argsort(-tq_scores, axis=1)[0, :3]
    
    # 5. Formulate Prompt for Qwen 2.5
    print("\n--- Retrieved Context ---")
    context_text = ""
    for rank, idx in enumerate(top_indices):
        chunk = payloads[idx]
        # Payload structure in Qdrant can vary, usually it has a 'text' or 'content' key
        text = chunk.get('text', chunk.get('content', str(chunk)))
        source = chunk.get('source_files', 'Unknown')
        print(f"[{rank+1}] Source: {source}")
        print(f"Content snippet: {text[:150]}...\n")
        context_text += f"\n--- Context {rank+1} ---\n{text}\n"

    final_prompt = f"""Use the following context to answer the question.
Context:
{context_text}

Question: {query_text}
Answer:"""

    print("\n--- Final Prompt for Qwen 2.5 ---")
    print(final_prompt)

if __name__ == "__main__":
    test_query = "What is the DCMI attack methodology?"
    run_rag_demo(test_query)
