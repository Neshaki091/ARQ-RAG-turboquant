import pandas as pd
import numpy as np
import requests
import json
import os

csv_path = 'f:/IT project/DoAn/Turboquant-rust demo/turboquant_v2/data/benchmark_queries_rows.csv'
output_path = 'f:/IT project/DoAn/Turboquant-rust demo/turboquant_v2/data/qwen_768_queries.npy'
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "nomic-embed-text"

def embed_questions():
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    questions = df['question'].tolist()
    print(f"Found {len(questions)} questions.")

    all_embeddings = []
    
    print(f"Connecting to Ollama via {OLLAMA_URL} using model {MODEL_NAME}...")
    
    for i, q in enumerate(questions):
        # Nomic on Ollama usually handles prefixes well, but we'll follow standard practice
        # For retrieval tasks, 'search_query: ' is the standard nomic prefix
        prompt = f"search_query: {q}"
        
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt
                }
            )
            response.raise_for_status()
            embedding = response.json()['embedding']
            all_embeddings.append(np.array(embedding, dtype=np.float32))
            
            if (i + 1) % 50 == 0:
                print(f"Progress: {i + 1}/{len(questions)} embedded...")
                
        except Exception as e:
            print(f"Error embedding question {i}: {e}")
            # Fallback to zero vector to maintain index if one fails (not ideal)
            all_embeddings.append(np.zeros(768, dtype=np.float32))

    embeddings = np.stack(all_embeddings)
    print(f"Final embeddings shape: {embeddings.shape}")
    
    np.save(output_path, embeddings)
    print(f"✅ Successfully saved Ollama query vectors to {output_path}")

if __name__ == "__main__":
    embed_questions()
