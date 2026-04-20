"""
rag_pq package — Product Quantization RAG
==========================================

Thuật toán nén vector tiên tiến nhất trong dự án:
- ProductQuantizer: huấn luyện codebook bằng K-Means (faiss-cpu)
- Indexer: pipeline ingest PDF → embed → encode PQ → upsert Qdrant
- Handler: query → ADC table → Qdrant retrieval → rerank → LLM

Tại sao PQ nhanh?
  Vector 768-dim float32 = 3072 bytes
  Sau PQ (M=8, K=256) = 8 bytes (384x nhỏ hơn!)
  Reranking bằng ADC tra bảng thay vì float multiply
"""
