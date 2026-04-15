import os
import logging
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger("VectorStore")

class VectorStoreManager:
    def __init__(self, host="qdrant", port=6333):
        self.client = QdrantClient(host=host, port=port)
        self.vector_size = 768 # nomic-embed-text dimension

    def create_collection_modular(self, name, storage_config=None):
        """Khởi tạo một collection với cấu hình nhận được từ Model Builder."""
        exists = self.client.collection_exists(name)
        if not exists:
            print(f"Đang tạo collection modular: {name}...")
            
            quantization = storage_config.get("quantization") if storage_config else None
            hnsw = storage_config.get("hnsw") if storage_config else None
            
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                ),
                quantization_config=quantization,
                hnsw_config=hnsw
            )
            return True
        return False

    def upsert_collection(self, name, chunks, embeddings, extra_payloads=None):
        """Đẩy dữ liệu vào một collection cụ thể với payload mở rộng."""
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            payload = {
                "file": chunk.get("file"),
                "chunk_id": chunk.get("chunk_id"),
                "content": chunk.get("content")
            }
            
            # Gộp thêm extra payloads (ví dụ: idx, qjl của ARQ hoặc codes của PQ)
            if extra_payloads is not None and i < len(extra_payloads):
                payload.update(extra_payloads[i])
            
            points.append(
                models.PointStruct(
                    id=i,
                    vector=vector.tolist() if hasattr(vector, "tolist") else vector,
                    payload=payload
                )
            )

        print(f"Đang upsert {len(points)} point vào {name}...")
        self.client.upsert(
            collection_name=name,
            points=points
        )
        return len(points)

    def search(self, collection_name, query_vector, limit=5):
        """Tìm kiếm trong một collection cụ thể."""
        logger.debug(f"Qdrant query: collection={collection_name}, limit={limit}, "
                      f"vector_norm={np.linalg.norm(query_vector):.4f}")
        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        )
        logger.debug(f"Qdrant response: {len(response.points)} points từ {collection_name}")
        return response.points

    def delete_all_collections(self, collections):
        """Xóa sạch các collections được chỉ định."""
        for name in collections:
            if self.client.collection_exists(name):
                print(f"Đang xóa collection: {name}...")
                self.client.delete_collection(name)
        return True
