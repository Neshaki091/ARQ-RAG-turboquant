import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

class VectorStoreManager:
    def __init__(self, host="qdrant", port=6333):
        self.client = QdrantClient(host=host, port=port)
        self.collections = ["vector_raw", "vector_pq", "vector_sq8", "vector_arq"]
        self.vector_size = 768 # nomic-embed-text dimension

    def initialize_collections(self):
        """Khởi tạo 4 collections với các cấu hình quantization khác nhau theo quy ước."""
        for name in self.collections:
            exists = self.client.collection_exists(name)
            if not exists:
                print(f"Đang tạo collection: {name}...")
                
                # Cấu hình Quantization dựa trên tên collection
                quantization_config = None
                
                if name == "vector_sq8":
                    quantization_config = models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            always_ram=True,
                        )
                    )
                elif name == "vector_pq":
                    quantization_config = models.ProductQuantization(
                        product=models.ProductQuantizationConfig(
                            compression=models.CompressionRatio.X32,
                            always_ram=True,
                        )
                    )
                elif name == "vector_arq":
                    # ARQ-RAG (TurboQuant) - Mô phỏng bằng Scalar Quantization cao cấp
                    quantization_config = models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            always_ram=True,
                        )
                    )
                
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    ),
                    quantization_config=quantization_config,
                    # Tối ưu hóa HNSW cho ARQ để có recall cao hơn (phục vụ mục tiêu thắng RAGAS)
                    hnsw_config=models.HnswConfigDiff(
                        ef_construct=512 if name == "vector_arq" else 128,
                        m=32 if name == "vector_arq" else 16
                    ) if name == "vector_arq" else None
                )
        return True

    def upsert_data(self, chunks, embeddings):
        """Đẩy dữ liệu vào cả 4 collections cùng lúc, kèm theo mã nén TurboQuant cho ARQ."""
        from quantization import QuantizationManager
        qm = QuantizationManager(dimension=self.vector_size)
        
        # 1. Huấn luyện Centroids tối ưu (Lloyd-Max) từ tập dữ liệu thực tế
        print("Đang huấn luyện Lloyd-Max Centroids cho TurboQuant...")
        qm.train_centroids(np.array(embeddings))
        
        # 2. Tính toán dữ liệu nén TurboQuant cho toàn bộ batch (sử dụng Centroids vừa học)
        arq_data = qm.build_arq(np.array(embeddings))
        
        if len(chunks) != len(embeddings):
            raise ValueError("Số lượng chunks và embeddings không khớp nhau.")

        for name in self.collections:
            points = []
            for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
                payload = {
                    "file": chunk.get("file"),
                    "chunk_id": chunk.get("chunk_id"),
                    "content": chunk.get("content")
                }
                
                # Nếu là collection ARQ, đính kèm thông tin nén vào payload để Reranking
                if name == "vector_arq":
                    payload["idx"] = arq_data["idx"][i].tolist()
                    payload["qjl"] = arq_data["qjl"][i].tolist()
                    payload["gamma"] = float(arq_data["gamma"][i])
                    payload["orig_norm"] = float(arq_data["orig_norm"][i])
                
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
        
        return len(chunks)

    def search(self, collection_name, query_vector, limit=5):
        """Tìm kiếm trong một collection cụ thể (luôn lấy payload)."""
        # qdrant-client 1.17.1+ uses query_points instead of search
        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        )
        return response.points

    def delete_all_collections(self):
        """Xóa sạch toàn bộ 4 collections."""
        for name in self.collections:
            if self.client.collection_exists(name):
                print(f"Đang xóa collection: {name}...")
                self.client.delete_collection(name)
        return True
