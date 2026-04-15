import numpy as np
from qdrant_client.http import models
from .quantization import ManualPQ

class PQBuilder:
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.pq = ManualPQ(d=dimension, m=32, nbits=8)

    def get_storage_config(self):
        """Cấu hình Product Quantization cho Qdrant."""
        return {
            "quantization": models.ProductQuantization(
                product=models.ProductQuantizationConfig(
                    compression=models.CompressionRatio.X32,
                    always_ram=True,
                )
            ),
            "hnsw": None
        }

    def build_index(self, embeddings):
        self.pq.train(embeddings)
        codes = self.pq.quantize(embeddings)
        return codes
