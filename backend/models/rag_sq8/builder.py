import numpy as np
from qdrant_client.http import models
from .quantization import ManualSQ8

class SQ8Builder:
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.sq8 = ManualSQ8(d=dimension)

    def get_storage_config(self):
        """Cấu hình Scalar Quantization cho Qdrant."""
        return {
            "quantization": models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    always_ram=True,
                )
            ),
            "hnsw": None
        }

    def build_index(self, embeddings):
        self.sq8.train(embeddings)
        codes = self.sq8.quantize(embeddings)
        return codes
