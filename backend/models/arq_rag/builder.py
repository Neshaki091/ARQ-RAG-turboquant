import numpy as np
import os
import faiss
from qdrant_client.http import models
from .quantization import TurboQuantProd

class ARQBuilder:
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.tq_prod = TurboQuantProd(d=dimension, b=4)

    def get_storage_config(self):
        """Trả về cấu hình lưu trữ tối ưu của ARQ cho Qdrant."""
        return {
            "quantization": models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    always_ram=True,
                )
            ),
            "hnsw": models.HnswConfigDiff(
                ef_construct=512,
                m=32
            )
        }

    def train_centroids(self, embeddings):
        X = embeddings.astype('float32')
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = X / (norms + 1e-10)
        Y = np.dot(X_norm, self.tq_prod.tq_mse.Pi.T)
        y_flat = Y.flatten().reshape(-1, 1)
        
        kmeans = faiss.Kmeans(d=1, k=self.tq_prod.tq_mse.num_centroids, niter=20)
        kmeans.train(y_flat)
        centroids = np.sort(kmeans.centroids.flatten())
        
        os.makedirs("backend/data", exist_ok=True)
        np.save("backend/data/centroids.npy", centroids)
        self.tq_prod.tq_mse.centroids = centroids
        return centroids

    def build_index(self, embeddings):
        X = embeddings.astype('float32')
        orig_norms = np.linalg.norm(X, axis=1)
        X_norm = X / (orig_norms[:, np.newaxis] + 1e-10)
        idx, qjl, gamma = self.tq_prod.quantize_batch(X_norm)
        return {
            "idx": idx,
            "qjl": qjl,
            "gamma": gamma,
            "orig_norm": orig_norms
        }
