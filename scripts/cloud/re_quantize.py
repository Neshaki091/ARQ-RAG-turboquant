import os
import pickle
import numpy as np
import logging
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger("ReQuantize")

# --- Cloud Storage Manager ---

class CloudSupabase:
    def __init__(self, url, key):
        self.client = create_client(url, key)

    def download_file(self, bucket, remote_path, local_path):
        res = self.client.storage.from_(bucket).download(remote_path)
        with open(local_path, "wb") as f:
            f.write(res)

# --- Classes (Ported for Standalone Application of Weights) ---

class TurboQuantMSE:
    def __init__(self, d, b, weights):
        self.d = d
        self.b = b
        self.Pi = weights['Pi']
        self.centroids = weights['centroids']

    def quantize_batch(self, X):
        Y = np.dot(X, self.Pi.T)
        diffs = np.abs(Y[:, :, np.newaxis] - self.centroids[np.newaxis, np.newaxis, :])
        idx = np.argmin(diffs, axis=2)
        return idx

    def dequantize_batch(self, idx):
        Y_tilde = self.centroids[idx]
        X_tilde = np.dot(Y_tilde, self.Pi)
        return X_tilde

class TurboQuantProd:
    def __init__(self, d, b, weights):
        self.d = d
        self.b = b
        self.tq_mse = TurboQuantMSE(d, b - 1, weights)
        self.S = weights['S']
        self.alpha = weights['alpha']

    def quantize_batch(self, X):
        idx = self.tq_mse.quantize_batch(X)
        X_tilde_mse = self.tq_mse.dequantize_batch(idx)
        R = X - X_tilde_mse
        gamma = np.linalg.norm(R, axis=1, ord=2) 
        TR = np.dot(R, self.S.T)
        qjl = np.sign(TR).astype(np.int8)
        qjl[qjl == 0] = 1 
        return idx, qjl, gamma

    def reconstruct_batch(self, idx, qjl, gamma):
        """Tái tạo vector xấp xỉ từ các mã nén ARQ."""
        X_mse = self.tq_mse.dequantize_batch(idx)
        # Residual approximation: R ~ alpha * gamma * (qjl * S)
        R_approx = self.alpha * gamma[:, np.newaxis] * np.dot(qjl.astype(float), self.S)
        return X_mse + R_approx

# --- Main Process ---

def main():
    Q_URL = os.getenv("QDRANT_CLOUD_URL")
    Q_KEY = os.getenv("QDRANT_CLOUD_API_KEY")
    S_URL = os.getenv("SUPABASE_URL")
    S_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))

    if not all([Q_URL, Q_KEY]):
        logger.error("Missing Qdrant credentials!")
        return

    weights_path = "backend/data/model_weights.pkl"
    os.makedirs("backend/data", exist_ok=True)
    
    # Try to download from Supabase first
    if S_URL and S_KEY:
        try:
            supabase = CloudSupabase(S_URL, S_KEY)
            logger.info("Downloading latest weights from Supabase ('centroids' bucket)...")
            supabase.download_file("centroids", "model_weights.pkl", weights_path)
            logger.info("✅ SUCCESS: Latest weights downloaded.")
        except Exception as e:
            logger.warning(f"Could not download weights from Supabase: {e}. Checking local...")

    if not os.path.exists(weights_path):
        logger.error(f"Weights file {weights_path} not found locally or on cloud!")
        return

    with open(weights_path, "rb") as f:
        weights = pickle.load(f)

    client = QdrantClient(url=Q_URL, api_key=Q_KEY, timeout=60.0)
    
    # Initialize ARQ model
    arq = TurboQuantProd(d=768, b=4, weights=weights['arq'])

    # Collection configuration (Only ARQ)
    # We use INT8 quantization in Qdrant for the RECONSTRUCTED vectors to save even more space
    name = "vector_arq"
    q_config = models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8, 
            always_ram=True
        )
    )

    if client.collection_exists(name):
        logger.info(f"Recreating collection {name}...")
        client.delete_collection(name)
    
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE, on_disk=True),
        on_disk_payload=True,
        quantization_config=q_config
    )

    logger.info("Starting batch re-quantization from 'vector_raw' focusing ONLY on ARQ...")
    
    scroll_token = None
    processed_count = 0

    while True:
        points, scroll_token = client.scroll(
            collection_name="vector_raw",
            limit=BATCH_SIZE,
            with_vectors=True,
            with_payload=True,
            scroll_filter=None,
            offset=scroll_token
        )
        
        if not points:
            break

        embeddings = np.array([p.vector for p in points], dtype='float32')
        chunks_payload = [p.payload for p in points]
        
        # 1. ARQ Quantization
        idx, qjl, gamma = arq.quantize_batch(embeddings)
        
        # 2. ARQ Reconstruction (Tái tạo vector từ miền nén)
        # Đây là bước quan trọng: Ta lưu vector xấp xỉ này vào Qdrant thay vì vector gốc
        reconstructed_vectors = arq.reconstruct_batch(idx, qjl, gamma)
        
        arq_points = []
        for i, p in enumerate(points):
            payload = chunks_payload[i].copy()
            payload.update({
                "idx": idx[i].tolist(),
                "qjl": qjl[i].tolist(),
                "gamma": float(gamma[i]),
                "orig_norm": float(np.linalg.norm(embeddings[i]))
            })
            # LƯU Ý: vector=reconstructed_vectors[i] (Không dùng p.vector gốc)
            arq_points.append(models.PointStruct(
                id=p.id, 
                vector=reconstructed_vectors[i].tolist(), 
                payload=payload
            ))
            
        client.upsert("vector_arq", arq_points)

        processed_count += len(points)
        logger.info(f"Processed {processed_count} points into vector_arq...")

        if scroll_token is None:
            break

    logger.info(f"✅ SUCCESS: Re-quantized {processed_count} points using ARQ Reconstruction.")


if __name__ == "__main__":
    main()
