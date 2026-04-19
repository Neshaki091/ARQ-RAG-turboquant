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

class ManualPQ:
    def __init__(self, d, m, weights, nbits=8):
        self.d = d
        self.m = m
        self.nbits = nbits
        self.k = 2 ** nbits
        self.ds = d // m
        self.centroids = weights['centroids']

    def quantize(self, X):
        N = X.shape[0]
        codes = np.zeros((N, self.m), dtype=np.uint8)
        for i in range(self.m):
            sub_X = X[:, i*self.ds : (i+1)*self.ds]
            diffs = np.linalg.norm(sub_X[:, np.newaxis, :] - self.centroids[i][np.newaxis, :, :], axis=2)
            codes[:, i] = np.argmin(diffs, axis=1)
        return codes

class ManualSQ8:
    def __init__(self, d, weights):
        self.d = d
        self.min_val = weights['min_val']
        self.max_val = weights['max_val']

    def quantize(self, X):
        X_scaled = (X - self.min_val) / (self.max_val - self.min_val)
        return (np.clip(X_scaled, 0, 1) * 255).astype(np.uint8)

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
            logger.info("Downloading latest weights from Supabase ('models' bucket)...")
            supabase.download_file("models", "model_weights.pkl", weights_path)
            logger.info("✅ SUCCESS: Latest weights downloaded.")
        except Exception as e:
            logger.warning(f"Could not download weights from Supabase: {e}. Checking local...")

    if not os.path.exists(weights_path):
        logger.error(f"Weights file {weights_path} not found locally or on cloud!")
        return

    with open(weights_path, "rb") as f:
        weights = pickle.load(f)

    client = QdrantClient(url=Q_URL, api_key=Q_KEY)
    
    # Initialize models
    sq8 = ManualSQ8(d=768, weights=weights['sq8'])
    pq = ManualPQ(d=768, m=32, weights=weights['pq'])
    arq = TurboQuantProd(d=768, b=4, weights=weights['arq'])

    collections = {
        "vector_sq8": models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)),
        "vector_pq": models.ProductQuantization(product=models.ProductQuantizationConfig(compression=models.CompressionRatio.X32, always_ram=True)),
        "vector_arq": models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True))
    }

    for name, q_config in collections.items():
        if client.collection_exists(name):
            logger.info(f"Recreating collection {name}...")
            client.delete_collection(name)
        
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE, on_disk=True),
            on_disk_payload=True,
            quantization_config=q_config
        )

    logger.info("Starting batch re-quantization from 'vector_raw'...")
    
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
        
        # 1. SQ8 Upsert
        codes_sq8 = sq8.quantize(embeddings)
        sq8_points = []
        for i, p in enumerate(points):
            payload = chunks_payload[i].copy()
            payload["codes"] = codes_sq8[i].tolist()
            sq8_points.append(models.PointStruct(id=p.id, vector=p.vector, payload=payload))
        client.upsert("vector_sq8", sq8_points)

        # 2. PQ Upsert
        codes_pq = pq.quantize(embeddings)
        pq_points = []
        for i, p in enumerate(points):
            payload = chunks_payload[i].copy()
            payload["codes"] = codes_pq[i].tolist()
            pq_points.append(models.PointStruct(id=p.id, vector=p.vector, payload=payload))
        client.upsert("vector_pq", pq_points)

        # 3. ARQ Upsert
        idx, qjl, gamma = arq.quantize_batch(embeddings)
        arq_points = []
        for i, p in enumerate(points):
            payload = chunks_payload[i].copy()
            payload.update({
                "idx": idx[i].tolist(),
                "qjl": qjl[i].tolist(),
                "gamma": float(gamma[i]),
                "orig_norm": float(np.linalg.norm(embeddings[i]))
            })
            arq_points.append(models.PointStruct(id=p.id, vector=p.vector, payload=payload))
        client.upsert("vector_arq", arq_points)

        processed_count += len(points)
        logger.info(f"Processed {processed_count} points...")

        if scroll_token is None:
            break

    logger.info(f"✅ SUCCESS: Re-quantized {processed_count} points into all collections.")

if __name__ == "__main__":
    main()
