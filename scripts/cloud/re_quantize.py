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

class ManualPQ:
    def __init__(self, d, m, weights, nbits=8):
        self.d = d
        self.m = m
        self.nbits = nbits
        self.ds = d // m
        self.centroids = weights['centroids']

    def quantize_batch(self, X):
        X = X.astype('float32')
        n_data = X.shape[0]
        codes = np.zeros((n_data, self.m), dtype='uint8')
        for i in range(self.m):
            sub_X = X[:, i*self.ds : (i+1)*self.ds]
            # Dùng faiss để tìm centroid gần nhất nhanh hơn
            import faiss
            index = faiss.IndexFlatL2(self.ds)
            index.add(self.centroids[i])
            _, I = index.search(sub_X, 1)
            codes[:, i] = I.reshape(-1)
        return codes

class ManualSQ8:
    def __init__(self, d, weights):
        self.d = d
        self.min_val = weights['min_val']
        self.max_val = weights['max_val']

    def quantize_batch(self, X):
        diff = self.max_val - self.min_val
        X_scaled = (X - self.min_val) / diff
        X_scaled = np.clip(X_scaled * 255, 0, 255).astype('uint8')
        return X_scaled

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
    
    # Initialize models
    arq = TurboQuantProd(d=768, b=4, weights=weights['arq'])
    pq = ManualPQ(d=768, m=32, weights=weights['pq'])
    sq8 = ManualSQ8(d=768, weights=weights['sq8'])

    # Define collections to sync
    target_collections = ["vector_adaptive", "vector_pq", "vector_sq8", "vector_arq"]

    for name in target_collections:
        logger.info(f"🔄 Preparing collection: {name}")
        
        # Cấu hình quantization tùy theo loại
        q_config = None
        if name == "vector_arq":
            q_config = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
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

    logger.info("Starting bulk re-quantization from 'vector_raw' for all models...")
    
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
        
        # 1. ARQ Quantization & Reconstruction
        idx, qjl, gamma = arq.quantize_batch(embeddings)
        reconstructed_arq = arq.reconstruct_batch(idx, qjl, gamma)

        # 2. PQ Codes
        pq_codes = pq.quantize_batch(embeddings)

        # 3. SQ8 Codes
        sq8_codes = sq8.quantize_batch(embeddings)

        # 4. Adaptive (Matryoshka - just uses first 256 dims if we simulate, but here we just copy)
        # In this project, 'adaptive' is treated as another float32 comparison but labeled differently
        
        # Prepare batches for upsert
        for name in target_collections:
            target_points = []
            for i, p in enumerate(points):
                new_payload = chunks_payload[i].copy()
                
                # Cập nhật payload đặc thù cho từng mô hình
                if name == "vector_arq":
                    new_payload.update({
                        "idx": idx[i].tolist(),
                        "qjl": qjl[i].tolist(),
                        "gamma": float(gamma[i]),
                        "orig_norm": float(np.linalg.norm(embeddings[i]))
                    })
                    # Native Engine chỉ dùng payload, ta xóa vector gốc khỏi Qdrant (thay bằng 0)
                    vector = [0.0] * 768 
                elif name == "vector_pq":
                    new_payload["codes"] = pq_codes[i].tolist()
                    vector = [0.0] * 768
                elif name == "vector_sq8":
                    new_payload["codes"] = sq8_codes[i].tolist()
                    vector = [0.0] * 768
                else: # vector_adaptive
                    vector = embeddings[i].tolist()

                target_points.append(models.PointStruct(id=p.id, vector=vector, payload=new_payload))
            
            client.upsert(name, target_points)


        processed_count += len(points)
        logger.info(f"Processed {processed_count} points...")

        if scroll_token is None:
            break

    logger.info(f"✅ SUCCESS: Re-quantized {processed_count} points for all 4 collections.")



if __name__ == "__main__":
    main()
