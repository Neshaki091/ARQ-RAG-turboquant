import os
import pickle
import numpy as np
import faiss
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger("GlobalTrain")

# --- Cloud Storage Manager ---

class CloudSupabase:
    def __init__(self, url, key):
        self.client = create_client(url, key)

    def upload_file(self, bucket, local_path, remote_path):
        with open(local_path, "rb") as f:
            return self.client.storage.from_(bucket).upload(remote_path, f, {"upsert": "true"})

    def download_file(self, bucket, remote_path, local_path):
        res = self.client.storage.from_(bucket).download(remote_path)
        with open(local_path, "wb") as f:
            f.write(res)

# --- Classes (Ported from cloud_ingest.py for consistency) ---

class TurboQuantMSE:
    def __init__(self, d, b, random_state=None):
        self.d = d
        self.b = b
        self.num_centroids = 2 ** b
        if random_state is None:
            random_state = np.random.RandomState(42)
        H = random_state.randn(d, d)
        Q, _ = np.linalg.qr(H)
        self.Pi = Q 
        self.centroids = np.zeros(self.num_centroids)

    def train(self, X):
        # Training centroids for projected residuals
        Y = np.dot(X, self.Pi.T)
        y_flat = Y.flatten().reshape(-1, 1)
        km = faiss.Kmeans(d=1, k=self.num_centroids, niter=20)
        km.train(y_flat)
        self.centroids = np.sort(km.centroids.flatten())
        logger.info(f"TurboQuantMSE trained. Centroids: {self.centroids}")

class TurboQuantProd:
    def __init__(self, d, b, random_state=None):
        self.d = d
        self.b = b
        if random_state is None:
            random_state = np.random.RandomState(42)
        self.tq_mse = TurboQuantMSE(d, b - 1, random_state=random_state)
        self.S = random_state.randn(d, d)
        self.alpha = np.sqrt(np.pi / 2.0) / d

    def train(self, X):
        self.tq_mse.train(X)
        logger.info("TurboQuantProd trained (MSE centroids learned).")

class ManualPQ:
    def __init__(self, d, m, nbits=8):
        self.d = d
        self.m = m
        self.nbits = nbits
        self.k = 2 ** nbits
        self.ds = d // m
        self.centroids = []

    def train(self, X):
        X = X.astype('float32')
        n_data = X.shape[0]
        self.centroids = []
        for i in range(self.m):
            sub_X = X[:, i*self.ds : (i+1)*self.ds]
            kmeans = faiss.Kmeans(d=self.ds, k=self.k, niter=20, verbose=False)
            kmeans.train(sub_X)
            self.centroids.append(kmeans.centroids)
        logger.info(f"ManualPQ trained for {self.m} subspaces.")

class ManualSQ8:
    def __init__(self, d):
        self.d = d
        self.min_val = None
        self.max_val = None

    def train(self, X):
        self.min_val = np.min(X, axis=0)
        self.max_val = np.max(X, axis=0)
        diff = self.max_val - self.min_val
        diff[diff == 0] = 1e-10
        self.max_val = self.min_val + diff
        logger.info("ManualSQ8 trained (Global Min/Max learned).")

def main():
    Q_URL = os.getenv("QDRANT_CLOUD_URL")
    Q_KEY = os.getenv("QDRANT_CLOUD_API_KEY")
    SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "20000"))

    if not all([Q_URL, Q_KEY]):
        logger.error("Missing Qdrant credentials!")
        return

    # Khởi tạo client với timeout lớn hơn để tránh lỗi ReadTimeout
    client = QdrantClient(url=Q_URL, api_key=Q_KEY, timeout=60.0)
    
    logger.info(f"Fetching {SAMPLE_SIZE} vectors from 'vector_raw' as a sample using iterative scroll...")
    
    # Lấy mẫu theo đợt (Iterative Scrolling) để tránh timeout
    all_sampled_points = []
    next_offset = None
    BATCH_FETCH = 1000  # Lấy 1000 cái mỗi lần
    
    while len(all_sampled_points) < SAMPLE_SIZE:
        limit_to_fetch = min(BATCH_FETCH, SAMPLE_SIZE - len(all_sampled_points))
        points, next_offset = client.scroll(
            collection_name="vector_raw",
            limit=limit_to_fetch,
            with_vectors=True,
            with_payload=False,
            offset=next_offset
        )
        
        if not points:
            break
            
        all_sampled_points.extend(points)
        logger.info(f"  Downloaded {len(all_sampled_points)}/{SAMPLE_SIZE} vectors...")
        
        if next_offset is None:
            break
    
    if not all_sampled_points:
        logger.error("No vectors found in 'vector_raw' to train on!")
        return
    
    embeddings = np.array([p.vector for p in all_sampled_points], dtype='float32')
    logger.info(f"Collected {len(embeddings)} vectors. Starting training...")

    # 1. SQ8
    sq8 = ManualSQ8(d=768)
    sq8.train(embeddings)

    # 2. PQ
    pq = ManualPQ(d=768, m=32, nbits=8)
    pq.train(embeddings)

    # 3. TurboQuant (ARQ)
    tq = TurboQuantProd(d=768, b=4)
    tq.train(embeddings)

    # Save weights
    weights = {
        "sq8": {"min_val": sq8.min_val, "max_val": sq8.max_val},
        "pq": {"centroids": pq.centroids},
        "arq": {
            "Pi": tq.tq_mse.Pi,
            "S": tq.S,
            "centroids": tq.tq_mse.centroids,
            "alpha": tq.alpha
        }
    }

    os.makedirs("backend/data", exist_ok=True)
    weights_path = "backend/data/model_weights.pkl"
    with open(weights_path, "wb") as f:
        pickle.dump(weights, f)
    
    logger.info(f"✅ Global weights saved locally to {weights_path}")

    # Upload to Supabase
    S_URL = os.getenv("SUPABASE_URL")
    S_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if S_URL and S_KEY:
        try:
            supabase = CloudSupabase(S_URL, S_KEY)
            logger.info("Uploading weights to Supabase Storage ('centroids' bucket)...")
            supabase.upload_file("centroids", weights_path, "model_weights.pkl")
            logger.info("✅ SUCCESS: Global weights uploaded to Cloud Storage.")
        except Exception as e:
            logger.error(f"Failed to upload weights: {e}")
    else:
        logger.warning("Supabase credentials missing, skipping cloud upload.")

if __name__ == "__main__":
    main()
