import os
import json
import logging
import httpx
import numpy as np
import fitz  # PyMuPDF
import faiss
from supabase import create_client, Client
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger("CloudIngest")

# --- Simplified Quantization Models (Ported for Standalone) ---

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
    def __init__(self, d, b, random_state=None):
        self.d = d
        self.b = b
        if random_state is None:
            random_state = np.random.RandomState(42)
        self.tq_mse = TurboQuantMSE(d, b - 1, random_state=random_state)
        self.S = random_state.randn(d, d)
        self.alpha = np.sqrt(np.pi / 2.0) / d

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
        k_actual = min(self.k, n_data)
        if k_actual < 1: return
        self.centroids = []
        for i in range(self.m):
            sub_X = X[:, i*self.ds : (i+1)*self.ds]
            if n_data >= k_actual and k_actual >= 2:
                kmeans = faiss.Kmeans(d=self.ds, k=k_actual, niter=20, verbose=False)
                kmeans.train(sub_X)
                self.centroids.append(kmeans.centroids)
            else:
                padding = np.zeros((max(0, k_actual - n_data), self.ds), dtype='float32')
                fallback_centroids = np.vstack([sub_X, padding]) if n_data > 0 else np.zeros((k_actual, self.ds), dtype='float32')
                self.centroids.append(fallback_centroids)

    def quantize(self, X):
        N = X.shape[0]
        codes = np.zeros((N, self.m), dtype=np.uint8)
        for i in range(self.m):
            sub_X = X[:, i*self.ds : (i+1)*self.ds]
            diffs = np.linalg.norm(sub_X[:, np.newaxis, :] - self.centroids[i][np.newaxis, :, :], axis=2)
            codes[:, i] = np.argmin(diffs, axis=1)
        return codes

class ManualSQ8:
    def __init__(self, d):
        self.d = d
        self.min_val = None
        self.max_val = None

    def train(self, X):
        if X.shape[0] == 0:
            self.min_val = np.zeros(self.d)
            self.max_val = np.ones(self.d)
            return
        self.min_val = np.min(X, axis=0)
        self.max_val = np.max(X, axis=0)
        diff = self.max_val - self.min_val
        diff[diff == 0] = 1e-10
        self.max_val = self.min_val + diff

    def quantize(self, X):
        X_scaled = (X - self.min_val) / (self.max_val - self.min_val)
        return (X_scaled * 255).astype(np.uint8)

# --- Managers for Cloud Context ---

class CloudSupabase:
    def __init__(self, url, key):
        self.client = create_client(url, key)

    def get_pending_papers(self):
        res = self.client.table("papers").select("*").eq("is_embedded", False).execute()
        return res.data if res.data else []

    def update_paper_status(self, paper_id, status=True):
        self.client.table("papers").update({"is_embedded": status}).eq("id", paper_id).execute()

    def get_file_content(self, bucket, filename):
        return self.client.storage.from_(bucket).download(filename)

    def list_files(self, bucket: str = "papers"):
        try:
            all_files = []
            offset = 0
            limit = 100
            while True:
                res = self.client.storage.from_(bucket).list(options={
                    'limit': limit,
                    'offset': offset,
                    'sortBy': {'column': 'name', 'order': 'asc'}
                })
                if not res:
                    break
                
                names = [f['name'] for f in res if f['name'] != '.emptyFolderPlaceholder']
                all_files.extend(names)
                
                if len(res) < limit:
                    break
                offset += limit
            return all_files
        except Exception as e:
            logger.error(f"Error listing files in bucket: {e}")
            return []

    def reset_all_paper_status(self):
        # Update is_embedded=False for ALL papers where it's True
        self.client.table("papers").update({"is_embedded": False}).neq("is_embedded", False).execute()

class CloudVectorStore:
    def __init__(self, url, api_key):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.dimension = 768

    def ensure_collection(self, name, quantization=None, hnsw=None):
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE),
                quantization_config=quantization,
                hnsw_config=hnsw
            )

    def upsert(self, name, chunks, embeddings, extra_payloads=None):
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            payload = {
                "file": chunk.get("file"),
                "chunk_id": chunk.get("chunk_id"),
                "content": chunk.get("content")
            }
            if extra_payloads and i < len(extra_payloads):
                payload.update(extra_payloads[i])
            # Qdrant client expects a list. If it's already a list, use it. If numpy, convert.
            vec_to_send = vector.tolist() if hasattr(vector, "tolist") else vector

            # Generate a valid UUID based on the chunk_id (deterministic)
            # Qdrant requires IDs to be either 64-bit integers or UUID strings.
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{name}_{chunk['chunk_id']}"))
            
            points.append(models.PointStruct(id=point_id, vector=vec_to_send, payload=payload))
        
        self.client.upsert(collection_name=name, points=points)

# --- Embedding Helper ---

def get_embeddings_batch(texts, ollama_url):
    embeddings = []
    for text in texts:
        payload = {"model": "nomic-embed-text", "prompt": text}
        try:
            with httpx.Client(timeout=120.0) as client:
                res = client.post(f"{ollama_url}/api/embeddings", json=payload)
                embeddings.append(res.json()["embedding"])
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            embeddings.append(np.random.rand(768).tolist())
    return embeddings

# --- Core Logic ---

def extract_text(pdf_stream):
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def main():
    # Load Env
    S_URL = os.getenv("SUPABASE_URL")
    S_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    Q_URL = os.getenv("QDRANT_CLOUD_URL")
    Q_KEY = os.getenv("QDRANT_CLOUD_API_KEY")
    O_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    PURGE = os.getenv("PURGE_MODE", "false").lower() == "true"

    if not all([S_URL, S_KEY, Q_URL, Q_KEY]):
        logger.error("Missing critical environment variables!")
        return

    supabase = CloudSupabase(S_URL, S_KEY)
    vector_store = CloudVectorStore(Q_URL, Q_KEY)

    if PURGE:
        logger.info("Purge mode detected. Resetting all paper statuses to false...")
        supabase.reset_all_paper_status()
        # Optionally delete collections from Qdrant Cloud too?
        # For safety, let's just reset metadata status.

    papers = supabase.get_pending_papers()
    if not papers:
        logger.info("No pending papers to embed.")
        return

    logger.info("Fetching actual file list from bucket...")
    actual_files = supabase.list_files("papers")
    logger.info(f"Bucket contains {len(actual_files)} files.")

    logger.info(f"Found {len(papers)} pending papers metadata.")

    for paper in papers:
        paper_id = paper['id']
        
        # Resolve target_file by prefix matching paper_id
        # Crawler saves as {arxiv_id}_{safe_title}.pdf
        target_file = next((f for f in actual_files if f.startswith(f"{paper_id}_") or f == f"{paper_id}.pdf" or f == paper_id), None)
        
        if not target_file:
            logger.warning(f"⚠️ Could not find file for {paper_id} in Storage (Prefix match failed)")
            continue

        logger.info(f"Processing paper: {paper['title']} (File found: {target_file})")

        try:
            pdf_content = supabase.get_file_content("papers", target_file)
            if not pdf_content:
                logger.warning(f"Could not download {target_file}")
                continue

            text = extract_text(pdf_content)
            raw_chunks = chunk_text(text)
            
            chunks = []
            for idx, content in enumerate(raw_chunks):
                chunks.append({
                    "file": target_file,
                    "chunk_id": f"{paper_id}_{idx}",
                    "content": content
                })

            # Get Embeddings
            embeddings = get_embeddings_batch([c['content'] for c in chunks], O_URL)
            emb_array = np.array(embeddings, dtype='float32')

            # 5 Models Ingestion
            
            # 1. RAW
            vector_store.ensure_collection("vector_raw")
            vector_store.upsert("vector_raw", chunks, embeddings)
            logger.info("   - RAW collection updated.")

            # 2. Adaptive
            vector_store.ensure_collection("vector_adaptive")
            vector_store.upsert("vector_adaptive", chunks, embeddings)
            logger.info("   - Adaptive collection updated.")

            # 3. SQ8
            sq8 = ManualSQ8(d=768)
            sq8.train(emb_array)
            codes_sq8 = sq8.quantize(emb_array)
            vector_store.ensure_collection("vector_sq8", quantization=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
            ))
            vector_store.upsert("vector_sq8", chunks, embeddings, extra_payloads=[{"codes": c.tolist()} for c in codes_sq8])
            logger.info("   - SQ8 collection updated.")

            # 4. PQ
            pq = ManualPQ(d=768, m=32, nbits=8)
            pq.train(emb_array)
            codes_pq = pq.quantize(emb_array)
            vector_store.ensure_collection("vector_pq", quantization=models.ProductQuantization(
                product=models.ProductQuantizationConfig(compression=models.CompressionRatio.X32, always_ram=True)
            ))
            vector_store.upsert("vector_pq", chunks, embeddings, extra_payloads=[{"codes": c.tolist()} for c in codes_pq])
            logger.info("   - PQ collection updated.")

            # 5. ARQ (TurboQuant)
            tq_prod = TurboQuantProd(d=768, b=4)
            # Training centroids in cloud might be simplified or skipped if not many points
            # But let's try a simple version
            Y = np.dot(emb_array, tq_prod.tq_mse.Pi.T)
            y_flat = Y.flatten().reshape(-1, 1)
            km = faiss.Kmeans(d=1, k=8, niter=20)
            km.train(y_flat)
            tq_prod.tq_mse.centroids = np.sort(km.centroids.flatten())
            
            idx, qjl, gamma = tq_prod.quantize_batch(emb_array)
            extra_arq = []
            for i in range(len(embeddings)):
                extra_arq.append({
                    "idx": idx[i].tolist(),
                    "qjl": qjl[i].tolist(),
                    "gamma": float(gamma[i]),
                    "orig_norm": float(np.linalg.norm(emb_array[i]))
                })
            
            vector_store.ensure_collection("vector_arq", quantization=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
            ))
            vector_store.upsert("vector_arq", chunks, embeddings, extra_payloads=extra_arq)
            logger.info("   - ARQ collection updated.")

            # Mark as embedded
            supabase.update_paper_status(paper_id, True)
            logger.info(f"✅ FINISHED: {paper_id} | Total chunks: {len(chunks)}")
            logger.info("-" * 40)

        except Exception as e:
            logger.error(f"Failed to process {paper_id}: {e}")

if __name__ == "__main__":
    main()
