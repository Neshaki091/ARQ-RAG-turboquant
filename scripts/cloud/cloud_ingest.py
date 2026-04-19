import os
import json
import logging
import httpx
import numpy as np
import fitz  # PyMuPDF
from supabase import create_client, Client
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger("CloudIngest")

# --- Cloud Context Managers ---


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
                vectors_config=models.VectorParams(
                    size=self.dimension, 
                    distance=models.Distance.COSINE,
                    on_disk=True # Enable on-disk storage for vectors
                ),
                on_disk_payload=True, # Enable on-disk storage for payload
                quantization_config=quantization,
                hnsw_config=hnsw
            )

    def upsert(self, name, chunks, embeddings, extra_payloads=None):
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            payload = {
                "file": chunk.get("file"),
                "chunk_id": chunk.get("chunk_id"),
                "content": chunk.get("content"),
                "topic": chunk.get("topic", "General") # Thêm nhãn topic vào payload Qdrant
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
    parser = argparse.ArgumentParser(description="Cloud Ingestion Script with Partitioning Support")
    parser.add_argument("--total_parts", type=int, default=1, help="Total number of partitions (runners)")
    parser.add_argument("--part_index", type=int, default=0, help="Index of the current partition (0-indexed)")
    parser.add_argument("--limit", type=int, default=0, help="Max papers to process in this runner (0 for all)")
    args = parser.parse_args()

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

    all_papers = supabase.get_pending_papers()
    if not all_papers:
        logger.info("No pending papers to embed.")
        return

    # Partitioning Logic
    logger.info(f"Total pending papers: {len(all_papers)}")
    papers = all_papers[args.part_index::args.total_parts]
    
    if args.limit > 0:
        papers = papers[:args.limit]
        
    logger.info(f"Runner {args.part_index}/{args.total_parts} assigned {len(papers)} papers.")


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
                    "content": content,
                    "topic": paper.get("topic", "General") # Lấy topic từ metadata bài báo
                })

            # Get Embeddings
            embeddings = get_embeddings_batch([c['content'] for c in chunks], O_URL)
            emb_array = np.array(embeddings, dtype='float32')

            # 5 Models Ingestion -> Optimized to 4 (Adaptive uses RAW)
            
            # 1. RAW (Used by both Standard and Adaptive RAG)
            vector_store.ensure_collection("vector_raw")
            vector_store.upsert("vector_raw", chunks, embeddings)
            logger.info("   - RAW collection updated (Standard & Adaptive).")


            # NOTE: SQ8, PQ, and ARQ ingestion is disabled here to avoid fragmented training.
            # These collections will be populated via a global re-quantization script
            # after enough raw data has been collected.

            # Mark as embedded
            supabase.update_paper_status(paper_id, True)
            logger.info(f"✅ FINISHED: {paper_id} | Total chunks: {len(chunks)}")
            logger.info("-" * 40)

        except Exception as e:
            logger.error(f"Failed to process {paper_id}: {e}")

if __name__ == "__main__":
    main()
