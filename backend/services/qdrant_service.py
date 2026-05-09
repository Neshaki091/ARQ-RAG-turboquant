from qdrant_client import QdrantClient
from qdrant_client.http import models
import os

class QdrantService:
    def __init__(self):
        # Sử dụng storage local (disk) để người dùng không cần chạy Docker Qdrant
        db_path = os.path.join("data", "qdrant_db")
        os.makedirs(db_path, exist_ok=True)
        self.client = QdrantClient(path=db_path)
        self.raw_col = "arq_rag_raw"
        self.tq_col = "arq_rag_tq"
        
        self._init_collections()

    def _init_collections(self):
        try:
            existing = [c.name for c in self.client.get_collections().collections]
            
            # Init Raw Collection
            if self.raw_col not in existing:
                self.client.create_collection(
                    collection_name=self.raw_col,
                    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
                )
            
            # Init TQ Collection (Chúng ta lưu TQ codes vào payload, nên vector config có thể để dummy hoặc rất nhỏ)
            if self.tq_col not in existing:
                self.client.create_collection(
                    collection_name=self.tq_col,
                    vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE) # Dummy vector
                )
        except Exception as e:
            print(f"Error initializing Qdrant: {e}")

    def upsert_raw(self, ids, vectors, payloads):
        self.client.upsert(
            collection_name=self.raw_col,
            points=models.Batch(ids=ids, vectors=vectors, payloads=payloads)
        )

    def upsert_tq(self, ids, payloads):
        # Lưu TQ codes vào payload của tq_col
        # Dùng dummy vector vì Qdrant yêu cầu phải có vector
        dummy_vectors = [[0.0] for _ in range(len(ids))]
        self.client.upsert(
            collection_name=self.tq_col,
            points=models.Batch(ids=ids, vectors=dummy_vectors, payloads=payloads)
        )

    def search_raw(self, query_vector, top_k=5):
        return self.client.search(
            collection_name=self.raw_col,
            query_vector=query_vector,
            limit=top_k
        )

    def get_by_ids(self, ids: list[int]):
        """
        Retrieve points from raw collection by integer IDs.
        """
        return self.client.retrieve(
            collection_name=self.raw_col,
            ids=ids
        )

    def delete_by_filename(self, filename: str):
        """
        Delete points from both collections by filename.
        """
        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=filename)
                )
            ]
        )
        self.client.delete(collection_name=self.raw_col, points_selector=filter)
        self.client.delete(collection_name=self.tq_col, points_selector=filter)

    def list_documents(self):
        """
        Get unique filenames from raw collection.
        """
        # Qdrant doesn't have a direct 'distinct' but we can scroll with payload and process
        res, _ = self.client.scroll(
            collection_name=self.raw_col,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        filenames = set([p.payload["source"] for p in res if "source" in p.payload])
        return list(filenames)

# Global instance
qdrant_service = QdrantService()
