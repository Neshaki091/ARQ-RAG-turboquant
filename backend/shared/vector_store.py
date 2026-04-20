"""
vector_store.py
---------------
QdrantManager: driver tập trung cho mọi thao tác với Qdrant.

Quản lý 5 collections tương ứng 5 variants thuật toán:
  - vector_raw    : Float32, không nén (baseline)
  - vector_pq     : Product Quantization codes (uint8 payload)
  - vector_sq8    : Scalar Quantization 8-bit
  - vector_arq    : ARQ = Float32 + QJL metadata
  - vector_adaptive: Adaptive RAG (dùng lại collection raw hoặc arq)

Lưu ý thiết kế:
- Qdrant lưu Float32 vectors trong collection (để tận dụng HNSW index)
- PQ codes và metadata lưu trong `payload` của mỗi point
- Reranking ADC được thực hiện ở tầng Python (backend), không phải Qdrant
"""

import os
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

VECTOR_DIM: int = int(os.getenv("PQ_VECTOR_DIM", "768"))

# Tên 5 collections
COLLECTION_NAMES = {
    "raw": "vector_raw",
    "pq": "vector_pq",
    "sq8": "vector_sq8",
    "arq": "vector_arq",
    "adaptive": "vector_raw",  # Adaptive dùng lại raw collection
}


class QdrantManager:
    """
    Wrapper tập trung cho Qdrant client.

    Mỗi collection dùng chung vector dimension (768) và HNSW index,
    nhưng payload của mỗi point sẽ khác nhau tùy thuật toán:

    - raw: payload = {text, source, chunk_id}
    - pq:  payload = {text, source, chunk_id, pq_codes: List[int]}
    - sq8: payload = {text, source, chunk_id, sq8_min, sq8_max}
    - arq: payload = {text, source, chunk_id, qjl_sketch: List[float]}
    """

    def __init__(self, qdrant_url: Optional[str] = None):
        url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY") or None
        self.client = QdrantClient(url=url, api_key=api_key)
        logger.info(f"QdrantManager kết nối tới: {url}")

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self, collection_name: str) -> bool:
        """
        Tạo collection nếu chưa tồn tại.
        Dùng Cosine distance để nhất quán với L2-normalized vectors.

        Args:
            collection_name: Tên collection Qdrant

        Returns:
            True nếu collection đã sẵn sàng
        """
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if collection_name not in collections:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=VECTOR_DIM,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Đã tạo collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Lỗi ensure_collection({collection_name}): {e}")
            return False

    def ensure_all_collections(self) -> None:
        """Khởi tạo tất cả 5 collections khi startup."""
        created = set()
        for name in COLLECTION_NAMES.values():
            if name not in created:
                self.ensure_collection(name)
                created.add(name)

    # ------------------------------------------------------------------
    # Insert / Upsert
    # ------------------------------------------------------------------

    def upsert_points(
        self,
        collection_name: str,
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        batch_size: int = 100,
    ) -> int:
        """
        Upsert batch vectors vào Qdrant collection.

        Args:
            collection_name: Tên collection
            vectors: np.ndarray shape (N, 768), float32
            payloads: List N dicts chứa metadata của mỗi point
            ids: List N UUID strings (tự tạo nếu None)
            batch_size: Số point mỗi batch insert

        Returns:
            Số point đã insert thành công
        """
        if len(vectors) != len(payloads):
            raise ValueError(f"vectors ({len(vectors)}) và payloads ({len(payloads)}) phải có cùng độ dài")

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

        total_inserted = 0
        for start in range(0, len(vectors), batch_size):
            end = min(start + batch_size, len(vectors))
            batch_vectors = vectors[start:end]
            batch_payloads = payloads[start:end]
            batch_ids = ids[start:end]

            points = [
                PointStruct(
                    id=str(uuid.UUID(pid)) if isinstance(pid, str) else pid,
                    vector=vec.tolist(),
                    payload=payload,
                )
                for pid, vec, payload in zip(batch_ids, batch_vectors, batch_payloads)
            ]

            try:
                self.client.upsert(collection_name=collection_name, points=points)
                total_inserted += len(points)
                logger.debug(f"Upserted {total_inserted}/{len(vectors)} points vào {collection_name}")
            except Exception as e:
                logger.error(f"Lỗi upsert batch [{start}:{end}]: {e}")

        return total_inserted

    # ------------------------------------------------------------------
    # Search / Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        limit: int = 40,
        score_threshold: Optional[float] = None,
        payload_filter: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredPoint]:
        """
        Tìm kiếm các vectors gần nhất trong collection.

        Args:
            collection_name: Tên collection
            query_vector: np.ndarray shape (768,), float32
            limit: Số ứng viên tối đa trả về
            score_threshold: Ngưỡng score tối thiểu (optional)
            payload_filter: Lọc theo payload field (optional)

        Returns:
            List ScoredPoint từ Qdrant, sắp xếp theo score giảm dần
        """
        qdrant_filter = None
        if payload_filter:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in payload_filter.items()
            ]
            qdrant_filter = Filter(must=conditions)

        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector.tolist(),
                limit=limit,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            return results
        except Exception as e:
            logger.error(f"Lỗi search trong {collection_name}: {e}")
            return []

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Trả về thông tin collection: số points, config, v.v."""
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": str(info.status),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_all_collections_info(self) -> List[Dict[str, Any]]:
        """Trả về info của tất cả collections."""
        return [
            self.get_collection_info(name)
            for name in set(COLLECTION_NAMES.values())
        ]
