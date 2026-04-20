"""
indexer.py
==========
PQIndexer — Pipeline nhồi dữ liệu vào Qdrant collection `vector_pq`.

Quy trình:
  PDF/Text → Chunk → Embed (float32) → PQ Encode (uint8) → Qdrant

Thiết kế lưu trữ trong Qdrant:
  - vector field: float32 gốc (để Qdrant HNSW index hoạt động)
  - payload field "pq_codes": PQ codes dưới dạng List[int]
    (Qdrant không hỗ trợ uint8 array natively → convert List[int])

Tại sao lưu cả float32 lẫn PQ codes?
  - float32 trong Qdrant: để tận dụng HNSW approximate nearest neighbor
    (lấy candidate nhanh, không cần scan toàn bộ collection)
  - PQ codes trong payload: để ADC reranking tại tầng Python
    (chính xác hơn cosine similarity của Qdrant)
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from models.rag_pq.quantizer import ProductQuantizer
from shared.supabase_client import SupabaseManager
from shared.vector_store import QdrantManager

logger = logging.getLogger(__name__)

COLLECTION_NAME = "vector_pq"
PQ_CODEBOOK_PATH = "pq/centroids.pkl"  # Đường dẫn trong Supabase bucket


class PQIndexer:
    """
    Quản lý việc index vectors vào Qdrant collection `vector_pq`.

    Workflow:
      1. Tải ProductQuantizer đã train từ Supabase
      2. Encode vectors → PQ codes
      3. Upsert vào Qdrant (vector + payload có pq_codes)
    """

    def __init__(
        self,
        qdrant: QdrantManager,
        supabase: SupabaseManager,
        pq: Optional[ProductQuantizer] = None,
    ):
        self.qdrant = qdrant
        self.supabase = supabase
        self.pq = pq

        # Đảm bảo collection tồn tại
        self.qdrant.ensure_collection(COLLECTION_NAME)

    def load_quantizer(self) -> bool:
        """
        Tải ProductQuantizer đã train từ Supabase Storage.

        Returns:
            True nếu tải thành công, False nếu chưa có codebook.
        """
        codebook_data = self.supabase.download_pickle(PQ_CODEBOOK_PATH)
        if codebook_data is None:
            logger.warning(
                "Chưa có codebook PQ trong Supabase. "
                "Chạy scripts/train_pq.py trước khi ingest."
            )
            return False

        self.pq = ProductQuantizer.from_dict(codebook_data)
        logger.info(f"Đã tải ProductQuantizer: {self.pq}")
        return True

    def index_documents(
        self,
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
        batch_size: int = 200,
    ) -> int:
        """
        Index batch documents vào Qdrant `vector_pq`.

        Với mỗi document:
          - Lưu float32 vector → Qdrant sử dụng HNSW index
          - Encode → PQ codes → lưu vào payload["pq_codes"]

        Args:
            vectors: np.ndarray shape (N, 768), float32
                     Đây là embedding float32 gốc từ OllamaEmbedder.
            payloads: List N dicts, mỗi dict chứa text/source/chunk_id/...
            batch_size: Số points mỗi batch upsert

        Returns:
            Số points đã index thành công
        """
        if self.pq is None or not self.pq.is_trained:
            raise RuntimeError(
                "ProductQuantizer chưa được load. Gọi load_quantizer() trước."
            )

        N = len(vectors)
        logger.info(f"Bắt đầu index {N} vectors vào {COLLECTION_NAME}...")

        # Encode toàn bộ sang PQ codes
        # encode() là fully vectorized (numpy), không for loop
        pq_codes_all = self.pq.encode(vectors)  # shape (N, M), uint8

        # Gắn pq_codes vào payload của từng document
        enriched_payloads = []
        for i, (payload, codes) in enumerate(zip(payloads, pq_codes_all)):
            enriched = dict(payload)
            # Qdrant payload lưu List[int] thay vì numpy array
            enriched["pq_codes"] = codes.tolist()
            enriched_payloads.append(enriched)

        # Upsert vào Qdrant
        inserted = self.qdrant.upsert_points(
            collection_name=COLLECTION_NAME,
            vectors=vectors,
            payloads=enriched_payloads,
            batch_size=batch_size,
        )

        logger.info(
            f"Index hoàn tất: {inserted}/{N} points vào {COLLECTION_NAME}. "
            f"Codebook size: {self.pq.codebook_size_mb:.2f} MB, "
            f"Compression ratio: {self.pq.compression_ratio:.0f}x"
        )
        return inserted
