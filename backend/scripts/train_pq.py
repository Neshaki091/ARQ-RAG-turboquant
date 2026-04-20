"""
train_pq.py
===========
Script huấn luyện ProductQuantizer và upload codebook lên Supabase.

PHẢI chạy TRƯỚC ingest_local.py!

Quy trình:
  1. Lấy sample vectors từ Qdrant vector_raw (hoặc embed PDF trực tiếp)
  2. Huấn luyện K-Means (faiss-cpu) trên sub-spaces
  3. Upload codebook (ProductQuantizer) lên Supabase bucket
  4. Thống kê: codebook size, compression ratio, reconstruction error

Sau khi train:
  - codebook lưu tại Supabase: pq/centroids.pkl
  - Mọi handler sẽ tải từ đây khi start

Usage:
  python scripts/train_pq.py
  python scripts/train_pq.py --M 8 --K 256 --sample-size 50000
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.embed import OllamaEmbedder
from shared.supabase_client import SupabaseManager
from shared.vector_store import QdrantManager, COLLECTION_NAMES
from models.rag_pq.quantizer import ProductQuantizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("train_pq")

PQ_CODEBOOK_PATH = "pq/centroids.pkl"


def sample_vectors_from_qdrant(
    qdrant: QdrantManager,
    collection_name: str,
    sample_size: int,
) -> np.ndarray:
    """
    Lấy sample vectors từ Qdrant để dùng cho K-Means training.

    Dùng Qdrant scroll API để lấy vectors theo batch.

    Args:
        qdrant: QdrantManager instance
        collection_name: Tên collection (vector_raw)
        sample_size: Số vectors tối đa cần lấy

    Returns:
        np.ndarray shape (N, D), float32
    """
    logger.info(f"Lấy {sample_size} vectors từ {collection_name}...")
    vectors = []
    offset = None
    batch_size = 500

    while len(vectors) < sample_size:
        records, next_offset = qdrant.client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_vectors=True,
            with_payload=False,
        )

        if not records:
            break

        for record in records:
            if record.vector:
                vectors.append(record.vector)
                if len(vectors) >= sample_size:
                    break

        offset = next_offset
        if offset is None:
            break

        logger.info(f"  Collected {len(vectors)}/{sample_size} vectors...")

    if not vectors:
        raise RuntimeError(
            f"Collection '{collection_name}' trống. "
            "Chạy ingest_local.py với vector_raw trước."
        )

    arr = np.array(vectors, dtype=np.float32)
    logger.info(f"Tổng cộng {len(arr)} training vectors, shape={arr.shape}")
    return arr


def compute_reconstruction_error(pq: ProductQuantizer, test_vectors: np.ndarray) -> float:
    """
    Tính MSE reconstruction error để đánh giá chất lượng codebook.

    MSE = mean(‖x - decode(encode(x))‖²)

    Args:
        pq: ProductQuantizer đã train
        test_vectors: np.ndarray (N, D), float32

    Returns:
        float — MSE score (thấp hơn = codebook tốt hơn)
    """
    sample = test_vectors[:500]  # Chỉ cần sample nhỏ để tính error
    codes = pq.encode(sample)
    reconstructed = pq.decode(codes)
    mse = float(np.mean((sample - reconstructed) ** 2))
    return mse


def main():
    parser = argparse.ArgumentParser(description="Train ProductQuantizer cho ARQ-RAG")
    parser.add_argument("--M", type=int, default=8, help="Số sub-quantizers (default: 8)")
    parser.add_argument("--K", type=int, default=256, help="Số centroids (default: 256)")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="Số vectors training (default: 100000)",
    )
    parser.add_argument(
        "--source",
        choices=["qdrant", "embed"],
        default="qdrant",
        help="Nguồn training data: 'qdrant' (từ collection) hoặc 'embed' (từ PDF)",
    )
    args = parser.parse_args()

    logger.info(
        f"Cấu hình PQ Training: M={args.M}, K={args.K}, "
        f"sample_size={args.sample_size:,}"
    )

    supabase = SupabaseManager()
    qdrant = QdrantManager()

    # ── Lấy training vectors ───────────────────────────────────────────
    if args.source == "qdrant":
        # Lấy từ Qdrant vector_raw (phải ingest raw trước)
        raw_collection = COLLECTION_NAMES["raw"]
        try:
            train_vectors = sample_vectors_from_qdrant(
                qdrant, raw_collection, args.sample_size
            )
        except RuntimeError as e:
            logger.error(str(e))
            sys.exit(1)
    else:
        logger.error("Chế độ 'embed' chưa được hỗ trợ trong script này.")
        sys.exit(1)

    # ── Huấn luyện ProductQuantizer ────────────────────────────────────
    logger.info(
        f"\n{'='*60}\n"
        f"Bắt đầu training ProductQuantizer\n"
        f"  D = {train_vectors.shape[1]} dims\n"
        f"  M = {args.M} sub-spaces\n"
        f"  K = {args.K} centroids each\n"
        f"  Training samples = {len(train_vectors):,}\n"
        f"  Compression ratio = {train_vectors.shape[1] * 4 // args.M}x\n"
        f"{'='*60}"
    )

    t_start = time.perf_counter()
    pq = ProductQuantizer(M=args.M, K=args.K, D=train_vectors.shape[1])
    pq.fit(train_vectors, verbose=True)
    train_time = time.perf_counter() - t_start

    # ── Đánh giá chất lượng codebook ──────────────────────────────────
    mse = compute_reconstruction_error(pq, train_vectors)
    logger.info(
        f"\n{'='*60}\n"
        f"✅ Training hoàn tất!\n"
        f"  Thời gian: {train_time:.1f}s\n"
        f"  Codebook size: {pq.codebook_size_mb:.2f} MB\n"
        f"  Compression ratio: {pq.compression_ratio:.0f}x\n"
        f"  Reconstruction MSE: {mse:.6f}\n"
        f"{'='*60}"
    )

    # ── Upload lên Supabase ────────────────────────────────────────────
    logger.info(f"Upload codebook lên Supabase ({PQ_CODEBOOK_PATH})...")
    success = supabase.upload_pickle(pq.to_dict(), PQ_CODEBOOK_PATH)

    if success:
        logger.info("🎉 Codebook upload thành công!")
        logger.info(
            "Tiếp theo: chạy scripts/ingest_local.py để index dữ liệu với PQ encoding"
        )
    else:
        logger.error("❌ Upload thất bại! Kiểm tra Supabase credentials.")
        sys.exit(1)


if __name__ == "__main__":
    main()
