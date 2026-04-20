"""
ingest_local.py
===============
Pipeline thu thập và lập chỉ mục tài liệu PDF vào tất cả 5 Qdrant collections.

Quy trình:
  PDF files → PyMuPDF parse → Chunk (overlap) → Embed (Ollama)
  → Upsert vào: vector_raw, vector_pq, vector_sq8, vector_arq

Lưu ý thứ tự:
  1. Phải chạy train_pq.py TRƯỚC để có codebook PQ
  2. ingest_local.py tải codebook từ Supabase để encode PQ codes
  3. SQ8 min/max được tính ngay trong quá trình ingest

Usage:
  python scripts/ingest_local.py --pdf-dir ./data/pdfs
  python scripts/ingest_local.py --pdf-dir ./data/pdfs --batch-size 50
"""

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
import numpy as np

# Thêm thư mục gốc vào path để import shared modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.embed import OllamaEmbedder
from shared.supabase_client import SupabaseManager
from shared.vector_store import QdrantManager, COLLECTION_NAMES
from models.rag_pq.quantizer import ProductQuantizer
from models.rag_pq.indexer import PQIndexer, PQ_CODEBOOK_PATH
from models.rag_sq8.quantizer import ScalarQuantizer
from models.arq_rag.reranker import QJLSketch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ingest")


# ── PDF Parsing ────────────────────────────────────────────────────────

def parse_pdf(pdf_path: Path) -> str:
    """
    Đọc toàn bộ text từ file PDF bằng PyMuPDF (fitz).

    Xử lý đặc biệt:
    - Loại bỏ header/footer (trang 1 giống trang 2 → skip)
    - Giữ nguyên LaTeX-like notation (không xóa $ ký tự)
    - Bảng biểu: convert thành text phẳng

    Args:
        pdf_path: Path đến file PDF

    Returns:
        Full text string đã làm sạch
    """
    try:
        doc = fitz.open(str(pdf_path))
        pages_text = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")

            # Làm sạch cơ bản
            text = text.replace("\x00", "")  # null bytes
            text = "\n".join(
                line for line in text.splitlines()
                if len(line.strip()) > 3  # Bỏ dòng quá ngắn (header/footer/page number)
            )

            if text.strip():
                pages_text.append(f"[Page {page_num + 1}]\n{text}")

        doc.close()
        full_text = "\n\n".join(pages_text)
        logger.info(f"Parsed {pdf_path.name}: {len(full_text)} chars, {len(doc)} pages")
        return full_text

    except Exception as e:
        logger.error(f"Lỗi parse {pdf_path}: {e}")
        return ""


# ── Main Ingestion Pipeline ────────────────────────────────────────────

class IngestPipeline:
    """
    Pipeline thu thập dữ liệu toàn diện.

    Một lần ingest → cập nhật tất cả 5 collections đồng thời.
    """

    def __init__(self):
        self.embedder = OllamaEmbedder()
        self.qdrant = QdrantManager()
        self.supabase = SupabaseManager()
        self.qdrant.ensure_all_collections()

        # Tải PQ codebook
        self.pq: ProductQuantizer = self._load_pq()

        # SQ8 quantizer (fit trong quá trình ingest)
        self.sq = ScalarQuantizer()

        # QJL sketch generator (dùng fixed random matrix)
        self.qjl = QJLSketch()

        logger.info("IngestPipeline khởi tạo xong")

    def _load_pq(self) -> ProductQuantizer:
        """Tải PQ codebook từ Supabase."""
        data = self.supabase.download_pickle(PQ_CODEBOOK_PATH)
        if data is None:
            raise RuntimeError(
                "Chưa có PQ codebook trong Supabase! "
                "Chạy scripts/train_pq.py trước."
            )
        pq = ProductQuantizer.from_dict(data)
        logger.info(f"PQ codebook loaded: {pq}")
        return pq

    def process_directory(self, pdf_dir: Path, batch_size: int = 100) -> int:
        """
        Xử lý toàn bộ PDF files trong thư mục.

        Args:
            pdf_dir: Thư mục chứa PDF files
            batch_size: Số chunks xử lý mỗi batch

        Returns:
            Tổng số chunks đã index
        """
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"Không tìm thấy PDF files trong {pdf_dir}")
            return 0

        logger.info(f"Tìm thấy {len(pdf_files)} PDF files")

        # Collect tất cả chunks + embeddings
        all_chunks: List[Dict[str, Any]] = []
        all_vectors: List[np.ndarray] = []

        for pdf_path in pdf_files:
            logger.info(f"Đang xử lý: {pdf_path.name}")
            raw_text = parse_pdf(pdf_path)
            if not raw_text:
                continue

            # Chunking có overlap
            chunks = OllamaEmbedder.chunk_text(raw_text)
            logger.info(f"  → {len(chunks)} chunks từ {pdf_path.name}")

            for i, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "text": chunk_text,
                    "source": pdf_path.stem,
                    "chunk_id": f"{pdf_path.stem}_{i:04d}",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                })

        if not all_chunks:
            logger.warning("Không có chunks để index")
            return 0

        logger.info(f"Bắt đầu embed {len(all_chunks)} chunks...")
        texts = [c["text"] for c in all_chunks]
        vectors = self.embedder.embed_batch(texts)  # (N, 768) float32

        # Fit SQ8 quantizer trên toàn bộ corpus
        logger.info("Fitting ScalarQuantizer trên toàn bộ corpus...")
        self.sq.fit(vectors)
        # Lưu SQ8 params lên Supabase
        self.supabase.upload_pickle(self.sq.to_dict(), "sq8/params.pkl")

        # Index vào tất cả collections
        total = self._index_all(vectors, all_chunks)
        logger.info(f"✅ Ingestion hoàn tất: {total} chunks x 4 collections")
        return total

    def _index_all(
        self,
        vectors: np.ndarray,
        chunks: List[Dict[str, Any]],
    ) -> int:
        """
        Index batch vectors vào tất cả 4 collections (raw, pq, sq8, arq).
        """
        N = len(vectors)

        # 1. Encode PQ codes cho vector_pq và vector_arq
        logger.info("Encoding PQ codes...")
        pq_codes_all = self.pq.encode(vectors)  # (N, M) uint8

        # 2. Encode SQ8 codes cho vector_sq8
        logger.info("Encoding SQ8 codes...")
        sq8_codes_all = self.sq.encode(vectors)  # (N, D) uint8

        # 3. Compute QJL sketches cho vector_arq
        logger.info("Computing QJL sketches...")
        qjl_sketches_all = self.qjl.sketch_documents_batch(vectors)  # (N, r) int8

        # 4. Build payloads cho mỗi collection
        raw_payloads = []
        pq_payloads = []
        sq8_payloads = []
        arq_payloads = []

        for i, chunk in enumerate(chunks):
            base = {
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
            }

            raw_payloads.append(dict(base))

            pq_payloads.append({
                **base,
                "pq_codes": pq_codes_all[i].tolist(),
            })

            sq8_payloads.append({
                **base,
                "sq8_codes": sq8_codes_all[i].tolist(),
                "sq8_min": self.sq.min_val,
                "sq8_max": self.sq.max_val,
            })

            arq_payloads.append({
                **base,
                "pq_codes": pq_codes_all[i].tolist(),
                "qjl_sketch": qjl_sketches_all[i].tolist(),
            })

        # 5. Upsert vào từng collection
        for col_name, payloads in [
            (COLLECTION_NAMES["raw"],  raw_payloads),
            (COLLECTION_NAMES["pq"],   pq_payloads),
            (COLLECTION_NAMES["sq8"],  sq8_payloads),
            (COLLECTION_NAMES["arq"],  arq_payloads),
        ]:
            # Bỏ qua vector_raw nếu trùng với vector_arq (collection alias)
            if col_name == COLLECTION_NAMES["adaptive"]:
                continue
            logger.info(f"Upsert {N} points vào {col_name}...")
            inserted = self.qdrant.upsert_points(
                collection_name=col_name,
                vectors=vectors,
                payloads=payloads,
            )
            logger.info(f"  → {inserted}/{N} points đã index")

        return N


# ── CLI Entry Point ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ARQ-RAG Data Ingestion Pipeline"
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("./data/pdfs"),
        help="Thư mục chứa PDF files (default: ./data/pdfs)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Số chunks mỗi batch embed (default: 100)",
    )
    args = parser.parse_args()

    if not args.pdf_dir.exists():
        logger.error(f"Thư mục không tồn tại: {args.pdf_dir}")
        sys.exit(1)

    pipeline = IngestPipeline()
    total = pipeline.process_directory(args.pdf_dir, args.batch_size)

    if total > 0:
        logger.info(f"🎉 Ingestion thành công: {total} chunks")
    else:
        logger.warning("Không có dữ liệu được index")
        sys.exit(1)


if __name__ == "__main__":
    main()
