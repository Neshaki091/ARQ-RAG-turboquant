"""
embed.py
--------
OllamaEmbedder: giao tiếp với Ollama server để tạo vector nhúng
từ mô hình nomic-embed-text-v1.5 (chiều vector: 768).

Hỗ trợ:
- Nhúng đơn lẻ (query embedding)
- Nhúng batch (document embedding khi ingest)
- Chunking văn bản với overlapping
"""

import os
import re
import logging
from typing import List, Optional

import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Cấu hình chunking từ .env
MAX_CHUNK_CHARS: int = int(os.getenv("MAX_CHUNK_CHARS", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text:v1.5")
VECTOR_DIM: int = int(os.getenv("PQ_VECTOR_DIM", "768"))


class OllamaEmbedder:
    """
    Client giao tiếp với Ollama API để tạo embedding vectors.

    Mô hình: nomic-embed-text-v1.5
    Output: vector 768 chiều (float32)

    Ưu điểm so với OpenAI Embedding:
    - Chạy hoàn toàn local (offline, không tính phí)
    - Tích hợp GPU qua Ollama Docker
    - Độ trễ thấp hơn khi cùng mạng nội bộ
    """

    def __init__(
        self,
        ollama_url: str = OLLAMA_URL,
        model: str = EMBED_MODEL,
        timeout: float = 60.0,
    ):
        self.api_url = f"{ollama_url.rstrip('/')}/api/embeddings"
        self.model = model
        self.timeout = timeout
        logger.info(f"OllamaEmbedder: model={model}, url={ollama_url}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Nhúng một đoạn văn bản đơn lẻ thành vector float32.

        Query embedding KHÔNG được nén (giữ nguyên float32) —
        đây là nguyên tắc cốt lõi của Asymmetric Distance Computation.

        Args:
            text: Văn bản cần nhúng (query hoặc document chunk)

        Returns:
            np.ndarray shape (768,), dtype=float32
        """
        if not text or not text.strip():
            logger.warning("embed_text: nhận được text rỗng, trả về zero vector")
            return np.zeros(VECTOR_DIM, dtype=np.float32)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    self.api_url,
                    json={"model": self.model, "prompt": text.strip()},
                )
                resp.raise_for_status()
                embedding = resp.json()["embedding"]
                vec = np.array(embedding, dtype=np.float32)

                # Normalize L2 (chuẩn hóa để cosine similarity = dot product)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm

                return vec

        except httpx.TimeoutException:
            logger.error(f"Timeout khi gọi Ollama embed ({self.timeout}s)")
            raise
        except Exception as e:
            logger.error(f"Lỗi embed_text: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Nhúng danh sách văn bản.

        Args:
            texts: Danh sách các đoạn text

        Returns:
            np.ndarray shape (N, 768), dtype=float32
        """
        if not texts:
            return np.zeros((0, VECTOR_DIM), dtype=np.float32)

        vectors = []
        for i, text in enumerate(texts):
            try:
                vec = self.embed_text(text)
                vectors.append(vec)
                if (i + 1) % 50 == 0:
                    logger.info(f"Embedded {i+1}/{len(texts)} chunks")
            except Exception as e:
                logger.error(f"Lỗi embed chunk {i}: {e}. Dùng zero vector.")
                vectors.append(np.zeros(VECTOR_DIM, dtype=np.float32))

        return np.vstack(vectors)

    # ------------------------------------------------------------------
    # Chunking: chia văn bản dài thành các đoạn nhỏ có overlapping
    # ------------------------------------------------------------------

    @staticmethod
    def chunk_text(
        text: str,
        max_chars: int = MAX_CHUNK_CHARS,
        overlap: int = CHUNK_OVERLAP,
    ) -> List[str]:
        """
        Phân cắt văn bản dài thành các chunk có overlapping.

        Chiến lược:
        1. Làm sạch whitespace thừa
        2. Tách theo câu (dấu chấm, xuống dòng)
        3. Gom câu vào chunk không vượt quá max_chars
        4. Overlapping: mỗi chunk mới bắt đầu lại từ cuối chunk trước
           (giúp không mất ngữ cảnh tại ranh giới chunk)

        Args:
            text: Văn bản gốc (output của PyMuPDF)
            max_chars: Giới hạn ký tự mỗi chunk (default: 1000)
            overlap: Số ký tự overlap giữa 2 chunk liên tiếp (default: 100)

        Returns:
            List các chunk text
        """
        if not text or not text.strip():
            return []

        # Làm sạch: loại bỏ whitespace thừa, giữ lại xuống dòng đơn
        text = re.sub(r"\n{3,}", "\n\n", text.strip())
        text = re.sub(r" {2,}", " ", text)

        # Tách thành các câu (split theo dấu câu + xuống dòng)
        sentences = re.split(r"(?<=[.!?])\s+|\n\n", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Nếu câu đơn lẻ đã dài hơn max_chars, cắt cứng
            if len(sentence) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                # Cắt câu dài thành nhiều phần
                for start in range(0, len(sentence), max_chars - overlap):
                    part = sentence[start : start + max_chars]
                    if part.strip():
                        chunks.append(part.strip())
                continue

            # Thêm câu vào chunk hiện tại
            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk = (current_chunk + " " + sentence).strip()
            else:
                # Chunk hiện tại đã đầy — lưu lại
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Bắt đầu chunk mới với overlap từ cuối chunk trước
                overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
                current_chunk = (overlap_text + " " + sentence).strip()

        # Chunk cuối cùng
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks
