"""
quantizer.py — Scalar Quantization 8-bit
==========================================
ScalarQuantizer: nén vector từ float32 → uint8 per dimension.

Nguyên lý SQ8:
  Với mỗi chiều d của vector:
    quantized[d] = round((x[d] - min_val) / (max_val - min_val) * 255)

  - min_val, max_val: computed trên toàn bộ training corpus
  - Nén: 768 × 4 bytes → 768 × 1 byte = 4x nhỏ hơn
  - Decode: x[d] ≈ quantized[d] / 255 * (max_val - min_val) + min_val

So sánh với PQ:
  - SQ8: đơn giản, không cần training, nén 4x
  - PQ:  phức tạp, cần training K-Means, nén 384x (M=8, K=256)
  - SQ8 mất ít thông tin hơn PQ per dimension, nhưng nén ít hơn nhiều
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ScalarQuantizer:
    """
    Scalar Quantizer 8-bit: per-dimension min-max normalization → uint8.

    Ưu điểm:
      - Không cần training (chỉ cần tính min/max)
      - Nén 4x: float32 (4 bytes/dim) → uint8 (1 byte/dim)
      - Decode nhanh: phép nhân + cộng đơn giản

    Nhược điểm so với PQ:
      - Nén ít hơn (4x vs 384x)
      - Không có ADC table → không có reranking hiệu quả
    """

    def __init__(self):
        self.min_val: float = 0.0
        self.max_val: float = 1.0
        self.is_fitted: bool = False

    def fit(self, vectors: np.ndarray) -> "ScalarQuantizer":
        """
        Tính min/max toàn cục từ tập training vectors.

        Args:
            vectors: np.ndarray shape (N, D), float32

        Returns:
            self
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        self.min_val = float(vectors.min())
        self.max_val = float(vectors.max())
        self.is_fitted = True

        range_val = self.max_val - self.min_val
        logger.info(
            f"ScalarQuantizer fitted: min={self.min_val:.4f}, "
            f"max={self.max_val:.4f}, range={range_val:.4f}"
        )
        return self

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode float32 vectors → uint8 (0-255).

        Args:
            vectors: np.ndarray shape (N, D) hoặc (D,), float32

        Returns:
            np.ndarray cùng shape, dtype=uint8
        """
        if not self.is_fitted:
            raise RuntimeError("ScalarQuantizer chưa được fit()")

        vectors = np.asarray(vectors, dtype=np.float32)
        range_val = self.max_val - self.min_val

        if range_val == 0:
            return np.zeros_like(vectors, dtype=np.uint8)

        # Clip để tránh overflow ngoài [min, max]
        clipped = np.clip(vectors, self.min_val, self.max_val)
        normalized = (clipped - self.min_val) / range_val  # [0, 1]
        quantized = np.round(normalized * 255).astype(np.uint8)
        return quantized

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode uint8 codes → approximate float32 vectors.

        Args:
            codes: np.ndarray dtype=uint8

        Returns:
            np.ndarray dtype=float32
        """
        if not self.is_fitted:
            raise RuntimeError("ScalarQuantizer chưa được fit()")

        codes = np.asarray(codes, dtype=np.float32)
        range_val = self.max_val - self.min_val
        return (codes / 255.0) * range_val + self.min_val

    def encode_and_score(
        self, query: np.ndarray, codes_batch: np.ndarray
    ) -> np.ndarray:
        """
        Tính cosine similarity giữa query float32 và batch SQ8 codes.
        Decode codes trước rồi dot product.

        Args:
            query: np.ndarray shape (D,), float32
            codes_batch: np.ndarray shape (N, D), uint8

        Returns:
            np.ndarray shape (N,), float32 — similarity scores
        """
        decoded = self.decode(codes_batch)  # (N, D) float32
        # Cosine similarity = dot product (vì đã L2-normalize khi embed)
        scores = decoded @ query  # (N,)
        return scores.astype(np.float32)

    def to_dict(self) -> dict:
        return {"min_val": self.min_val, "max_val": self.max_val}

    @classmethod
    def from_dict(cls, data: dict) -> "ScalarQuantizer":
        sq = cls()
        sq.min_val = data["min_val"]
        sq.max_val = data["max_val"]
        sq.is_fitted = True
        return sq
