"""
quantizer.py
============
ProductQuantizer — Thuật toán nén vector lõi của RAG-PQ.

═══════════════════════════════════════════════════════════════
PRODUCT QUANTIZATION — LÝ THUYẾT
═══════════════════════════════════════════════════════════════

Vector gốc x ∈ ℝ^D được chia thành M sub-vectors:
    x = [x₁ | x₂ | ... | xₘ]   (mỗi xᵢ ∈ ℝ^(D/M))

Mỗi sub-space có một codebook Cᵢ = {c₁, c₂, ..., c_K}
    với K centroids (K=256 → mã hóa bằng 1 byte)

Encoding:   aᵢ = arg min_k ‖xᵢ - cᵢₖ‖²   (index centroid gần nhất)
Encoded vector: a = [a₁, a₂, ..., aₘ] ∈ {0,...,K-1}^M (dtype=uint8)

═══════════════════════════════════════════════════════════════
ASYMMETRIC DISTANCE COMPUTATION (ADC)
═══════════════════════════════════════════════════════════════

Query q ∈ ℝ^D (KHÔNG nén — giữ nguyên float32):
    qᵢ = q[i*(D/M) : (i+1)*(D/M)]  (sub-vector thứ i)

Precompute ADC table T ∈ ℝ^(M×K):
    T[i][k] = dot(qᵢ, Cᵢ[k])   (dot product với centroid)

Distance giữa q và encoded vector a:
    sim(q, a) ≈ Σᵢ T[i][aᵢ]     (chỉ M phép cộng!)

So sánh:
  - Exact dot product: D=768 phép nhân+cộng
  - ADC lookup: M=8 phép cộng (tra bảng)
  - Speedup lý thuyết: 768/8 = 96x (thực tế ~20-50x do cache)

═══════════════════════════════════════════════════════════════
THAM SỐ
═══════════════════════════════════════════════════════════════
  M = 8   sub-quantizers (768 / 8 = 96 dims mỗi sub-space)
  K = 256 centroids per sub-space (uint8, 1 byte per code)
  Training: K-Means với faiss-cpu (nhanh hơn sklearn ~10-50x)
"""

import io
import logging
import os
import pickle
from typing import Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# Cấu hình PQ từ .env (hoặc giá trị mặc định)
DEFAULT_M = int(os.getenv("PQ_M", "8"))       # số sub-spaces
DEFAULT_K = int(os.getenv("PQ_K", "256"))      # số centroids mỗi sub-space
VECTOR_DIM = int(os.getenv("PQ_VECTOR_DIM", "768"))  # chiều vector gốc


class ProductQuantizer:
    """
    Product Quantizer với Asymmetric Distance Computation.

    Attributes:
        M (int): Số sub-quantizers. Mỗi sub-space có D/M chiều.
        K (int): Số centroids mỗi sub-space. K=256 → dùng uint8.
        D (int): Chiều vector gốc (768).
        d_sub (int): Chiều mỗi sub-vector = D // M.
        codebook (np.ndarray): Centroids, shape (M, K, d_sub), float32.
        is_trained (bool): True sau khi fit() hoàn thành.
    """

    def __init__(self, M: int = DEFAULT_M, K: int = DEFAULT_K, D: int = VECTOR_DIM):
        if D % M != 0:
            raise ValueError(
                f"Chiều vector D={D} phải chia hết cho M={M}. "
                f"Hiện tại D/M = {D/M:.2f}. Thử M=6 (768/6=128) hoặc M=8 (768/8=96)."
            )
        if K > 256:
            raise ValueError(f"K={K} > 256. uint8 chỉ lưu được 0-255.")

        self.M = M
        self.K = K
        self.D = D
        self.d_sub = D // M  # chiều mỗi sub-vector

        # Codebook: sẽ được gán sau khi fit()
        self.codebook: Optional[np.ndarray] = None  # shape (M, K, d_sub)
        self.is_trained: bool = False

        logger.info(
            f"ProductQuantizer khởi tạo: D={D}, M={M}, K={K}, "
            f"d_sub={self.d_sub}, nén {D*4}B → {M}B "
            f"(compression ratio={D*4//M}x)"
        )

    # ------------------------------------------------------------------
    # Training: K-Means clustering với faiss-cpu
    # ------------------------------------------------------------------

    def fit(self, vectors: np.ndarray, verbose: bool = True) -> "ProductQuantizer":
        """
        Huấn luyện codebook bằng K-Means (faiss-cpu) trên tập training vectors.

        Quy trình:
          1. Tách mỗi vector thành M sub-vectors
          2. Chạy faiss.Kmeans trên sub-vectors của mỗi sub-space
          3. Lưu tâm cụm (centroids) vào self.codebook

        Tại sao dùng faiss thay vì sklearn KMeans?
          - faiss thực thi bằng C++/BLAS → nhanh hơn 10-50x
          - Hỗ trợ GPU acceleration (nếu có)
          - Xử lý tốt với dataset lớn (>100k vectors)

        Args:
            vectors: Ma trận float32, shape (N, D). N ≥ K để K-Means hội tụ.
            verbose: In tiến trình mỗi sub-space

        Returns:
            self (để chain: pq.fit(data).encode(data))
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        N, D = vectors.shape

        if D != self.D:
            raise ValueError(f"Chiều vector {D} không khớp với D={self.D}")
        if N < self.K:
            raise ValueError(
                f"Cần ít nhất {self.K} vectors để train {self.K} centroids. "
                f"Hiện có {N} vectors."
            )

        logger.info(f"Bắt đầu huấn luyện PQ codebook: {N} vectors, M={self.M} sub-spaces")
        codebook = np.zeros((self.M, self.K, self.d_sub), dtype=np.float32)

        for m in range(self.M):
            # Cắt sub-vectors của sub-space thứ m
            start = m * self.d_sub
            end = start + self.d_sub
            sub_vectors = np.ascontiguousarray(vectors[:, start:end])

            if verbose:
                logger.info(f"  Sub-space {m+1}/{self.M}: K-Means trên {N} vectors dim={self.d_sub}")

            # Huấn luyện K-Means bằng faiss
            # niter=20: số vòng lặp (đủ hội tụ cho embedding space)
            # nredo=3: restart K-Means 3 lần, lấy kết quả tốt nhất
            kmeans = faiss.Kmeans(
                d=self.d_sub,
                k=self.K,
                niter=20,
                nredo=3,
                verbose=False,
                seed=42,
            )
            kmeans.train(sub_vectors)

            # Lưu centroids vào codebook
            codebook[m] = kmeans.centroids  # shape (K, d_sub)

        self.codebook = codebook
        self.is_trained = True
        logger.info(
            f"Huấn luyện hoàn tất. Codebook size: {codebook.nbytes / 1024:.1f} KB"
        )
        return self

    # ------------------------------------------------------------------
    # Encoding: Vector → PQ codes (uint8)
    # ------------------------------------------------------------------

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode vectors thành mảng PQ codes (uint8).

        Với mỗi vector x và sub-space m:
            code[m] = arg min_k ‖x[m*d_sub:(m+1)*d_sub] - codebook[m][k]‖²

        Tận dụng numpy broadcasting để TRÁNH vòng lặp for trên vectors:
          sub_vecs shape: (N, d_sub)
          centroids shape: (K, d_sub)
          distances shape: (N, K)  — broadcast cực nhanh

        Args:
            vectors: np.ndarray shape (N, D) hoặc (D,), float32

        Returns:
            np.ndarray shape (N, M) hoặc (M,), dtype=uint8
        """
        self._check_trained()
        single = vectors.ndim == 1
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        N = vectors.shape[0]
        codes = np.zeros((N, self.M), dtype=np.uint8)

        for m in range(self.M):
            start = m * self.d_sub
            end = start + self.d_sub
            sub_vecs = vectors[:, start:end]  # (N, d_sub)
            centroids = self.codebook[m]       # (K, d_sub)

            # Tính khoảng cách L2 squared bằng numpy broadcasting (vectorized, không for loop)
            # ‖x - c‖² = ‖x‖² - 2x·c + ‖c‖²
            x_sq = np.sum(sub_vecs ** 2, axis=1, keepdims=True)   # (N, 1)
            c_sq = np.sum(centroids ** 2, axis=1)                  # (K,)
            xc   = sub_vecs @ centroids.T                           # (N, K)
            dists = x_sq + c_sq - 2 * xc                           # (N, K)

            codes[:, m] = np.argmin(dists, axis=1).astype(np.uint8)

        return codes[0] if single else codes

    # ------------------------------------------------------------------
    # Decoding: PQ codes → Approximate vector (float32)
    # ------------------------------------------------------------------

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Tái tạo xấp xỉ vector từ PQ codes.
        Kết quả là vector đã mất thông tin (lossy), dùng cho debug/visualization.

        Args:
            codes: np.ndarray shape (N, M) hoặc (M,), dtype=uint8

        Returns:
            np.ndarray shape (N, D) hoặc (D,), dtype=float32
        """
        self._check_trained()
        single = codes.ndim == 1
        codes = np.atleast_2d(np.asarray(codes, dtype=np.int32))
        N = codes.shape[0]
        reconstructed = np.zeros((N, self.D), dtype=np.float32)

        for m in range(self.M):
            start = m * self.d_sub
            end = start + self.d_sub
            # Tra codebook theo index — fancy indexing numpy
            reconstructed[:, start:end] = self.codebook[m][codes[:, m]]

        return reconstructed[0] if single else reconstructed

    # ------------------------------------------------------------------
    # ADC: Asymmetric Distance Computation
    # ------------------------------------------------------------------

    def compute_adc_table(self, query: np.ndarray) -> np.ndarray:
        """
        Tính bảng ADC cho một query vector — TÍNH 1 LẦN cho mỗi query.

        ADC table T ∈ ℝ^(M×K):
            T[m][k] = dot(query_sub_m, codebook[m][k])

        Tại sao dot product thay vì L2 distance?
          - Vectors đã được L2-normalize khi embed (embed.py)
          - Với unit vectors: cosine_sim = dot_product
          - dot product nhanh hơn L2 (bỏ -2xc term, tránh sqrt)

        Args:
            query: np.ndarray shape (D,), float32 — KHÔNG nén

        Returns:
            np.ndarray shape (M, K), float32 — ADC lookup table
        """
        self._check_trained()
        query = np.asarray(query, dtype=np.float32)

        if query.shape != (self.D,):
            raise ValueError(f"Query shape {query.shape} != ({self.D},)")

        # (M, K) = dot(query sub-vectors, codebook)
        # Tối ưu hoàn toàn bằng numpy matmul, không dùng vòng lặp Python
        adc_table = np.zeros((self.M, self.K), dtype=np.float32)
        for m in range(self.M):
            start = m * self.d_sub
            end = start + self.d_sub
            q_sub = query[start:end]          # (d_sub,)
            adc_table[m] = self.codebook[m] @ q_sub  # (K,) = (K, d_sub) @ (d_sub,)

        return adc_table  # (M, K)

    def adc_score_batch(
        self, adc_table: np.ndarray, codes_batch: np.ndarray
    ) -> np.ndarray:
        """
        Tính approximate similarity scores cho batch PQ codes bằng ADC.

        Đây là bước Reranking cực nhanh — chỉ dùng M phép cộng mỗi vector.

        sim(q, aᵢ) ≈ Σₘ ADC_table[m][aᵢₘ]

        Triển khai vectorized hoàn toàn = không dùng for loop Python:
          - codes_batch shape: (N, M)
          - adc_table shape:   (M, K)
          - Kết quả shape:     (N,)

        Kỹ thuật: fancy indexing + sum reduction
          scores = adc_table[range(M), codes_batch].sum(axis=1)
          → Với mỗi hàng i: sum(adc_table[m, codes_batch[i,m]] for m in range(M))

        Args:
            adc_table:   np.ndarray shape (M, K), float32
            codes_batch: np.ndarray shape (N, M), uint8

        Returns:
            np.ndarray shape (N,), float32 — similarity scores (cao hơn = tốt hơn)
        """
        if codes_batch.ndim == 1:
            codes_batch = codes_batch.reshape(1, -1)

        codes_batch = np.asarray(codes_batch, dtype=np.int32)
        N, M = codes_batch.shape

        # Vectorized lookup + sum — KHÔNG for loop
        # adc_table[m_indices, code_values] shape: (N, M)
        m_indices = np.arange(M)  # [0, 1, 2, ..., M-1]
        scores = adc_table[m_indices, codes_batch].sum(axis=1)  # (N,)

        return scores.astype(np.float32)

    # ------------------------------------------------------------------
    # Serialization: lưu/tải codebook
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize ProductQuantizer thành dict để pickle."""
        self._check_trained()
        return {
            "M": self.M,
            "K": self.K,
            "D": self.D,
            "d_sub": self.d_sub,
            "codebook": self.codebook,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProductQuantizer":
        """Khôi phục ProductQuantizer từ dict (được download từ Supabase)."""
        pq = cls(M=data["M"], K=data["K"], D=data["D"])
        pq.codebook = data["codebook"]
        pq.is_trained = True
        logger.info(
            f"ProductQuantizer loaded: M={pq.M}, K={pq.K}, D={pq.D}, "
            f"codebook shape={pq.codebook.shape}"
        )
        return pq

    def save_local(self, path: str) -> None:
        """Lưu codebook ra file local (dùng trong dev/debug)."""
        with open(path, "wb") as f:
            pickle.dump(self.to_dict(), f)
        logger.info(f"Đã lưu ProductQuantizer vào {path}")

    @classmethod
    def load_local(cls, path: str) -> "ProductQuantizer":
        """Tải codebook từ file local."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        if not self.is_trained or self.codebook is None:
            raise RuntimeError(
                "ProductQuantizer chưa được huấn luyện! Gọi fit() hoặc from_dict() trước."
            )

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return (
            f"ProductQuantizer(M={self.M}, K={self.K}, D={self.D}, "
            f"d_sub={self.d_sub}, status={status})"
        )

    # ------------------------------------------------------------------
    # Compression ratio info
    # ------------------------------------------------------------------

    @property
    def compression_ratio(self) -> float:
        """Tỉ lệ nén: số byte vector gốc / số byte sau PQ encode."""
        bytes_original = self.D * 4   # float32 = 4 bytes
        bytes_encoded = self.M        # 1 byte per sub-quantizer (uint8)
        return bytes_original / bytes_encoded

    @property
    def codebook_size_mb(self) -> float:
        """Kích thước codebook (MB): M × K × d_sub × 4 bytes."""
        if self.codebook is None:
            return 0.0
        return self.codebook.nbytes / (1024 * 1024)
