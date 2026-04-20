"""
reranker.py — ARQ Reranker (ADC + QJL TurboQuant)
===================================================
Kết hợp 2 kỹ thuật reranking:

1. ADC (Asymmetric Distance Computation) — từ Product Quantization
   sim_adc(q, a) ≈ Σₘ T[m][aₘ]
   (như trong rag_pq/quantizer.py)

2. QJL (Quantized Johnson-Lindenstrauss) sketch
   Ý tưởng từ bài báo TurboQuant:
   - Project vector vào không gian chiều thấp hơn bằng random matrix R
   - Sketch s = sign(R @ x) ∈ {-1, +1}^r   (r chiều, r << D)
   - Ước lượng dot(q, x) ≈ (π/4) * dot(q_sign, x_sign) * ‖q‖ * ‖x‖_approx
   - Sai số: E[|est - true|] = O(1/√r)

Tổng hợp score TurboQuant:
   score = α * adc_score + (1-α) * qjl_score
   α = 0.7 (ADC chiếm ưu thế, qjl bổ sung)

Tại sao QJL cải thiện ADC?
   - ADC bị sai lêch khi sub-space assignment sai centroid
   - QJL độc lập với PQ codebook → bù trừ lỗi của ADC
   - Kết hợp giảm phương sai tổng → Top-K chính xác hơn
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Tham số QJL
DEFAULT_QJL_DIM = int(os.getenv("QJL_DIM", "64"))   # r: chiều sketch
QJL_ALPHA = float(os.getenv("QJL_ALPHA", "0.7"))     # trọng số ADC vs QJL
QJL_MATRIX_PATH = "arq/qjl_matrix.pkl"               # Supabase bucket path


class QJLSketch:
    """
    Johnson-Lindenstrauss Sketch (Binary) cho TurboQuant reranking.

    Projection matrix R ∈ ℝ^(r×D):
      - Random Gaussian: R[i,j] ~ N(0, 1/r)
      - Sau đó sign() → binary sketch {-1, +1}

    Thuộc tính JL cơ bản:
      E[⟨Rq, Rx⟩] = ⟨q, x⟩
      Var ≈ 2⟨q,x⟩²/r  → giảm khi tăng r
    """

    def __init__(self, D: int = 768, r: int = DEFAULT_QJL_DIM, seed: int = 42):
        self.D = D
        self.r = r
        rng = np.random.RandomState(seed)

        # Ma trận projection: chia cho sqrt(r) để ổn định phương sai
        self.R: np.ndarray = rng.randn(r, D).astype(np.float32) / np.sqrt(r)
        logger.info(f"QJLSketch: D={D}, r={r}, R shape={self.R.shape}")

    def sketch_query(self, query: np.ndarray) -> np.ndarray:
        """
        Tính QJL projection của query (giữ dấu actual, không binarize).
        Query KHÔNG binarize để tính dot product chính xác hơn.

        Args:
            query: np.ndarray shape (D,), float32

        Returns:
            np.ndarray shape (r,), float32  — projected query
        """
        return (self.R @ query).astype(np.float32)  # (r,)

    def sketch_documents_batch(self, vectors: np.ndarray) -> np.ndarray:
        """
        Tính binary QJL sketch cho batch documents.
        Documents được binarize: sign(R @ x) ∈ {-1, +1}

        Args:
            vectors: np.ndarray shape (N, D), float32

        Returns:
            np.ndarray shape (N, r), int8  — binary sketches
        """
        projected = (vectors @ self.R.T).astype(np.float32)  # (N, r)
        return np.sign(projected).astype(np.int8)

    def qjl_scores(
        self, q_sketch: np.ndarray, doc_sketches_batch: np.ndarray
    ) -> np.ndarray:
        """
        Ước lượng dot product bằng QJL sketches.

        Ước lượng: dot(q, x) ≈ (π/4) * dot(q_sketch, sign(x_sketch))

        Args:
            q_sketch:          (r,) float32 — query projection (không binarize)
            doc_sketches_batch:(N, r) int8  — document binary sketches

        Returns:
            (N,) float32 — QJL approximate dot product scores
        """
        doc_sketches_f = doc_sketches_batch.astype(np.float32)   # (N, r)
        raw_scores = doc_sketches_f @ q_sketch                    # (N,)
        # Hệ số (π/4) ≈ 0.785: từ lý thuyết Hamming distance → cosine
        return (np.pi / 4) * raw_scores


class ARQReranker:
    """
    ARQ Reranker: kết hợp ADC scores và QJL scores theo trọng số α.

    Dùng bởi ARQRAGHandler để reranker candidates từ Qdrant.
    """

    def __init__(self, D: int = 768, r: int = DEFAULT_QJL_DIM, alpha: float = QJL_ALPHA):
        """
        Args:
            D: Chiều vector gốc
            r: Chiều QJL sketch (lớn hơn = chính xác hơn, chậm hơn)
            alpha: Trọng số ADC (0.0-1.0). alpha=1.0 → pure ADC (= PQ)
        """
        self.D = D
        self.r = r
        self.alpha = alpha
        self.qjl = QJLSketch(D=D, r=r)
        logger.info(f"ARQReranker: alpha={alpha}, QJL r={r}")

    def compute_combined_scores(
        self,
        query_vec: np.ndarray,
        adc_table: np.ndarray,
        pq_codes_batch: np.ndarray,
        qjl_sketches_batch: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Tính combined TurboQuant scores cho batch candidates.

        score = α * adc_score + (1-α) * qjl_score

        Args:
            query_vec:          (D,) float32 — query gốc
            adc_table:          (M, K) float32 — từ ProductQuantizer.compute_adc_table()
            pq_codes_batch:     (N, M) uint8 — PQ codes của candidates
            qjl_sketches_batch: (N, r) int8 hoặc None — QJL sketches (optional)

        Returns:
            (N,) float32 — combined scores (cao hơn = tốt hơn)
        """
        N = len(pq_codes_batch)

        # ADC scores (từ ProductQuantizer lookup table)
        m_indices = np.arange(adc_table.shape[0])
        pq_codes_i32 = pq_codes_batch.astype(np.int32)
        adc_scores = adc_table[m_indices, pq_codes_i32].sum(axis=1)  # (N,)

        if qjl_sketches_batch is None or len(qjl_sketches_batch) != N:
            # Fallback: chỉ dùng ADC (= rag_pq behavior)
            logger.debug("Không có QJL sketches → dùng pure ADC")
            return adc_scores.astype(np.float32)

        # QJL scores
        q_sketch = self.qjl.sketch_query(query_vec)                           # (r,)
        qjl_scores = self.qjl.qjl_scores(q_sketch, qjl_sketches_batch)       # (N,)

        # Normalize cả 2 về cùng scale trước khi combine
        def _safe_normalize(arr: np.ndarray) -> np.ndarray:
            std = arr.std()
            if std < 1e-8:
                return arr - arr.mean()
            return (arr - arr.mean()) / std

        adc_norm = _safe_normalize(adc_scores)
        qjl_norm = _safe_normalize(qjl_scores)

        combined = self.alpha * adc_norm + (1 - self.alpha) * qjl_norm
        return combined.astype(np.float32)
