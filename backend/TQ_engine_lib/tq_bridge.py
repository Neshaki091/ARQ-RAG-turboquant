import numpy as np
import os
try:
    from . import tq_native_lib  # Local pyd
except ImportError:
    import tq_native_lib  # Global fallback

class TQNative:
    """
    TurboQuant Native Bridge — Rust SIMD Backend
    
    Cung cấp:
    - tq_scan: Lõi chuẩn mới SQ(b-1) + QJL(1)
    - sq8_scan, pq_scan: Baselines cho so sánh
    """
    def __init__(self):
        self.lib = tq_native_lib
        print("TurboQuant Native (Rust SIMD) Active - SQ+QJL Mode")

    # ========================
    # NEW: Unified SQ+QJL Scan
    # ========================
    def tq_scan(self, query, sq_codes, centroids, norms,
                qjl_signs, res_norms, qjl_query, qjl_scale, dim, mse_bits):
        """
        Tính điểm tương tự giữa query và N vectors nén SQ+QJL.
        """
        return self.lib.tq_scan(
            query, sq_codes, centroids, norms,
            qjl_signs, res_norms, qjl_query,
            qjl_scale, dim, mse_bits
        )

    def tq_batch_scan(self, queries, sq_codes, centroids, norms,
                      qjl_signs, res_norms, qjl_queries,
                      qjl_scale, dim, mse_bits):
        """
        Batch Search Flat (không IVF) trên toàn bộ dữ liệu nén.
        """
        return self.lib.tq_batch_scan(
            queries, sq_codes, centroids, norms,
            qjl_signs, res_norms, qjl_queries,
            qjl_scale, dim, mse_bits
        )
    
    def tq_ivf_online_scan(self, queries, sq_codes, centroids, norms,
                            qjl_signs, res_norms, qjl_queries,
                            list_offsets, coarse_centroids,
                            n_probe, qjl_scale, dim, mse_bits, top_k):
        """
        Batch Search trên toàn bộ chỉ mục IVF (Chính là Core Production).
        Trả về (top_scores, top_indices)
        """
        return self.lib.tq_ivf_online_scan(
            queries, sq_codes, centroids, norms,
            qjl_signs, res_norms, qjl_queries,
            list_offsets, coarse_centroids,
            n_probe, qjl_scale, dim, mse_bits, top_k
        )

    # ========================
    # Index-time quantization
    # ========================
    def tq_quantize_rotated(self, x_rot, sq_centroids, sq_bits):
        """
        Quantize rotated vectors into (sq_codes_packed, qjl_signs_packed, res_norms).
        - x_rot: (N, D) float32 (rotated residual space)
        - sq_centroids: (K,) float32, K=2 (sq_bits=1) or K=8 (sq_bits=3)
        """
        return self.lib.tq_quantize_rotated(x_rot, sq_centroids, int(sq_bits))
    
    # ========================
    # Baselines
    # ========================
    def sq8_scan(self, query, keys, norms):
        return self.lib.sq8_score_simd(query, keys, norms)

    def pq_scan(self, query, codes, precomputed_dist):
        return self.lib.pq_score_simd(query, codes, precomputed_dist)

# Global instance
tq_native = TQNative()
