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
    
    # ========================
    # Baselines
    # ========================
    def sq8_scan(self, query, keys, norms):
        return self.lib.sq8_score_simd(query, keys, norms)

    def pq_scan(self, query, codes, precomputed_dist):
        return self.lib.pq_score_simd(query, codes, precomputed_dist)

# Global instance
tq_native = TQNative()
