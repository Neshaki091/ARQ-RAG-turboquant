from __future__ import annotations
import torch
import numpy as np
from typing import Optional, NamedTuple, Any
from tq_engine.quantizer import TQEngine, ProdQuantized

# =============================================================================
# MODULE: LƯU TRỮ KV CACHE NÉN
# Quản lý việc lưu trữ và truy xuất các Keys/Values sau khi đã nén TurboQuant.
# =============================================================================

class FlatCache(NamedTuple):
    """Cấu trúc dữ liệu KV cache sau khi đã gộp các chunk (Flattened View)."""
    prod_q: ProdQuantized       # Keys đã nén
    value_q: Any                # Values đã nén
    num_tokens: int

class CompressedKVStore:
    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        key_bits: int = 3,
        value_bits: int = 2,
        device: torch.device = None,
        layer_idx: int = 0,
    ):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.key_bits = key_bits
        self.device = device or torch.device("cpu")
        
        self.quantizer = TQEngine(
            dim=head_dim,
            bits=key_bits,
            device=self.device,
            seed=42 + layer_idx * 7,
        )

        self._key_chunks: list[ProdQuantized] = []
        self._value_chunks: list[Any] = []
        self._chunk_lengths: list[int] = []
        self._flat: Optional[FlatCache] = None

    @property
    def total_tokens(self) -> int:
        return sum(self._chunk_lengths)

    def append_chunk(self, key: torch.Tensor, value: torch.Tensor):
        key_q = self.quantizer.quantize(key) # Assuming key is (N, D)
        self._key_chunks.append(key_q)
        self._value_chunks.append(value)
        self._chunk_lengths.append(key.shape[0])
        self._flat = None

    def get_flat_cache(self) -> Optional[FlatCache]:
        if not self._key_chunks: return None
        if self._flat is not None: return self._flat

        flat_kq = _concat_prod_q(self._key_chunks)
        self._flat = FlatCache(prod_q=flat_kq, value_q=None, num_tokens=self.total_tokens)
        return self._flat

def _concat_prod_q(chunks: list[ProdQuantized]) -> ProdQuantized:
    """Ghép các chunk nén (SQ+QJL) thành một khối duy nhất."""
    return ProdQuantized(
        sq_codes=np.concatenate([c.sq_codes for c in chunks], axis=0),
        qjl_signs=np.concatenate([c.qjl_signs for c in chunks], axis=0),
        norms=np.concatenate([c.norms for c in chunks], axis=0),
        centroids=chunks[0].centroids,
        dim=chunks[0].dim,
        total_bits=chunks[0].total_bits,
        qjl_scale=chunks[0].qjl_scale,
        rotation_signs=chunks[0].rotation_signs,
        res_norms=np.concatenate([c.res_norms for c in chunks], axis=0) if chunks[0].res_norms is not None else None
    )
