from __future__ import annotations
import torch
from typing import Optional, NamedTuple, Any
from tq_engine.quantizer import TQEngine, ProdQuantized

# =============================================================================
# MODULE: LƯU TRỮ KV CACHE NÉN
# Quản lý việc lưu trữ và truy xuất các Keys/Values sau khi đã nén TurboQuant.
# Sử dụng cơ chế Lazy Flattening: Chỉ gộp các chunk dữ liệu khi cần truy vấn.
# =============================================================================

class FlatCache(NamedTuple):
    """Cấu trúc dữ liệu KV cache sau khi đã gộp các chunk (Flattened View)."""
    prod_q: ProdQuantized       # Keys đã nén (H, T, D)
    value_q: Any                # Values đã nén
    num_tokens: int

class CompressedKVStore:
    """
    Lớp quản lý KV Store nén.
    Keys được nén bằng TQ (MSE + QJL).
    Values có thể dùng Scalar Quantization hoặc Symmetric Quantization.
    """

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
        self.value_bits = value_bits
        self.device = device or torch.device("cuda")
        
        # Khởi tạo quantizer cho Keys
        self.quantizer = TQEngine(
            dim=head_dim,
            bits=key_bits,
            device=self.device,
            seed=42 + layer_idx * 7,
        )

        # Danh sách các chunks dữ liệu
        self._key_chunks: list[ProdQuantized] = []
        self._value_chunks: list[Any] = []
        self._chunk_lengths: list[int] = []

        # Cache cho vùng nhớ phẳng (flattened)
        self._flat: Optional[FlatCache] = None

    @property
    def total_tokens(self) -> int:
        return sum(self._chunk_lengths)

    def append_chunk(self, key: torch.Tensor, value: torch.Tensor):
        """Nén và thêm một chunk KV mới vào store."""
        chunk_len = key.shape[0]
        
        # Chuyển đổi shape cho đúng định dạng nén (H, T, D)
        # key: (T, H, D) -> k: (1, H, T, D)
        k = key.transpose(0, 1).unsqueeze(0)
        
        # Thực hiện nén
        key_q = self.quantizer.quantize(k)
        
        # Lưu trữ
        self._key_chunks.append(key_q)
        self._value_chunks.append(value) # Tạm thời giữ raw value hoặc nén đơn giản
        self._chunk_lengths.append(chunk_len)
        
        # Vô hiệu hóa cache cũ
        self._flat = None

    def get_flat_cache(self) -> Optional[FlatCache]:
        """Gộp tất cả các chunks thành một khối dữ liệu liên tục để truy vấn nhanh."""
        if not self._key_chunks:
            return None

        if self._flat is not None:
            return self._flat

        # Thực hiện ghép nối (Concatenate)
        # kq: ProdQuantized
        flat_kq = _concat_prod_q([_flatten_pq(c) for c in self._key_chunks])
        
        self._flat = FlatCache(
            prod_q=flat_kq,
            value_q=None, # Update logic cho value nếu cần
            num_tokens=self.total_tokens,
        )
        return self._flat

    def reset(self):
        """Xóa sạch bộ nhớ cache của layer."""
        self._key_chunks.clear()
        self._value_chunks.clear()
        self._chunk_lengths.clear()
        self._flat = None


def _flatten_pq(pq: ProdQuantized) -> ProdQuantized:
    """Xóa bỏ chiều batch (1, H, T, ...) -> (H, T, ...)."""
    return ProdQuantized(
        mse_indices=pq.mse_indices.squeeze(0).contiguous(),
        qjl_signs=pq.qjl_signs.squeeze(0).contiguous(),
        residual_norms=pq.residual_norms.squeeze(0).contiguous(),
        norms=pq.norms.squeeze(0).contiguous(),
        mse_bits=pq.mse_bits,
    )

def _concat_prod_q(chunks: list[ProdQuantized]) -> ProdQuantized:
    """Ghép các chunk nén thành một tensor lớn duy nhất."""
    return ProdQuantized(
        mse_indices=torch.cat([c.mse_indices for c in chunks], dim=-2),
        qjl_signs=torch.cat([c.qjl_signs for c in chunks], dim=-2),
        residual_norms=torch.cat([c.residual_norms for c in chunks], dim=-1),
        norms=torch.cat([c.norms for c in chunks], dim=-1),
        mse_bits=chunks[0].mse_bits,
    )
