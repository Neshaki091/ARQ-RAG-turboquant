import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, NamedTuple

from tq_engine.codebook import get_codebook_tensors
from tq_engine.rotation import (
    generate_rotation_matrix,
    generate_qjl_matrix,
    rotate_forward,
    rotate_backward,
)

# =============================================================================
# Cấu trúc dữ liệu kết quả nén
# =============================================================================

class MSEQuantized(NamedTuple):
    """Kết quả nén MSE (Algorithm 1)"""
    indices: torch.Tensor       # Các chỉ số centroids (đã đóng gói bit)
    norms: torch.Tensor         # Độ dài L2 của vector gốc
    bits: int                   # Số bit dùng cho mỗi phần tử


class ProdQuantized(NamedTuple):
    """Kết quả nén cho Inner Product (Algorithm 2 - MSE + QJL)"""
    mse_indices: torch.Tensor   # Chỉ số nén MSE stage 1
    qjl_signs: torch.Tensor    # Các bit dấu (+1/-1) nén từ stage 2 QJL
    residual_norms: torch.Tensor  # Độ dài vector phần dư (residual)
    norms: torch.Tensor         # Độ dài vector gốc
    mse_bits: int               # Số bit MSE


# =============================================================================
# Các hàm bổ trợ đóng gói Bit (Bit-packing) để tiết kiệm bộ nhớ
# =============================================================================

def _pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Đóng gói các chỉ số integer vào uint8 để tiết kiệm RAM."""
    d = indices.shape[-1]
    batch_shape = indices.shape[:-1]

    if bits == 1:
        vals_per_byte = 8
    elif bits == 2:
        vals_per_byte = 4
    elif bits <= 4:
        vals_per_byte = 2
        bits = 4
    else:
        return indices.to(torch.uint8)

    padded_d = ((d + vals_per_byte - 1) // vals_per_byte) * vals_per_byte
    if padded_d > d:
        indices = F.pad(indices.to(torch.uint8), (0, padded_d - d), value=0)

    reshaped = indices.to(torch.uint8).reshape(*batch_shape, -1, vals_per_byte)
    shifts = torch.arange(vals_per_byte, device=indices.device, dtype=torch.uint8) * bits
    packed = (reshaped << shifts).sum(dim=-1, dtype=torch.uint8)
    return packed


def _unpack_indices(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    """Giải nén các chỉ số từ uint8 về lại tensor integer."""
    batch_shape = packed.shape[:-1]

    if bits == 1:
        vals_per_byte = 8
    elif bits == 2:
        vals_per_byte = 4
    elif bits <= 4:
        vals_per_byte = 2
        bits = 4
    else:
        return packed.long()

    mask = (1 << bits) - 1
    shifts = torch.arange(vals_per_byte, device=packed.device, dtype=torch.uint8) * bits
    unpacked = ((packed.unsqueeze(-1) >> shifts) & mask)
    unpacked = unpacked.reshape(*batch_shape, -1)
    return unpacked[..., :d].long()


# =============================================================================
# ENGINE 1: Nén MSE (Tối ưu cho việc tái tạo vector)
# =============================================================================

class TQEngineMSE(torch.nn.Module):
    def __init__(self, dim: int, bits: int = 3, device: torch.device = None, dtype: torch.dtype = torch.float32, seed: int = 42):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Khởi tạo ma trận quay Pi
        self.register_buffer("Pi", generate_rotation_matrix(dim, self.device, dtype, seed=seed))

        # Khởi tạo bảng mã
        centroids, boundaries = get_codebook_tensors(dim, bits, self.device, dtype)
        self.register_buffer("centroids", centroids)
        self.register_buffer("decision_boundaries", boundaries[1:-1].contiguous())

    @torch.no_grad()
    def quantize(self, x: torch.Tensor, micro_batch: int = 100000) -> MSEQuantized:
        """Nén vector sang dạng indices với cơ chế micro-batching chống tràn RAM."""
        norms = x.norm(dim=-1)
        packed_list = []
        
        for i in range(0, x.size(0), micro_batch):
            chunk = x[i : i + micro_batch]
            chunk_unit = chunk / (chunk.norm(dim=-1, keepdim=True) + 1e-10)
            
            # Quay và tìm centroid gần nhất
            y = rotate_forward(chunk_unit.float(), self.Pi)
            indices = torch.searchsorted(self.decision_boundaries, y.contiguous())
            
            packed_list.append(_pack_indices(indices, self.bits))
            
        return MSEQuantized(indices=torch.cat(packed_list, dim=0), norms=norms, bits=self.bits)

    def dequantize(self, q: MSEQuantized) -> torch.Tensor:
        """Giải nén và quay ngược về không gian gốc."""
        indices = _unpack_indices(q.indices, q.bits, self.dim)
        y_hat = self.centroids[indices]
        x_hat = rotate_backward(y_hat, self.Pi)
        return x_hat * q.norms.unsqueeze(-1)


# =============================================================================
# ENGINE 2: Nén Inner Product (Tối ưu cho tính toán Attention)
# =============================================================================

class TQEngine(torch.nn.Module):
    def __init__(self, dim: int, bits: int = 3, use_qjl: bool = True, device: torch.device = None, dtype: torch.dtype = torch.float32, seed: int = 42):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.use_qjl = use_qjl
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Phân bổ bit: Nếu dùng QJL thì MSE chiếm b-1 bits, 1 bit dành cho QJL
        mse_bits = (bits - 1 if use_qjl else bits) if bits > 1 else 1
        self.mse_quantizer = TQEngineMSE(dim=dim, bits=mse_bits, device=self.device, dtype=dtype, seed=seed)

        if self.use_qjl:
            self.register_buffer("S", generate_qjl_matrix(dim, self.device, dtype, seed=seed + 1000))
            self.qjl_scale = math.sqrt(math.pi / 2.0) / dim
        else:
            self.S, self.qjl_scale = None, 0.0

    def _pack_qjl_signs(self, projected: torch.Tensor) -> torch.Tensor:
        signs = (projected > 0).to(torch.uint8)
        d = signs.shape[-1]
        if d % 8 != 0: signs = F.pad(signs, (0, 8 - d % 8), value=0)
        signs_reshaped = signs.reshape(*signs.shape[:-1], -1, 8)
        powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=signs.device, dtype=torch.uint8)
        return (signs_reshaped * powers).sum(dim=-1, dtype=torch.uint8)

    def _unpack_qjl_signs(self, packed: torch.Tensor) -> torch.Tensor:
        powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=packed.device, dtype=torch.uint8)
        unpacked = ((packed.unsqueeze(-1) & powers) > 0).float()
        return 2.0 * unpacked.reshape(*packed.shape[:-1], -1)[..., :self.dim] - 1.0

    @torch.no_grad()
    def quantize(self, x: torch.Tensor, micro_batch: int = 100000) -> ProdQuantized:
        """Thực hiện nén 2 giai đoạn: MSE + QJL."""
        mse_indices_list, qjl_signs_list, residual_norms_list = [], [], []
        norms = x.norm(dim=-1)

        for i in range(0, x.size(0), micro_batch):
            chunk = x[i : i + micro_batch]
            mse_q = self.mse_quantizer.quantize(chunk, micro_batch=micro_batch)
            x_hat = self.mse_quantizer.dequantize(mse_q)
            
            residual = chunk - x_hat
            res_norms = residual.norm(dim=-1)
            
            if self.use_qjl and self.S is not None:
                projected = torch.matmul(residual.float(), self.S.T)
                packed_signs = self._pack_qjl_signs(projected)
            else:
                packed_signs = torch.zeros((*residual.shape[:-1], (self.dim + 7) // 8), device=self.device, dtype=torch.uint8)
            
            mse_indices_list.append(mse_q.indices)
            qjl_signs_list.append(packed_signs)
            residual_norms_list.append(res_norms)

        return ProdQuantized(
            mse_indices=torch.cat(mse_indices_list, dim=0),
            qjl_signs=torch.cat(qjl_signs_list, dim=0),
            residual_norms=torch.cat(residual_norms_list, dim=0),
            norms=norms,
            mse_bits=self.mse_quantizer.bits
        )

    def dequantize(self, q: ProdQuantized) -> torch.Tensor:
        """Giải nén vector từ cả 2 stage."""
        x_mse = self.mse_quantizer.dequantize(MSEQuantized(q.mse_indices, q.norms, q.mse_bits))
        if self.use_qjl and self.S is not None:
            signs = self._unpack_qjl_signs(q.qjl_signs)
            x_qjl = torch.matmul(signs, self.S) * (self.qjl_scale * q.residual_norms.unsqueeze(-1))
            return x_mse + x_qjl
        return x_mse

    def attention_score(self, query: torch.Tensor, quantized_key: ProdQuantized) -> torch.Tensor:
        """Tính toán tích vô hướng query-key trực tiếp trên dạng nén (Asymmetric)."""
        k_mse = self.mse_quantizer.dequantize(MSEQuantized(quantized_key.mse_indices, quantized_key.norms, quantized_key.mse_bits))
        scores_mse = torch.matmul(query.float(), k_mse.float().transpose(-2, -1))

        if self.use_qjl and self.S is not None:
            q_sketched = torch.matmul(query.float(), self.S.T)
            signs = self._unpack_qjl_signs(quantized_key.qjl_signs)
            scores_qjl = torch.matmul(q_sketched, signs.transpose(-2, -1)) * (self.qjl_scale * quantized_key.residual_norms.unsqueeze(-2))
            return scores_mse + scores_qjl.to(scores_mse.dtype)
        return scores_mse
