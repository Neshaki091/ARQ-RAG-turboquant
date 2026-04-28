import torch
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

# =============================================================================
# Scalar Quantizer - Stage 1 (MSE)
# =============================================================================

@dataclass
class SQQuantized:
    indices: torch.Tensor  # (N, D) uint8/int - indices to centroids
    centroids: torch.Tensor # (K,) - centroids values

class ScalarQuantizer:
    """
    Scalar Quantizer (STAGE 1): Nén từng chiều độc lập dựa trên codebook.
    Tối ưu hóa bằng thuật toán Max-Lloyd cho phân phối Gaussian.
    """
    def __init__(self, dim: int, bits: int = 4, device: str = "cpu", dtype=torch.float32, seed: int = 42):
        self.dim = dim
        self.bits = bits
        self.n_clusters = 2**bits
        self.device = device
        self.dtype = dtype
        
        # Mặc định: Centroids tối ưu cho phân phối Gaussian N(0, 1/d)
        # sau khi nhân với ma trận xoay Π (QJL).
        # Các giá trị được tính sẵn từ thuật toán Max-Lloyd (Continuous K-Means).
        scale = 1.0 / np.sqrt(dim)
        
        if bits == 1:
            # 1-bit MSE (2 states): ±sqrt(2/pi) * (1/sqrt(d)) ≈ ±0.798/sqrt(d)
            val = 0.79788456 * scale
            centroids = torch.tensor([-val, val], device=self.device, dtype=self.dtype)
        elif bits == 3:
            # 3-bit MSE (8 states): ±0.245, ±0.756, ±1.344, ±2.152
            vals = np.array([-2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152]) * scale
            centroids = torch.from_numpy(vals).to(device=self.device, dtype=self.dtype)
        else:
            raise ValueError(f"ScalarQuantizer only supports 1-bit and 3-bit MSE configurations. Received: {bits}")
            
        self.centroids = centroids
        self.boundaries = self._get_boundaries(centroids)

    def _get_boundaries(self, centroids: torch.Tensor) -> torch.Tensor:
        """Tính các mốc ranh giới (Decision Boundaries) giữa các centroids."""
        boundaries = torch.zeros(len(centroids) + 1, device=self.device)
        boundaries[0], boundaries[-1] = -1e10, 1e10
        if len(centroids) > 1:
            # Ranh giới tối ưu Voronoi là trung điểm của 2 centroids kế tiếp
            boundaries[1:-1] = (centroids[:-1] + centroids[1:]) / 2
        return boundaries

    def fit(self, x: torch.Tensor, iterations: int = 50):
        """
        Học centroids tối ưu bằng thuật toán Max-Lloyd (1D K-Means).
        LƯU Ý: Với dữ liệu TurboQuant đã qua phép xoay Π, dữ liệu đã hội tụ về 
        phân phối chuẩn N(0, 1/d). Các centroids khởi tạo trong __init__ đã là 
        tối ưu lý thuyết, việc chạy fit() có thể không cần thiết hoặc chỉ điều chỉnh nhẹ.
        """
        data_flat = x.flatten()
        if data_flat.device.type != self.device:
            data_flat = data_flat.to(self.device)
        
        # Sampling 1M points để tối ưu tốc độ huấn luyện
        if len(data_flat) > 1_000_000:
            indices = torch.randperm(len(data_flat), device=self.device)[:1_000_000]
            subset = data_flat[indices]
        else:
            subset = data_flat

        # 1. Khởi tạo Centroids bằng Quantiles (Gần với tối ưu Gaussian ngay từ đầu)
        p = torch.linspace(0, 1, self.n_clusters + 1, device=self.device)
        p_mid = (p[:-1] + p[1:]) / 2
        centroids = torch.quantile(subset, p_mid).sort()[0]
        
        # 2. Vòng lặp Max-Lloyd
        for _ in range(iterations):
            boundaries = self._get_boundaries(centroids)
            
            # Gán điểm vào các buckets (Searchsorted tìm ranh giới nhanh trong 1D)
            bucket_indices = torch.searchsorted(boundaries, subset) - 1
            bucket_indices = bucket_indices.clamp(0, self.n_clusters - 1)
            
            # Tính Centroids mới = Mean của từng bucket
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(self.n_clusters, device=self.device)
            sums = torch.zeros(self.n_clusters, device=self.device)
            
            sums.scatter_add_(0, bucket_indices, subset)
            counts.scatter_add_(0, bucket_indices, torch.ones_like(subset))
            
            mask = counts > 0
            new_centroids[mask] = sums[mask] / counts[mask]
            
            # Xử lý các bucket trống (nếu có) bằng cách nội suy từ centroids cũ
            new_centroids[~mask] = centroids[~mask]
            
            # Kiểm tra hội tụ
            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids.sort()[0]
            
        self.centroids = centroids
        self.boundaries = self._get_boundaries(centroids)

    def quantize(self, x: torch.Tensor) -> SQQuantized:
        """Lượng tử hóa vector x thành các chỉ số (indices)."""
        # Đảm bảo contiguous cho performance
        # Sửa lỗi: Cần trừ 1 vì searchsorted trả về vị trí chèn, index của bucket là searchsorted - 1
        indices = (torch.searchsorted(self.boundaries, x.contiguous()) - 1).clamp(0, self.n_clusters - 1)
        indices_np = indices.cpu().numpy().astype(np.uint8)
        
        # Bit-Packing Logic (2 values per byte for 4-bit)
        n, d = indices_np.shape
        bits = self.bits
        vals_per_byte = 1
        if bits == 1: vals_per_byte = 8
        elif bits == 2: vals_per_byte = 4
        elif bits in [3, 4]: vals_per_byte = 2
        
        if vals_per_byte > 1:
            packed_d = (d + vals_per_byte - 1) // vals_per_byte
            packed_codes = np.zeros((n, packed_d), dtype=np.uint8)
            for i in range(vals_per_byte):
                # Pack values into bytes
                subset = indices_np[:, i::vals_per_byte]
                curr_d = subset.shape[1]
                packed_codes[:, :curr_d] |= (subset << (i * bits))
            return SQQuantized(indices=torch.from_numpy(packed_codes), centroids=self.centroids)
        
        return SQQuantized(indices=indices.to(torch.uint8), centroids=self.centroids)

    def reconstruct(self, codes: torch.Tensor) -> torch.Tensor:
        """Giải nén vector (De-quantize) từ mã đã nén."""
        codes_np = codes.cpu().numpy()
        n, packed_d = codes_np.shape
        bits = self.bits
        vals_per_byte = 1
        if bits == 1: vals_per_byte = 8
        elif bits == 2: vals_per_byte = 4
        elif bits in [3, 4]: vals_per_byte = 2
        
        bit_mask = (1 << bits) - 1
        
        if vals_per_byte > 1:
            indices = np.zeros((n, self.dim), dtype=np.int64)
            for i in range(vals_per_byte):
                # Unpack
                subset = (codes_np >> (i * bits)) & bit_mask
                # i::vals_per_byte unpacking
                indices[:, i::vals_per_byte] = subset[:, :((self.dim - i + vals_per_byte - 1) // vals_per_byte)]
            
            indices_t = torch.from_numpy(indices).to(self.device)
            return self.centroids[indices_t]
        
