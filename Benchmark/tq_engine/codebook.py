import torch
import numpy as np
import math
from typing import Tuple

# =============================================================================
# MODUNE: BẢNG MÃ (CODEBOOK)
# Chứa logic tạo các điểm centroids tối ưu cho việc nén TurboQuant.
# Sử dụng phân phối Gaussian N(0, 1/dim) và thuật toán Lloyd-Max.
# =============================================================================

def get_codebook_tensors(dim: int, bits: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hàm tạo bảng mã (centroids) và các ranh giới quyết định (boundaries).
    Dùng để tìm index nhanh nhất cho mỗi tọa độ vector.
    """
    n_clusters = 2**bits
    
    # 1. Khởi tạo Centroids dựa trên số bit (Lloyd-Max Gaussian)
    if bits == 1:
        # 1-bit MSE (thường dùng cho các bản siêu nén)
        centroids = torch.tensor([-0.7979, 0.7979])
    elif bits == 2:
        # 2-bit MSE (cho cấu hình 3-bit TQ)
        centroids = torch.tensor([-1.51, -0.452, 0.452, 1.51])
    elif bits == 4:
        # 4-bit MSE (cho cấu hình 5-bit TQ) - 16 mức tối ưu
        centroids = torch.tensor([
            -2.733, -2.069, -1.618, -1.256, -0.942, -0.657, -0.388, -0.128,
             0.128,  0.388,  0.657,  0.942,  1.256,  1.618,  2.069,  2.733
        ])
    elif bits == 8:
        # 8-bit MSE (cho cấu hình 9-bit TQ) - Tạo 256 điểm qua hàm Inverse CDF
        p = torch.linspace(1/(2*n_clusters), 1 - 1/(2*n_clusters), n_clusters)
        centroids = torch.erfinv(2*p - 1) * math.sqrt(2)
    else:
        # Các trường hợp bit khác (fallback)
        centroids = torch.linspace(-3, 3, n_clusters)
        
    # 2. Rescale theo số chiều không gian (D=128 hoặc 768)
    # Vì vector sau khi quay có phương sai xấp xỉ 1/dim
    scaling = 1.0 / math.sqrt(dim)
    centroids = centroids * scaling
    centroids = centroids.to(device=device, dtype=dtype)
    
    # 3. Tạo Boundaries: Ranh giới để gán index (Nearest Centroid)
    boundaries = torch.zeros(n_clusters + 1, device=device, dtype=dtype)
    boundaries[0] = -1e10
    boundaries[-1] = 1e10
    boundaries[1:-1] = (centroids[:-1] + centroids[1:]) / 2
    
    return centroids, boundaries
