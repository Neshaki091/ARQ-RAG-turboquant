import torch
from typing import Tuple, List

def recursive_polar_transform(x: torch.Tensor, levels: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Biến đổi Cartesian -> Polar đệ quy (Algorithm 1 của PolarQuant).
    
    Tại mỗi level l, cặp (r_i, r_{i+1}) được biến đổi thành:
        R = sqrt(r_i^2 + r_{i+1}^2)
        theta = atan2(r_{i+1}, r_i)
    
    Args:
        x: Input tensor (..., dim)
        levels: Số level đệ quy (L). dim phải chia hết cho 2^L.
        
    Returns:
        final_radius: (..., dim // 2**levels) - bán kính cuối
        angles: List of L tensors, angles[l-1] có shape (..., dim // 2**l)
    """
    angles = []
    current = x.float()
    
    for l in range(levels):
        # current: (..., N)
        shape = current.shape
        paired = current.reshape(*shape[:-1], -1, 2)  # (..., N//2, 2)
        
        r1 = paired[..., 0]
        r2 = paired[..., 1]
        
        # Góc theo atan2: range [-pi, pi]
        angle = torch.atan2(r2, r1)
        angles.append(angle)
        
        # Bán kính mới
        current = torch.sqrt(r1 * r1 + r2 * r2)
    
    return current, angles  # final_radius, [angle_L1, angle_L2, ..., angle_LL]


def recursive_polar_reconstruct(radius: torch.Tensor, angles: List[torch.Tensor]) -> torch.Tensor:
    """
    Tái tạo Cartesian từ Polar đệ quy (nghịch đảo của transform).
    
    Args:
        radius: (..., final_dim) - bán kính cuối cùng
        angles: List of L tensors (thứ tự giống transform)
    
    Returns:
        x_hat: (..., original_dim) - vector tái tạo
    """
    current = radius.float()
    levels = len(angles)
    
    # Xử lý ngược từ level cuối về level đầu
    for l in range(levels - 1, -1, -1):
        angle = angles[l]
        r1 = current * torch.cos(angle)
        r2 = current * torch.sin(angle)
        combined = torch.stack([r1, r2], dim=-1)  # (..., N, 2)
        current = combined.reshape(*combined.shape[:-2], -1)  # (..., 2*N)
    
    return current
