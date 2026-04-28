import torch
import math

def get_orthogonal_matrix(dim: int, device: torch.device = torch.device('cpu'), seed: int = 42):
    """
    Creates a perfectly orthogonal d x d matrix using QR decomposition.
    Follows the core TurboQuant paper approach for memory efficiency.
    """
    torch.manual_seed(seed)
    # 1. Initialize random Gaussian matrix
    H = torch.randn(dim, dim, device=device)
    # 2. QR Decomposition to get Orthogonal Q
    Q, _ = torch.linalg.qr(H)
    return Q.float()

def rotate_forward(x: torch.Tensor, rot_mat: torch.Tensor) -> torch.Tensor:
    """Perfectly orthogonal rotation: x' = x * Pi"""
    return torch.matmul(x, rot_mat)

def rotate_backward(y: torch.Tensor, rot_mat: torch.Tensor) -> torch.Tensor:
    """Inverse rotation: x = y * Pi^T"""
    return torch.matmul(y, rot_mat.T)
