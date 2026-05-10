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

def train_optimized_rotation(x_sample: torch.Tensor, dim: int, iters: int = 20) -> torch.Tensor:
    """
    Thuật toán ITQ (Iterative Quantization / OPQ)
    Tìm ma trận xoay tối ưu để phân phối đều phương sai trên tất cả các chiều,
    tối đa hóa hiệu suất cho Scalar Quantization.
    """
    device = x_sample.device
    
    # 1. Khởi tạo ngẫu nhiên
    R = get_orthogonal_matrix(dim, device)
    
    # Zero-mean data
    mean = x_sample.mean(dim=0)
    x_centered = x_sample - mean
    
    print("  Training Optimized Rotation Matrix (ITQ/OPQ)...")
    for i in range(iters):
        # B1: Xoay dữ liệu
        V = torch.matmul(x_centered, R)
        
        # B2: Lượng tử hóa giả lập bằng hàm Sign
        B = torch.sign(V)
        B[B == 0] = 1.0 
        
        # B3: Cập nhật R bằng SVD (Procrustes problem)
        cov = torch.matmul(x_centered.t(), B)
        U, _, V_svd = torch.linalg.svd(cov, full_matrices=False)
        R = torch.matmul(U, V_svd)
        
    return R.float()

def rotate_forward(x: torch.Tensor, rot_mat: torch.Tensor) -> torch.Tensor:
    """Perfectly orthogonal rotation: x' = x * Pi"""
    return torch.matmul(x, rot_mat)

def rotate_backward(y: torch.Tensor, rot_mat: torch.Tensor) -> torch.Tensor:
    """Inverse rotation: x = y * Pi^T"""
    return torch.matmul(y, rot_mat.T)
