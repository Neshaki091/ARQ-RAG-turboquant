import math
import torch

# =============================================================================
# MODULE: PHÉP QUAY NGẪU NHIÊN (ROTATION)
# Chứa các hàm tạo ma trận quay Pi và ma trận chiếu QJL (S).
# Ma trận quay giúp phân phối lại phương sai đồng đều trên các chiều không gian.
# =============================================================================

def generate_rotation_matrix(
    d: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> torch.Tensor:
    """
    Tạo ma trận trực giao Pi cỡ (d x d) bằng phân rã QR.
    Đây là bước tiền xử lý quan trọng trước khi nén MSE.
    """
    # Dùng Generator cố định seed trên CPU để đảm bảo kết quả tái lập (Reproducibility)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    # 1. Khởi tạo ma trận Gaussian ngẫu nhiên
    G = torch.randn(d, d, generator=rng, dtype=torch.float32)
    
    # 2. Phân rã QR để lấy ma trận trực giao Q
    Q, R = torch.linalg.qr(G)

    # 3. Chỉnh dấu để đảm bảo det = 1 (phép quay thuần túy)
    diag_sign = torch.sign(torch.diag(R))
    Q = Q * diag_sign.unsqueeze(0)

    return Q.to(device=device, dtype=dtype)


def generate_qjl_matrix(
    d: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 12345,
) -> torch.Tensor:
    """
    Tạo ma trận chiếu ngẫu nhiên S cho thuật toán QJL.
    Các phần tử tuân theo phân phối i.i.d N(0, 1).
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    S = torch.randn(d, d, generator=rng, dtype=torch.float32)
    return S.to(device=device, dtype=dtype)


def rotate_forward(x: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """Thực hiện phép quay xuôi: y = x @ Pi^T"""
    return torch.matmul(x, Pi.T)


def rotate_backward(y: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """Thực hiện phép quay ngược để giải nén: x = y @ Pi"""
    return torch.matmul(y, Pi)
