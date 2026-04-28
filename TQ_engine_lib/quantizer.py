import torch
import numpy as np
import math
from dataclasses import dataclass
from .codebook import ScalarQuantizer
from .rotation import get_orthogonal_matrix, rotate_forward, rotate_backward
from .tq_bridge import tq_native

@dataclass
class ProdQuantized:
    sq_codes: np.ndarray
    qjl_signs: np.ndarray
    norms: np.ndarray
    centroids: np.ndarray
    dim: int
    sq_bits: int
    total_bits: int
    qjl_scale: float
    rot_op: np.ndarray
    res_norms: np.ndarray

class TQEngine:
    def __init__(self, dim: int = 768, bits: int = 4, device: str = None):
        if bits not in [2, 4]:
            raise ValueError(f"TurboQuant currently only supports 2-bit (1+1) and 4-bit (3+1) configurations. Received: {bits}")
            
        self.dim = dim
        self.bits = bits
        self.sq_bits = bits - 1
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Initialize Scalar Quantizer (Stage 1)
        self.sq_quantizer = ScalarQuantizer(dim=dim, bits=self.sq_bits, device=self.device)
        
        # 2. Pure TurboQuant: Exact Dimension Orthogonal Rotation
        rot_op_t = get_orthogonal_matrix(dim, device=self.device)
        self.rot_op_np = rot_op_t.cpu().numpy().astype(np.float32)
        
        # 3. Calculate QJL Scale
        if self.sq_bits == 1:
            self.qjl_scale = 0.585 / math.sqrt(self.dim)
        else:
            self.qjl_scale = math.sqrt(2.0 / math.pi) / math.sqrt(self.dim)

    def quantize(self, x: torch.Tensor, online_clustering: bool = False) -> ProdQuantized:
        """
        Nén bộ dữ liệu vector x (N, D) sang định dạng TurboQuant (SQ+QJL).
        
        Args:
            x: Tensor đầu vào (N, D).
            online_clustering: Nếu True, sẽ chạy Max-Lloyd để fit centroids vào dữ liệu. 
                               Nếu False (mặc định), sẽ dùng centroids Gaussian tối ưu lý thuyết.
        """
        if x.device.type != self.device:
            x = x.to(self.device)
        
        dist_device = x.device
        rot_op_t = torch.from_numpy(self.rot_op_np).to(dist_device)

        # 1. ROTATE
        x_rot = rotate_forward(x, rot_op_t)
        
        # 2. Extract Norms
        norms = torch.norm(x, dim=-1)
        
        # 3. Stage 1: SQ
        if online_clustering:
            self.sq_quantizer.fit(x_rot)
        
        sq_q = self.sq_quantizer.quantize(x_rot)
        x_hat_1 = self.sq_quantizer.reconstruct(sq_q.indices)
        
        # 4. Stage 2: QJL Residual (Pure TQ: Signs in Rotated Space)
        residual = x_rot - x_hat_1
        res_norms = torch.norm(residual, dim=-1)
        
        # Signs of residual vector directly in rotated space
        signs = (residual > 0).to(torch.uint8).cpu().numpy()
        qjl_signs = np.packbits(signs, axis=-1, bitorder='little')
        
        return ProdQuantized(
            sq_codes=sq_q.indices.cpu().numpy().astype(np.uint8),
            qjl_signs=qjl_signs.astype(np.uint8),
            norms=norms.cpu().numpy().astype(np.float32),
            centroids=self.sq_quantizer.centroids.cpu().numpy().astype(np.float32),
            dim=self.dim,
            sq_bits=self.sq_bits,
            total_bits=self.bits,
            qjl_scale=self.qjl_scale,
            rot_op=self.rot_op_np,
            res_norms=res_norms.cpu().numpy().astype(np.float32)
        )

    def native_cosine_search(self, query: torch.Tensor, pq: ProdQuantized, top_k: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
        q_t = query.to(self.device).float()
        rot_op_t = torch.from_numpy(pq.rot_op).to(self.device)
        
        # 1. Query Rotation (Stage 2 query is the SAME as Stage 1 query)
        q_rot = rotate_forward(q_t.unsqueeze(0), rot_op_t).squeeze(0)
        q_np = q_rot.cpu().numpy().astype(np.float32)
        
        # 2. Preparation for Rust (Strict C-Contiguous)
        query_1d = np.array(q_np, dtype=np.float32, order='C')
        sq_codes_2d = np.array(pq.sq_codes, dtype=np.uint8, order='C')
        centroids_1d = np.array(pq.centroids, dtype=np.float32, order='C')
        norms_1d = np.array(pq.norms, dtype=np.float32, order='C')
        qjl_signs_2d = np.array(pq.qjl_signs, dtype=np.uint8, order='C')
        res_norms_1d = np.array(pq.res_norms, dtype=np.float32, order='C')
        
        # Pure TQ: qjl_query IS query_1d
        scores = tq_native.tq_scan(
            query_1d, sq_codes_2d, centroids_1d, norms_1d,
            qjl_signs_2d, res_norms_1d, query_1d,
            float(pq.qjl_scale), int(self.dim), int(self.sq_bits)
        )
        
        scores_t = torch.from_numpy(scores).view(-1)
        top_scores, top_indices = torch.topk(scores_t, min(top_k, len(scores_t)))
        return top_indices, top_scores
