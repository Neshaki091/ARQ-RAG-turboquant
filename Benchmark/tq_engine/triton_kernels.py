import math
import torch
import triton
import triton.language as tl

# =============================================================================
# MODULE: TRITON GIA TỐC HÓA GPU
# Chứa các Kernel Triton để tính toán Attention Score trực tiếp trên dữ liệu nén.
# Giúp bỏ qua bước giải nén trung gian, tiết kiệm băng thông bộ nhớ GPU.
# =============================================================================

# ─── Kernel 1: Tính toán Score cho giai đoạn MSE ─────────────────────────
@triton.jit
def _mse_score_kernel(
    Q_ptr, MSE_ptr, NORMS_ptr, CENTROIDS_ptr, OUT_ptr,
    stride_q_bh, stride_q_d,
    stride_m_bh, stride_m_n, stride_m_d,
    stride_n_bh, stride_n_n,
    stride_o_bh, stride_o_n,
    N, D: tl.constexpr, PACKED_D: tl.constexpr,
    BITS: tl.constexpr, VALS_PER_BYTE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Tính toán tích vô hướng query @ key_mse ngay trên dữ liệu bit-packed."""
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # Tích lũy score cho từng token trong block
    scores = tl.zeros([BLOCK_N], dtype=tl.float32)
    BIT_MASK: tl.constexpr = (1 << BITS) - 1

    for byte_idx in range(PACKED_D):
        packed = tl.load(
            MSE_ptr + pid_bh * stride_m_bh + n_offs * stride_m_n + byte_idx * stride_m_d,
            mask=n_mask, other=0
        ).to(tl.int32)

        for sub in range(VALS_PER_BYTE):
            coord_idx = byte_idx * VALS_PER_BYTE + sub
            if coord_idx < D:
                # Giải nén chỉ số centroid ngay trong thanh ghi
                idx = (packed >> (sub * BITS)) & BIT_MASK
                centroid_val = tl.load(CENTROIDS_ptr + idx)
                q_val = tl.load(Q_ptr + pid_bh * stride_q_bh + coord_idx * stride_q_d)
                scores += q_val * centroid_val

    # Nhân với Norm gốc để khôi phục biên độ
    norms = tl.load(NORMS_ptr + pid_bh * stride_n_bh + n_offs * stride_n_n, mask=n_mask, other=0.0)
    tl.store(OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n, scores * norms, mask=n_mask)


# ─── Kernel 2: Tính toán Score cho giai đoạn QJL ─────────────────────────
@triton.jit
def _qjl_score_kernel(
    Q_SKETCH_ptr, SIGNS_ptr, RES_NORMS_ptr, OUT_ptr,
    stride_qs_bh, stride_qs_d,
    stride_s_bh, stride_s_n, stride_s_d,
    stride_rn_bh, stride_rn_n,
    stride_o_bh, stride_o_n,
    N, D: tl.constexpr, PACKED_D_SIGNS: tl.constexpr,
    QJL_SCALE, BLOCK_N: tl.constexpr,
):
    """Tính toán phần bù QJL để tăng độ chính xác cho Inner Product."""
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    dot = tl.zeros([BLOCK_N], dtype=tl.float32)

    for byte_idx in range(PACKED_D_SIGNS):
        packed = tl.load(
            SIGNS_ptr + pid_bh * stride_s_bh + n_offs * stride_s_n + byte_idx * stride_s_d,
            mask=n_mask, other=0
        ).to(tl.int32)

        for bit in range(8):
            coord_idx = byte_idx * 8 + bit
            if coord_idx < D:
                # Giải nén bit dấu: 1 -> +1, 0 -> -1
                sign_val = tl.where((packed >> bit) & 1 == 1, 1.0, -1.0)
                q_val = tl.load(Q_SKETCH_ptr + pid_bh * stride_qs_bh + coord_idx * stride_qs_d)
                dot += q_val * sign_val

    res_norms = tl.load(RES_NORMS_ptr + pid_bh * stride_rn_bh + n_offs * stride_rn_n, mask=n_mask, other=0.0)
    qjl_scores = dot * res_norms * QJL_SCALE

    # Cộng dồn vào score MSE đã tính trước đó
    existing = tl.load(OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n, mask=n_mask, other=0.0)
    tl.store(OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n, existing + qjl_scores, mask=n_mask)


# ─── Python API Wrappers ──────────────────────────────────────────────────

def compute_mse_score_triton(q_rot, mse_packed, norms, centroids, bits):
    """Giao diện Python để gọi Kernel MSE Triton."""
    BH, D = q_rot.shape[0], q_rot.shape[-1]
    N = mse_packed.shape[1]
    
    # Logic xác định số phần tử đóng gói trong 1 byte
    vals_per_byte = 8 if bits == 1 else (4 if bits == 2 else 2)
    eff_bits = bits if bits in [1, 2, 4] else 4

    out = torch.zeros(BH, N, device=q_rot.device, dtype=torch.float32)
    BLOCK_N = 128
    grid = (BH, triton.cdiv(N, BLOCK_N))

    _mse_score_kernel[grid](
        q_rot, mse_packed, norms, centroids, out,
        q_rot.stride(0), q_rot.stride(1),
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        norms.stride(0), norms.stride(1),
        out.stride(0), out.stride(1),
        N=N, D=D, PACKED_D=mse_packed.shape[2],
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte,
        BLOCK_N=BLOCK_N,
    )
    return out

def compute_qjl_score_triton(q_sketch, qjl_signs, res_norms, qjl_scale, out):
    """Giao diện Python để gọi Kernel QJL Triton."""
    BH, N = qjl_signs.shape[0], qjl_signs.shape[1]
    D = q_sketch.shape[-1]
    
    BLOCK_N = 128
    grid = (BH, triton.cdiv(N, BLOCK_N))

    _qjl_score_kernel[grid](
        q_sketch, qjl_signs, res_norms, out,
        q_sketch.stride(0), q_sketch.stride(1),
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        res_norms.stride(0), res_norms.stride(1),
        out.stride(0), out.stride(1),
        N=N, D=D, PACKED_D_SIGNS=qjl_signs.shape[2],
        QJL_SCALE=qjl_scale,
        BLOCK_N=BLOCK_N,
    )
    return out
