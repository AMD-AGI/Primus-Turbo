###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Acknowledgement:
#   The persistent FP8 GEMM kernels in this file are adapted from tritonBLAS
#   (https://github.com/ROCm/tritonBLAS). We thank the tritonBLAS authors
#   for their high-quality Triton kernel implementations on AMD GPUs.
###############################################################################

"""
FP8 GEMM Triton persistent kernels.

Contains:
  Tensorwise (per-tensor) scaling:
    - _fp8_persistent_gemm_kernel: Core persistent kernel
    - gemm_fp8_tensorwise_triton_kernel: Public API

  Rowwise (per-row/per-col vector) scaling:
    - _fp8_rowwise_persistent_gemm_kernel: Core persistent kernel
    - gemm_fp8_rowwise_triton_kernel: Public API

  Blockwise scaling:
    - _blockwise_fp8_autotune_kernel: Core autotuned persistent kernel
    - gemm_fp8_blockwise_triton_kernel: Unified public API (NT/NN/TN)

Environment variable: PRIMUS_TURBO_GEMM_BACKEND=TRITON activates these kernels.
"""

from __future__ import annotations

import itertools

import torch
import triton
import triton.language as tl

from primus_turbo.triton.gemm.gemm_kernel import (
    NUM_XCDS,
    _chiplet_transform_chunked,
    _compute_sk_grid,
    _get_hardware,
    _select_params_origami,
)

# ═══════════════════════════════════════════════════════════════════════════════
# AMD knobs helper (blockwise-specific)
# ═══════════════════════════════════════════════════════════════════════════════


def _set_amd_knobs(enable: bool = True):
    """Set AMD-specific Triton knobs.
    NOTE: use_async_copy and scalarize_packed_fops HURT CRR/TN performance
    by ~5-8% on MI300X. Only enable for NT/RCR and NN/RRR.
    """
    if hasattr(triton, "knobs") and hasattr(triton.knobs, "amd"):
        triton.knobs.amd.use_async_copy = enable
        triton.knobs.amd.scalarize_packed_fops = enable


# ###########################################################################
#
#  PART 1 — TENSORWISE (per-tensor) FP8 GEMM
#
# ###########################################################################


def offline_select_fp8(M, N, K, s_ak, s_bk):
    """FP8 config selection from MI300X bench data (out_fp8_gemm_persistent_full.yaml, 184 entries).

    Stride → layout:
      TN (trans_a=False, trans_b=True):  s_ak=1, s_bk=1   → C = A @ B^T
      TT (trans_a=False, trans_b=False): s_ak=1, s_bk≠1   → C = A @ B
      NT (trans_a=True,  trans_b=False): s_ak≠1, s_bk≠1   → C = A^T @ B

    Returns (BM, BN, BK, GM, NUM_SMS, CHUNK, CA, CB).
    """
    is_tn = s_ak == 1 and s_bk == 1

    tiles_m = (M + 255) // 256
    tiles_n = (N + 255) // 256

    # ── Block sizes (97% = 256×256, BK depends on layout) ──
    BM, BN = 256, 256
    BK = 128 if is_tn else 64  # TN→128 (84%), TT→64 (100%), NT→64 (91%)

    # TN with N≈3584 (tiles_n≤14) + M≥8192 (tiles_m≥32): 128×128×128
    # quadruples tile count for CU utilisation (448→1792)
    if is_tn and tiles_n <= 14 and tiles_m >= 32:
        BM, BN, BK = 128, 128, 128

    # Recalculate tiles with actual block sizes
    tiles_m = (M + BM - 1) // BM
    tiles_n = (N + BN - 1) // BN
    total_tiles = tiles_m * tiles_n

    cu_count = _get_hardware().N_CU

    # ── NUM_SMS ──
    # tiles <= ~5 waves (cu*5): sk_grid for wave efficiency
    # tiles > ~5 waves: data-parallel (avoids regression at tiles>=1792)
    if total_tiles <= cu_count * 5:
        num_sms = _compute_sk_grid(M, N, K, BM, BN, BK, cu_count)
    else:
        num_sms = total_tiles

    # ── GROUP_SIZE_M ──
    # small tiles: GM=8 (87%); TN large: GM=4 (62%); TT/NT large: GM=5 (68-75%)
    if min(tiles_m, tiles_n) < 16:
        group_m = 8
    elif is_tn:
        group_m = 4
    else:
        group_m = 5

    # ── CHUNK_SIZE ──
    # persistent mode benefits from smaller chunks; data-parallel prefers 64
    if num_sms < total_tiles:
        chunk = min(32, max(1, num_sms // NUM_XCDS))
    else:
        chunk = 64

    return BM, BN, BK, group_m, num_sms, chunk, ".ca", ".ca"


@triton.jit()
def _fp8_persistent_gemm_kernel(
    A,
    B,
    C,
    A_scale_ptr,
    B_scale_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # Load per-tensor scales once (scalar)
    scale_a = tl.load(A_scale_ptr)
    scale_b = tl.load(B_scale_ptr)
    scale = scale_a * scale_b

    acc_dtype = tl.float32

    for tile_id in range(pid, total_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        # Use int64 offsets for pointer arithmetic to prevent int32 overflow with large matrices
        A_BASE = A + rm[:, None].to(tl.int64) * stride_am + rk[None, :].to(tl.int64) * stride_ak
        B_BASE = B + rk[:, None].to(tl.int64) * stride_bk + rn[None, :].to(tl.int64) * stride_bn

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1
        tl.assume(loop_k > 1)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_A)
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_A)

            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_B)
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_B)

            acc += tl.dot(a, b, input_precision="ieee")
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None].to(tl.int64) * stride_am + rk[None, :].to(tl.int64) * stride_ak
            B_BASE = B + rk[:, None].to(tl.int64) * stride_bk + rn[None, :].to(tl.int64) * stride_bn
            if stride_ak == 1:
                A_BASE = tl.multiple_of(A_BASE, (1, 16))
            else:
                A_BASE = tl.multiple_of(A_BASE, (16, 1))
            if stride_bk == 1:
                B_BASE = tl.multiple_of(B_BASE, (16, 1))
            else:
                B_BASE = tl.multiple_of(B_BASE, (1, 16))
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0, cache_modifier=CACHE_MODIFIER_A)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0, cache_modifier=CACHE_MODIFIER_B)
            acc += tl.dot(a, b, input_precision="ieee")

        # Apply per-tensor scale
        acc *= scale
        c = acc.to(C.type.element_ty)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None].to(tl.int64) * stride_cm + rn[None, :].to(tl.int64) * stride_cn
        tl.store(C_, c, c_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# Tensorwise FP8 GEMM Public API
# ═══════════════════════════════════════════════════════════════════════════════


def gemm_fp8_tensorwise_triton_kernel(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
) -> torch.Tensor:
    """General-purpose FP8 GEMM with per-tensor scaling.

    Computes: C = op(A) @ op(B) * a_scale_inv * b_scale_inv
    If trans_c=True, returns C^T (contiguous, shape N×M).

    Args:
        a: FP8 input matrix.
        a_scale_inv: Per-tensor dequantization scale for A, shape (1,), fp32.
        b: FP8 input matrix.
        b_scale_inv: Per-tensor dequantization scale for B, shape (1,), fp32.
        trans_a: Whether A is transposed.
        trans_b: Whether B is transposed.
        out_dtype: Output dtype (default bfloat16).
        trans_c: If True, return transposed output C^T (shape N×M).

    Returns:
        C of shape (M, N) if trans_c=False, or (N, M) if trans_c=True.
    """
    if trans_a:
        K, M = a.shape
        A_view = a.T
    else:
        M, K = a.shape
        A_view = a

    if trans_b:
        N, K2 = b.shape
        B_view = b.T
    else:
        K2, N = b.shape
        B_view = b

    assert K == K2, f"K mismatch: A gives K={K}, B gives K={K2}"

    # Ensure views have proper strides (no broadcast/expand zeros from autograd)
    if A_view.stride(0) == 0 or A_view.stride(1) == 0:
        A_view = A_view.contiguous()
    if B_view.stride(0) == 0 or B_view.stride(1) == 0:
        B_view = B_view.contiguous()

    # Handle trans_c by writing to (N, M) buffer with swapped strides
    if trans_c:
        out = torch.empty((N, M), device=a.device, dtype=out_dtype)
        stride_cm = out.stride(1)  # = 1
        stride_cn = out.stride(0)  # = M
    else:
        out = torch.empty((M, N), device=a.device, dtype=out_dtype)
        stride_cm = out.stride(0)  # = N
        stride_cn = out.stride(1)  # = 1

    # Stride constexprs for compiler optimisation
    s_ak = A_view.stride(1)
    s_bk = B_view.stride(0)

    block_m, block_n, block_k, group_m, num_sms, chunk_size, cache_a, cache_b = offline_select_fp8(
        M, N, K, s_ak, s_bk
    )
    origami_params = _select_params_origami(
        M,
        N,
        K,
        out_dtype,
        A_view.dtype,
        B_view.dtype,
        trans_a=trans_a,
        trans_b=trans_b,
    )
    if origami_params is not None:
        om, on, ok, ogm, oca, ocb = origami_params
        if (om, on, ok) == (block_m, block_n, block_k):
            group_m = ogm
            cache_a = oca
            cache_b = ocb

    args = (
        A_view,
        B_view,
        out,
        a_scale_inv,
        b_scale_inv,
        M,
        N,
        K,
        A_view.stride(0),
        B_view.stride(1),
        stride_cm,
        stride_cn,
    )
    even_k = K % block_k == 0
    _fp8_persistent_gemm_kernel[(num_sms,)](
        *args,
        stride_ak=s_ak,
        stride_bk=s_bk,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_m,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=chunk_size,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=cache_a,
        CACHE_MODIFIER_B=cache_b,
        num_warps=8,
        num_stages=2,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return out


# ###########################################################################
#
#  PART 1.5 — ROWWISE (per-row / per-col vector) FP8 GEMM
#
# ###########################################################################


# ═══════════════════════════════════════════════════════════════════════════════
# Rowwise FP8 persistent kernel
#
# Identical to the tensorwise kernel except:
#   - A_scale_ptr is (M,) fp32 and B_scale_ptr is (N,) fp32
#   - Scale is applied as  acc *= a_scale[rm][:, None] * b_scale[rn][None, :]
# ═══════════════════════════════════════════════════════════════════════════════


@triton.jit()
def _fp8_rowwise_persistent_gemm_kernel(
    A,
    B,
    C,
    A_scale_ptr,  # (M,) fp32 — per output-row scale
    B_scale_ptr,  # (N,) fp32 — per output-col scale
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
):
    """Persistent FP8 GEMM kernel with per-row/per-col vector scaling.

    Computes: C[m,n] = (sum_k A[m,k] * B[k,n]) * a_scale[m] * b_scale[n]

    Kernel structure mirrors ``_fp8_persistent_gemm_kernel`` (tensorwise)
    and the standalone ``fp8_gemm_8192_mi300.py`` benchmark kernel.
    """
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32

    for tile_id in range(pid, total_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        # Use int64 offsets for pointer arithmetic to prevent int32 overflow with large matrices
        A_BASE = A + rm[:, None].to(tl.int64) * stride_am + rk[None, :].to(tl.int64) * stride_ak
        B_BASE = B + rk[:, None].to(tl.int64) * stride_bk + rn[None, :].to(tl.int64) * stride_bn

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1
        tl.assume(loop_k > 1)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_A)
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_A)

            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_B)
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_B)

            acc += tl.dot(a, b, input_precision="ieee")
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None].to(tl.int64) * stride_am + rk[None, :].to(tl.int64) * stride_ak
            B_BASE = B + rk[:, None].to(tl.int64) * stride_bk + rn[None, :].to(tl.int64) * stride_bn
            if stride_ak == 1:
                A_BASE = tl.multiple_of(A_BASE, (1, 16))
            else:
                A_BASE = tl.multiple_of(A_BASE, (16, 1))
            if stride_bk == 1:
                B_BASE = tl.multiple_of(B_BASE, (16, 1))
            else:
                B_BASE = tl.multiple_of(B_BASE, (1, 16))
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0, cache_modifier=CACHE_MODIFIER_A)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0, cache_modifier=CACHE_MODIFIER_B)
            acc += tl.dot(a, b, input_precision="ieee")

        # Apply per-row/per-col vector scales
        a_scale = tl.load(A_scale_ptr + rm)
        b_scale = tl.load(B_scale_ptr + rn)
        acc *= a_scale[:, None] * b_scale[None, :]
        c = acc.to(C.type.element_ty)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None].to(tl.int64) * stride_cm + rn[None, :].to(tl.int64) * stride_cn
        tl.store(C_, c, c_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# Rowwise FP8 GEMM Public API
# ═══════════════════════════════════════════════════════════════════════════════


def gemm_fp8_rowwise_triton_kernel(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
) -> torch.Tensor:
    """General-purpose FP8 GEMM with per-row / per-col (rowwise) scaling.

    Computes: C = op(A) @ op(B)  element-wise scaled by a_scale and b_scale.
    Scale semantics (after transposing into [M,K]@[K,N] form):
      a_scale_inv: (M,) fp32 — per output-row
      b_scale_inv: (N,) fp32 — per output-col

    Typical caller scale shapes from ``FP8GemmRowFunction``:
      NT (forward):  a=[M,K] axis=-1 → (M,);  b=[N,K] axis=-1 → (N,)
      NN (grad_X):   grad_out=[M,N] axis=-1 → (M,);  b_col=[N,K] axis=-2 → (K,)
      TN (grad_W):   a_col=[M,K] axis=-2 → (K,);  grad_out_col=[M,N] axis=-2 → (N,)

    Args:
        a: FP8 input matrix.
        a_scale_inv: Per-row dequantization scale for A (after transpose), shape (M,), fp32.
        b: FP8 input matrix.
        b_scale_inv: Per-col dequantization scale for B (after transpose), shape (N,), fp32.
        trans_a: Whether A is transposed.
        trans_b: Whether B is transposed.
        out_dtype: Output dtype (default bfloat16).
        trans_c: If True, return transposed output C^T (shape N×M).

    Returns:
        C of shape (M, N) if trans_c=False, or (N, M) if trans_c=True.
    """
    if trans_a:
        K, M = a.shape
        A_view = a.T
    else:
        M, K = a.shape
        A_view = a

    if trans_b:
        N, K2 = b.shape
        B_view = b.T
    else:
        K2, N = b.shape
        B_view = b

    assert K == K2, f"K mismatch: A gives K={K}, B gives K={K2}"

    # Ensure views have proper strides (no broadcast/expand zeros from autograd)
    if A_view.stride(0) == 0 or A_view.stride(1) == 0:
        A_view = A_view.contiguous()
    if B_view.stride(0) == 0 or B_view.stride(1) == 0:
        B_view = B_view.contiguous()

    # Handle trans_c by writing to (N, M) buffer with swapped strides
    if trans_c:
        out = torch.empty((N, M), device=a.device, dtype=out_dtype)
        stride_cm = out.stride(1)  # = 1
        stride_cn = out.stride(0)  # = M
    else:
        out = torch.empty((M, N), device=a.device, dtype=out_dtype)
        stride_cm = out.stride(0)  # = N
        stride_cn = out.stride(1)  # = 1

    # Stride constexprs for compiler optimisation
    s_ak = A_view.stride(1)
    s_bk = B_view.stride(0)

    block_m, block_n, block_k, group_m, num_sms, chunk_size, cache_a, cache_b = offline_select_fp8(
        M, N, K, s_ak, s_bk
    )
    origami_params = _select_params_origami(
        M,
        N,
        K,
        out_dtype,
        A_view.dtype,
        B_view.dtype,
        trans_a=trans_a,
        trans_b=trans_b,
    )
    if origami_params is not None:
        om, on, ok, ogm, oca, ocb = origami_params
        if (om, on, ok) == (block_m, block_n, block_k):
            group_m = ogm
            cache_a = oca
            cache_b = ocb

    args = (
        A_view,
        B_view,
        out,
        a_scale_inv,
        b_scale_inv,
        M,
        N,
        K,
        A_view.stride(0),
        B_view.stride(1),
        stride_cm,
        stride_cn,
    )
    even_k = K % block_k == 0
    _fp8_rowwise_persistent_gemm_kernel[(num_sms,)](
        *args,
        stride_ak=s_ak,
        stride_bk=s_bk,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_m,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=chunk_size,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=cache_a,
        CACHE_MODIFIER_B=cache_b,
        num_warps=8,
        num_stages=2,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return out


# ###########################################################################
#
#  PART 2 — BLOCKWISE FP8 GEMM
#
# ###########################################################################


# ═══════════════════════════════════════════════════════════════════════════════
# Blockwise autotune configs (32 total)
# Fixed: BLOCK_N=128, BLOCK_K=128, wpe=2, mfma=16
# Variable: BLOCK_M, kpack, GROUP_M, CHUNK, num_stages
# ═══════════════════════════════════════════════════════════════════════════════
def _get_blockwise_autotune_configs():
    configs = []
    for block_m, kp, gm, chunk, ns in itertools.product(
        [128, 256],
        [1, 2],
        [4, 8],
        [32, 64],
        [1, 2],
    ):
        nw = 4 if block_m == 128 else 8
        configs.append(
            triton.Config(
                {
                    "BLOCK_M": block_m,
                    "BLOCK_N": 128,
                    "BLOCK_K": 128,
                    "GROUP_M": gm,
                    "NUM_XCDS": 8,
                    "CHUNK": chunk,
                },
                num_warps=nw,
                num_stages=ns,
                pre_hook=None,
            )
        )
    return configs


# ═══════════════════════════════════════════════════════════════════════════════
# Autotuned block-wise FP8 GEMM kernel
# ═══════════════════════════════════════════════════════════════════════════════
@triton.autotune(
    configs=_get_blockwise_autotune_configs(),
    key=["M", "N", "K", "A_K_CONTIGUOUS", "B_K_CONTIGUOUS", "SCALE_2D_B"],
)
@triton.jit
def _blockwise_fp8_autotune_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    A_scales_ptr,
    B_scales_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak_val,
    stride_bk_val,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_as_k,
    stride_as_m,
    stride_bs_0,
    stride_bs_1,
    NUM_SMS,
    NUM_K_BLOCKS,
    A_K_CONTIGUOUS: tl.constexpr,
    B_K_CONTIGUOUS: tl.constexpr,
    SCALE_2D_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)

    # XCD-aware PID transform (inlined from _chiplet_transform_chunked
    # because NUM_SMS is not constexpr in this kernel)
    if NUM_XCDS != 1:
        full_chunk_pids = (NUM_SMS // (NUM_XCDS * CHUNK)) * (NUM_XCDS * CHUNK)
        if pid <= full_chunk_pids:
            local_pid = pid // NUM_XCDS
            chunk_idx = local_pid // CHUNK
            pos_in_chunk = local_pid % CHUNK
            xcd = pid % NUM_XCDS
            pid = chunk_idx * NUM_XCDS * CHUNK + xcd * CHUNK + pos_in_chunk

    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    total = num_m * num_n
    grp = GROUP_M * num_n

    tl.assume(stride_am > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    for tid in range(pid, total, NUM_SMS):
        gid = tid // grp
        fm = gid * GROUP_M
        gs = min(num_m - fm, GROUP_M)
        pm = fm + (tid % grp) % gs
        pn = (tid % grp) // gs
        tl.assume(pm >= 0)
        tl.assume(pn >= 0)

        rm = tl.max_contiguous(tl.multiple_of((pm * BLOCK_M + tl.arange(0, BLOCK_M)) % M, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of((pn * BLOCK_N + tl.arange(0, BLOCK_N)) % N, BLOCK_N), BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        if A_K_CONTIGUOUS:
            a_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :]
        else:
            a_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak_val

        if B_K_CONTIGUOUS:
            b_ptrs = B_ptr + rk[:, None] + rn[None, :] * stride_bn
        else:
            b_ptrs = B_ptr + rk[:, None] * stride_bk_val + rn[None, :] * stride_bn

        as_ptrs = A_scales_ptr + rm * stride_as_m

        if SCALE_2D_B:
            bs_ptr_base = B_scales_ptr + pn * stride_bs_0
        else:
            bs_ptrs = B_scales_ptr + rn * stride_bs_0

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for ki in range(NUM_K_BLOCKS):
            # Mask for partial K blocks (last iteration when K % BLOCK_K != 0)
            k_remaining = K - ki * BLOCK_K
            mask_k_col = rk[None, :] < k_remaining  # [1, BLOCK_K] for A
            mask_k_row = rk[:, None] < k_remaining  # [BLOCK_K, 1] for B

            if A_K_CONTIGUOUS:
                a = tl.load(a_ptrs, mask=mask_k_col, other=0.0, cache_modifier=".ca")
            else:
                a = tl.load(a_ptrs, mask=mask_k_col, other=0.0, cache_modifier=".ca")

            if B_K_CONTIGUOUS:
                b = tl.load(b_ptrs, mask=mask_k_row, other=0.0, cache_modifier=".ca")
            else:
                b = tl.load(b_ptrs, mask=mask_k_row, other=0.0, cache_modifier=".ca")

            partial = tl.dot(a, b, input_precision="ieee")

            a_s = tl.load(as_ptrs + ki * stride_as_k)

            if SCALE_2D_B:
                b_s = tl.load(bs_ptr_base + ki * stride_bs_1)
                acc += partial * (a_s * b_s)[:, None]
            else:
                b_s = tl.load(bs_ptrs + ki * stride_bs_1)
                acc += partial * a_s[:, None] * b_s[None, :]

            if A_K_CONTIGUOUS:
                a_ptrs += BLOCK_K
            else:
                a_ptrs += BLOCK_K * stride_ak_val

            if B_K_CONTIGUOUS:
                b_ptrs += BLOCK_K
            else:
                b_ptrs += BLOCK_K * stride_bk_val

        offs_m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(C_ptr.type.element_ty), mask)


# ═══════════════════════════════════════════════════════════════════════════════
# Unified Public API — Block-wise FP8 GEMM
# Interface consistent with CK blockwise backend.
# ═══════════════════════════════════════════════════════════════════════════════


def gemm_fp8_blockwise_triton_kernel(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
) -> torch.Tensor:
    """Unified block-wise FP8 GEMM Triton kernel.

    Interface consistent with the CK blockwise backend.
    Dispatches internally by (trans_a, trans_b) to the optimal layout.

    Supported layouts:
      NT/RCR (forward):  trans_a=False, trans_b=True
        A: [M, K], A_scales: [M, K//128]
        B: [N, K], B_scales: [N//128, K//128]  (2D block scaling)

      NN/RRR (grad_X):   trans_a=False, trans_b=False
        A: [M, K], A_scales: [M, K//128]
        B: [N, K], B_scales: [N//128, K//128]  (2D, transposed internally)

      TN/CRR (grad_W):   trans_a=True, trans_b=False
        A: [K, M], A_scales: [K//128, M]
        B: [K, N], B_scales: [K//128, N]

    Args:
        a: FP8 input matrix.
        a_scale_inv: Block-wise scale for A, shape depends on layout.
        trans_a: Whether A is transposed.
        b: FP8 input matrix.
        b_scale_inv: Block-wise scale for B, shape depends on layout.
        trans_b: Whether B is transposed.
        out_dtype: Output dtype (default bfloat16).
        trans_c: If True, return transposed output.

    Returns:
        C of shape (M, N) if trans_c=False, or (N, M) if trans_c=True.
    """
    if not trans_a and trans_b:
        return _blockwise_nt(a, a_scale_inv, b, b_scale_inv, out_dtype, trans_c)
    elif not trans_a and not trans_b:
        return _blockwise_nn(a, a_scale_inv, b, b_scale_inv, out_dtype, trans_c)
    elif trans_a and not trans_b:
        return _blockwise_tn(a, a_scale_inv, b, b_scale_inv, out_dtype, trans_c)
    else:
        raise ValueError(f"Unsupported layout for blockwise FP8 Triton: trans_a={trans_a}, trans_b={trans_b}")


# ═══════════════════════════════════════════════════════════════════════════════
# Internal layout-specific helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _blockwise_nt(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    out_dtype: torch.dtype,
    trans_c: bool,
) -> torch.Tensor:
    """NT/RCR forward: C = A[M,K] @ B[N,K].T, 2D B_scales."""
    _set_amd_knobs(enable=True)

    M, K = a.shape
    N = b.shape[0]
    num_k = (K + 127) // 128

    if trans_c:
        out = torch.empty((N, M), device=a.device, dtype=out_dtype)
        stride_cm, stride_cn = out.stride(1), out.stride(0)
    else:
        out = torch.empty((M, N), device=a.device, dtype=out_dtype)
        stride_cm, stride_cn = out.stride(0), out.stride(1)

    A_scales_t = a_scale_inv.T.contiguous()  # [K//128, M]
    B_t = b.T  # view [K, N]

    num_m = (M + 127) // 128
    num_n = (N + 127) // 128
    NUM_SMS = num_m * num_n

    _blockwise_fp8_autotune_kernel[(NUM_SMS,)](
        a,
        B_t,
        out,
        A_scales_t,
        b_scale_inv,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        B_t.stride(0),
        B_t.stride(1),
        stride_cm,
        stride_cn,
        A_scales_t.stride(0),
        A_scales_t.stride(1),
        b_scale_inv.stride(0),
        b_scale_inv.stride(1),
        NUM_SMS,
        num_k,
        A_K_CONTIGUOUS=True,
        B_K_CONTIGUOUS=True,
        SCALE_2D_B=True,
    )
    return out


def _blockwise_nn(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    out_dtype: torch.dtype,
    trans_c: bool,
) -> torch.Tensor:
    """NN/RRR grad_X: C = A[M,K] @ B[K,N], 2D B_scales (transposed internally)."""
    _set_amd_knobs(enable=True)

    M, K = a.shape
    _, N = b.shape
    num_k = (K + 127) // 128

    if trans_c:
        out = torch.empty((N, M), device=a.device, dtype=out_dtype)
        stride_cm, stride_cn = out.stride(1), out.stride(0)
    else:
        out = torch.empty((M, N), device=a.device, dtype=out_dtype)
        stride_cm, stride_cn = out.stride(0), out.stride(1)

    A_scales_t = a_scale_inv.T.contiguous()  # [K//128, M]
    # B_scales from quantization: [dim0_blocks, dim1_blocks] for weight stored as [N_fwd, K_fwd].
    # Kernel expects [N_output_blocks, K_inner_blocks] indexing → transpose.
    b_scale_inv_t = b_scale_inv.T.contiguous()

    num_m = (M + 127) // 128
    num_n = (N + 127) // 128
    NUM_SMS = num_m * num_n

    _blockwise_fp8_autotune_kernel[(NUM_SMS,)](
        a,
        b,
        out,
        A_scales_t,
        b_scale_inv_t,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        stride_cm,
        stride_cn,
        A_scales_t.stride(0),
        A_scales_t.stride(1),
        b_scale_inv_t.stride(0),
        b_scale_inv_t.stride(1),
        NUM_SMS,
        num_k,
        A_K_CONTIGUOUS=True,
        B_K_CONTIGUOUS=False,
        SCALE_2D_B=True,
    )
    return out


def _blockwise_tn(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    out_dtype: torch.dtype,
    trans_c: bool,
) -> torch.Tensor:
    """TN/CRR grad_W: C = A[K,M].T @ B[K,N], 1D+1D B_scales.

    Scale layouts (from axis=0 / column-wise quantization):
      a_scale_inv: [K//128, M]
      b_scale_inv: [K//128, N]
    """
    _set_amd_knobs(enable=False)

    K, M = a.shape
    _, N = b.shape
    num_k = (K + 127) // 128

    if trans_c:
        out = torch.empty((N, M), device=a.device, dtype=out_dtype)
        stride_cm, stride_cn = out.stride(1), out.stride(0)
    else:
        out = torch.empty((M, N), device=a.device, dtype=out_dtype)
        stride_cm, stride_cn = out.stride(0), out.stride(1)

    A_view = a.T  # [M, K] view with strided K

    num_m = (M + 127) // 128
    num_n = (N + 127) // 128
    NUM_SMS = num_m * num_n

    _blockwise_fp8_autotune_kernel[(NUM_SMS,)](
        A_view,
        b,
        out,
        a_scale_inv,
        b_scale_inv,
        M,
        N,
        K,
        A_view.stride(0),
        A_view.stride(1),
        b.stride(0),
        b.stride(1),
        stride_cm,
        stride_cn,
        a_scale_inv.stride(0),
        a_scale_inv.stride(1),  # [K//128, M]: stride_as_k=M, stride_as_m=1
        b_scale_inv.stride(1),
        b_scale_inv.stride(0),  # [K//128, N]: stride_bs_0=1(rn), stride_bs_1=N(ki)
        NUM_SMS,
        num_k,
        A_K_CONTIGUOUS=False,
        B_K_CONTIGUOUS=False,
        SCALE_2D_B=False,
    )
    return out
