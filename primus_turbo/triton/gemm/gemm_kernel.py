###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Optimized GEMM kernels for MI300X GPUs (Triton persistent kernels).

Contains:
  - BF16/FP16 persistent GEMM kernel
    * 256×256×64 tiles, 2-config autotune for (CHUNK_SIZE, waves_per_eu)
    * Heuristic GROUP_SIZE_M (not autotuned): TN large→5, small→8, default→4
  - FP8 per-tensor scaling persistent GEMM kernel
    * 256×256 tiles, 4-config autotune for (BLOCK_SIZE_K, CHUNK_SIZE), pruned by layout
    * Heuristic GROUP_SIZE_M: small→8, default→4
  - StreamK grid computation utility

Supported layouts: NN, NT, TN

Environment variable: PRIMUS_TURBO_GEMM_BACKEND=TRITON activates these kernels.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

# ═══════════════════════════════════════════════════════════════════════════════
# Grid Utilities (StreamK)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_sk_grid(
    M: int,
    N: int,
    K: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_cus: int,
    max_workspace_bytes: int = 128 * 1024 * 1024,
) -> int:
    """Compute optimal StreamK grid size based on problem dimensions and hardware.

    Implements the dynamic grid logic from tritonBLAS/Origami:
    - If tiles > CUs: Try fractional splits to balance work
    - If tiles < CUs: Split along K-dimension
    - Consider workspace constraints for partial tiles
    """
    tiles = math.ceil(M / block_m) * math.ceil(N / block_n)
    iters_per_tile = max(1, math.ceil(K / block_k))

    sk_grid = tiles

    tile_fractions = [0.0, 1.0 / 2, 1.0 / 8, 1.0 / 5, 1.0 / 4, 1.0 / 3]
    split_factors = [8, 6, 4, 3, 2, 1]

    if tiles > num_cus:
        min_even_tiles = tiles / num_cus
        for frac in tile_fractions:
            frac_grid = int((tiles / (min_even_tiles + frac)) + 0.5)
            partial_tile_bytes = block_m * block_n * 2 * frac_grid
            if tiles % frac_grid != 0 and partial_tile_bytes > max_workspace_bytes:
                continue
            if frac_grid <= num_cus:
                sk_grid = frac_grid
                break
    elif tiles < num_cus:
        for factor in split_factors:
            split_grid = tiles * factor
            iters_per_cu = iters_per_tile // factor
            if split_grid <= num_cus and iters_per_cu >= 8:
                sk_grid = split_grid
                break

    if tiles % sk_grid != 0:
        sk_grid = tiles

    if tiles >= num_cus:
        last_wave_remainder = tiles % num_cus
        if last_wave_remainder < 128 and last_wave_remainder > 0 and num_cus == 304:
            sk_grid = 256

    return sk_grid


# ═══════════════════════════════════════════════════════════════════════════════
# Hardware constants (lazy init)
# ═══════════════════════════════════════════════════════════════════════════════

NUM_XCDS = 8
_NUM_CUS: Optional[int] = None


def _get_num_cus() -> int:
    """Lazy initialization of CU count (avoids import-time CUDA calls)."""
    global _NUM_CUS
    if _NUM_CUS is None:
        _NUM_CUS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return _NUM_CUS


# ═══════════════════════════════════════════════════════════════════════════════
# Chiplet Transform (shared helper)
# ═══════════════════════════════════════════════════════════════════════════════


@triton.jit
def _chiplet_transform_chunked(
    pid,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    if pid > (NUM_SMS // (NUM_XCDS * CHUNK_SIZE)) * (NUM_XCDS * CHUNK_SIZE):
        return pid
    local_pid = pid // NUM_XCDS
    chunk_idx = local_pid // CHUNK_SIZE
    pos_in_chunk = local_pid % CHUNK_SIZE
    xcd = pid % NUM_XCDS
    return chunk_idx * NUM_XCDS * CHUNK_SIZE + xcd * CHUNK_SIZE + pos_in_chunk


# ═══════════════════════════════════════════════════════════════════════════════
# BF16 Persistent GEMM Kernel
# ═══════════════════════════════════════════════════════════════════════════════


def _get_bf16_autotune_configs():
    """BF16 autotune configs — only 2 (fast compilation).

    Key insight from 94-entry exhaustive tuning on MI300X:
      - CHUNK_SIZE and waves_per_eu are strongly correlated:
        * CHUNK=32 → waves_per_eu=2  (93.6%)
        * CHUNK=64 → waves_per_eu=0  (100%)
      - GROUP_SIZE_M: determined by heuristic (not autotuned)
    """
    configs = []
    for chunk_size, waves in [(32, 2), (64, 0)]:
        configs.append(
            triton.Config(
                {
                    "CHUNK_SIZE": chunk_size,
                    "waves_per_eu": waves,
                    "matrix_instr_nonkdim": 16,
                    "kpack": 1,
                },
                num_warps=8,
                num_stages=2,
            )
        )
    return configs


def _select_group_size_m_bf16(M, N, stride_ak, stride_bk):
    """Deterministic GROUP_SIZE_M from 94-entry BF16 MI300 tuning data.

    Patterns:
    - min_tile < 16 (non-standard dims like 3584): GROUP=8
    - TN layout (stride_ak!=1, stride_bk!=1) + both tiles >= 32: GROUP=5
      (BF16 tuning: 8 of 8 large TN cases prefer GROUP=5)
    - Default: 4
    """
    tiles_m = (M + 255) // 256
    tiles_n = (N + 255) // 256
    min_tile = min(tiles_m, tiles_n)

    is_tn = (stride_ak != 1) and (stride_bk != 1)

    if min_tile < 16:
        return 8
    elif is_tn and tiles_m >= 32 and tiles_n >= 32:
        return 5
    else:
        return 4


@triton.autotune(
    configs=_get_bf16_autotune_configs(),
    key=["M", "N", "K", "GROUP_SIZE_M", "NUM_SMS", "EVEN_K", "stride_ak", "stride_bk"],
    warmup=20,
    rep=80,
)
@triton.jit()
def _bf16_persistent_gemm_kernel(
    A,
    B,
    C,
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
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
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
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

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

            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
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
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

        c = acc.to(C.type.element_ty)
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, c_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# FP8 Persistent GEMM Kernel (Per-Tensor Scaling)
# ═══════════════════════════════════════════════════════════════════════════════


def _get_fp8_autotune_configs():
    """FP8 autotune configs — 4 total, pruned to 2 by layout.

    Hardcoded from 98-entry exhaustive tuning on MI300X:
      - waves_per_eu=0 (85% of cases)
      - GROUP_SIZE_M: computed by heuristic, passed directly

    Autotuned:
      - BLOCK_SIZE_K ∈ {64, 128}
      - CHUNK_SIZE ∈ {32, 64}

    Total: 4 configs.
    """
    configs = []
    for blk_k in [64, 128]:
        for chunk in [32, 64]:
            configs.append(
                triton.Config(
                    {
                        "BLOCK_SIZE_K": blk_k,
                        "CHUNK_SIZE": chunk,
                        "waves_per_eu": 0,
                        "matrix_instr_nonkdim": 16,
                        "kpack": 1,
                    },
                    num_warps=8,
                    num_stages=2,
                )
            )
    return configs


def _select_group_size_m_fp8(M, N, stride_ak, stride_bk):
    """Deterministic GROUP_SIZE_M from FP8 MI300 tuning data.

    FP8 patterns:
    - min_tile < 16 (non-standard dims like 3584): GROUP=8
    - Default (including TN): GROUP=4
      (29/32 FP8 TN entries use GROUP=4)
    """
    tiles_m = (M + 255) // 256
    tiles_n = (N + 255) // 256
    min_tile = min(tiles_m, tiles_n)

    if min_tile < 16:
        return 8
    else:
        return 4


def _prune_fp8_configs(configs, named_args, **kwargs):
    """Prune FP8 configs — placeholder for future layout-specific pruning.

    Currently keeps all 4 configs for NN/NT/TN layouts.
    """
    return configs


@triton.autotune(
    configs=_get_fp8_autotune_configs(),
    key=["M", "N", "K", "GROUP_SIZE_M", "NUM_SMS", "EVEN_K", "stride_ak", "stride_bk"],
    prune_configs_by={"early_config_prune": _prune_fp8_configs},
    warmup=20,
    rep=80,
)
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
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

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
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
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
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, c_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — BF16 GEMM
# ═══════════════════════════════════════════════════════════════════════════════


def gemm_triton_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
) -> torch.Tensor:
    """General-purpose BF16/FP16 GEMM on MI300 using optimized persistent kernel.

    Computes: C = op(A) @ op(B), where op(X) = X^T if trans else X.
    If trans_c=True, returns C^T (contiguous, shape N×M).

    Args:
        a: Input matrix (BF16 or FP16).
        b: Input matrix (BF16 or FP16).
        trans_a: Whether A is transposed.
        trans_b: Whether B is transposed.
        out_dtype: Output dtype (default bfloat16).
        trans_c: If True, return transposed output C^T (shape N×M).

    Returns:
        C of shape (M, N) if trans_c=False, or (N, M) if trans_c=True.
    """
    assert a.dtype in (torch.bfloat16, torch.float16), f"Unsupported dtype: {a.dtype}"
    assert b.dtype in (torch.bfloat16, torch.float16), f"Unsupported dtype: {b.dtype}"
    # Determine logical (M, K) and (K, N) views
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

    # Handle trans_c by writing to a (N, M) buffer with swapped strides
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

    # Heuristic GROUP_SIZE_M (not autotuned)
    group_m = _select_group_size_m_bf16(M, N, s_ak, s_bk)

    num_cus = _get_num_cus()
    num_sms = compute_sk_grid(M, N, K, 256, 256, 64, num_cus)
    even_k = K % 64 == 0

    _bf16_persistent_gemm_kernel[(num_sms,)](
        A_view,
        B_view,
        out,
        M,
        N,
        K,
        A_view.stride(0),
        B_view.stride(1),
        stride_cm,
        stride_cn,
        stride_ak=s_ak,
        stride_bk=s_bk,
        BLOCK_SIZE_M=256,
        BLOCK_SIZE_N=256,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=group_m,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=".ca",
        CACHE_MODIFIER_B=".ca",
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — FP8 GEMM (Per-Tensor Scaling)
# ═══════════════════════════════════════════════════════════════════════════════


def fp8_gemm_pertensor(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
) -> torch.Tensor:
    """General-purpose FP8 GEMM on MI300 with per-tensor scaling.

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

    # Heuristic GROUP_SIZE_M (FP8-specific, not autotuned)
    group_m = _select_group_size_m_fp8(M, N, s_ak, s_bk)

    num_cus = _get_num_cus()
    num_sms = compute_sk_grid(M, N, K, 256, 256, 64, num_cus)
    even_k = K % 128 == 0  # Safe for both BLK_K=64 and 128

    _fp8_persistent_gemm_kernel[(num_sms,)](
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
        stride_ak=s_ak,
        stride_bk=s_bk,
        BLOCK_SIZE_M=256,
        BLOCK_SIZE_N=256,
        GROUP_SIZE_M=group_m,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=".ca",
        CACHE_MODIFIER_B=".ca",
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy API (backward compatibility with gemm_triton_impl.py)
# ═══════════════════════════════════════════════════════════════════════════════
