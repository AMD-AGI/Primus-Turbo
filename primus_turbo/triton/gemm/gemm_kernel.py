###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Acknowledgement:
#   The persistent GEMM kernels in this file are adapted from tritonBLAS
#   (https://github.com/ROCm/tritonBLAS). We thank the tritonBLAS authors
#   for their high-quality Triton kernel implementations on AMD GPUs.
###############################################################################

"""
GEMM Triton persistent kernels — BF16/FP16.

Contains:
  - _bf16_persistent_gemm_kernel: BF16/FP16 persistent kernel
  - StreamK grid computation utility

Public API:
  - gemm_triton_kernel  — BF16/FP16 GEMM

FP8 kernels (tensorwise + blockwise) are in gemm_fp8_kernel.py.

Environment variable: PRIMUS_TURBO_GEMM_BACKEND=TRITON activates these kernels.
"""

from __future__ import annotations

import functools
import itertools
import math
import os
from typing import Optional

import torch
import triton
import triton.language as tl

try:
    import origami

    _HAS_ORIGAMI = not os.environ.get("PRIMUS_TURBO_DISABLE_ORIGAMI", "")
except ModuleNotFoundError:
    _HAS_ORIGAMI = False

# ═══════════════════════════════════════════════════════════════════════════════
# Grid Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def compute_sk_grid(M, N, K, block_m, block_n, block_k, num_cus, max_workspace_bytes=128 * 1024 * 1024):
    """Compute optimal StreamK grid size based on problem dimensions and hardware.

    Implements the dynamic grid logic from tritonBLAS/Origami:
    - If tiles > CUs: Try fractional splits to balance work
    - If tiles < CUs: Split along K-dimension
    - Consider workspace constraints for partial tiles

    Args:
        M, N, K: Matrix dimensions
        block_m, block_n, block_k: Tile sizes
        num_cus: Number of compute units (e.g., 256 for MI300, 304 for MI325X)
        max_workspace_bytes: Maximum workspace size for partial tiles (default 128MB)

    Returns:
        int: Optimal grid size for StreamK kernel
    """
    tiles = math.ceil(M / block_m) * math.ceil(N / block_n)
    iters_per_tile = max(1, math.ceil(K / block_k))

    sk_grid = tiles

    tile_fractions = [0.0, 1.0 / 2.0, 1.0 / 8.0, 1.0 / 5.0, 1.0 / 4.0, 1.0 / 3.0]
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
# Hardware constants
# ═══════════════════════════════════════════════════════════════════════════════

NUM_XCDS = 8
_NUM_CUS: Optional[int] = None

# Kept for FP8 kernel compatibility (gemm_fp8_kernel.py imports this).
_AUTOTUNE_MN_THRESHOLD = 128 * 1024 * 1024


def _get_num_cus() -> int:
    """Lazy initialization of CU count (avoids import-time CUDA calls)."""
    global _NUM_CUS
    if _NUM_CUS is None:
        _NUM_CUS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return _NUM_CUS


# ═══════════════════════════════════════════════════════════════════════════════
# Chiplet Transform
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


def _select_group_size_m_bf16(M, N, stride_ak, stride_bk):
    """Fallback heuristic GROUP_SIZE_M (used when origami is unavailable).

    Patterns from 94-entry BF16 tuning data:
    - min_tile < 16 (non-standard dims like 3584): GROUP=8
    - TN layout (stride_ak!=1, stride_bk!=1) + both tiles >= 32: GROUP=5
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


# ─── Origami analytical config selection ──────────────────────────────────────

_ORIGAMI_CONFIGS: Optional[list] = None


def _get_origami_configs():
    global _ORIGAMI_CONFIGS
    if _ORIGAMI_CONFIGS is None:
        _ORIGAMI_CONFIGS = _build_origami_configs()
    return _ORIGAMI_CONFIGS


def _build_origami_configs():
    """Build candidate configs for origami (gfx942 BF16)."""
    block_mn = [16, 32, 64, 128, 256]
    block_k = [16, 32, 64, 128, 256, 512]
    configs = []
    for bm, bn, bk in itertools.product(block_mn, block_mn, block_k):
        for wpe in [0, 1, 2]:
            for nw in [4, 8]:
                for ns in [1, 2]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "waves_per_eu": wpe},
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
    return configs


@functools.lru_cache(maxsize=4096)
def _select_params_origami(M, N, K, a_stride, b_stride, out_dtype):
    """Use origami to select macrotile, GROUP_SIZE_M, CHUNK_SIZE, waves_per_eu.

    Results are cached by (M, N, K, strides, dtype).

    Returns:
        (block_m, block_n, block_k, wgm, chunk, wpe) or None.
    """
    if not _HAS_ORIGAMI:
        return None

    try:
        selector = origami.OrigamiMatmulSelector(
            config_gen=_get_origami_configs(),
            m=M,
            n=N,
            k=K,
            a_dtype=torch.bfloat16,
            b_dtype=torch.bfloat16,
            out_dtype=out_dtype,
            device=torch.device(f"cuda:{torch.cuda.current_device()}"),
            a_stride=a_stride,
            b_stride=b_stride,
            batch=1,
        )
        block_m = selector.macrotile_m
        block_n = selector.macrotile_n
        block_k = selector.macrotile_k
        wgm = selector.wgm
        chunk = selector.wgmxccchunk if selector.wgmxccchunk > 0 else 32
        wpe = selector.occupancy if selector.occupancy > 0 else 2
        print(
            f"[gemm/origami] M={M} N={N} K={K} "
            f"a_stride={a_stride} b_stride={b_stride} → "
            f"tile={block_m}x{block_n}x{block_k} wgm={wgm} chunk={chunk} wpe={wpe}"
        )
        return block_m, block_n, block_k, wgm, chunk, wpe
    except Exception as e:
        print(f"[gemm/origami] ERROR M={M} N={N} K={K}: {e}")
        return None


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

            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
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
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

        c = acc.to(C.type.element_ty)
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None].to(tl.int64) * stride_cm + rn[None, :].to(tl.int64) * stride_cn
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
    """General-purpose BF16/FP16 GEMM using optimized persistent kernel.

    Uses origami analytical model for config selection (GROUP_SIZE_M, CHUNK_SIZE)
    when available, otherwise falls back to heuristic.

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

    num_cus = _get_num_cus()

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

    # Config selection: origami analytical model > heuristic fallback
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    chunk_size = 32
    waves_per_eu = 2
    group_m = _select_group_size_m_bf16(M, N, s_ak, s_bk)
    origami_params = _select_params_origami(M, N, K, A_view.stride(), B_view.stride(), out_dtype)
    if origami_params is not None:
        BLOCK_M, BLOCK_N, BLOCK_K, group_m, chunk_size, waves_per_eu = origami_params
        chunk_size = max(chunk_size, 1)

    num_sms = compute_sk_grid(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_cus)
    even_k = K % BLOCK_K == 0

    args = (A_view, B_view, out, M, N, K, A_view.stride(0), B_view.stride(1), stride_cm, stride_cn)

    _bf16_persistent_gemm_kernel[(num_sms,)](
        *args,
        stride_ak=s_ak,
        stride_bk=s_bk,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=group_m,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=chunk_size,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=".ca",
        CACHE_MODIFIER_B=".ca",
        num_warps=8,
        num_stages=2,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return out
