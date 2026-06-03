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
  - _bf16_persistent_gemm_kernel: BF16/FP16 persistent kernel (data-parallel grid)

Public API:
  - gemm_triton_kernel  — BF16/FP16 GEMM

FP8 kernels (tensorwise + blockwise) are in gemm_fp8_kernel.py.

Environment variable: PRIMUS_TURBO_GEMM_BACKEND=TRITON activates these kernels.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.triton.utils.origami import (
    origama_compute_sk_grid,
    origama_hardware_info,
    origama_select_params,
)
from primus_turbo.triton.utils.triton_knobs_helper import set_triton_knobs_gfx950

# ═══════════════════════════════════════════════════════════════════════════════
# Hardware constants & chiplet transform
# ═══════════════════════════════════════════════════════════════════════════════
# TODO(ruibin): remove this once we have a better way to determine the number of XCDs
NUM_XCDS = 8


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


def offline_select_bf16(
    M: int, N: int, K: int, s_ak: int, s_bk: int
) -> tuple[int, int, int, int, int, int, str, str]:
    """BF16 config selection from MI300X bench data (out_bf16_gemm.yaml, 186 entries).

    Stride → layout:
      NT (trans_a=False, trans_b=True):  s_ak=1, s_bk=1   → C = A @ B^T
      NN (trans_a=False, trans_b=False): s_ak=1, s_bk≠1   → C = A @ B
      TN (trans_a=True,  trans_b=False): s_ak≠1, s_bk≠1   → C = A^T @ B
      TT (trans_a=True,  trans_b=True):  s_ak≠1, s_bk=1   → C = A^T @ B^T

    Returns (BM, BN, BK, GM, NUM_SMS, CHUNK, CA, CB).
    """
    # ── Block sizes (256×256×64 covers ~93% of bench entries) ──
    BM, BN, BK = 256, 256, 64

    tiles_m = (M + BM - 1) // BM
    tiles_n = (N + BN - 1) // BN
    total_tiles = tiles_m * tiles_n

    cu_count = origama_hardware_info().N_CU

    # ── NUM_SMS ──
    # Small grids: sk_grid for wave efficiency (persistent, NUM_SMS=256/304)
    # Large grids: data-parallel (NUM_SMS=total_tiles) to keep all CUs busy
    if total_tiles <= cu_count * 4:
        num_sms = origama_compute_sk_grid(M, N, K, BM, BN, BK, cu_count)
    else:
        num_sms = total_tiles

    # ── GROUP_SIZE_M ──
    if min(tiles_m, tiles_n) < 16:
        group_m = 8
    else:
        group_m = 4

    # ── CHUNK_SIZE ──
    # persistent mode: small chunks for XCD load-balance
    # data-parallel: 64 for large tile counts, 32 for small
    if num_sms < total_tiles:
        chunk = min(32, max(1, num_sms // NUM_XCDS))
    else:
        chunk = 64 if total_tiles > 1024 else 32

    return BM, BN, BK, group_m, num_sms, chunk, ".ca", ".ca"


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
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    A_LOAD_ALIGNED: tl.constexpr,
    B_LOAD_ALIGNED: tl.constexpr,
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

        rm_raw = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn_raw = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rm = rm_raw % M
        rn = rn_raw % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        if EVEN_M:
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        if EVEN_N:
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        # Use int64 offsets for pointer arithmetic to prevent int32 overflow with large matrices
        A_BASE = A + rm[:, None].to(tl.int64) * stride_am + rk[None, :].to(tl.int64) * stride_ak
        B_BASE = B + rk[:, None].to(tl.int64) * stride_bk + rn[None, :].to(tl.int64) * stride_bn

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1
        tl.assume(loop_k >= 0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            if EVEN_M and A_LOAD_ALIGNED:
                if stride_ak == 1:
                    a = tl.load(tl.multiple_of(A_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_A)
                else:
                    a = tl.load(tl.multiple_of(A_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_A)
            else:
                a = tl.load(A_BASE, cache_modifier=CACHE_MODIFIER_A)

            if EVEN_N and B_LOAD_ALIGNED:
                if stride_bk == 1:
                    b = tl.load(tl.multiple_of(B_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_B)
                else:
                    b = tl.load(tl.multiple_of(B_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_B)
            else:
                b = tl.load(B_BASE, cache_modifier=CACHE_MODIFIER_B)

            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None].to(tl.int64) * stride_am + rk[None, :].to(tl.int64) * stride_ak
            B_BASE = B + rk[:, None].to(tl.int64) * stride_bk + rn[None, :].to(tl.int64) * stride_bn
            a_mask_k = rk[None, :] < K
            b_mask_k = rk[:, None] < K
            if EVEN_M and A_LOAD_ALIGNED:
                if stride_ak == 1:
                    A_BASE = tl.multiple_of(A_BASE, (1, 16))
                else:
                    A_BASE = tl.multiple_of(A_BASE, (16, 1))
            a = tl.load(A_BASE, mask=a_mask_k, other=0.0, cache_modifier=CACHE_MODIFIER_A)
            if EVEN_N and B_LOAD_ALIGNED:
                if stride_bk == 1:
                    B_BASE = tl.multiple_of(B_BASE, (16, 1))
                else:
                    B_BASE = tl.multiple_of(B_BASE, (1, 16))
            b = tl.load(B_BASE, mask=b_mask_k, other=0.0, cache_modifier=CACHE_MODIFIER_B)
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

        c = acc.to(C.type.element_ty)
        c_mask = (rm_raw[:, None] < M) & (rn_raw[None, :] < N)
        rm_s = rm_raw % M
        rn_s = rn_raw % N
        if EVEN_M:
            rm_s = tl.max_contiguous(tl.multiple_of(rm_s, BLOCK_SIZE_M), BLOCK_SIZE_M)
        if EVEN_N:
            rn_s = tl.max_contiguous(tl.multiple_of(rn_s, BLOCK_SIZE_N), BLOCK_SIZE_N)
        C_ = C + rm_s[:, None].to(tl.int64) * stride_cm + rn_s[None, :].to(tl.int64) * stride_cn
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

    Uses offline heuristic for block sizes / NUM_SMS, then origami analytical
    model to override GROUP_SIZE_M and cache modifiers.

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

    if is_gfx950():
        set_triton_knobs_gfx950()

        # gfx950 BF16 config from 164-entry tuning data.
        # TN layout with large K → BLK_K=64, stages=2; all other cases → 32/3.
        # Small TN (K≤3584, dims≤16384, min dim≤4608) stays on 32/3.
        is_tn = (s_ak == 1) and (s_bk == 1)
        use_bk64 = is_tn and (K > 3584 or min(M, N) > 4608 or max(M, N) > 16384)

        BLOCK_M, BLOCK_N = 256, 256
        BLOCK_K, num_stages = (64, 2) if use_bk64 else (32, 3)
        chunk_size, waves_per_eu = 32, 0
        cache_a, cache_b = ".ca", ".ca"

        tiles_m = (M + BLOCK_M - 1) // BLOCK_M
        tiles_n = (N + BLOCK_N - 1) // BLOCK_N
        min_tile = min(tiles_m, tiles_n)
        group_m = 7 if min_tile < 16 else 4

        cu_count = origama_hardware_info().N_CU

        origami_params = origama_select_params(
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
            if min(om, on) >= 128 and ok == BLOCK_K:
                BLOCK_M, BLOCK_N, group_m = om, on, ogm

        # Occupancy: when TN BLK_K=64 tiles land in 1–2 wave zone, halve
        # BLOCK_N for better CU utilisation (keeps BLOCK_M=256 for A locality).
        if use_bk64:
            tm = (M + BLOCK_M - 1) // BLOCK_M
            tn = (N + BLOCK_N - 1) // BLOCK_N
            if cu_count < tm * tn < 2 * cu_count and tn >= tm:
                new_tn = (N + 127) // 128
                if tm * new_tn >= 2 * cu_count:
                    BLOCK_N, group_m = 128, 8

        num_sms = origama_compute_sk_grid(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, cu_count)
    else:
        # ── gfx942 path (unchanged) ──────────────────────────────────────────
        BLOCK_M, BLOCK_N, BLOCK_K, group_m, num_sms, chunk_size, cache_a, cache_b = offline_select_bf16(
            M, N, K, s_ak, s_bk
        )
        num_stages, waves_per_eu = 2, 0
        origami_params = origama_select_params(
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
            if (om, on, ok) == (BLOCK_M, BLOCK_N, BLOCK_K):
                group_m = ogm
                cache_a, cache_b = oca, ocb

    even_k = K % BLOCK_K == 0
    even_m = M % BLOCK_M == 0
    even_n = N % BLOCK_N == 0

    # For partial M tiles with non-unit K stride (e.g. A comes from .T),
    # force C-contiguous so s_ak becomes 1, avoiding non-deterministic
    # interactions between strided access and modular index wrapping.
    if not even_m and A_view.stride(1) != 1:
        A_view = A_view.contiguous()
        s_ak = A_view.stride(1)
    # For partial N tiles with non-unit K stride (e.g. B comes from .T),
    # the K dim is dim-0; C-contiguous gives stride (N, 1) so s_bk = N,
    # NOT 1.  We still materialise a dense copy to eliminate any exotic
    # stride pattern, but this does NOT make K contiguous for B.
    if not even_n and B_view.stride(0) != 1:
        B_view = B_view.contiguous()
        s_bk = B_view.stride(0)

    # tl.multiple_of hints for vectorised loads are only valid when BOTH
    # the base pointer AND the non-contiguous stride are 16-element-aligned.
    # Subviews from TP/MoE weight slicing can have aligned strides but a
    # misaligned base address, which would cause garbage loads and NaN loss.
    stride_am = A_view.stride(0)
    stride_bn = B_view.stride(1)
    elem_bytes = A_view.element_size()
    ptr_aligned_a = A_view.data_ptr() % (16 * elem_bytes) == 0
    ptr_aligned_b = B_view.data_ptr() % (16 * elem_bytes) == 0
    if s_ak == 1:
        a_load_aligned = ptr_aligned_a and stride_am % 16 == 0
    else:
        a_load_aligned = ptr_aligned_a and s_ak % 16 == 0
    if s_bk == 1:
        b_load_aligned = ptr_aligned_b and stride_bn % 16 == 0
    else:
        b_load_aligned = ptr_aligned_b and s_bk % 16 == 0

    args = (A_view, B_view, out, M, N, K, stride_am, stride_bn, stride_cm, stride_cn)

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
        EVEN_M=even_m,
        EVEN_N=even_n,
        A_LOAD_ALIGNED=a_load_aligned,
        B_LOAD_ALIGNED=b_load_aligned,
        CACHE_MODIFIER_A=cache_a,
        CACHE_MODIFIER_B=cache_b,
        num_warps=8,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return out
