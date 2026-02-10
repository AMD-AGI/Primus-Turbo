###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Grouped GEMM Triton persistent kernels (CPU-sync-free).

Contains:
  - BF16/FP16 grouped GEMM (forward): _grouped_bf16_persistent_gemm_kernel
  - FP8 grouped GEMM (forward, per-tensor scaling): _grouped_fp8_persistent_gemm_kernel
  - Variable-K grouped GEMM (backward, BF16/FP8): _grouped_variable_k_gemm_kernel

Public API:
  - grouped_gemm_triton_kernel              — BF16/FP16 forward
  - grouped_gemm_fp8_tensorwise_triton_kernel          — FP8 forward (per-tensor scaling)
  - grouped_gemm_variable_k_triton_kernel   — BF16/FP16 backward (variable-K)
  - grouped_gemm_fp8_tensorwise_variable_k_triton_kernel — FP8 backward (variable-K, per-tensor scaling)

Environment variable: PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON activates these kernels.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

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
# Grouped GEMM — Persistent Kernel (CPU-sync-free)
#
# Computes: out[offs[g]:offs[g+1], :] = A[offs[g]:offs[g+1], :] @ B_view[g]
#   for g = 0, 1, ..., G-1
#
# Design:
#   Single persistent kernel processes ALL groups × ALL tiles.
#   Each CU computes total_tiles and group mapping on-the-fly via O(G)
#   linear scan of group_offs (G is small, ≤256, data cached in L2).
#   Zero CPU synchronization — group_offs read entirely on GPU.
# ═══════════════════════════════════════════════════════════════════════════════


@triton.jit()
def _grouped_bf16_persistent_gemm_kernel(
    # Pointers
    A,  # [M_total, K]
    B,  # [G, ?, ?]  — (K,N) or (N,K) depending on trans_b
    C,  # [M_total, N]
    group_offs_ptr,  # [G+1] int64
    # Dimensions
    G,  # number of groups (runtime)
    N,
    K,
    # Strides
    stride_am,  # A row stride
    stride_bg,  # B group stride: b.stride(0)
    stride_bn,  # B N-stride (within a group)
    stride_cm,  # C row stride
    stride_cn,  # C col stride
    # Constexpr strides (for compiler optimisation)
    stride_ak: tl.constexpr,  # A K-stride (=1 when trans_a=False, contiguous)
    stride_bk: tl.constexpr,  # B K-stride (=1 when trans_b=True)
    # Tile config
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
    """Persistent grouped GEMM kernel (CPU-sync-free).

    One kernel launch processes ALL groups × ALL tiles.
    Each persistent CU computes total_tiles and maps global tile IDs to
    (group, local_tile) on the fly via O(G) linear scan of group_offs.
    """
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # ── Compute total tiles across all groups (O(G) per CU, group_offs cached in L2) ──
    # Cast int64 group_offs to int32 for tile arithmetic (tile counts fit in int32)
    total_tiles: tl.int32 = 0
    for _g in range(G):
        m_g = (tl.load(group_offs_ptr + _g + 1) - tl.load(group_offs_ptr + _g)).to(tl.int32)
        total_tiles += tl.cdiv(m_g, BLOCK_SIZE_M) * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32

    for global_tile_id in range(pid, total_tiles, NUM_SMS):
        # ── Find group via linear scan (O(G)) ──
        group_idx: tl.int32 = 0
        tile_start: tl.int32 = 0
        cumsum: tl.int32 = 0
        for _g in range(G):
            m_g_i = (tl.load(group_offs_ptr + _g + 1) - tl.load(group_offs_ptr + _g)).to(tl.int32)
            tiles_g = tl.cdiv(m_g_i, BLOCK_SIZE_M) * num_pid_n
            new_cumsum = cumsum + tiles_g
            if global_tile_id >= new_cumsum:
                group_idx = _g + 1
                tile_start = new_cumsum
            cumsum = new_cumsum

        # ── Group-local tile → (pid_m, pid_n) with GROUP_SIZE_M swizzle ──
        local_tile = global_tile_id - tile_start
        m_start_g = tl.load(group_offs_ptr + group_idx)  # keep int64 to avoid address overflow
        M_g = (tl.load(group_offs_ptr + group_idx + 1) - tl.load(group_offs_ptr + group_idx)).to(tl.int32)
        tiles_m_g = tl.cdiv(M_g, BLOCK_SIZE_M)

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        swizzle_group = local_tile // num_pid_in_group
        first_pid_m = swizzle_group * GROUP_SIZE_M
        group_size_m = min(tiles_m_g - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((local_tile % num_pid_in_group) % group_size_m)
        pid_n = (local_tile % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        # ── Address computation ──
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        A_BASE = A + m_start_g * stride_am + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + group_idx * stride_bg + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        # ── K-loop (identical to single GEMM) ──
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
            rk_last = loop_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_LAST = A + m_start_g * stride_am + rm[:, None] * stride_am + rk_last[None, :] * stride_ak
            B_LAST = B + group_idx * stride_bg + rk_last[:, None] * stride_bk + rn[None, :] * stride_bn
            if stride_ak == 1:
                A_LAST = tl.multiple_of(A_LAST, (1, 16))
            else:
                A_LAST = tl.multiple_of(A_LAST, (16, 1))
            if stride_bk == 1:
                B_LAST = tl.multiple_of(B_LAST, (16, 1))
            else:
                B_LAST = tl.multiple_of(B_LAST, (1, 16))
            a = tl.load(A_LAST, mask=rk_last[None, :] < K, other=0.0, cache_modifier=CACHE_MODIFIER_A)
            b = tl.load(B_LAST, mask=rk_last[:, None] < K, other=0.0, cache_modifier=CACHE_MODIFIER_B)
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

        # ── Store ──
        c = acc.to(C.type.element_ty)
        rm_s = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
        rn_s = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rn_s = tl.max_contiguous(tl.multiple_of(rn_s, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm_s[:, None] < M_g) & (rn_s[None, :] < N)
        C_ = C + m_start_g * stride_cm + rm_s[:, None] * stride_cm + rn_s[None, :] * stride_cn
        tl.store(C_, c, c_mask)


def grouped_gemm_triton_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    group_offs: torch.Tensor,
    trans_b: bool = False,
) -> torch.Tensor:
    """Persistent grouped GEMM (CPU-sync-free) using Triton.

    Computes: out[offs[g]:offs[g+1], :] = a[offs[g]:offs[g+1], :] @ B_view[g]
    for g = 0, ..., G-1, where B_view[g] is b[g] or b[g]^T depending on trans_b.

    Single kernel launch, zero CPU synchronization.

    Args:
        a: [M_total, K] BF16/FP16 input (trans_a=False always).
        b: [G, K, N] or [G, N, K] (if trans_b) BF16/FP16 weights.
        group_offs: [G+1] int64 prefix sum of group lengths.
        trans_b: If True, b[g] is [N, K] (transposed).

    Returns:
        [M_total, N] BF16/FP16 output.
    """
    assert a.ndim == 2, f"a must be 2D, got {a.shape}"
    assert b.ndim == 3, f"b must be 3D, got {b.shape}"
    assert a.dtype in (torch.bfloat16, torch.float16), f"Unsupported dtype: {a.dtype}"
    assert b.dtype in (torch.bfloat16, torch.float16), f"Unsupported dtype: {b.dtype}"

    M_total, K_a = a.shape
    G = b.shape[0]

    if trans_b:
        N, K_b = b.shape[1], b.shape[2]
        stride_bk = b.stride(2)  # K is the fast dimension (=1 for contiguous)
        stride_bn = b.stride(1)  # N-stride
    else:
        K_b, N = b.shape[1], b.shape[2]
        stride_bk = b.stride(1)  # K-stride
        stride_bn = b.stride(2)  # N is the fast dimension (=1 for contiguous)

    assert K_a == K_b, f"K mismatch: a has K={K_a}, b has K={K_b}"
    K = K_a

    stride_bg = b.stride(0)  # Group stride
    stride_ak = a.stride(1)  # =1 for contiguous a

    # Output
    out = torch.empty((M_total, N), device=a.device, dtype=a.dtype)

    # Kernel config
    num_sms = _get_num_cus()
    even_k = K % 64 == 0
    group_m = 4  # Default GROUP_SIZE_M for grouped GEMM

    _grouped_bf16_persistent_gemm_kernel[(num_sms,)](
        a,
        b,
        out,
        group_offs,
        G,
        N,
        K,
        a.stride(0),  # stride_am
        stride_bg,  # B group stride
        stride_bn,  # B N-stride
        out.stride(0),  # stride_cm
        out.stride(1),  # stride_cn
        stride_ak=stride_ak,
        stride_bk=stride_bk,
        BLOCK_SIZE_M=256,
        BLOCK_SIZE_N=256,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=group_m,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=32,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=".ca",
        CACHE_MODIFIER_B=".ca",
        num_warps=8,
        num_stages=2,
        waves_per_eu=2,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Grouped FP8 GEMM — Persistent Kernel (CPU-sync-free, per-tensor scaling)
#
# Computes: out[offs[g]:offs[g+1], :] = A[offs[g]:offs[g+1], :] @ B_view[g] * a_scale * b_scale
#   for g = 0, 1, ..., G-1
#
# Same persistent design as BF16 grouped GEMM but with FP8 inputs + per-tensor scaling.
# ═══════════════════════════════════════════════════════════════════════════════


@triton.jit()
def _grouped_fp8_persistent_gemm_kernel(
    # Pointers
    A,  # [M_total, K] FP8
    B,  # [G, ?, ?]  FP8 — (K,N) or (N,K) depending on trans_b
    C,  # [M_total, N] output (BF16/FP16)
    A_scale_ptr,  # per-tensor scale for A (scalar, fp32)
    B_scale_ptr,  # per-tensor scale for B (scalar, fp32)
    group_offs_ptr,  # [G+1] int64
    # Dimensions
    G,  # number of groups (runtime)
    N,
    K,
    # Strides
    stride_am,  # A row stride
    stride_bg,  # B group stride: b.stride(0)
    stride_bn,  # B N-stride (within a group)
    stride_cm,  # C row stride
    stride_cn,  # C col stride
    # Constexpr strides (for compiler optimisation)
    stride_ak: tl.constexpr,  # A K-stride (=1 when trans_a=False, contiguous)
    stride_bk: tl.constexpr,  # B K-stride (=1 when trans_b=True)
    # Tile config
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
    """Persistent grouped FP8 GEMM kernel (CPU-sync-free, per-tensor scaling)."""
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # ── Compute total tiles across all groups ──
    total_tiles: tl.int32 = 0
    for _g in range(G):
        m_g = (tl.load(group_offs_ptr + _g + 1) - tl.load(group_offs_ptr + _g)).to(tl.int32)
        total_tiles += tl.cdiv(m_g, BLOCK_SIZE_M) * num_pid_n

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

    for global_tile_id in range(pid, total_tiles, NUM_SMS):
        # ── Find group via linear scan (O(G)) ──
        group_idx: tl.int32 = 0
        tile_start: tl.int32 = 0
        cumsum: tl.int32 = 0
        for _g in range(G):
            m_g_i = (tl.load(group_offs_ptr + _g + 1) - tl.load(group_offs_ptr + _g)).to(tl.int32)
            tiles_g = tl.cdiv(m_g_i, BLOCK_SIZE_M) * num_pid_n
            new_cumsum = cumsum + tiles_g
            if global_tile_id >= new_cumsum:
                group_idx = _g + 1
                tile_start = new_cumsum
            cumsum = new_cumsum

        # ── Group-local tile → (pid_m, pid_n) with GROUP_SIZE_M swizzle ──
        local_tile = global_tile_id - tile_start
        m_start_g = tl.load(group_offs_ptr + group_idx)  # keep int64 to avoid address overflow
        M_g = (tl.load(group_offs_ptr + group_idx + 1) - tl.load(group_offs_ptr + group_idx)).to(tl.int32)
        tiles_m_g = tl.cdiv(M_g, BLOCK_SIZE_M)

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        swizzle_group = local_tile // num_pid_in_group
        first_pid_m = swizzle_group * GROUP_SIZE_M
        group_size_m = min(tiles_m_g - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((local_tile % num_pid_in_group) % group_size_m)
        pid_n = (local_tile % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        # ── Address computation ──
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        A_BASE = A + m_start_g * stride_am + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + group_idx * stride_bg + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        # ── K-loop ──
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
            rk_last = loop_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_LAST = A + m_start_g * stride_am + rm[:, None] * stride_am + rk_last[None, :] * stride_ak
            B_LAST = B + group_idx * stride_bg + rk_last[:, None] * stride_bk + rn[None, :] * stride_bn
            if stride_ak == 1:
                A_LAST = tl.multiple_of(A_LAST, (1, 16))
            else:
                A_LAST = tl.multiple_of(A_LAST, (16, 1))
            if stride_bk == 1:
                B_LAST = tl.multiple_of(B_LAST, (16, 1))
            else:
                B_LAST = tl.multiple_of(B_LAST, (1, 16))
            a = tl.load(A_LAST, mask=rk_last[None, :] < K, other=0.0, cache_modifier=CACHE_MODIFIER_A)
            b = tl.load(B_LAST, mask=rk_last[:, None] < K, other=0.0, cache_modifier=CACHE_MODIFIER_B)
            acc += tl.dot(a, b, input_precision="ieee")

        # ── Apply per-tensor scale and store ──
        acc *= scale
        c = acc.to(C.type.element_ty)
        rm_s = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
        rn_s = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rn_s = tl.max_contiguous(tl.multiple_of(rn_s, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm_s[:, None] < M_g) & (rn_s[None, :] < N)
        C_ = C + m_start_g * stride_cm + rm_s[:, None] * stride_cm + rn_s[None, :] * stride_cn
        tl.store(C_, c, c_mask)


def grouped_gemm_fp8_tensorwise_triton_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    group_offs: torch.Tensor,
    trans_b: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Persistent grouped FP8 GEMM (CPU-sync-free, per-tensor scaling) using Triton.

    Computes: out[offs[g]:offs[g+1], :] = a[offs[g]:offs[g+1], :] @ B_view[g] * a_scale * b_scale
    for g = 0, ..., G-1, where B_view[g] is b[g] or b[g]^T depending on trans_b.

    Single kernel launch, zero CPU synchronization.

    Args:
        a: [M_total, K] FP8 input (trans_a=False always).
        b: [G, K, N] or [G, N, K] (if trans_b) FP8 weights.
        a_scale: Per-tensor dequantization scale for A, scalar fp32.
        b_scale: Per-tensor dequantization scale for B, scalar fp32.
        group_offs: [G+1] int64 prefix sum of group lengths.
        trans_b: If True, b[g] is [N, K] (transposed).
        out_dtype: Output dtype (default bfloat16).

    Returns:
        [M_total, N] output in out_dtype.
    """
    assert a.ndim == 2, f"a must be 2D, got {a.shape}"
    assert b.ndim == 3, f"b must be 3D, got {b.shape}"

    M_total, K_a = a.shape
    G = b.shape[0]

    if trans_b:
        N, K_b = b.shape[1], b.shape[2]
        stride_bk = b.stride(2)
        stride_bn = b.stride(1)
    else:
        K_b, N = b.shape[1], b.shape[2]
        stride_bk = b.stride(1)
        stride_bn = b.stride(2)

    assert K_a == K_b, f"K mismatch: a has K={K_a}, b has K={K_b}"
    K = K_a

    stride_bg = b.stride(0)
    stride_ak = a.stride(1)  # =1 for contiguous a

    # Output
    out = torch.empty((M_total, N), device=a.device, dtype=out_dtype)

    # Kernel config (fixed, no autotune)
    # FP8 uses BLOCK_SIZE_K=128 when both A and B have contiguous K-strides
    # (forward: stride_ak=1, stride_bk=1 → BLK_K=128)
    # (backward grad_a: stride_bk=N ≠ 1 → fall back to BLK_K=64)
    num_sms = _get_num_cus()
    blk_k = 128 if (stride_ak == 1 and stride_bk == 1) else 64
    even_k = K % blk_k == 0

    # GROUP_SIZE_M heuristic: use 8 for many-group / small per-group tiles
    tiles_m_per_group = (M_total + G * 256 - 1) // (G * 256)  # approx ceil(M/G/256)
    tiles_n = (N + 255) // 256
    group_m = 8 if min(tiles_m_per_group, tiles_n) < 16 else 4

    _grouped_fp8_persistent_gemm_kernel[(num_sms,)](
        a,
        b,
        out,
        a_scale,
        b_scale,
        group_offs,
        G,
        N,
        K,
        a.stride(0),  # stride_am
        stride_bg,  # B group stride
        stride_bn,  # B N-stride
        out.stride(0),  # stride_cm
        out.stride(1),  # stride_cn
        stride_ak=stride_ak,
        stride_bk=stride_bk,
        BLOCK_SIZE_M=256,
        BLOCK_SIZE_N=256,
        BLOCK_SIZE_K=blk_k,
        GROUP_SIZE_M=group_m,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=32,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=".ca",
        CACHE_MODIFIER_B=".ca",
        num_warps=8,
        num_stages=2,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Grouped Variable-K GEMM — Persistent Kernel (backward pass, CPU-sync-free)
#
# Computes: C[g] = LHS[offs[g]:offs[g+1]]^T @ RHS[offs[g]:offs[g+1]] [* scale]
#   for g = 0, 1, ..., G-1
#
# Used in backward pass where both LHS and RHS are 2D tensors sliced by groups.
# Output is 3D: [G, OUT_M, OUT_N].
# All groups share the same output dimensions; only the inner product dim (M_g)
# varies per group, making group→tile mapping a simple div/mod.
# ═══════════════════════════════════════════════════════════════════════════════


@triton.jit()
def _grouped_variable_k_gemm_kernel(
    # C[g] = LHS_g^T @ RHS_g [* scale if IS_FP8]
    # LHS: [M_total, OUT_M] (2D), RHS: [M_total, OUT_N] (2D)
    # C: [G, OUT_M, OUT_N] (3D)
    LHS,
    RHS,
    C,
    LHS_scale_ptr,
    RHS_scale_ptr,  # only used if IS_FP8
    group_offs_ptr,  # [G+1] int64
    G,  # number of groups
    OUT_M,
    OUT_N,  # output dimensions (fixed across groups)
    # Strides
    stride_lhs_m,  # LHS row stride (along M_total)
    stride_rhs_m,  # RHS row stride (along M_total)
    stride_cg,  # C group stride
    stride_cm,  # C row stride (along OUT_M)
    stride_cn,  # C col stride (along OUT_N)
    # Constexpr strides
    stride_lhs_n: tl.constexpr,  # LHS col stride (=1 for row-major)
    stride_rhs_n: tl.constexpr,  # RHS col stride (=1 for row-major)
    # Tile config
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # inner loop block over M_g
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    IS_FP8: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
):
    """Persistent grouped variable-K GEMM kernel for backward pass (CPU-sync-free).

    All groups share the same output dims (OUT_M × OUT_N), only the inner product
    dimension M_g varies per group. Group→tile mapping is simple div/mod.
    """
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    tiles_m = tl.cdiv(OUT_M, BLOCK_SIZE_M)
    tiles_n = tl.cdiv(OUT_N, BLOCK_SIZE_N)
    tiles_per_group = tiles_m * tiles_n
    total_tiles = G * tiles_per_group

    tl.assume(stride_lhs_m > 0)
    tl.assume(stride_lhs_n > 0)
    tl.assume(stride_rhs_m > 0)
    tl.assume(stride_rhs_n > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    if IS_FP8:
        scale = tl.load(LHS_scale_ptr) * tl.load(RHS_scale_ptr)

    acc_dtype = tl.float32

    for global_tile in range(pid, total_tiles, NUM_SMS):
        # ── Map to (group, local_tile) — simple div/mod ──
        group_idx = global_tile // tiles_per_group
        local_tile = global_tile - group_idx * tiles_per_group

        # ── Swizzle local tile → (pid_m, pid_n) ──
        num_pid_in_group = GROUP_SIZE_M * tiles_n
        swizzle_group = local_tile // num_pid_in_group
        first_pid_m = swizzle_group * GROUP_SIZE_M
        group_size_m = min(tiles_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((local_tile % num_pid_in_group) % group_size_m)
        pid_n = (local_tile % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        # ── Group boundaries ──
        m_start = tl.load(group_offs_ptr + group_idx)  # int64 to avoid overflow
        M_g = (tl.load(group_offs_ptr + group_idx + 1) - m_start).to(tl.int32)

        # ── Output indices ──
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % OUT_M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % OUT_N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # ── Base pointers ──
        # LHS^T[rm, rk] = LHS[m_start + rk, rm]
        LHS_BASE = LHS + m_start * stride_lhs_m + rm[:, None] * stride_lhs_n + rk[None, :] * stride_lhs_m
        # RHS[rk, rn] = RHS[m_start + rk, rn]
        RHS_BASE = RHS + m_start * stride_rhs_m + rk[:, None] * stride_rhs_m + rn[None, :] * stride_rhs_n

        # ── K-loop over M_g (variable per group, always masked for correctness) ──
        loop_k = tl.cdiv(M_g, BLOCK_SIZE_K)
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for k in range(loop_k):
            k_start = k * BLOCK_SIZE_K
            mask_k = (k_start + tl.arange(0, BLOCK_SIZE_K)) < M_g

            if stride_lhs_n == 1:
                a = tl.load(
                    tl.multiple_of(LHS_BASE, (16, 1)),
                    mask=mask_k[None, :],
                    other=0.0,
                    cache_modifier=CACHE_MODIFIER_A,
                )
            else:
                a = tl.load(
                    tl.multiple_of(LHS_BASE, (1, 16)),
                    mask=mask_k[None, :],
                    other=0.0,
                    cache_modifier=CACHE_MODIFIER_A,
                )

            if stride_rhs_n == 1:
                b = tl.load(
                    tl.multiple_of(RHS_BASE, (1, 16)),
                    mask=mask_k[:, None],
                    other=0.0,
                    cache_modifier=CACHE_MODIFIER_B,
                )
            else:
                b = tl.load(
                    tl.multiple_of(RHS_BASE, (16, 1)),
                    mask=mask_k[:, None],
                    other=0.0,
                    cache_modifier=CACHE_MODIFIER_B,
                )

            if IS_FP8:
                acc += tl.dot(a, b, input_precision="ieee")
            else:
                acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

            LHS_BASE += BLOCK_SIZE_K * stride_lhs_m
            RHS_BASE += BLOCK_SIZE_K * stride_rhs_m

        # ── Apply scaling and store ──
        if IS_FP8:
            acc *= scale
        c = acc.to(C.type.element_ty)
        rm_s = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn_s = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn_s = tl.max_contiguous(tl.multiple_of(rn_s % OUT_N, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm_s[:, None] < OUT_M) & (rn_s[None, :] < OUT_N)
        C_ = C + group_idx * stride_cg + rm_s[:, None] * stride_cm + rn_s[None, :] * stride_cn
        tl.store(C_, c, c_mask)


# ── Public API — Variable-K BF16 grouped GEMM (backward) ──


def grouped_gemm_variable_k_triton_kernel(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_offs: torch.Tensor,
) -> torch.Tensor:
    """Variable-K grouped BF16/FP16 GEMM (backward) using Triton.

    Computes C[g] = lhs[offs[g]:offs[g+1]]^T @ rhs[offs[g]:offs[g+1]]
    Output: [G, OUT_M, OUT_N].

    Args:
        lhs: [M_total, OUT_M] BF16/FP16 (after trans_c swap, this is grad_out).
        rhs: [M_total, OUT_N] BF16/FP16 (after trans_c swap, this is a).
        group_offs: [G+1] int64 prefix sum.

    Returns:
        [G, OUT_M, OUT_N] output.
    """
    assert lhs.ndim == 2 and rhs.ndim == 2
    assert lhs.shape[0] == rhs.shape[0]
    OUT_M = lhs.shape[1]
    OUT_N = rhs.shape[1]
    G = group_offs.shape[0] - 1

    out = torch.empty((G, OUT_M, OUT_N), device=lhs.device, dtype=lhs.dtype)
    num_sms = _get_num_cus()
    dummy_scale = torch.empty(1, device=lhs.device, dtype=torch.float32)

    _grouped_variable_k_gemm_kernel[(num_sms,)](
        lhs,
        rhs,
        out,
        dummy_scale,
        dummy_scale,  # unused for BF16
        group_offs,
        G,
        OUT_M,
        OUT_N,
        lhs.stride(0),
        rhs.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        stride_lhs_n=lhs.stride(1),
        stride_rhs_n=rhs.stride(1),
        BLOCK_SIZE_M=256,
        BLOCK_SIZE_N=256,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=4,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=32,
        IS_FP8=False,
        CACHE_MODIFIER_A=".ca",
        CACHE_MODIFIER_B=".ca",
        num_warps=8,
        num_stages=2,
        waves_per_eu=2,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return out


# ── Public API — Variable-K FP8 grouped GEMM (backward) ──


def grouped_gemm_fp8_tensorwise_variable_k_triton_kernel(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_scale: torch.Tensor,
    group_offs: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Variable-K grouped FP8 GEMM (backward, per-tensor scaling) using Triton.

    Computes C[g] = lhs[offs[g]:offs[g+1]]^T @ rhs[offs[g]:offs[g+1]] * lhs_scale * rhs_scale
    Output: [G, OUT_M, OUT_N].

    Args:
        lhs: [M_total, OUT_M] FP8 (after trans_c swap, this is grad_out_fp8).
        rhs: [M_total, OUT_N] FP8 (after trans_c swap, this is a_fp8).
        lhs_scale: Per-tensor scale for LHS, scalar fp32.
        rhs_scale: Per-tensor scale for RHS, scalar fp32.
        group_offs: [G+1] int64 prefix sum.
        out_dtype: Output dtype (default bfloat16).

    Returns:
        [G, OUT_M, OUT_N] output.
    """
    assert lhs.ndim == 2 and rhs.ndim == 2
    assert lhs.shape[0] == rhs.shape[0]
    OUT_M = lhs.shape[1]
    OUT_N = rhs.shape[1]
    G = group_offs.shape[0] - 1

    out = torch.empty((G, OUT_M, OUT_N), device=lhs.device, dtype=out_dtype)
    num_sms = _get_num_cus()

    _grouped_variable_k_gemm_kernel[(num_sms,)](
        lhs,
        rhs,
        out,
        lhs_scale,
        rhs_scale,
        group_offs,
        G,
        OUT_M,
        OUT_N,
        lhs.stride(0),
        rhs.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        stride_lhs_n=lhs.stride(1),
        stride_rhs_n=rhs.stride(1),
        BLOCK_SIZE_M=256,
        BLOCK_SIZE_N=256,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=4,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=32,
        IS_FP8=True,
        CACHE_MODIFIER_A=".ca",
        CACHE_MODIFIER_B=".ca",
        num_warps=8,
        num_stages=2,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return out
