###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Grouped FP8 GEMM Triton persistent kernels (CPU-sync-free).

Contains:
  Tensorwise (per-tensor) scaling:
    - _grouped_fp8_persistent_gemm_kernel: Forward
    - grouped_gemm_fp8_tensorwise_triton_kernel: Forward public API
    - grouped_gemm_fp8_tensorwise_variable_k_triton_kernel: Backward public API

  Rowwise (per-row/per-col vector) scaling:
    - _grouped_fp8_rowwise_persistent_gemm_kernel: Forward
    - _grouped_fp8_rowwise_variable_k_gemm_kernel: Backward variable-K
    - grouped_gemm_fp8_rowwise_triton_kernel: Forward public API
    - grouped_gemm_fp8_rowwise_variable_k_triton_kernel: Backward public API

  Blockwise scaling:
    - _grouped_blockwise_fp8_persistent_gemm_kernel: Forward (EVEN_K handles
      both K-aligned and K-unaligned shapes)
    - _grouped_blockwise_fp8_variable_k_gemm_kernel: Backward variable-K
    - grouped_gemm_fp8_blockwise_triton_kernel: Forward public API
    - grouped_gemm_fp8_blockwise_variable_k_triton_kernel: Backward public API

Environment variable: PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON activates these kernels.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from primus_turbo.pytorch.core.utils import get_num_cus, is_gfx950
from primus_turbo.triton.grouped_gemm.grouped_gemm_kernel import (
    NUM_XCDS,
    _chiplet_transform_chunked,
    _grouped_variable_k_gemm_kernel,
)
from primus_turbo.triton.utils.origami import (
    origama_calculate_lds_usage,
    origama_hardware_info,
    origama_select_params,
)
from primus_turbo.triton.utils.triton_knobs_helper import set_triton_knobs_gfx950

_grouped_blockwise_warmed: set = set()
_grouped_blockwise_vk_warmed: set = set()

# ═══════════════════════════════════════════════════════════════════════════════
# AMD knobs helper
# ═══════════════════════════════════════════════════════════════════════════════


def _set_amd_knobs(enable: bool = True):
    """Set AMD-specific Triton knobs."""
    if hasattr(triton, "knobs") and hasattr(triton.knobs, "amd"):
        triton.knobs.amd.use_async_copy = enable
        triton.knobs.amd.scalarize_packed_fops = enable


def offline_select_gg_fp8(M_total, G, N, K, s_ak, s_bk):
    """FP8 grouped GEMM config from MI300X bench (out_gg_fp8_persistent_full.yaml).

    Returns (BM, BN, BK, GM, NUM_SMS, CHUNK, CA, CB).
    """
    is_tn = s_ak == 1 and s_bk == 1
    avg_m = max(M_total // max(G, 1), 1)

    BM, BN = 256, 256
    BK = 128 if is_tn else 64

    tiles_m_g = max(1, (avg_m + BM - 1) // BM)
    tiles_n = (N + BN - 1) // BN
    tiles_m_total = (M_total + BM - 1) // BM
    total_tiles = tiles_m_total * tiles_n

    if min(tiles_m_g, tiles_n) < 16:
        group_m = 8
    elif is_tn:
        group_m = 4
    else:
        group_m = 5

    cu_count = get_num_cus()
    num_sms = min(total_tiles, cu_count)
    chunk = 64 if num_sms >= NUM_XCDS * 64 else 32

    return BM, BN, BK, group_m, num_sms, chunk, ".ca", ".ca"


# ═══════════════════════════════════════════════════════════════════════════════
# Cached config selection (avoids per-call origami / LDS overhead)
# ═══════════════════════════════════════════════════════════════════════════════


@functools.lru_cache(maxsize=256)
def _get_gg_fp8_tw_fwd_config(
    avg_m,
    N,
    K,
    out_dtype,
    a_dtype,
    b_dtype,
    trans_b,
    G,
    num_sms,
    M_total,
    stride_ak,
    stride_bk,
):
    """Cached kernel config for FP8 tensorwise grouped GEMM forward."""
    if is_gfx950():
        blk_m, blk_n = 256, 256
        blk_k = 128
        num_stages_val = 2
        group_m = 4
        cache_a, cache_b = ".ca", ".ca"
        chunk_size = 32
        grid_sms = num_sms

        origami_params = origama_select_params(
            avg_m,
            N,
            K,
            out_dtype,
            a_dtype,
            b_dtype,
            trans_a=False,
            trans_b=trans_b,
        )
        if origami_params is not None:
            om, on, ok, ogm, oc_a, oc_b = origami_params
            tiles_default = G * ((avg_m + 255) // 256) * ((N + 255) // 256)
            if min(om, on) >= 128 and ok == blk_k:
                blk_m, blk_n, group_m = om, on, ogm
                cache_a, cache_b = oc_a, oc_b
            elif tiles_default < num_sms and min(om, on) >= 64:
                proposed_stages = 2 if ok >= 128 else 3
                lds = origama_calculate_lds_usage(om, on, ok, 1, 1, proposed_stages)
                if lds <= origama_hardware_info().lds_capacity:
                    blk_m, blk_n, blk_k = om, on, ok
                    group_m = ogm
                    cache_a, cache_b = oc_a, oc_b
                    num_stages_val = proposed_stages
    else:
        blk_m, blk_n, blk_k, group_m, grid_sms, chunk_size, cache_a, cache_b = offline_select_gg_fp8(
            M_total, G, N, K, stride_ak, stride_bk
        )
        num_stages_val = 2

        origami_params = origama_select_params(
            avg_m,
            N,
            K,
            out_dtype,
            a_dtype,
            b_dtype,
            trans_a=False,
            trans_b=trans_b,
        )
        if origami_params is not None:
            om, on, ok, ogm, oca, ocb = origami_params
            if (om, on, ok) == (blk_m, blk_n, blk_k):
                group_m = ogm
                cache_a = oca
                cache_b = ocb

    return blk_m, blk_n, blk_k, group_m, cache_a, cache_b, num_stages_val, chunk_size, grid_sms


@functools.lru_cache(maxsize=256)
def _get_gg_fp8_tw_vk_config(OUT_M, OUT_N, avg_k, a_dtype, b_dtype, G, num_sms):
    """Cached kernel config for FP8 tensorwise grouped GEMM variable-K backward."""
    if is_gfx950():
        blk_m, blk_n = 256, 256
        blk_k, num_stages_val = 64, 3
        group_m = 4
        cache_a, cache_b = ".ca", ".ca"
        chunk_size = 32

        origami_params = origama_select_params(
            OUT_M,
            OUT_N,
            avg_k,
            a_dtype,
            a_dtype,
            b_dtype,
            trans_a=True,
            trans_b=False,
        )
        if origami_params is not None:
            om, on, ok, ogm, oc_a, oc_b = origami_params
            tiles_default = G * ((OUT_M + 255) // 256) * ((OUT_N + 255) // 256)
            if min(om, on) >= 128 and ok in (64, 128):
                blk_m, blk_n, blk_k, group_m = om, on, ok, ogm
                num_stages_val = 3 if ok <= 64 else 2
                cache_a, cache_b = oc_a, oc_b
            elif tiles_default < num_sms and min(om, on) >= 64:
                proposed_stages = 2 if ok >= 128 else 3
                lds = origama_calculate_lds_usage(om, on, ok, 1, 1, proposed_stages)
                if lds <= origama_hardware_info().lds_capacity:
                    blk_m, blk_n, blk_k = om, on, ok
                    group_m = ogm
                    cache_a, cache_b = oc_a, oc_b
                    num_stages_val = proposed_stages
    else:
        blk_m, blk_n, blk_k = 256, 256, 64
        group_m = 4
        num_stages_val = 2
        cache_a, cache_b = ".ca", ".ca"
        chunk_size = 32

    return blk_m, blk_n, blk_k, group_m, cache_a, cache_b, num_stages_val, chunk_size


@functools.lru_cache(maxsize=256)
def _get_gg_fp8_rw_fwd_config(
    avg_m,
    N,
    K,
    out_dtype,
    a_dtype,
    b_dtype,
    trans_b,
    G,
    num_sms,
    M_total,
    stride_ak,
    stride_bk,
):
    """Cached kernel config for FP8 rowwise grouped GEMM forward."""
    if is_gfx950():
        blk_m, blk_n = 256, 256
        blk_k = 128
        num_stages_val = 2
        group_m = 4
        cache_a, cache_b = ".ca", ".ca"
        chunk_size = 32

        origami_params = origama_select_params(
            avg_m,
            N,
            K,
            out_dtype,
            a_dtype,
            b_dtype,
            trans_a=False,
            trans_b=trans_b,
        )
        if origami_params is not None:
            om, on, ok, ogm, oc_a, oc_b = origami_params
            tiles_default = G * ((avg_m + 255) // 256) * ((N + 255) // 256)
            if min(om, on) >= 128 and ok == blk_k:
                blk_m, blk_n, group_m = om, on, ogm
                cache_a, cache_b = oc_a, oc_b
            elif tiles_default < num_sms and min(om, on) >= 64:
                proposed_stages = 2 if ok >= 128 else 3
                lds = origama_calculate_lds_usage(om, on, ok, 1, 1, proposed_stages)
                if lds <= origama_hardware_info().lds_capacity:
                    blk_m, blk_n, blk_k = om, on, ok
                    group_m = ogm
                    cache_a, cache_b = oc_a, oc_b
                    num_stages_val = proposed_stages
    else:
        blk_m, blk_n = 256, 256
        blk_k = 128 if (stride_ak == 1 and stride_bk == 1) else 64
        num_stages_val = 2
        cache_a, cache_b = ".ca", ".ca"
        chunk_size = 32

        tiles_m_per_group = (M_total + G * blk_m - 1) // (G * blk_m)
        tiles_n = (N + blk_n - 1) // blk_n
        group_m = 8 if min(tiles_m_per_group, tiles_n) < 16 else 4

    return blk_m, blk_n, blk_k, group_m, cache_a, cache_b, num_stages_val, chunk_size


@functools.lru_cache(maxsize=256)
def _get_gg_fp8_rw_vk_config(OUT_M, OUT_N, avg_k, a_dtype, b_dtype, G, num_sms):
    """Cached kernel config for FP8 rowwise grouped GEMM variable-K backward."""
    if is_gfx950():
        blk_m, blk_n = 256, 256
        blk_k, num_stages_val = 64, 3
        group_m = 4
        cache_a, cache_b = ".ca", ".ca"
        chunk_size = 32

        origami_params = origama_select_params(
            OUT_M,
            OUT_N,
            avg_k,
            a_dtype,
            a_dtype,
            b_dtype,
            trans_a=True,
            trans_b=False,
        )
        if origami_params is not None:
            om, on, ok, ogm, oc_a, oc_b = origami_params
            tiles_default = G * ((OUT_M + 255) // 256) * ((OUT_N + 255) // 256)
            if min(om, on) >= 128 and ok in (64, 128):
                blk_m, blk_n, blk_k, group_m = om, on, ok, ogm
                num_stages_val = 3 if ok <= 64 else 2
                cache_a, cache_b = oc_a, oc_b
            elif tiles_default < num_sms and min(om, on) >= 64:
                proposed_stages = 2 if ok >= 128 else 3
                lds = origama_calculate_lds_usage(om, on, ok, 1, 1, proposed_stages)
                if lds <= origama_hardware_info().lds_capacity:
                    blk_m, blk_n, blk_k = om, on, ok
                    group_m = ogm
                    cache_a, cache_b = oc_a, oc_b
                    num_stages_val = proposed_stages
    else:
        blk_m, blk_n, blk_k = 256, 256, 64
        group_m = 4
        num_stages_val = 2
        cache_a, cache_b = ".ca", ".ca"
        chunk_size = 32

    return blk_m, blk_n, blk_k, group_m, cache_a, cache_b, num_stages_val, chunk_size


# ###########################################################################
#
#  PART 1 — TENSORWISE (per-tensor) FP8 Grouped GEMM
#
# ###########################################################################


# ═══════════════════════════════════════════════════════════════════════════════
# Tensorwise FP8 Forward Kernel (persistent, CPU-sync-free)
#
# Computes: out[offs[g]:offs[g+1], :] = A[offs[g]:offs[g+1], :] @ B_view[g] * a_scale * b_scale
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

        # Cast group_idx to int64 to prevent overflow in B group offset
        group_offset_b = group_idx.to(tl.int64) * stride_bg

        A_BASE = A + m_start_g * stride_am + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + group_offset_b + rk[:, None] * stride_bk + rn[None, :] * stride_bn

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

            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            rk_last = loop_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_LAST = A + m_start_g * stride_am + rm[:, None] * stride_am + rk_last[None, :] * stride_ak
            B_LAST = B + group_offset_b + rk_last[:, None] * stride_bk + rn[None, :] * stride_bn
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
            acc += tl.dot(a, b)

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

    # Kernel config (cached — origami + LDS check run only on first call per shape)
    num_sms = get_num_cus()
    avg_m = max(M_total // max(G, 1), 256)
    if is_gfx950():
        set_triton_knobs_gfx950()
    blk_m, blk_n, blk_k, group_m, cache_a, cache_b, num_stages_val, chunk_size, num_sms = (
        _get_gg_fp8_tw_fwd_config(
            avg_m,
            N,
            K,
            out_dtype,
            a.dtype,
            b.dtype,
            trans_b,
            G,
            num_sms,
            M_total,
            stride_ak,
            stride_bk,
        )
    )
    even_k = K % blk_k == 0

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
        stride_bg,
        stride_bn,
        out.stride(0),  # stride_cm
        out.stride(1),  # stride_cn
        stride_ak=stride_ak,
        stride_bk=stride_bk,
        BLOCK_SIZE_M=blk_m,
        BLOCK_SIZE_N=blk_n,
        BLOCK_SIZE_K=blk_k,
        GROUP_SIZE_M=group_m,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=chunk_size,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=cache_a,
        CACHE_MODIFIER_B=cache_b,
        num_warps=8,
        num_stages=num_stages_val,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return out


# ── Tensorwise FP8 Variable-K Backward ──


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
    num_sms = get_num_cus()

    if is_gfx950():
        set_triton_knobs_gfx950()
    avg_m_g = max(lhs.shape[0] // max(G, 1), 256)
    blk_m, blk_n, blk_k, group_m, cache_a, cache_b, num_stages_val, chunk_size = _get_gg_fp8_tw_vk_config(
        OUT_M, OUT_N, avg_m_g, lhs.dtype, rhs.dtype, G, num_sms
    )

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
        BLOCK_SIZE_M=blk_m,
        BLOCK_SIZE_N=blk_n,
        BLOCK_SIZE_K=blk_k,
        GROUP_SIZE_M=group_m,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=chunk_size,
        IS_FP8=True,
        CACHE_MODIFIER_A=cache_a,
        CACHE_MODIFIER_B=cache_b,
        num_warps=8,
        num_stages=num_stages_val,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return out


# ###########################################################################
#
#  PART 1.5 — ROWWISE (per-row / per-col vector) FP8 Grouped GEMM
#
# ###########################################################################


# ═══════════════════════════════════════════════════════════════════════════════
# Rowwise FP8 Forward Kernel (persistent, CPU-sync-free)
#
# Identical to the tensorwise forward except scale application:
#   a_scale: (M_total,) fp32 — per output-row (indexed by absolute row)
#   b_scale: (G, N)    fp32 — per output-col per group
# ═══════════════════════════════════════════════════════════════════════════════


@triton.jit()
def _grouped_fp8_rowwise_persistent_gemm_kernel(
    # Pointers
    A,  # [M_total, K] FP8
    B,  # [G, ?, ?]  FP8 — (K,N) or (N,K) depending on trans_b
    C,  # [M_total, N] output (BF16/FP16)
    A_scale_ptr,  # (M_total,) fp32 — per-row scale for A
    B_scale_ptr,  # (G, N)    fp32 — per-col scale per group for B
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
    stride_bscale_g,  # B_scale group stride
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
    """Persistent grouped FP8 GEMM kernel (CPU-sync-free, per-row/per-col vector scaling)."""
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

        # Cast group_idx to int64 to prevent overflow in B group offset
        group_offset_b = group_idx.to(tl.int64) * stride_bg

        A_BASE = A + m_start_g * stride_am + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + group_offset_b + rk[:, None] * stride_bk + rn[None, :] * stride_bn

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

            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            rk_last = loop_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_LAST = A + m_start_g * stride_am + rm[:, None] * stride_am + rk_last[None, :] * stride_ak
            B_LAST = B + group_offset_b + rk_last[:, None] * stride_bk + rn[None, :] * stride_bn
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
            acc += tl.dot(a, b)

        # ── Apply per-row/per-col vector scales and store ──
        a_scale = tl.load(A_scale_ptr + (m_start_g + rm).to(tl.int64))
        b_scale = tl.load(B_scale_ptr + group_idx.to(tl.int64) * stride_bscale_g + rn)
        acc *= a_scale[:, None] * b_scale[None, :]
        c = acc.to(C.type.element_ty)
        rm_s = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
        rn_s = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rn_s = tl.max_contiguous(tl.multiple_of(rn_s, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm_s[:, None] < M_g) & (rn_s[None, :] < N)
        C_ = C + m_start_g * stride_cm + rm_s[:, None] * stride_cm + rn_s[None, :] * stride_cn
        tl.store(C_, c, c_mask)


def grouped_gemm_fp8_rowwise_triton_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    group_offs: torch.Tensor,
    trans_b: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Persistent grouped FP8 GEMM (CPU-sync-free, per-row/per-col scaling) using Triton.

    Scale semantics (same as non-grouped rowwise):
      a_scale: (M_total,) fp32 — per output-row
      b_scale: (G, N)     fp32 — per output-col per group

    Args:
        a: [M_total, K] FP8 input (trans_a=False always).
        b: [G, K, N] or [G, N, K] (if trans_b) FP8 weights.
        a_scale: (M_total,) per-row dequantization scale for A, fp32.
        b_scale: (G, N) per-col per-group dequantization scale for B, fp32.
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

    # Kernel config (cached — origami + LDS check run only on first call per shape)
    num_sms = get_num_cus()
    avg_m = max(M_total // max(G, 1), 256)
    if is_gfx950():
        set_triton_knobs_gfx950()
    blk_m, blk_n, blk_k, group_m, cache_a, cache_b, num_stages_val, chunk_size = _get_gg_fp8_rw_fwd_config(
        avg_m,
        N,
        K,
        out_dtype,
        a.dtype,
        b.dtype,
        trans_b,
        G,
        num_sms,
        M_total,
        stride_ak,
        stride_bk,
    )
    even_k = K % blk_k == 0

    _grouped_fp8_rowwise_persistent_gemm_kernel[(num_sms,)](
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
        stride_bg,
        stride_bn,
        out.stride(0),  # stride_cm
        out.stride(1),  # stride_cn
        b_scale.stride(0),  # stride_bscale_g
        stride_ak=stride_ak,
        stride_bk=stride_bk,
        BLOCK_SIZE_M=blk_m,
        BLOCK_SIZE_N=blk_n,
        BLOCK_SIZE_K=blk_k,
        GROUP_SIZE_M=group_m,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=chunk_size,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=cache_a,
        CACHE_MODIFIER_B=cache_b,
        num_warps=8,
        num_stages=num_stages_val,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Rowwise FP8 Variable-K Backward Kernel
#
# C[g] = LHS_g^T @ RHS_g  with vector scaling:
#   lhs_scale: (OUT_M,) fp32, rhs_scale: (OUT_N,) fp32
# ═══════════════════════════════════════════════════════════════════════════════


@triton.jit()
def _grouped_fp8_rowwise_variable_k_gemm_kernel(
    # C[g] = LHS_g^T @ RHS_g  with per-row/per-col vector scales
    # LHS: [M_total, OUT_M] (2D), RHS: [M_total, OUT_N] (2D)
    # C: [G, OUT_M, OUT_N] (3D)
    LHS,
    RHS,
    C,
    LHS_scale_ptr,  # (OUT_M,) fp32
    RHS_scale_ptr,  # (OUT_N,) fp32
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
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
):
    """Persistent grouped variable-K FP8 GEMM kernel (backward, per-row/per-col vector scaling)."""
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

            acc += tl.dot(a, b)

            LHS_BASE += BLOCK_SIZE_K * stride_lhs_m
            RHS_BASE += BLOCK_SIZE_K * stride_rhs_m

        # ── Apply per-row/per-col vector scales and store ──
        lhs_scale = tl.load(LHS_scale_ptr + rm)
        rhs_scale = tl.load(RHS_scale_ptr + rn)
        acc *= lhs_scale[:, None] * rhs_scale[None, :]
        c = acc.to(C.type.element_ty)
        rm_s = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn_s = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn_s = tl.max_contiguous(tl.multiple_of(rn_s % OUT_N, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm_s[:, None] < OUT_M) & (rn_s[None, :] < OUT_N)
        # Cast group_idx to int64 to prevent overflow in C group offset
        C_ = C + group_idx.to(tl.int64) * stride_cg + rm_s[:, None] * stride_cm + rn_s[None, :] * stride_cn
        tl.store(C_, c, c_mask)


def grouped_gemm_fp8_rowwise_variable_k_triton_kernel(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_scale: torch.Tensor,
    group_offs: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Variable-K grouped FP8 GEMM (backward, per-row/per-col scaling) using Triton.

    Computes C[g] = lhs[offs[g]:offs[g+1]]^T @ rhs[offs[g]:offs[g+1]]
    with vector scaling: result *= lhs_scale[rm] * rhs_scale[rn]

    Scale semantics:
      lhs_scale: (OUT_M,) fp32 — per output-row
      rhs_scale: (OUT_N,) fp32 — per output-col

    Args:
        lhs: [M_total, OUT_M] FP8 (after trans_c swap).
        rhs: [M_total, OUT_N] FP8 (after trans_c swap).
        lhs_scale: (OUT_M,) per-row scale for LHS, fp32.
        rhs_scale: (OUT_N,) per-col scale for RHS, fp32.
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
    num_sms = get_num_cus()

    if is_gfx950():
        set_triton_knobs_gfx950()
    avg_m_g = max(lhs.shape[0] // max(G, 1), 256)
    blk_m, blk_n, blk_k, group_m, cache_a, cache_b, num_stages_val, chunk_size = _get_gg_fp8_rw_vk_config(
        OUT_M, OUT_N, avg_m_g, lhs.dtype, rhs.dtype, G, num_sms
    )

    _grouped_fp8_rowwise_variable_k_gemm_kernel[(num_sms,)](
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
        BLOCK_SIZE_M=blk_m,
        BLOCK_SIZE_N=blk_n,
        BLOCK_SIZE_K=blk_k,
        GROUP_SIZE_M=group_m,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=chunk_size,
        CACHE_MODIFIER_A=cache_a,
        CACHE_MODIFIER_B=cache_b,
        num_warps=8,
        num_stages=num_stages_val,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return out


# ###########################################################################
#
#  PART 2 — BLOCKWISE FP8 Grouped GEMM
#
# ###########################################################################


# ═══════════════════════════════════════════════════════════════════════════════
# Blockwise FP8 Forward Kernel (persistent, CPU-sync-free)
#
# Computes: out[offs[g]:offs[g+1], :] = A[offs[g]:offs[g+1], :] @ B_view[g]
#   with block-wise scaling: A_scales[M, K//128], B_scales[G, n_blocks, k_blocks]
# ═══════════════════════════════════════════════════════════════════════════════


# 8-config curated set covering small/medium/large M and N-major/M-major reductions.
# All BK=128 (winners). Trimmed from a 32-config sweep that always picked one of these.
def _get_grouped_blockwise_autotune_configs():
    # BLOCK_SIZE_N pinned to 128: the persistent kernel loads ONE b_scale per
    # tile (assumes BN == SCALE_BLOCK_N == 128). With BN=256 the kernel applies
    # the wrong scale to half the output cols → ~10dB SNR drop. There's also a
    # known MI300X MFMA layout bug on FP8 e4m3fnuz with BN=64/256.
    return [
        # small/medium fallback
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "CHUNK_SIZE": 32,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "CHUNK_SIZE": 32,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "CHUNK_SIZE": 64,
            },
            num_warps=4,
            num_stages=2,
        ),
        # large-M, M-major
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "CHUNK_SIZE": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "CHUNK_SIZE": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "CHUNK_SIZE": 32,
            },
            num_warps=8,
            num_stages=1,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "CHUNK_SIZE": 64,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "CHUNK_SIZE": 64,
            },
            num_warps=8,
            num_stages=1,
        ),
    ]


# Single persistent kernel for both K-aligned (EVEN_K) and K-unaligned (masked
# tail) blockwise FP8. BLOCK_N pinned to 128 (BN=64/256 hit an MFMA layout bug on
# MI300X FP8 e4m3fnuz); BLOCK_M=256 must use BLOCK_K=128 (same bug).
# BLOCK_M=256 must use BLOCK_K=128 (same bug otherwise).
@triton.autotune(configs=_get_grouped_blockwise_autotune_configs(), key=["G", "N", "K"])
@triton.jit()
def _grouped_blockwise_fp8_persistent_gemm_kernel(
    A,
    B,
    C,
    A_scales_ptr,
    B_scales_ptr,
    group_offs_ptr,
    G: tl.constexpr,
    N,
    K,
    stride_am,
    stride_bg,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_as_k,
    stride_as_m,
    stride_bs_g,
    stride_bs_n,
    stride_bs_k,
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
    CACHE_MODIFIER: tl.constexpr,
    G_POW2: tl.constexpr,
):
    """Persistent grouped block-wise FP8 GEMM (fwd/dgrad), CPU-sync-free.

    Handles K % BLOCK_SIZE_K == 0 (EVEN_K=True, no tail) and the masked K-tail
    (EVEN_K=False) in one body. BLOCK_SIZE_K == SCALE_BLOCK_K == 128: one block
    scale per K-block. A scales are pre-shuffled [K//128, M] for coalesced reads."""
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Vectorise the per-group cumsum so the per-tile group lookup is a single
    # masked compare+reduce instead of an O(G) carry-dependent scan. tl.arange
    # needs a power-of-2 bound, so pad to G_POW2 and mask the tail to 0 tiles
    # (padded groups contribute nothing to cumsum / group_idx selection).
    g_arange = tl.arange(0, G_POW2)
    g_valid = g_arange < G
    g_starts = tl.load(group_offs_ptr + g_arange, mask=g_valid, other=0)
    g_ends = tl.load(group_offs_ptr + g_arange + 1, mask=g_valid, other=0)
    m_per_g_v = (g_ends - g_starts).to(tl.int32)
    tiles_per_g_v = tl.cdiv(m_per_g_v, BLOCK_SIZE_M) * num_pid_n
    cum_incl_v = tl.cumsum(tiles_per_g_v, axis=0)
    total_tiles = tl.sum(tiles_per_g_v, axis=0)

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32

    for global_tile_id in range(pid, total_tiles, NUM_SMS):
        # group_idx via vector compare; gather (m_start, M_g, tile_start) via mask+sum.
        group_idx = tl.sum((cum_incl_v <= global_tile_id).to(tl.int32), axis=0)
        is_cur = (g_arange == group_idx).to(tl.int32)
        # Keep m_start_g int64: it is a row offset into A/C and feeds pointer
        # arithmetic (m_start_g * stride), which overflows int32 once M_total*K
        # exceeds 2^31. M_g (per-group rows) stays int32 (bounded).
        m_start_g = tl.sum(g_starts * is_cur.to(tl.int64), axis=0)
        M_g = tl.sum(m_per_g_v * is_cur, axis=0)
        tiles_m_g = tl.cdiv(M_g, BLOCK_SIZE_M)
        tile_start = tl.sum(tl.where(g_arange < group_idx, tiles_per_g_v, 0), axis=0)
        local_tile = global_tile_id - tile_start

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        swizzle_group = local_tile // num_pid_in_group
        first_pid_m = swizzle_group * GROUP_SIZE_M
        group_size_m = min(tiles_m_g - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((local_tile % num_pid_in_group) % group_size_m)
        pid_n = (local_tile % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        group_offset_b = group_idx.to(tl.int64) * stride_bg
        A_BASE = A + m_start_g * stride_am + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + group_offset_b + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        as_ptrs_base = A_scales_ptr + (m_start_g + rm.to(tl.int64)) * stride_as_m
        bs_ptr_base = B_scales_ptr + group_idx.to(tl.int64) * stride_bs_g + pid_n * stride_bs_n

        # ── K-loop with per-block scaling (one block scale per BLOCK_SIZE_K) ──
        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1
        tl.assume(loop_k > 1)
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for ki in range(0, loop_k):
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER)
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER)
            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER)
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER)
            partial = tl.dot(a, b)
            a_s = tl.load(as_ptrs_base + ki * stride_as_k)
            b_s = tl.load(bs_ptr_base + ki * stride_bs_k)
            acc += partial * (a_s * b_s)[:, None]
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            # ── Last partial K-block (masked) ──
            rk_last = loop_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_LAST = A + m_start_g * stride_am + rm[:, None] * stride_am + rk_last[None, :] * stride_ak
            B_LAST = B + group_offset_b + rk_last[:, None] * stride_bk + rn[None, :] * stride_bn
            if stride_ak == 1:
                A_LAST = tl.multiple_of(A_LAST, (1, 16))
            else:
                A_LAST = tl.multiple_of(A_LAST, (16, 1))
            if stride_bk == 1:
                B_LAST = tl.multiple_of(B_LAST, (16, 1))
            else:
                B_LAST = tl.multiple_of(B_LAST, (1, 16))
            a = tl.load(A_LAST, mask=rk_last[None, :] < K, other=0.0, cache_modifier=CACHE_MODIFIER)
            b = tl.load(B_LAST, mask=rk_last[:, None] < K, other=0.0, cache_modifier=CACHE_MODIFIER)
            partial = tl.dot(a, b)
            a_s = tl.load(as_ptrs_base + loop_k * stride_as_k)
            b_s = tl.load(bs_ptr_base + loop_k * stride_bs_k)
            acc += partial * (a_s * b_s)[:, None]

        c = acc.to(C.type.element_ty)
        rm_s = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
        rn_s = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rn_s = tl.max_contiguous(tl.multiple_of(rn_s, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm_s[:, None] < M_g) & (rn_s[None, :] < N)
        C_ = C + m_start_g * stride_cm + rm_s[:, None] * stride_cm + rn_s[None, :] * stride_cn
        tl.store(C_, c, c_mask)


# Blockwise FP8 variable-K backward (persistent, CPU-sync-free):
#   C[g] = LHS[g]^T @ RHS[g] with 1D+1D blockwise scales.
# Autotune is keyed on shape-stable (G, OUT_M, OUT_N) only — per-group M varies.
def _bwd_autotune_configs():
    # 8-config curated set for variable-K wgrad. BK=128 always wins; covers M-major and N-major.
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "CHUNK_SIZE": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "CHUNK_SIZE": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "CHUNK_SIZE": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "CHUNK_SIZE": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "CHUNK_SIZE": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "CHUNK_SIZE": 32,
            },
            num_warps=8,
            num_stages=1,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "CHUNK_SIZE": 32,
            },
            num_warps=8,
            num_stages=1,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "CHUNK_SIZE": 32,
            },
            num_warps=4,
            num_stages=2,
        ),
    ]


@triton.autotune(
    configs=_bwd_autotune_configs(),
    key=["G", "OUT_M", "OUT_N", "A_K_CONTIGUOUS", "B_K_CONTIGUOUS"],
)
@triton.jit()
def _grouped_blockwise_fp8_variable_k_gemm_kernel(
    LHS,
    RHS,
    C,
    LHS_scales_ptr,
    RHS_scales_ptr,
    group_offs_ptr,
    G,
    OUT_M,
    OUT_N,
    stride_lhs_m,
    stride_rhs_m,
    stride_cg,
    stride_cm,
    stride_cn,
    stride_ls_0,
    stride_ls_1,
    stride_rs_0,
    stride_rs_1,
    stride_lhs_n: tl.constexpr,
    stride_rhs_n: tl.constexpr,
    A_K_CONTIGUOUS: tl.constexpr,
    B_K_CONTIGUOUS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    CACHE_MODIFIER: tl.constexpr,
):
    """Variable-K BWD: C[g] = LHS_g^T @ RHS_g. M_g is segment-padded to BLOCK_SIZE_K
    by quant_fp8_blockwise_segment_m_row_col_impl, so no K-loop masking is needed."""
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

    acc_dtype = tl.float32

    for global_tile in range(pid, total_tiles, NUM_SMS):
        # ── Map to (group, local_tile) ──
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
        m_start = tl.load(group_offs_ptr + group_idx)  # int64
        M_g = (tl.load(group_offs_ptr + group_idx + 1) - m_start).to(tl.int32)

        # ── Output indices ──
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % OUT_M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % OUT_N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # ── Base pointers ──
        # K-contig path: LHS layout is [OUT_M, M_padded_max], rk dim has stride 1.
        # Strided path: LHS layout is [M_padded_total, OUT_M], rk dim has stride OUT_M.
        if A_K_CONTIGUOUS:
            LHS_BASE = LHS + m_start + rm[:, None] * stride_lhs_n + rk[None, :]
        else:
            LHS_BASE = LHS + m_start * stride_lhs_m + rm[:, None] * stride_lhs_n + rk[None, :] * stride_lhs_m
        if B_K_CONTIGUOUS:
            RHS_BASE = RHS + m_start + rk[:, None] + rn[None, :] * stride_rhs_n
        else:
            RHS_BASE = RHS + m_start * stride_rhs_m + rk[:, None] * stride_rhs_m + rn[None, :] * stride_rhs_n

        scale_row_start = m_start // BLOCK_SIZE_K

        # ── K-loop over M_g with block-wise 1D+1D scaling ──
        # M_g is always a multiple of BLOCK_SIZE_K (data padded), so no masking needed.
        loop_k = M_g // BLOCK_SIZE_K
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for k in range(loop_k):
            if A_K_CONTIGUOUS:
                a = tl.load(tl.multiple_of(LHS_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER)
            elif stride_lhs_n == 1:
                a = tl.load(tl.multiple_of(LHS_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER)
            else:
                a = tl.load(tl.multiple_of(LHS_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER)

            if B_K_CONTIGUOUS:
                b = tl.load(tl.multiple_of(RHS_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER)
            elif stride_rhs_n == 1:
                b = tl.load(tl.multiple_of(RHS_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER)
            else:
                b = tl.load(tl.multiple_of(RHS_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER)

            partial = tl.dot(a, b)

            # 1D+1D block-wise scales
            scale_row = scale_row_start + k
            a_s = tl.load(LHS_scales_ptr + scale_row * stride_ls_0 + rm * stride_ls_1)
            b_s = tl.load(RHS_scales_ptr + scale_row * stride_rs_0 + rn * stride_rs_1)
            acc += partial * a_s[:, None] * b_s[None, :]

            if A_K_CONTIGUOUS:
                LHS_BASE += BLOCK_SIZE_K
            else:
                LHS_BASE += BLOCK_SIZE_K * stride_lhs_m
            if B_K_CONTIGUOUS:
                RHS_BASE += BLOCK_SIZE_K
            else:
                RHS_BASE += BLOCK_SIZE_K * stride_rhs_m

        # ── Store output ──
        c = acc.to(C.type.element_ty)
        rm_s = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn_s = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn_s = tl.max_contiguous(tl.multiple_of(rn_s % OUT_N, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm_s[:, None] < OUT_M) & (rn_s[None, :] < OUT_N)
        C_ = C + group_idx.to(tl.int64) * stride_cg + rm_s[:, None] * stride_cm + rn_s[None, :] * stride_cn
        tl.store(C_, c, c_mask)


# ── Blockwise FP8 Forward Public API ──


def grouped_gemm_fp8_blockwise_triton_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_offs: torch.Tensor,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Persistent grouped block-wise FP8 GEMM (CPU-sync-free) using Triton.

    Computes: out[offs[g]:offs[g+1], :] = A[offs[g]:offs[g+1], :] @ B_view[g]
    with block-wise scaling for each K-block.

    Args:
        a: [M_total, K] FP8 input (trans_a=False always).
        b: [G, N, K] (if trans_b=True) or [G, K, N] FP8 weights.
        a_scales: [K//128, M_total] float32, block-wise scale for A (pre-shuffled).
        b_scales: [G, ceil(N/128), ceil(K/128)] or [G, ceil(K/128), ceil(N/128)] float32.
        group_offs: [G+1] int64 prefix sum of group lengths.
        trans_b: If True, b[g] is [N, K] (transposed).
        out_dtype: Output dtype (default bfloat16).

    Returns:
        [M_total, N] output in out_dtype.
    """
    if is_gfx950():
        set_triton_knobs_gfx950()
    else:
        _set_amd_knobs(enable=True)

    assert a.ndim == 2, f"a must be 2D, got {a.shape}"
    assert b.ndim == 3, f"b must be 3D, got {b.shape}"
    assert b_scales.ndim == 3, f"b_scales must be 3D, got {b_scales.shape}"

    M_total, K = a.shape
    G = b.shape[0]

    if trans_b:
        N = b.shape[1]
        stride_bk = b.stride(2)
        stride_bn = b.stride(1)
        stride_bs_n = b_scales.stride(1)
        stride_bs_k = b_scales.stride(2)
    else:
        N = b.shape[2]
        stride_bk = b.stride(1)
        stride_bn = b.stride(2)
        stride_bs_n = b_scales.stride(2)
        stride_bs_k = b_scales.stride(1)

    stride_bg = b.stride(0)
    stride_ak = a.stride(1)

    out = torch.empty((M_total, N), device=a.device, dtype=out_dtype)
    # a_scales arrives pre-shuffled as [K_blocks, M_total] (quant op + CK both emit
    # this layout); kernel reads it coalesced with no runtime .T.contiguous().
    num_sms = get_num_cus()

    aligned = K % 128 == 0
    # First call for this (G, N, K, aligned) shape: prime autotune with a balanced
    # group_offs distribution. Triton's autotune key is (G, N, K) only — group_lens
    # is not part of the key, so whichever distribution happens to drive the first
    # call gets baked into the cached config. In MoE training, per-step routing is
    # uneven and varies, so without this warm-up we'd cache a config tuned to one
    # accidental imbalance. Run a synthetic balanced-offs trial into a scratch out
    # so the chosen config generalizes across the lifetime of the process.
    if (G, N, K, aligned) not in _grouped_blockwise_warmed:
        _grouped_blockwise_warmed.add((G, N, K, aligned))
        per = M_total // G
        bal_offs = torch.arange(G + 1, device=group_offs.device, dtype=group_offs.dtype) * per
        bal_offs[-1] = M_total
        out_warm = torch.empty_like(out)
        if hasattr(triton, "knobs") and hasattr(triton.knobs, "amd"):
            triton.knobs.amd.use_async_copy = False
        _grouped_blockwise_fp8_persistent_gemm_kernel[(num_sms,)](
            a,
            b,
            out_warm,
            a_scales,
            b_scales,
            bal_offs,
            G,
            N,
            K,
            a.stride(0),
            stride_bg,
            stride_bn,
            out_warm.stride(0),
            out_warm.stride(1),
            a_scales.stride(0),
            a_scales.stride(1),
            b_scales.stride(0),
            stride_bs_n,
            stride_bs_k,
            stride_ak=stride_ak,
            stride_bk=stride_bk,
            NUM_SMS=num_sms,
            NUM_XCDS=NUM_XCDS,
            EVEN_K=aligned,
            CACHE_MODIFIER=".ca",
            G_POW2=triton.next_power_of_2(G),
            waves_per_eu=0,
            matrix_instr_nonkdim=16,
            kpack=2,
        )

    if hasattr(triton, "knobs") and hasattr(triton.knobs, "amd"):
        triton.knobs.amd.use_async_copy = False
    _grouped_blockwise_fp8_persistent_gemm_kernel[(num_sms,)](
        a,
        b,
        out,
        a_scales,
        b_scales,
        group_offs,
        G,
        N,
        K,
        a.stride(0),
        stride_bg,
        stride_bn,
        out.stride(0),
        out.stride(1),
        a_scales.stride(0),
        a_scales.stride(1),
        b_scales.stride(0),
        stride_bs_n,
        stride_bs_k,
        stride_ak=stride_ak,
        stride_bk=stride_bk,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        EVEN_K=aligned,
        CACHE_MODIFIER=".ca",
        G_POW2=triton.next_power_of_2(G),
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=2,
    )
    return out


# ── Blockwise FP8 Variable-K Backward Public API ──


def grouped_gemm_fp8_blockwise_variable_k_triton_kernel(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_scales: torch.Tensor,
    rhs_scales: torch.Tensor,
    group_offs: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    a_k_contig: bool = False,
    b_k_contig: bool = False,
    out_M: int = -1,
    out_N: int = -1,
) -> torch.Tensor:
    """Variable-K grouped block-wise FP8 GEMM (backward, 1D+1D scaling) using Triton.

    Computes: C[g] = lhs[offs[g]:offs[g+1]]^T @ rhs[offs[g]:offs[g+1]]
    with 1D+1D block-wise scaling applied in the K-loop.

    Output: [G, OUT_M, OUT_N].

    Args:
        lhs: [M_padded_total, OUT_M] FP8 if a_k_contig=False (default).
             [OUT_M, M_padded_max] FP8 if a_k_contig=True (K-axis contig).
        rhs: [M_padded_total, OUT_N] or [OUT_N, M_padded_max] (mirror of lhs).
        lhs_scales: [ceil(M_padded/128), OUT_M] float32.
        rhs_scales: [ceil(M_padded/128), OUT_N] float32.
        group_offs: [G+1] int64 padded segment offsets.
        out_dtype: Output dtype (default bfloat16).
        a_k_contig: If True, lhs is the transposed layout [OUT_M, M_padded_max]
                    so the K-axis is contiguous → kernel uses vectorized K-contig
                    loads. Combined with the fused-quant-with-xpose op, replaces
                    the strided-A pattern at no extra cost.
        b_k_contig: Same for rhs.
        out_M, out_N: Required when *_k_contig=True (cannot infer from transposed
                      shape because it equals OUT_M/N).

    Returns:
        [G, OUT_M, OUT_N] output.
    """
    if is_gfx950():
        set_triton_knobs_gfx950()
    else:
        _set_amd_knobs(enable=False)

    assert lhs.ndim == 2 and rhs.ndim == 2
    G = group_offs.shape[0] - 1

    if a_k_contig:
        OUT_M = out_M if out_M > 0 else lhs.shape[0]
    else:
        OUT_M = lhs.shape[1]
    if b_k_contig:
        OUT_N = out_N if out_N > 0 else rhs.shape[0]
    else:
        OUT_N = rhs.shape[1]

    out = torch.empty((G, OUT_M, OUT_N), device=lhs.device, dtype=out_dtype)
    num_sms = get_num_cus()

    # K-axis stride: for K-contig, dim 1 of [OUT_M, M_padded_max] = 1.
    # For strided, dim 0 of [M_padded_total, OUT_M] = OUT_M.
    stride_lhs_m = lhs.stride(1) if a_k_contig else lhs.stride(0)
    stride_rhs_m = rhs.stride(1) if b_k_contig else rhs.stride(0)
    # Output-row stride: for K-contig, dim 0 = M_padded_max. For strided, dim 1 = 1.
    stride_lhs_n = lhs.stride(0) if a_k_contig else lhs.stride(1)
    stride_rhs_n = rhs.stride(0) if b_k_contig else rhs.stride(1)

    # Same balanced-warmup logic as the fwd kernel: autotune key is
    # (G, OUT_M, OUT_N, A_K_CONTIGUOUS, B_K_CONTIGUOUS) — group_offs is not part
    # of the key, so prime the cache once with a balanced distribution so the
    # chosen config generalizes across MoE per-step routing variation.
    warm_key = (G, OUT_M, OUT_N, a_k_contig, b_k_contig)
    if warm_key not in _grouped_blockwise_vk_warmed:
        _grouped_blockwise_vk_warmed.add(warm_key)
        # Padded segment lens for variable-K need to respect BLOCK_K alignment;
        # use M_padded // G rounded down to BLOCK_K (=128) granularity.
        M_padded = lhs.shape[1] if a_k_contig else lhs.shape[0]
        per = max((M_padded // G) // 128 * 128, 128)
        bal_offs = torch.arange(G + 1, device=group_offs.device, dtype=group_offs.dtype) * per
        bal_offs[-1] = M_padded
        out_warm = torch.empty_like(out)
        _grouped_blockwise_fp8_variable_k_gemm_kernel[(num_sms,)](
            lhs,
            rhs,
            out_warm,
            lhs_scales,
            rhs_scales,
            bal_offs,
            G,
            OUT_M,
            OUT_N,
            stride_lhs_m,
            stride_rhs_m,
            out_warm.stride(0),
            out_warm.stride(1),
            out_warm.stride(2),
            lhs_scales.stride(0),
            lhs_scales.stride(1),
            rhs_scales.stride(0),
            rhs_scales.stride(1),
            stride_lhs_n=stride_lhs_n,
            stride_rhs_n=stride_rhs_n,
            A_K_CONTIGUOUS=a_k_contig,
            B_K_CONTIGUOUS=b_k_contig,
            NUM_SMS=num_sms,
            NUM_XCDS=NUM_XCDS,
            CACHE_MODIFIER=".ca",
            waves_per_eu=2,
            matrix_instr_nonkdim=16,
            kpack=2,
        )

    _grouped_blockwise_fp8_variable_k_gemm_kernel[(num_sms,)](
        lhs,
        rhs,
        out,
        lhs_scales,
        rhs_scales,
        group_offs,
        G,
        OUT_M,
        OUT_N,
        stride_lhs_m,
        stride_rhs_m,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lhs_scales.stride(0),
        lhs_scales.stride(1),
        rhs_scales.stride(0),
        rhs_scales.stride(1),
        stride_lhs_n=stride_lhs_n,
        stride_rhs_n=stride_rhs_n,
        A_K_CONTIGUOUS=a_k_contig,
        B_K_CONTIGUOUS=b_k_contig,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CACHE_MODIFIER=".ca",
        waves_per_eu=2,
        matrix_instr_nonkdim=16,
        kpack=2,
    )
    return out
