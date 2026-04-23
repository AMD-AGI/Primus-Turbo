###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Grouped MX-FP8 GEMM Triton persistent kernel (CPU-sync-free).

MX-FP8 (per OCP Microscaling spec): FP8 E4M3 values with e8m0 power-of-2 scales
at group_size=32 along K. Maps to the native CDNA4 (gfx950) MFMA instruction
``v_mfma_scale_f32_32x32x64_f8f6f4`` via Triton's ``tl.dot_scaled``.

Contains:
  - _grouped_mxfp8_persistent_gemm_kernel: Forward kernel (acc-seeded MFMA chain)
  - grouped_gemm_mxfp8_triton_kernel    : Forward public API

MX-FP8 scale layout (what tl.dot_scaled expects):
  a_scale [M_total, K//32] uint8 (e8m0)
  b_scale [G, N, K//32]    uint8 (e8m0)   — N-first; do NOT transpose

Environment variable: PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON activates this kernel
when the FP8 path is dispatched with ScalingGranularity.MX_BLOCKWISE.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from primus_turbo.triton.gemm.gemm_kernel import _is_gfx950, _set_knobs_gfx950
from primus_turbo.triton.grouped_gemm.grouped_gemm_kernel import (
    NUM_XCDS,
    _chiplet_transform_chunked,
    _get_num_cus,
)


# MX-FP8 is defined at K-group size 32 (OCP spec, mandatory for tl.dot_scaled w/ e8m0 scales)
MX_GROUP_SIZE = 32


# ═══════════════════════════════════════════════════════════════════════════════
# MX-FP8 Forward Kernel (persistent, CPU-sync-free)
#
# Computes: out[offs[g]:offs[g+1], :] = A[offs[g]:offs[g+1], :] @ B[g]
#   where A/B are FP8 E4M3 with e8m0 scales at group_size=32 along K.
# ═══════════════════════════════════════════════════════════════════════════════


@triton.jit()
def _grouped_mxfp8_persistent_gemm_kernel(
    # Pointers
    A,              # [M_total, K] fp8 e4m3
    B,              # [G, K, N]   fp8 e4m3
    C,              # [M_total, N] bf16/fp16
    A_scale_ptr,    # [M_total, K//32] uint8 (e8m0)
    B_scale_ptr,    # [G, N, K//32]    uint8 (e8m0)
    group_offs_ptr, # [G+1] int64
    # Dimensions
    G,              # number of groups (runtime)
    N,
    K,
    # Strides
    stride_am,
    stride_bg,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_as_m,
    stride_as_k,
    stride_bs_g,
    stride_bs_n,
    stride_bs_k,
    # Constexpr tile config
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_K: tl.constexpr,       # MX group size (32)
    GROUP_SIZE_M: tl.constexpr,  # M-swizzle size
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Persistent grouped MX-FP8 GEMM kernel. Uses tl.dot_scaled to emit native
    v_mfma_scale_f32_32x32x64_f8f6f4 on gfx950 (fallback to bf16 emulation on
    older archs)."""
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Compute total tiles across all groups (persistent dispatch).
    total_tiles: tl.int32 = 0
    for _g in range(G):
        m_g = (tl.load(group_offs_ptr + _g + 1) - tl.load(group_offs_ptr + _g)).to(tl.int32)
        total_tiles += tl.cdiv(m_g, BLOCK_SIZE_M) * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    for global_tile_id in range(pid, total_tiles, NUM_SMS):
        # Find group via linear scan (O(G)).
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

        # Group-local tile -> (pid_m, pid_n) with GROUP_SIZE_M swizzle.
        local_tile = global_tile_id - tile_start
        m_start_g = tl.load(group_offs_ptr + group_idx)  # int64
        M_g = (tl.load(group_offs_ptr + group_idx + 1) - m_start_g).to(tl.int32)
        tiles_m_g = tl.cdiv(M_g, BLOCK_SIZE_M)

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
        rks = tl.arange(0, BLOCK_SIZE_K // GROUP_K)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        group_offset_b = group_idx.to(tl.int64) * stride_bg
        A_ptr = A + (m_start_g + rm.to(tl.int64))[:, None] * stride_am + rk[None, :]
        B_ptr = B + group_offset_b + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        AS_ptr = A_scale_ptr + (m_start_g + rm.to(tl.int64))[:, None] * stride_as_m + rks[None, :] * stride_as_k
        BS_ptr = (B_scale_ptr + group_idx.to(tl.int64) * stride_bs_g
                  + rn[:, None] * stride_bs_n + rks[None, :] * stride_bs_k)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # Full K-block iterations.
        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1
        tl.assume(loop_k > 0)
        for ki in range(0, loop_k):
            a = tl.load(A_ptr)
            b = tl.load(B_ptr)
            a_s = tl.load(AS_ptr)
            b_s = tl.load(BS_ptr)
            acc = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc=acc, fast_math=True)
            A_ptr += BLOCK_SIZE_K
            B_ptr += BLOCK_SIZE_K * stride_bk
            AS_ptr += (BLOCK_SIZE_K // GROUP_K) * stride_as_k
            BS_ptr += (BLOCK_SIZE_K // GROUP_K) * stride_bs_k

        # Masked tail iter for K not divisible by BLOCK_SIZE_K.
        if not EVEN_K:
            k_base = loop_k * BLOCK_SIZE_K
            mask_k = (k_base + tl.arange(0, BLOCK_SIZE_K)) < K
            mask_ks = (k_base // GROUP_K + tl.arange(0, BLOCK_SIZE_K // GROUP_K)) < tl.cdiv(K, GROUP_K)
            a = tl.load(A_ptr, mask=mask_k[None, :], other=0.0)
            b = tl.load(B_ptr, mask=mask_k[:, None], other=0.0)
            # e8m0 scale = 0 -> real exponent -127 -> effectively zero contribution
            a_s = tl.load(AS_ptr, mask=mask_ks[None, :], other=0)
            b_s = tl.load(BS_ptr, mask=mask_ks[None, :], other=0)
            acc = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc=acc, fast_math=True)

        rm_s = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
        rn_s = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rn_s = tl.max_contiguous(tl.multiple_of(rn_s, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm_s[:, None] < M_g) & (rn_s[None, :] < N)
        C_ptr = C + m_start_g * stride_cm + rm_s[:, None] * stride_cm + rn_s[None, :] * stride_cn
        tl.store(C_ptr, acc.to(C.type.element_ty), c_mask)


def grouped_gemm_mxfp8_triton_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    group_offs: torch.Tensor,
    trans_b: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Persistent grouped MX-FP8 GEMM (CPU-sync-free) using tl.dot_scaled.

    Computes:
        out[offs[g]:offs[g+1], :] = A[offs[g]:offs[g+1], :] @ B_view[g]
    with e8m0 per-32-K-element scales applied natively by the MFMA instruction.
    Layouts (for B) match the rest of the Primus-Turbo FP8 grouped GEMM family:
      trans_b=False -> B_view[g] is b[g] interpreted as [K, N]
      trans_b=True  -> B_view[g] is b[g] interpreted as [N, K], transposed into [K, N]

    Args:
        a:        [M_total, K]       fp8 e4m3 input.
        b:        [G, K, N] if trans_b=False else [G, N, K]  fp8 e4m3 weights.
        a_scale:  [M_total, K//32]   uint8 e8m0 activation scale.
        b_scale:  [G, N, K//32]      uint8 e8m0 weight scale (N-first; do NOT transpose).
        group_offs: [G+1] int64 prefix sum of group lengths.
        trans_b:  Whether B is stored in [G, N, K] (NT) layout.
        out_dtype: Output dtype (bf16 or fp16).

    Returns:
        [M_total, N] output in ``out_dtype``.
    """
    assert a.ndim == 2, f"a must be 2D, got {a.shape}"
    assert b.ndim == 3, f"b must be 3D, got {b.shape}"

    # Opt into the gfx950 AMD compiler knobs used by the rest of the Triton
    # FP8 family (use_async_copy, use_block_pingpong, scalarize_packed_fops).
    # Consistency with sibling kernels — matters when mxfp8 is the first (or
    # only) FP8 kernel invoked in a process, before any sibling kernel has
    # flipped the global _KNOBS_SET flag.
    if _is_gfx950():
        _set_knobs_gfx950()

    M_total, K = a.shape
    G = b.shape[0]
    if trans_b:
        # B is [G, N, K]; stride_bk along the K axis (axis 2), stride_bn along N axis (axis 1).
        N = b.shape[1]
        assert b.shape[2] == K, f"K mismatch: a={K}, b={b.shape[2]} (trans_b=True)"
        stride_bk = b.stride(2)
        stride_bn = b.stride(1)
    else:
        # B is [G, K, N]; stride_bk along K axis (axis 1), stride_bn along N axis (axis 2).
        assert b.shape[1] == K, f"K mismatch: a={K}, b={b.shape[1]} (trans_b=False)"
        N = b.shape[2]
        stride_bk = b.stride(1)
        stride_bn = b.stride(2)

    assert K % MX_GROUP_SIZE == 0, f"K={K} must be multiple of {MX_GROUP_SIZE}"

    out = torch.empty((M_total, N), device=a.device, dtype=out_dtype)
    num_sms = _get_num_cus()

    # Fixed config — tuned for gpt_oss_20B gate_up shape on MI355X. Full sweep
    # (BLK_K × stages × GROUP_M) showed BLK=256×256×128, stages=2, GM=4 is the
    # LDS-fitting optimum. Larger BLK_K or stages=3 exceeds 160 KB LDS budget;
    # smaller BLK_K=64 regresses (1 MFMA/iter loses loop-overhead amortization
    # — the scaled MFMA instruction already covers K=64).
    blk_m = 256
    blk_n = 256
    blk_k = 128
    group_m = 4
    even_k = (K % blk_k) == 0

    _grouped_mxfp8_persistent_gemm_kernel[(num_sms,)](
        a, b, out, a_scale, b_scale, group_offs,
        G, N, K,
        a.stride(0),
        b.stride(0), stride_bk, stride_bn,
        out.stride(0), out.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        b_scale.stride(0), b_scale.stride(1), b_scale.stride(2),
        BLOCK_SIZE_M=blk_m,
        BLOCK_SIZE_N=blk_n,
        BLOCK_SIZE_K=blk_k,
        GROUP_K=MX_GROUP_SIZE,
        GROUP_SIZE_M=group_m,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=32,
        EVEN_K=even_k,
        num_warps=8,
        num_stages=2,
        waves_per_eu=0,
        matrix_instr_nonkdim=32,
    )
    return out
