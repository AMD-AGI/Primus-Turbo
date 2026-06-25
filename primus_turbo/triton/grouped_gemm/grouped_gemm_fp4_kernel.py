###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Grouped MXFP4 (MX_BLOCKWISE) GEMM Triton persistent kernels (gfx950).

Mirrors the MXFP8 grouped kernels in ``grouped_gemm_fp8_kernel.py`` but the
operands are E2M1 FP4 packed two-values-per-byte along the contraction (K)
axis, fed to the hardware block-scaled MMA via ``tl.dot_scaled(..., "e2m1", ...)``
with E8M0 (VEC_SIZE=32) scales.

  - _grouped_mxfp4_persistent_gemm_kernel:    Forward (NT)  C[g] = A[g] @ B[g]^T
  - grouped_gemm_mxfp4_triton_kernel:         Forward public API
  - _grouped_mxfp4_variable_k_gemm_kernel:    Backward wgrad C[g] = LHS[g] @ RHS[g]^T
  - grouped_gemm_mxfp4_variable_k_triton_kernel: Backward (variable-K) public API

FP4 packing notes:
  * data tensors store K/2 (resp. M/2) bytes along the contraction axis;
  * scale tensors store K/32 (resp. M/32) E8M0 bytes (one per 1x32 block);
  * scales are stored transposed (K/32, free) / (G, K/32, free) for a coalesced
    per-K-iter load, transposed back in-reg for tl.dot_scaled.

Only EVEN contraction (a multiple of BLOCK_SIZE_K = 128 logical elements) is
supported; the FP4 quantizer pads K to 128 and the MX wrapper pads per-group M
to 128, so this always holds in practice.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.triton.grouped_gemm.grouped_gemm_kernel import (
    NUM_XCDS,
    _chiplet_transform_chunked,
)
from primus_turbo.triton.utils.triton_knobs_helper import set_triton_knobs_gfx950

# E8M0 block size: one scale per 1x32 logical-element block (same as MXFP8).
# tl.dot_scaled consumes the "e2m1" FP4 format string (inlined in the kernels;
# @triton.jit cannot read module globals).
VEC_SIZE = 32


# ###########################################################################
#  Forward (NT):   C[g] = A[g] @ B[g]^T
#    A (total_M, K/2) fp4-packed, A_scale (K/32, total_M) e8m0
#    B (G, N, K/2)    fp4-packed, B_scale (G, K/32, N)   e8m0
# ###########################################################################


@triton.jit
def _grouped_mxfp4_persistent_gemm_kernel(
    A,
    B,
    C,
    A_scale,  # (total_M, K//32) uint8 e8m0
    B_scale,  # (G, N, K//32) uint8 e8m0
    rd_offs_ptr,  # (G+1) int64 — read base along M (A / A_scale)
    out_offs_ptr,  # (G+1) int64 — write base along M (C, tile count)
    G,
    N,
    K,  # logical contraction size (multiple of BLOCK_SIZE_K)
    stride_am,
    stride_ak,  # along packed K bytes
    stride_bg,
    stride_bn,
    stride_bk,  # along packed K bytes
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsg,
    stride_bsn,
    stride_bsk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # logical K per iter
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    CACHE_MODIFIER: tl.constexpr,
    VEC: tl.constexpr,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    total_tiles: tl.int32 = 0
    for _g in range(G):
        m_g = (tl.load(out_offs_ptr + _g + 1) - tl.load(out_offs_ptr + _g)).to(tl.int32)
        total_tiles += tl.cdiv(m_g, BLOCK_SIZE_M) * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_cm > 0)

    BK_PACK: tl.constexpr = BLOCK_SIZE_K // 2  # packed bytes per K-iter
    BK_SCALE: tl.constexpr = BLOCK_SIZE_K // VEC  # scale entries per K-iter

    for global_tile_id in range(pid, total_tiles, NUM_SMS):
        # ── locate group (linear scan) ──
        group_idx: tl.int32 = 0
        tile_start: tl.int32 = 0
        cumsum: tl.int32 = 0
        for _g in range(G):
            m_g_i = (tl.load(out_offs_ptr + _g + 1) - tl.load(out_offs_ptr + _g)).to(tl.int32)
            tiles_g = tl.cdiv(m_g_i, BLOCK_SIZE_M) * num_pid_n
            new_cumsum = cumsum + tiles_g
            if global_tile_id >= new_cumsum:
                group_idx = _g + 1
                tile_start = new_cumsum
            cumsum = new_cumsum

        local_tile = global_tile_id - tile_start
        m_rd = tl.load(rd_offs_ptr + group_idx)  # int64 read base (A / A_scale)
        m_out = tl.load(out_offs_ptr + group_idx)  # int64 write base (C)
        M_g = (tl.load(out_offs_ptr + group_idx + 1) - m_out).to(tl.int32)
        tiles_m_g = tl.cdiv(M_g, BLOCK_SIZE_M)

        # ── swizzle ──
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
        rk = tl.arange(0, BK_PACK)  # packed K bytes
        rks = tl.arange(0, BK_SCALE)  # scale entries
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        A_BASE = A + (m_rd + rm[:, None]) * stride_am + rk[None, :] * stride_ak
        B_BASE = B + group_idx.to(tl.int64) * stride_bg + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        AS_BASE = A_scale + rks[:, None] * stride_ask + (m_rd + rm[None, :]) * stride_asm
        BS_BASE = (
            B_scale
            + group_idx.to(tl.int64) * stride_bsg
            + rks[:, None] * stride_bsk
            + rn[None, :] * stride_bsn
        )

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        loop_k = K // BLOCK_SIZE_K
        for ki in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER)  # (BM, BK/2)
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER)  # (BK/2, BN)
            a_s = tl.trans(tl.load(AS_BASE))  # (BK/32, BM) -> (BM, BK/32)
            b_s = tl.trans(tl.load(BS_BASE))  # (BK/32, BN) -> (BN, BK/32)
            acc = tl.dot_scaled(a, a_s, "e2m1", b, b_s, "e2m1", acc)
            A_BASE += BK_PACK * stride_ak
            B_BASE += BK_PACK * stride_bk
            AS_BASE += BK_SCALE * stride_ask
            BS_BASE += BK_SCALE * stride_bsk

        c = acc.to(C.type.element_ty)
        rm_s = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
        rn_s = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rn_s = tl.max_contiguous(tl.multiple_of(rn_s, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm_s[:, None] < M_g) & (rn_s[None, :] < N)
        C_ = C + (m_out + rm_s[:, None]) * stride_cm + rn_s[None, :] * stride_cn
        tl.store(C_, c, c_mask)


def grouped_gemm_mxfp4_triton_kernel(
    a,
    a_scale,
    b,
    b_scale,
    group_offs,
    N,
    K,
    group_offs_out=None,
    out_dtype=torch.bfloat16,
    num_cu=None,
):
    """A(total_M, K/2) @ B(G, N, K/2)^T -> C. FP4-packed data, e8m0 uint8 scales.

    group_offs:     read offsets along M for A / A_scale.
    group_offs_out: write offsets along M for C (defaults to group_offs).
    K is the logical contraction size (must be a multiple of BLOCK_SIZE_K=128).
    """
    if group_offs_out is None:
        group_offs_out = group_offs
    if is_gfx950():
        set_triton_knobs_gfx950()
    G = b.shape[0]
    c = torch.empty((a.shape[0], N), dtype=out_dtype, device=a.device)
    a_s = a_scale.view(torch.uint8)
    b_s = b_scale.view(torch.uint8)
    a_u8 = a.view(torch.uint8)
    b_u8 = b.view(torch.uint8)
    cu = num_cu if num_cu is not None else torch.cuda.get_device_properties(a.device).multi_processor_count
    BM, BN, BK = 256, 256, 128
    m_alloc = a.shape[0]
    avg_m = max(m_alloc // max(G, 1), 1)
    tiles_n = (N + BN - 1) // BN
    GM = 8 if min((avg_m + BM - 1) // BM, tiles_n) < 16 else 4
    total_tiles = ((m_alloc + BM - 1) // BM + G) * tiles_n
    num_sms = min(total_tiles, cu)
    chunk = 64 if num_sms >= NUM_XCDS * 64 else 32
    grid = (num_sms,)
    _grouped_mxfp4_persistent_gemm_kernel[grid](
        a_u8,
        b_u8,
        c,
        a_s,
        b_s,
        group_offs,
        group_offs_out,
        G,
        N,
        K,
        a_u8.stride(0),
        a_u8.stride(1),
        b_u8.stride(0),
        b_u8.stride(1),
        b_u8.stride(2),
        c.stride(0),
        c.stride(1),
        a_s.stride(0),  # stride_asm (M);   a_s is (total_M, K/32)
        a_s.stride(1),  # stride_ask (K/32 contiguous)
        b_s.stride(0),  # stride_bsg
        b_s.stride(1),  # stride_bsn (N);   b_s is (G, N, K/32)
        b_s.stride(2),  # stride_bsk (K/32 contiguous)
        BLOCK_SIZE_M=BM,
        BLOCK_SIZE_N=BN,
        BLOCK_SIZE_K=BK,
        GROUP_SIZE_M=GM,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=chunk,
        CACHE_MODIFIER=".ca",
        VEC=VEC_SIZE,
        num_warps=8,
        num_stages=2,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return c


# ###########################################################################
#  Variable-K backward (wgrad): C[g] = LHS[g] @ RHS[g]^T, reduction over M_g
#    LHS (OUT_M, M_total/2) fp4-packed, LHS_scale (OUT_M, M_total/32) e8m0
#    RHS (OUT_N, M_total/2) fp4-packed, RHS_scale (OUT_N, M_total/32) e8m0
#    C   (G, OUT_M, OUT_N)
#    go_pad: padded per-group offsets along M (each M_g a multiple of 128)
# ###########################################################################


@triton.jit
def _grouped_mxfp4_variable_k_gemm_kernel(
    LHS,
    RHS,
    C,
    LHS_scale,
    RHS_scale,
    go_pad_ptr,
    G,
    OUT_M,
    OUT_N,
    stride_lm,
    stride_lk,  # along packed M bytes
    stride_rm,
    stride_rk,  # along packed M bytes
    stride_cg,
    stride_cm,
    stride_cn,
    stride_lsm,
    stride_lsk,
    stride_rsm,
    stride_rsk,
    BLOCK_SIZE_M: tl.constexpr,  # over OUT_M
    BLOCK_SIZE_N: tl.constexpr,  # over OUT_N
    BLOCK_SIZE_K: tl.constexpr,  # logical reduction over M_g
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    CACHE_MODIFIER: tl.constexpr,
    VEC: tl.constexpr,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)
    tiles_m = tl.cdiv(OUT_M, BLOCK_SIZE_M)
    tiles_n = tl.cdiv(OUT_N, BLOCK_SIZE_N)
    tiles_per_group = tiles_m * tiles_n
    total_tiles = G * tiles_per_group

    tl.assume(stride_lm > 0)
    tl.assume(stride_rm > 0)
    tl.assume(stride_cm > 0)

    BK_PACK: tl.constexpr = BLOCK_SIZE_K // 2
    BK_SCALE: tl.constexpr = BLOCK_SIZE_K // VEC

    for global_tile in range(pid, total_tiles, NUM_SMS):
        group_idx = global_tile // tiles_per_group
        local_tile = global_tile - group_idx * tiles_per_group

        num_pid_in_group = GROUP_SIZE_M * tiles_n
        swizzle_group = local_tile // num_pid_in_group
        first_pid_m = swizzle_group * GROUP_SIZE_M
        group_size_m = min(tiles_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((local_tile % num_pid_in_group) % group_size_m)
        pid_n = (local_tile % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        m_start = tl.load(go_pad_ptr + group_idx)  # int64 (logical M, multiple of 128)
        M_g = (tl.load(go_pad_ptr + group_idx + 1) - m_start).to(tl.int32)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % OUT_M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % OUT_N
        rk = tl.arange(0, BK_PACK)
        rks = tl.arange(0, BK_SCALE)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # packed M offset = m_start // 2 (M_g multiple of 128 -> even)
        mp0 = (m_start // 2).to(tl.int64)
        L_BASE = LHS + rm[:, None] * stride_lm + (mp0 + rk[None, :]) * stride_lk
        R_BASE = RHS + (mp0 + rk[:, None]) * stride_rk + rn[None, :] * stride_rm
        sk0 = (m_start // VEC).to(tl.int32)
        LS_BASE = LHS_scale + (sk0 + rks[:, None]) * stride_lsk + rm[None, :] * stride_lsm
        RS_BASE = RHS_scale + (sk0 + rks[:, None]) * stride_rsk + rn[None, :] * stride_rsm

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        loop_k = M_g // BLOCK_SIZE_K  # padded -> no mask
        for _ in range(loop_k):
            l = tl.load(tl.multiple_of(L_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER)  # (BM, BK/2)
            r = tl.load(tl.multiple_of(R_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER)  # (BK/2, BN)
            ls = tl.trans(tl.load(LS_BASE))  # (BK/32, BM) -> (BM, BK/32)
            rs = tl.trans(tl.load(RS_BASE))
            acc = tl.dot_scaled(l, ls, "e2m1", r, rs, "e2m1", acc)
            L_BASE += BK_PACK * stride_lk
            R_BASE += BK_PACK * stride_rk
            LS_BASE += BK_SCALE * stride_lsk
            RS_BASE += BK_SCALE * stride_rsk

        c = acc.to(C.type.element_ty)
        rm_s = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn_s = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        cmask = (rm_s[:, None] < OUT_M) & (rn_s[None, :] < OUT_N)
        C_ = C + group_idx.to(tl.int64) * stride_cg + rm_s[:, None] * stride_cm + rn_s[None, :] * stride_cn
        tl.store(C_, c, cmask)


def grouped_gemm_mxfp4_variable_k_triton_kernel(
    lhs, lhs_scale, rhs, rhs_scale, go_pad, OUT_M, OUT_N, G, out_dtype=torch.bfloat16, num_cu=None
):
    """C[g] (OUT_M, OUT_N) = lhs[:,g] @ rhs[:,g]^T.

    lhs (OUT_M, M_total/2) fp4-packed, rhs (OUT_N, M_total/2) fp4-packed.
    go_pad: padded per-group offsets along M (each M_g a multiple of 128).
    """
    if is_gfx950():
        set_triton_knobs_gfx950()
    c = torch.empty((G, OUT_M, OUT_N), dtype=out_dtype, device=lhs.device)
    ls = lhs_scale.view(torch.uint8)
    rs = rhs_scale.view(torch.uint8)
    l_u8 = lhs.view(torch.uint8)
    r_u8 = rhs.view(torch.uint8)
    cu = num_cu if num_cu is not None else torch.cuda.get_device_properties(lhs.device).multi_processor_count
    BM, BN, BK = 256, 128, 128
    tiles_m = (OUT_M + BM - 1) // BM
    tiles_n = (OUT_N + BN - 1) // BN
    GM = 8 if min(tiles_m, tiles_n) < 16 else 4
    total_tiles = G * tiles_m * tiles_n
    num_sms = min(total_tiles, cu)
    chunk = 64 if num_sms >= NUM_XCDS * 64 else 32
    _grouped_mxfp4_variable_k_gemm_kernel[(num_sms,)](
        l_u8,
        r_u8,
        c,
        ls,
        rs,
        go_pad,
        G,
        OUT_M,
        OUT_N,
        l_u8.stride(0),
        l_u8.stride(1),
        r_u8.stride(0),
        r_u8.stride(1),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        ls.stride(0),  # stride_lsm (OUT_M); ls is (OUT_M, M/32)
        ls.stride(1),  # stride_lsk (M/32 contiguous)
        rs.stride(0),  # stride_rsm (OUT_N); rs is (OUT_N, M/32)
        rs.stride(1),  # stride_rsk (M/32 contiguous)
        BLOCK_SIZE_M=BM,
        BLOCK_SIZE_N=BN,
        BLOCK_SIZE_K=BK,
        GROUP_SIZE_M=GM,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=chunk,
        CACHE_MODIFIER=".ca",
        VEC=VEC_SIZE,
        num_warps=8,
        num_stages=3,
        waves_per_eu=2,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return c
