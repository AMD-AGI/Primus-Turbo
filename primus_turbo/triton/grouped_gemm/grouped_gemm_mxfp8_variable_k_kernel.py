###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Grouped MX-FP8 variable-K GEMM Triton kernel (wgrad, CPU-sync-free).

Computes per-group (one expert):

    dB[g][k, n] = sum_{m in g} A[m, k] * grad_out[m, n]

With MX-FP8 (e4m3 values + e8m0 scales @ K-group=32) where the MX reduction
axis is M. Maps to the native CDNA4 (gfx950) MFMA instruction
``v_mfma_scale_f32_32x32x64_f8f6f4`` via Triton's ``tl.dot_scaled``.

Contains:
  - _grouped_mxfp8_variable_k_kernel           — Persistent wgrad kernel
  - grouped_gemm_mxfp8_variable_k_triton_kernel — Public wrapper

Layouts:
  lhs        [M_total, K]          fp8 e4m3    (A col-quantised; reduction dim = M)
  rhs        [M_total, N]          fp8 e4m3    (grad_out col-quantised)
  lhs_scale  [K, M_total // 32]    uint8 e8m0  (scales grouped along M)
  rhs_scale  [N, M_total // 32]    uint8 e8m0  (scales grouped along M, N-first)
  out        [G, K, N]             bf16/fp16

Requirements:
  - K % 32 == 0 is irrelevant; K is the OUTPUT dim here (not reduction).
  - Each per-group segment length M_g must be a multiple of 32 so that no MX
    scale group ever spans two expert groups. In the target training shape
    (M_total = 65536 / G = 32 => M_g = 2048) this holds trivially.

Notes vs the forward kernel:
  - K (output row dim, originally hidden) and N (output col dim) are BOTH
    compile-time constants shared across groups. Only the reduction length M_g
    varies — group→tile mapping is a simple div/mod (similar to the existing
    bf16 variable-K kernel), not the O(G) linear scan the forward does.
  - The inner K-loop reduces along M; we use a masked tail iter for
    M_g % BLOCK_SIZE_K != 0 (BLOCK_SIZE_K here = the reduction-M block).
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


# MX-FP8 is defined at K-group size 32 (OCP spec, mandatory for tl.dot_scaled w/ e8m0).
MX_GROUP_SIZE = 32


# ═══════════════════════════════════════════════════════════════════════════════
# MX-FP8 wgrad Variable-K Kernel (persistent, CPU-sync-free)
#
# C[g][rk, rn] = sum_{m in g} LHS[m, rk] * RHS[m, rn]
#              = LHS[g]^T  @ RHS[g]   where rows are reduced over M.
#
# Naming in the kernel:
#   BLOCK_SIZE_M ≡ output-rows tile (along K, the output row dim)
#   BLOCK_SIZE_N ≡ output-cols tile (along N)
#   BLOCK_SIZE_K ≡ reduction-M tile (MX-group multiple, always >=32)
# ═══════════════════════════════════════════════════════════════════════════════


@triton.jit()
def _grouped_mxfp8_variable_k_kernel(
    # Pointers
    LHS,              # [M_total, K] fp8 e4m3   (A col-quantised)
    RHS,              # [M_total, N] fp8 e4m3   (grad_out col-quantised)
    C,                # [G, K, N]    bf16/fp16
    LHS_scale_ptr,    # [K, total_scale_cols] uint8 e8m0, jagged per-group
    RHS_scale_ptr,    # [N, total_scale_cols] uint8 e8m0, jagged per-group
    group_offs_ptr,   # [G+1] int64
    scale_offs_ptr,   # [G+1] int64 (jagged scale offsets per expert)
    # Dimensions
    G,                # number of groups (runtime; small constant really, but
                      #  we don't need it as constexpr — only used for total_tiles)
    OUT_M,            # = K (output-row dim, original hidden dim)
    OUT_N,            # = N (output-col dim, original interm dim)
    # Strides (lhs, rhs, C, scales)
    stride_lhs_m,     # LHS row stride along M_total
    stride_rhs_m,     # RHS row stride along M_total
    stride_cg,        # C group stride
    stride_cm,        # C row stride along OUT_M (=K)
    stride_cn,        # C col stride along OUT_N (=N)
    stride_ls_k,      # LHS_scale stride along K (outer / operand-row)
    stride_ls_m,      # LHS_scale stride along M//32 (inner / scale-group)
    stride_rs_n,      # RHS_scale stride along N (outer / operand-row)
    stride_rs_m,      # RHS_scale stride along M//32 (inner / scale-group)
    # Constexpr strides (mostly ==1 for row-major contiguous inputs)
    stride_lhs_n: tl.constexpr,   # LHS col stride (=1 for contiguous fp8)
    stride_rhs_n: tl.constexpr,   # RHS col stride (=1 for contiguous fp8)
    # Tile config
    BLOCK_SIZE_M: tl.constexpr,   # output-K tile
    BLOCK_SIZE_N: tl.constexpr,   # output-N tile
    BLOCK_SIZE_K: tl.constexpr,   # reduction-M tile (multiple of GROUP_K)
    GROUP_K: tl.constexpr,        # MX group size (32)
    GROUP_SIZE_M: tl.constexpr,   # swizzle size along K-tiles
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """Persistent grouped MX-FP8 variable-K wgrad kernel.

    Uses tl.dot_scaled to emit native v_mfma_scale_f32_32x32x64_f8f6f4 on gfx950.
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
    tl.assume(stride_ls_k > 0)
    tl.assume(stride_ls_m > 0)
    tl.assume(stride_rs_n > 0)
    tl.assume(stride_rs_m > 0)

    for global_tile in range(pid, total_tiles, NUM_SMS):
        # ── Map to (group, local_tile) via div/mod (shared OUT_M, OUT_N) ──
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

        # ── Group boundaries (in the reduction-M dimension) ──
        m_start = tl.load(group_offs_ptr + group_idx)   # int64
        M_g = (tl.load(group_offs_ptr + group_idx + 1) - m_start).to(tl.int32)

        # ── Output / operand-row indices ──
        # rm indexes along OUT_M (=K, the output rows). This is *also* the
        # operand-row of the LHS tile since LHS tile = A^T[rk=rm, rm=red].
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % OUT_M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % OUT_N
        rk = tl.arange(0, BLOCK_SIZE_K)                    # reduction-M within tile
        rks = tl.arange(0, BLOCK_SIZE_K // GROUP_K)        # scale-group indices within tile
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # ── Base pointers ──
        # LHS tile to feed tl.dot_scaled is A^T[rm, rk] = A[m_start+rk, rm].
        # Layout in LHS = [M_total, K]:  stride_lhs_m along M, stride_lhs_n along K.
        # So operand-row ↔ rm (goes along K axis of LHS), operand-col ↔ rk (goes along M axis).
        LHS_BASE = LHS + m_start * stride_lhs_m + rm[:, None] * stride_lhs_n + rk[None, :] * stride_lhs_m
        # RHS tile is grad_out[rk, rn] = RHS[m_start+rk, rn].
        RHS_BASE = RHS + m_start * stride_rhs_m + rk[:, None] * stride_rhs_m + rn[None, :] * stride_rhs_n

        # Scales: both are [outer, total_scale_cols] with outer == K (LHS) / N (RHS).
        # Jagged layout — per-expert scale base is loaded from scale_offs_ptr.
        # ms_start is the flat-scale-column index where this expert's scales begin.
        ms_start = tl.load(scale_offs_ptr + group_idx).to(tl.int32)
        LHS_S_BASE = LHS_scale_ptr + rm[:, None] * stride_ls_k + (ms_start + rks[None, :]) * stride_ls_m
        RHS_S_BASE = RHS_scale_ptr + rn[:, None] * stride_rs_n + (ms_start + rks[None, :]) * stride_rs_m

        # ── K-loop over M_g (variable per group) ──
        # Precondition: segment lengths are multiples of GROUP_K=32 (enforced
        # by the wrapper), so scale-group boundaries are always respected.
        # For the fp8/scale loads in the tail iter we mask out-of-range M elements.
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        loop_k = M_g // BLOCK_SIZE_K                 # full-tile iters
        tail_m = M_g - loop_k * BLOCK_SIZE_K         # residual (<BLOCK_SIZE_K, multiple of GROUP_K)
        tl.assume(loop_k >= 0)

        for k in range(loop_k):
            a = tl.load(LHS_BASE)
            b = tl.load(RHS_BASE)
            a_s = tl.load(LHS_S_BASE)
            b_s = tl.load(RHS_S_BASE)
            acc = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc=acc, fast_math=True)
            LHS_BASE += BLOCK_SIZE_K * stride_lhs_m
            RHS_BASE += BLOCK_SIZE_K * stride_rhs_m
            LHS_S_BASE += (BLOCK_SIZE_K // GROUP_K) * stride_ls_m
            RHS_S_BASE += (BLOCK_SIZE_K // GROUP_K) * stride_rs_m

        if tail_m > 0:
            mask_k = rk < tail_m
            # ceil-divide so a partial last scale group (M_g % 32 != 0) is
            # still loaded; padded-fp8 rows outside M_g are zero so the extra
            # scale slot contributes 0 regardless of its value.
            mask_ks = rks < ((tail_m + GROUP_K - 1) // GROUP_K)
            a = tl.load(LHS_BASE, mask=mask_k[None, :], other=0.0)
            b = tl.load(RHS_BASE, mask=mask_k[:, None], other=0.0)
            a_s = tl.load(LHS_S_BASE, mask=mask_ks[None, :], other=0)
            b_s = tl.load(RHS_S_BASE, mask=mask_ks[None, :], other=0)
            acc = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc=acc, fast_math=True)

        # ── Store ──
        c = acc.to(C.type.element_ty)
        rm_s = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn_s = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn_s = tl.max_contiguous(tl.multiple_of(rn_s % OUT_N, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm_s[:, None] < OUT_M) & (rn_s[None, :] < OUT_N)
        # Cast group_idx to int64 to prevent overflow in C group offset.
        C_ = C + group_idx.to(tl.int64) * stride_cg + rm_s[:, None] * stride_cm + rn_s[None, :] * stride_cn
        tl.store(C_, c, c_mask)


def grouped_gemm_mxfp8_variable_k_triton_kernel(
    lhs: torch.Tensor,        # [M_total, K]       fp8 e4m3
    rhs: torch.Tensor,        # [M_total, N]       fp8 e4m3
    lhs_scale: torch.Tensor,  # [K, total_scale_cols] uint8 e8m0 (jagged)
    rhs_scale: torch.Tensor,  # [N, total_scale_cols] uint8 e8m0 (jagged)
    group_offs: torch.Tensor, # [G+1] int64
    scale_offs: torch.Tensor, # [G+1] int64 — jagged scale-space prefix sums
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Persistent grouped MX-FP8 variable-K wgrad GEMM with jagged scale layout.

    Computes, for each expert group g:

        dB[g][k, n] = sum_{m=offs[g]}^{offs[g+1]-1} lhs[m, k] * rhs[m, n]

    Scales are stored per-group consecutively (jagged) so any ``M_g`` is
    supported — no padding of the fp8 operand required. Use
    ``quant_mxfp8_colwise_jagged`` to produce ``(lhs, lhs_scale, scale_offs)``.

    Args:
        lhs:        [M_total, K]            fp8 e4m3       (col-quantised A)
        rhs:        [M_total, N]            fp8 e4m3       (col-quantised grad_out)
        lhs_scale:  [K, total_scale_cols]   uint8 e8m0
        rhs_scale:  [N, total_scale_cols]   uint8 e8m0
        group_offs: [G+1]                   int64 prefix sum (M-space)
        scale_offs: [G+1]                   int64 prefix sum (scale-space)
        out_dtype:  Output dtype (bf16 or fp16).

    Returns:
        [G, K, N] output in ``out_dtype``.
    """
    assert lhs.ndim == 2, f"lhs must be 2D, got {lhs.shape}"
    assert rhs.ndim == 2, f"rhs must be 2D, got {rhs.shape}"
    assert lhs.shape[0] == rhs.shape[0], f"M mismatch: lhs={lhs.shape[0]} rhs={rhs.shape[0]}"
    assert lhs_scale.ndim == 2 and rhs_scale.ndim == 2, "scales must be 2D"
    assert lhs.dtype == torch.float8_e4m3fn, f"lhs must be fp8_e4m3fn, got {lhs.dtype}"
    assert rhs.dtype == torch.float8_e4m3fn, f"rhs must be fp8_e4m3fn, got {rhs.dtype}"

    K = lhs.shape[1]
    N = rhs.shape[1]
    G = group_offs.shape[0] - 1
    assert scale_offs.shape[0] == G + 1, (
        f"scale_offs must have shape [G+1]=[{G + 1}], got {tuple(scale_offs.shape)}"
    )
    assert lhs_scale.shape[0] == K, f"lhs_scale outer dim {lhs_scale.shape[0]} != K={K}"
    assert rhs_scale.shape[0] == N, f"rhs_scale outer dim {rhs_scale.shape[0]} != N={N}"
    assert lhs_scale.shape[1] == rhs_scale.shape[1], (
        f"scale inner dims must match: lhs={lhs_scale.shape[1]} rhs={rhs_scale.shape[1]}"
    )

    # Opt into the gfx950 AMD compiler knobs used by the rest of the Triton
    # FP8 family (use_async_copy, use_block_pingpong, scalarize_packed_fops).
    if _is_gfx950():
        _set_knobs_gfx950()

    out = torch.empty((G, K, N), device=lhs.device, dtype=out_dtype)
    num_sms = _get_num_cus()

    # Fixed config — mirror the forward kernel. BLK_M=256 (output K-dim),
    # BLK_N=256 (output N-dim), BLK_K=128 (reduction-M, 4 MX-groups per iter).
    # GROUP_M=4, num_warps=8, num_stages=2, matrix_instr_nonkdim=32 — all
    # required for the native v_mfma_scale_f32_32x32x64_f8f6f4 codegen path.
    blk_m = 256
    blk_n = 256
    blk_k = 128
    group_m = 4

    _grouped_mxfp8_variable_k_kernel[(num_sms,)](
        lhs, rhs, out,
        lhs_scale, rhs_scale,
        group_offs, scale_offs,
        G, K, N,
        lhs.stride(0),
        rhs.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
        lhs_scale.stride(0), lhs_scale.stride(1),
        rhs_scale.stride(0), rhs_scale.stride(1),
        stride_lhs_n=lhs.stride(1),
        stride_rhs_n=rhs.stride(1),
        BLOCK_SIZE_M=blk_m,
        BLOCK_SIZE_N=blk_n,
        BLOCK_SIZE_K=blk_k,
        GROUP_K=MX_GROUP_SIZE,
        GROUP_SIZE_M=group_m,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=32,
        num_warps=8,
        num_stages=2,
        waves_per_eu=0,
        matrix_instr_nonkdim=32,
    )
    return out
