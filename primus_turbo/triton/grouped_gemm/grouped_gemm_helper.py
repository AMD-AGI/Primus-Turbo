###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared plumbing for the grouped-GEMM Triton persistent kernels.

The persistent grouped-GEMM kernels (plain BF16/FP16, the fused GLU-activation
epilogue variant, FP8, FP4) all wrap the same scaffolding around their
per-tile compute body:

  - map the AMD round-robin ``program_id`` onto XCD-contiguous chunks,
  - count how many output tiles the whole batch of groups needs,
  - resolve one global tile id to ``(group, pid_m, pid_n)``.

Only the compute body in between actually differs per kernel, so that
scaffolding lives here and every kernel module imports it. The same goes for
the post-pass that zeroes an over-allocated output's padding tail, which every
dtype's dispatcher runs and none of them implement differently.

Anything specific to one kernel stays in that kernel's module -- the fused GLU
epilogue's pair-tile addressing, register-tile surgery and activation math all
live in ``grouped_gemm_fp8_glu_kernel``, since it is their only consumer.

Contains:
  - NUM_XCDS                        -- MI300/MI350 chiplet count
  - _chiplet_transform_chunked      -- pid -> XCD-chunked pid
  - _get_gg_bf16_fwd_config         -- cached tile config for the BF16 forward
  - _count_group_tiles              -- total output tiles across all groups
  - _locate_group                   -- global tile id -> owning group (O(G) scan)
  - _gemm_tile_dot                  -- per-tile K-loop shared by the GEMM bodies
  - _swizzle_tile                   -- group-local tile id -> (pid_m, pid_n)
  - grouped_gemm_output_tail_kernel -- zero the uncovered padding rows

``grouped_gemm_kernel`` re-exposes ``NUM_XCDS`` and
``_chiplet_transform_chunked`` in its own namespace, so the FP8/FP4 modules
that import them from there keep working.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from primus_turbo.pytorch.core.utils import get_num_cus, is_gfx950
from primus_turbo.triton.utils.origami import origama_select_params

# ===============================================================================
# Hardware constants
# ===============================================================================

NUM_XCDS = 8


# ===============================================================================
# Cached config selection (avoids per-call origami / LDS overhead)
# ===============================================================================


@functools.lru_cache(maxsize=256)
def _get_gg_bf16_fwd_config(avg_m, N, K, dtype_a, dtype_b, trans_b, G, num_sms):
    """Cached kernel config for BF16 grouped GEMM forward."""
    if is_gfx950():
        is_tn = not trans_b
        BLOCK_M, BLOCK_N = 256, 256
        if is_tn:
            BLOCK_K, num_stages_val = 64, 2
        else:
            BLOCK_K, num_stages_val = 32, 3
        group_m = 4
        cache_a, cache_b = ".ca", ".ca"
        chunk_size = 32

        origami_params = origama_select_params(
            avg_m,
            N,
            K,
            dtype_a,
            dtype_a,
            dtype_b,
            trans_a=False,
            trans_b=trans_b,
        )
        if origami_params is not None:
            om, on, ok, ogm, oc_a, oc_b = origami_params
            if min(om, on) >= 128 and ok == BLOCK_K:
                BLOCK_M, BLOCK_N, group_m = om, on, ogm
                cache_a, cache_b = oc_a, oc_b
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
        group_m = 4
        num_stages_val = 2
        cache_a, cache_b = ".ca", ".ca"
        chunk_size = 64 if num_sms >= NUM_XCDS * 64 else 32

        origami_params = origama_select_params(
            avg_m,
            N,
            K,
            dtype_a,
            dtype_a,
            dtype_b,
            trans_a=False,
            trans_b=trans_b,
        )
        if origami_params is not None:
            om, on, ok, ogm, oc_a, oc_b = origami_params
            if ogm >= 2 and om * on >= 256 * 256:
                BLOCK_M, BLOCK_N, BLOCK_K, group_m, cache_a, cache_b = (om, on, ok, ogm, oc_a, oc_b)

    return BLOCK_M, BLOCK_N, BLOCK_K, group_m, cache_a, cache_b, num_stages_val, chunk_size


# ===============================================================================
# Chiplet transform
# ===============================================================================


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


# ===============================================================================
# Tile <-> group mapping
#
# The grouped GEMM's M extent is ragged (one variable-length slice of A per
# group) while N and K are shared, so the tile grid is a concatenation of
# per-group grids. Both the tile count and the tile -> group lookup are an
# O(G) scan over ``group_offs``; G is small (<=256) and the prefix sum stays
# resident in L2, so every CU can afford to redo the scan and no CPU sync or
# host-side tile table is needed.
#
# ``num_pid_n`` is passed in rather than derived, because it is not always
# ``cdiv(N, BLOCK_N)``: the fused GLU kernel tiles over gate/up *pairs*, so
# its N grid spans half the output width.
# ===============================================================================


@triton.jit
def _count_group_tiles(
    group_offs_ptr,
    G,
    num_pid_n,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Total number of output tiles across all G groups."""
    total_tiles: tl.int32 = 0
    for _g in range(G):
        m_g = (tl.load(group_offs_ptr + _g + 1) - tl.load(group_offs_ptr + _g)).to(tl.int32)
        total_tiles += tl.cdiv(m_g, BLOCK_SIZE_M) * num_pid_n
    return total_tiles


@triton.jit
def _locate_group(
    global_tile_id,
    group_offs_ptr,
    G,
    num_pid_n,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Resolve a global tile id to the group that owns it.

    Returns ``(group_idx, tile_start, total_tiles)``, where ``tile_start`` is
    the first global tile id belonging to ``group_idx`` (so the group-local id
    is ``global_tile_id - tile_start``) and ``total_tiles`` is the full count
    across all groups.

    Callers must treat ``global_tile_id >= total_tiles`` as out of range and
    skip the tile: past the end, ``group_idx`` has walked off group_offs and
    loading ``group_offs_ptr + group_idx + 1`` would be OOB.
    """
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
    return group_idx, tile_start, cumsum


@triton.jit
def _gemm_tile_dot(
    A,
    B,
    m_start_g,
    group_idx,
    rm,
    rn,
    K,
    stride_am,
    stride_bg,
    stride_bn,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """``A[m_start_g + rm, :] @ B[group_idx][:, rn]`` as an fp32 accumulator.

    Takes ``rm``/``rn`` already reduced modulo the group's M and the output N,
    so the caller owns the tile-to-group mapping and any epilogue masking.
    """
    rk = tl.arange(0, BLOCK_SIZE_K)
    # Cast group_idx to int64 to prevent overflow in B group offset
    group_offset_b = group_idx.to(tl.int64) * stride_bg

    A_BASE = A + m_start_g * stride_am + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_BASE = B + group_offset_b + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    loop_k = tl.cdiv(K, BLOCK_SIZE_K)
    if not EVEN_K:
        loop_k -= 1
    # ``tl.assume(loop_k > 1)`` would be a false assertion for shapes where
    # K <= BLOCK_SIZE_K (loop_k == 1 when EVEN_K, 0 when not). Relax to a
    # condition that always holds, so the compiler can still know the loop
    # count is non-negative without risking miscompilation on tiny K.
    tl.assume(loop_k >= 0)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
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
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

    return acc


@triton.jit
def _swizzle_tile(
    local_tile,
    tiles_m,
    tiles_n,
    GROUP_SIZE_M: tl.constexpr,
):
    """Map a group-local tile id to ``(pid_m, pid_n)``, M-major in bands of GROUP_SIZE_M.

    Walking GROUP_SIZE_M rows of tiles before advancing along N lets a band of
    programs share the same B columns, which is what keeps B resident in L2.
    ``tiles_m`` is clamped per band so a short final band stays rectangular.
    """
    num_pid_in_group = GROUP_SIZE_M * tiles_n
    swizzle_group = local_tile // num_pid_in_group
    first_pid_m = swizzle_group * GROUP_SIZE_M
    group_size_m = min(tiles_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((local_tile % num_pid_in_group) % group_size_m)
    pid_n = (local_tile % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    return pid_m, pid_n


# ===============================================================================
# Output padding tail
# ===============================================================================


@triton.jit
def _grouped_gemm_output_tail_kernel(
    C,
    group_offs_ptr,
    G,
    M_total,
    N,
    stride_cm,
    stride_cn,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    covered = tl.load(group_offs_ptr + G).to(tl.int64)
    m_total = tl.cast(M_total, tl.int64)
    num_pad = m_total - covered
    if num_pad <= 0:
        return

    num_m_blocks = tl.cdiv(num_pad, BLOCK_SIZE_M)
    num_n_blocks = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_m_blocks * num_n_blocks

    zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=C.dtype.element_ty)

    pid = tl.program_id(0)
    for tile in range(pid, total_tiles, NUM_SMS):
        bm = tile // num_n_blocks
        bn = tile % num_n_blocks
        rows = covered + bm * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        cols = bn * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = (rows[:, None] < m_total) & (cols[None, :] < N)
        C_ = C + rows[:, None] * stride_cm + cols[None, :] * stride_cn
        tl.store(C_, zeros, mask)


def grouped_gemm_output_tail_kernel(
    out: torch.Tensor,
    group_offs: torch.Tensor,
) -> torch.Tensor:
    """Zero the uncovered padding tail of a grouped GEMM output, in place.

    Clears rows ``[group_offs[-1], out.shape[0])`` -- the rows the persistent
    forward/dgrad kernel never writes when the output is over-allocated to
    padded token counts. CPU-sync-free (the covered boundary is read on device)
    and a no-op when there is no padding.

    Args:
        out: [M_total, N] grouped GEMM output.
        group_offs: [G+1] int64 prefix sum the kernel used to write ``out``.

    Returns:
        The same ``out`` tensor.
    """
    assert out.ndim == 2, f"expected 2D grouped GEMM output, got {tuple(out.shape)}"
    M_total, N = out.shape
    G = group_offs.shape[0] - 1
    if G <= 0 or M_total == 0 or N == 0:
        return out

    num_sms = get_num_cus()
    _grouped_gemm_output_tail_kernel[(num_sms,)](
        out,
        group_offs,
        G,
        M_total,
        N,
        out.stride(0),
        out.stride(1),
        NUM_SMS=num_sms,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=256,
        num_warps=4,
    )
    return out
