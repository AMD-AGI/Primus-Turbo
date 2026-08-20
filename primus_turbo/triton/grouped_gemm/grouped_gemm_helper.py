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

The fused GLU kernels add a second, larger shared layer: ``grouped_gemm_glu_kernel``
(BF16/FP16) and ``grouped_gemm_fp8_glu_kernel`` (FP8 tensorwise) run the same
gate/up pair-tile addressing and the same activation math, and diverge only in
operand dtype handling, tile-config tuning, and the one dequantisation multiply
FP8 applies to the accumulator. Everything they agree on lives here.

Contains:
  - NUM_XCDS                        -- MI300/MI350 chiplet count
  - _chiplet_transform_chunked      -- pid -> XCD-chunked pid
  - _get_gg_bf16_fwd_config         -- cached tile config for the BF16 forward
  - _count_group_tiles              -- total output tiles across all groups
  - _locate_group                   -- global tile id -> owning group (O(G) scan)
  - _gemm_tile_dot                  -- per-tile K-loop shared by the GEMM bodies
  - _swizzle_tile                   -- group-local tile id -> (pid_m, pid_n)
  - SUPPORTED_ACTIVATIONS           -- GLU variants the fused epilogues accept
  - _glu_activation{,_probs,_grad}  -- constexpr-dispatched activation math
  - _pair_tile_locate / _pair_tile_cols / _pair_tile_dot
                                    -- gate/up pair-tile mapping and K-loop
  - _split_pair_cols / _chunk_rows  -- register-tile surgery for the epilogues
  - resolve_pair_tile_shape / resolve_probs_launch
                                    -- launch-side validation for the above
  - grouped_gemm_output_tail_kernel -- zero the uncovered padding rows

``grouped_gemm_kernel`` re-exposes ``NUM_XCDS`` and
``_chiplet_transform_chunked`` in its own namespace, so the FP8/FP4 modules
that import them from there keep working.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

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
# Fused GLU epilogue: activations
#
# The gated activations and their gradients, as pure register-tile math. Both
# the BF16/FP16 and the FP8 GLU kernel modules dequantise (or not) into fp32
# accumulators before reaching here, so these see the same inputs either way.
# ===============================================================================

# GLU-family activations the fused kernels can apply. The launchers validate
# against this tuple so ``_glu_activation`` never sees an unknown name.
SUPPORTED_ACTIVATIONS = ("swiglu",)


@triton.jit
def _silu_mul(gate, up):
    """``silu(gate) * up`` on fp32 accumulators.

    Every high-level spelling of ``x / (1 + exp(-x))`` -- ``tl.fdiv`` (with or
    without ``ieee_rounding``), ``tl.sigmoid``, plain ``/`` -- lowers to the
    same IEEE-exact divide: 70 VALU ops per element, dominated by
    v_div_scale/v_div_fmas/v_div_fixup. That is the same order as the tile's
    MFMA count, and it sits in the epilogue where (at BLOCK_M=256 the kernel
    runs 1 wave/SIMD) nothing hides it.

    Folding exp's log2e into exp2 and taking the raw hardware reciprocal with
    no Newton fixup costs 14 VALU ops instead. v_rcp_f32 is ~1 ulp, several
    orders below bf16's 8-bit mantissa, so the stored result is unchanged in
    practice. AMDGCN-only (inline asm).
    """
    d = 1.0 + tl.exp2(-gate * 1.4426950408889634)
    r = tl.inline_asm_elementwise("v_rcp_f32_e32 $0, $1", "=v,v", [d], dtype=tl.float32, is_pure=True, pack=1)
    return gate * r * up


@triton.jit
def _silu_mul_probs(gate, up, probs_row):
    """``silu(gate) * up * probs`` with ``probs_row`` [BLOCK_M] broadcast over columns."""
    d = 1.0 + tl.exp2(-gate * 1.4426950408889634)
    r = tl.inline_asm_elementwise("v_rcp_f32_e32 $0, $1", "=v,v", [d], dtype=tl.float32, is_pure=True, pack=1)
    return gate * r * up * probs_row[:, None]


@triton.jit
def _glu_activation(gate, up, ACTIVATION: tl.constexpr):
    """Apply the selected gate activation to a pair of fp32 register tiles.

    This and its two siblings -- :func:`_glu_activation_probs` and
    :func:`_glu_activation_grad` -- are the only places the epilogues name a
    concrete activation. ``ACTIVATION`` is a constexpr string, so the branch is
    folded at compile time and each variant gets its own specialisation, with
    no runtime dispatch in the epilogue. Only SwiGLU is implemented; anything
    else fails to compile rather than silently falling back, and the launchers
    reject it earlier still by asserting against ``SUPPORTED_ACTIVATIONS``.

    Adding a variant means a branch in all three, plus a name in
    ``SUPPORTED_ACTIVATIONS``. Note that the activation shares registers with a
    live BLOCK_M x BLOCK_N fp32 accumulator that already sits at the 512-VGPR
    ceiling: a GeGLU spelled with ``tl.erf`` crashes the AMDGPU backend from
    here ("Virtual register defs don't dominate all uses") while compiling fine
    in the standalone geglu kernel. So a new activation needs its own compile
    check, not just a numerical one.
    """
    if ACTIVATION == "swiglu":
        out = _silu_mul(gate, up)
    else:
        tl.static_assert(False, "ACTIVATION must be one of SUPPORTED_ACTIVATIONS")
        out = gate
    return out


@triton.jit
def _glu_activation_probs(gate, up, probs_row, ACTIVATION: tl.constexpr):
    """:func:`_glu_activation` with a per-row routing scale folded in.

    Separate from ``_glu_activation`` rather than an extra constexpr flag on it
    because the caller already branches on ``USE_PROBS`` to skip the load.
    """
    if ACTIVATION == "swiglu":
        out = _silu_mul_probs(gate, up, probs_row)
    else:
        tl.static_assert(False, "ACTIVATION must be one of SUPPORTED_ACTIVATIONS")
        out = gate
    return out


@triton.jit
def _dsilu_mul(gate, up, dout):
    """Gradients of ``silu(gate) * up`` w.r.t. both halves.

    Returns ``(dgate, dup)``. Same exp2 + raw-reciprocal sigmoid as
    :func:`_silu_mul`, for the same reason: the IEEE divide costs ~70 VALU ops
    per element in an epilogue that nothing hides.
    """
    d = 1.0 + tl.exp2(-gate * 1.4426950408889634)
    s = tl.inline_asm_elementwise("v_rcp_f32_e32 $0, $1", "=v,v", [d], dtype=tl.float32, is_pure=True, pack=1)
    silu = s * gate
    dup = dout * silu
    # s * (1 + gate * (1 - s)) rewritten as s * (1 + gate - silu): silu is
    # already live, so this drops a multiply per element.
    dgate = dout * up * s * (1.0 + gate - silu)
    return dgate, dup


@triton.jit
def _glu_activation_grad(gate, up, dout, ACTIVATION: tl.constexpr):
    """Backward counterpart of :func:`_glu_activation`; returns ``(dgate, dup)``.

    ``probs`` does not appear here: both backward epilogues fold it into
    ``dout`` beforehand, which is activation-agnostic.
    """
    if ACTIVATION == "swiglu":
        dgate, dup = _dsilu_mul(gate, up, dout)
    else:
        tl.static_assert(False, "ACTIVATION must be one of SUPPORTED_ACTIVATIONS")
        dgate, dup = gate, up
    return dgate, dup


# ===============================================================================
# Fused GLU epilogue: pair-tile addressing and register-tile surgery
#
# A gated activation needs columns ``j`` and ``j + I`` of the same row, which a
# plain contiguous BLOCK_N slice of the 2I-wide fc1 output splits across two
# programs. The fused kernels instead hand each program ``HALF = BLOCK_N // 2``
# gate/up *pairs*. Everything below implements that remapping and the register
# reshuffles the epilogues need to take the tile apart again; none of it
# depends on the operand dtype, so the BF16/FP16 and FP8 modules share it.
# ===============================================================================


@triton.jit
def _pair_tile_locate(
    global_tile_id,
    group_offs_ptr,
    G,
    num_pid_n,
    BLOCK_SIZE_M: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Resolve a pair-tile id to ``(in_range, m_start_g, M_g, pid_m, pid_n)``.

    Split out from the tile bodies so the out-of-range early return stays in
    the caller: wrapping the dot in a conditional instead would put the whole
    K-loop under a branch.
    """
    group_idx, tile_start, total_tiles = _locate_group(
        global_tile_id, group_offs_ptr, G, num_pid_n, BLOCK_SIZE_M
    )
    if global_tile_id >= total_tiles:
        return False, tl.cast(0, tl.int64), 0, 0, 0, 0

    local_tile = global_tile_id - tile_start
    m_start_g = tl.load(group_offs_ptr + group_idx)  # keep int64 to avoid address overflow
    M_g = (tl.load(group_offs_ptr + group_idx + 1) - tl.load(group_offs_ptr + group_idx)).to(tl.int32)
    pid_m, pid_n = _swizzle_tile(local_tile, tl.cdiv(M_g, BLOCK_SIZE_M), num_pid_n, GROUP_SIZE_M)
    return True, m_start_g, M_g, pid_m, pid_n, group_idx


@triton.jit
def _pair_tile_cols(
    pid_n,
    INTER_N,
    BLOCK_SIZE_N: tl.constexpr,
    N_ALIGNED: tl.constexpr,
    PAIR_CONTIG: tl.constexpr,
):
    """Column indices for one pair-tile: ``(rn_g, pair)`` over an ``I``-wide tensor.

    ``PAIR_CONTIG`` is ``gcd(HALF, I)``: the longest run the wrapped index is
    still guaranteed contiguous over, which is what the store vectoriser needs.
    When ``HALF`` divides ``I`` nothing wraps and it degenerates to ``HALF``.
    """
    HALF: tl.constexpr = BLOCK_SIZE_N // 2
    pair = pid_n * HALF + tl.arange(0, HALF)
    if N_ALIGNED:
        rn_g = tl.max_contiguous(tl.multiple_of(pair, HALF), HALF)
    else:
        rn_g = tl.max_contiguous(tl.multiple_of(pair % INTER_N, PAIR_CONTIG), PAIR_CONTIG)
    return rn_g, pair


@triton.jit
def _split_rows(t, BM: tl.constexpr, W: tl.constexpr):
    """Halve a (BM, W) tile into its top and bottom row blocks."""
    return tl.split(tl.permute(tl.reshape(t, (2, BM // 2, W)), (1, 2, 0)))


@triton.jit
def _chunk_rows(t, c: tl.constexpr, BM: tl.constexpr, W: tl.constexpr, C: tl.constexpr):
    """Chunk ``c`` of ``C`` contiguous row slices of a (BM, W) register tile."""
    if C == 1:
        out = t
    elif C == 2:
        lo, hi = _split_rows(t, BM, W)
        out = lo if c == 0 else hi
    elif C == 4:
        lo, hi = _split_rows(t, BM, W)
        half = lo if c < 2 else hi
        a, b = _split_rows(half, BM // 2, W)
        out = a if c % 2 == 0 else b
    else:
        tl.static_assert(C == 8, "EPI_CHUNKS must be 1, 2, 4 or 8")
        lo, hi = _split_rows(t, BM, W)
        half = lo if c < 4 else hi
        a, b = _split_rows(half, BM // 2, W)
        quarter = a if (c // 2) % 2 == 0 else b
        x, y = _split_rows(quarter, BM // 4, W)
        out = x if c % 2 == 0 else y
    return out


@triton.jit
def _split_pair_cols(t, BM: tl.constexpr, HALF: tl.constexpr):
    """Peel a (BM, 2*HALF) gate||up tile into its two (BM, HALF) halves.

    The permute is what makes the gate/up axis innermost so ``tl.split`` can
    take it, and it is not the shuffle it looks like -- two ways of removing it
    both measure worse than paying it:

    * A bare ``reshape(BM, HALF, 2)`` split, same shapes and no lane crossing
      (values are wrong, but it prices the permute): 0.93x vs 1.02x. The
      permuted layout is what lets the halves store as wide contiguous writes.
    * Pre-packing B with gate/up interleaved, so that reshape would be correct:
      0.91x. It scatters the B loads, which costs more than it saves. Blocked
      pre-packing keeps the permute and gains +1%, not worth a weight transform.

    Both on 131072 x 5760 x 2880.
    """
    return tl.split(tl.permute(tl.reshape(t, (BM, 2, HALF)), (0, 2, 1)))


@triton.jit
def _pair_tile_dot(
    A,
    B,
    m_start_g,
    M_g,
    group_idx,
    pid_m,
    pid_n,
    INTER_N,
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
    N_ALIGNED: tl.constexpr,
    PAIR_CONTIG: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Accumulate one pair-tile of fc1, still laid out as ``gate||up`` columns.

    Shared by the forward and backward epilogues -- they differ only in what
    they do with the accumulator, not in how the tile is addressed or summed.
    Returns ``(acc, rn_g, pair)``, where ``acc`` is (BLOCK_M, BLOCK_N) with the
    gate slice in ``[0, HALF)`` and the up slice in ``[HALF, BLOCK_N)``,
    ``rn_g`` indexes the tile's ``HALF`` columns within an ``I``-wide tensor,
    and ``pair`` is the unwrapped version used for bounds masks.

    The accumulator is fp32 and the operands are read as-is, so FP8 callers get
    a *quantised* sum back and must apply their dequantisation scale to it
    before the activation.

    The halves are peeled off by :func:`_split_pair_cols`. Callers decide when:
    splitting the full width makes both halves live at once on top of a full
    accumulator, so an epilogue that walks the tile in row chunks is better off
    splitting each chunk instead.
    """
    HALF: tl.constexpr = BLOCK_SIZE_N // 2

    # The tile owns gate columns [p, p+HALF) and up columns [I+p, I+p+HALF).
    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
    rn_g, pair = _pair_tile_cols(pid_n, INTER_N, BLOCK_SIZE_N, N_ALIGNED, PAIR_CONTIG)
    rk = tl.arange(0, BLOCK_SIZE_K)

    group_offset_b = group_idx.to(tl.int64) * stride_bg
    A_BASE = A + m_start_g * stride_am + rm[:, None] * stride_am + rk[None, :] * stride_ak

    loop_k = tl.cdiv(K, BLOCK_SIZE_K)
    if not EVEN_K:
        loop_k -= 1
    tl.assume(loop_k >= 0)

    # One full-width dot, same MFMA shape as the unfused kernel. rn holds the
    # gate slice in [0, HALF) and the up slice in [HALF, BLOCK_N), so each half
    # is a contiguous run of B columns.
    cols = tl.arange(0, BLOCK_SIZE_N)
    rn = pid_n * HALF + (cols % HALF)
    if not N_ALIGNED:
        rn = rn % INTER_N
    rn += (cols // HALF) * INTER_N
    B_BASE = B + group_offset_b + rk[:, None] * stride_bk + rn[None, :] * stride_bn

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
        a = tl.load(A_LAST, mask=rk_last[None, :] < K, other=0.0, cache_modifier=CACHE_MODIFIER_A)
        b = tl.load(B_LAST, mask=rk_last[:, None] < K, other=0.0, cache_modifier=CACHE_MODIFIER_B)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

    return acc, rn_g, pair


# ===============================================================================
# Fused GLU epilogue: launch-side resolution
#
# Shape validation and the optional-probs argument are identical across the
# dtypes; only the tile-config tuning differs, and that stays in each kernel
# module.
# ===============================================================================


class PairTileShape(NamedTuple):
    """Shapes and strides a pair-tile launch derives from its fc1 operands."""

    M_total: int
    G: int
    K: int
    inter_n: int
    stride_bg: int
    stride_bn: int
    stride_ak: int
    stride_bk: int


class PairTileLaunch(NamedTuple):
    """Everything the pair-tile kernels need from shape + config resolution."""

    inter_n: int
    K: int
    num_sms: int
    stride_bg: int
    stride_bn: int
    stride_ak: int
    stride_bk: int
    even_k: bool
    n_aligned: bool
    pair_contig: int
    out_dtype: torch.dtype
    grid: dict
    knobs: dict
    cache_a: str
    cache_b: str
    cache_out: str
    cache_l1: str
    epi_chunks: int


def resolve_pair_tile_shape(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_b: bool,
    activation: str,
) -> PairTileShape:
    """Validate the fc1 operand shapes for a pair-tile launch.

    Dtype validation is deliberately absent: the BF16/FP16 and FP8 modules
    accept disjoint operand dtypes and check their own.
    """
    assert a.ndim == 2, f"a must be 2D, got {a.shape}"
    assert b.ndim == 3, f"b must be 3D, got {b.shape}"
    assert activation in SUPPORTED_ACTIVATIONS, (
        f"Unsupported activation: {activation!r}, expected one of {SUPPORTED_ACTIVATIONS}"
    )

    M_total, K_a = a.shape
    G = b.shape[0]
    if trans_b:
        N, K_b = b.shape[1], b.shape[2]
        stride_bk, stride_bn = b.stride(2), b.stride(1)
    else:
        K_b, N = b.shape[1], b.shape[2]
        stride_bk, stride_bn = b.stride(1), b.stride(2)

    assert K_a == K_b, f"K mismatch: a has K={K_a}, b has K={K_b}"
    assert N % 2 == 0, f"fc1 output width must be even (gate||up), got {N}"

    return PairTileShape(
        M_total=M_total,
        G=G,
        K=K_a,
        inter_n=N // 2,
        stride_bg=b.stride(0),
        stride_bn=stride_bn,
        stride_ak=a.stride(1),
        stride_bk=stride_bk,
    )


def resolve_probs_launch(probs: torch.Tensor | None, m_total: int, device: torch.device):
    """Return (probs_ptr, stride_probs, use_probs) for optional routing probs."""
    if probs is None:
        return torch.empty(1, device=device, dtype=torch.float32), 0, False
    assert probs.ndim == 1 and probs.shape[0] == m_total, (
        f"probs must be [{m_total}], got {tuple(probs.shape)}"
    )
    assert probs.dtype == torch.float32, f"probs must be float32, got {probs.dtype}"
    return probs, probs.stride(0), True


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
