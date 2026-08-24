###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Grouped GEMM Triton persistent kernels with a fused GLU-activation epilogue -- FP8 tensorwise.

The operands are FP8 and the fp32 accumulator is dequantised by the per-tensor
scale product before the epilogue sees it::

    l1[g] = (a[g] @ B_view[g]) * a_scale * b_scale        # [M_g, 2I], gate||up
    act[g] = f(l1[g][:, :I]) * l1[g][:, I:] * probs[g]    # [M_g, I]

Only SwiGLU (``f = silu``) is implemented; see ``_glu_activation`` below.

Two kernels: the fc1 forward above, and the backward that hangs the activation
gradient off fc2's dgrad. The forward always writes ``l1``, so there is no
variant that recomputes fc1 inside the gradient epilogue.

The generic grouped-GEMM scaffolding -- the pid/XCD remapping, the tile to group
lookup, the plain per-tile K-loop -- comes from ``grouped_gemm_helper``, and the
activation math from ``primus_turbo.triton.utils.silu``. The pair-tile
addressing and register-tile surgery this epilogue needs are local to this
module.

``act`` and ``l1`` are written as BF16/FP16, not FP8: re-quantising here would
need a per-tensor amax that only exists after every tile has run, so the FP8
cast for the fc2 GEMM stays a separate pass.

The scale is applied to the whole accumulator right after the K-loop -- a
per-tensor scale cannot fold into it -- so everything downstream is defined on
real values rather than the quantised sum.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from primus_turbo.pytorch.core.utils import get_num_cus
from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import (
    _get_gg_fp8_tw_fwd_config,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_helper import (
    NUM_XCDS,
    _chiplet_transform_chunked,
    _count_group_tiles,
    _gemm_tile_dot,
    _locate_group,
    _swizzle_tile,
)
from primus_turbo.triton.utils.silu import silu_mul_bwd_act, silu_mul_probs

# Operand dtypes these kernels accept. Both the OCP (gfx950) and FNUZ (gfx942)
# spellings are listed because the encoding is picked per architecture by
# ``primus_turbo.pytorch.core.low_precision``; the kernels themselves only ever
# see the raw bits and the dequantisation scale.
_SUPPORTED_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
)

# The activation and pre-activation this module writes. FP8 is deliberately
# absent: see the module docstring.
_SUPPORTED_OUT_DTYPES = (torch.bfloat16, torch.float16)


def _check_fp8_operands(a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.dtype in _SUPPORTED_FP8_DTYPES, f"a must be FP8, got {a.dtype}"
    assert b.dtype in _SUPPORTED_FP8_DTYPES, f"b must be FP8, got {b.dtype}"


def _check_scales(a_scale: torch.Tensor, b_scale: torch.Tensor) -> None:
    """Per-tensor scales: one fp32 element each, on device."""
    for name, s in (("a_scale", a_scale), ("b_scale", b_scale)):
        assert s.numel() == 1, f"{name} must be a scalar tensor (tensorwise scaling), got {tuple(s.shape)}"
        assert s.dtype == torch.float32, f"{name} must be float32, got {s.dtype}"


def _check_out_dtype(out_dtype: torch.dtype) -> None:
    assert out_dtype in _SUPPORTED_OUT_DTYPES, (
        f"FP8 GLU writes a high-precision activation; out_dtype must be one of "
        f"{_SUPPORTED_OUT_DTYPES}, got {out_dtype}"
    )


# ===============================================================================
# Activation dispatch
#
# The activation math itself lives in ``primus_turbo.triton.utils.silu``; these
# two wrappers are the only places the epilogues name a concrete activation.
# ===============================================================================

# GLU-family activations the fused kernels can apply. The launchers validate
# against this tuple so ``_glu_activation`` never sees an unknown name.
SUPPORTED_ACTIVATIONS = ("silu",)


@triton.jit
def _glu_activation(gate, up, probs_row, ACTIVATION: tl.constexpr):
    """Gate activation on a pair of fp32 register tiles, scaled by ``probs_row``.

    ``ACTIVATION`` is a constexpr string, so the branch folds at compile time
    and each variant gets its own specialisation. Only SwiGLU is implemented;
    anything else fails to compile rather than silently falling back.

    Adding a variant means a branch here, one in :func:`_glu_activation_grad`,
    and a name in ``SUPPORTED_ACTIVATIONS``. Note that the activation shares
    registers with a live BLOCK_M x BLOCK_N fp32 accumulator already at the VGPR
    ceiling: a GeGLU spelled with ``tl.erf`` crashes the AMDGPU backend from here
    ("Virtual register defs don't dominate all uses") while compiling fine in the
    standalone geglu kernel. A new activation needs its own compile check, not
    just a numerical one.
    """
    if ACTIVATION == "silu":
        out = silu_mul_probs(gate, up, probs_row)
    else:
        tl.static_assert(False, "ACTIVATION must be one of SUPPORTED_ACTIVATIONS")
        out = gate
    return out


@triton.jit
def _glu_activation_grad(gate, up, dout, ACTIVATION: tl.constexpr):
    """Backward counterpart of :func:`_glu_activation`.

    Returns ``(dgate, dup, dout_act)``, the last being the per-element grad_probs
    term. ``probs`` does not appear: ``dout`` has to arrive *unscaled* for that
    term to be the routing gradient, so the caller scales the halves afterwards.
    """
    if ACTIVATION == "silu":
        dgate, dup, dout_act = silu_mul_bwd_act(gate, up, dout)
    else:
        tl.static_assert(False, "ACTIVATION must be one of SUPPORTED_ACTIVATIONS")
        dgate, dup, dout_act = gate, up, dout
    return dgate, dup, dout_act


# ===============================================================================
# Pair-tile addressing and register-tile surgery
#
# A gated activation needs columns ``j`` and ``j + I`` of the same row, which a
# plain contiguous BLOCK_N slice of the 2I-wide fc1 output splits across two
# programs. The forward instead hands each program ``HALF = BLOCK_N // 2``
# gate/up *pairs*; what follows implements that remapping and the register
# reshuffles the epilogues need to take the tile apart again.
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
    """Resolve a pair-tile id to ``(in_range, m_start_g, M_g, pid_m, pid_n, group_idx)``.

    Split out from the tile body so the out-of-range early return stays in the
    caller: wrapping the dot in a conditional instead would put the whole
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
    """Chunk ``c`` of ``C`` contiguous row slices of a (BM, W) register tile.

    Not free: ``_split_rows`` permutes, and a permute crosses lanes through LDS
    behind a barrier. Callers that only need the chunk at output precision
    should round *before* chunking rather than after -- half the dtype is half
    the LDS traffic, and for both epilogues here it also matches the unfused
    path bit for bit, since that one round-trips the same tile through memory
    in the stored dtype.
    """
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
    elif C == 8:
        lo, hi = _split_rows(t, BM, W)
        half = lo if c < 4 else hi
        a, b = _split_rows(half, BM // 2, W)
        quarter = a if (c // 2) % 2 == 0 else b
        x, y = _split_rows(quarter, BM // 4, W)
        out = x if c % 2 == 0 else y
    else:
        tl.static_assert(C == 16, "EPI_CHUNKS must be 1, 2, 4, 8 or 16")
        lo, hi = _split_rows(t, BM, W)
        half = lo if c < 8 else hi
        a, b = _split_rows(half, BM // 2, W)
        quarter = a if (c // 4) % 2 == 0 else b
        x, y = _split_rows(quarter, BM // 4, W)
        eighth = x if (c // 2) % 2 == 0 else y
        p, q = _split_rows(eighth, BM // 8, W)
        out = p if c % 2 == 0 else q
    return out


@triton.jit
def _split_pair_cols(t, BM: tl.constexpr, HALF: tl.constexpr):
    """Peel a (BM, 2*HALF) gate||up tile into its two (BM, HALF) halves.

    The permute is what makes the gate/up axis innermost so ``tl.split`` can
    take it, and it is the epilogue's single largest cost: ``silu(gate) * up``
    needs columns ``c`` and ``c + HALF`` in one lane, and a 2*HALF-wide MFMA
    result puts them in different ones. Three ways of removing it -- a bare
    ``reshape(BM, HALF, 2)`` split, pre-packing B with gate/up interleaved, and
    two HALF-wide dots into separate accumulators -- all measured worse than
    paying it, the last one badly (two B operands live at once through the
    k-loop). The permuted layout is also what lets the halves store as wide
    contiguous writes.
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

    Returns ``(acc, rn_g, pair)``, where ``acc`` is (BLOCK_M, BLOCK_N) with the
    gate slice in ``[0, HALF)`` and the up slice in ``[HALF, BLOCK_N)``,
    ``rn_g`` indexes the tile's ``HALF`` columns within an ``I``-wide tensor,
    and ``pair`` is the unwrapped version used for bounds masks.

    The accumulator is fp32 and the operands are read as-is, so the caller gets
    a *quantised* sum back and must apply the dequantisation scale to it before
    the activation. The halves are peeled off by :func:`_split_pair_cols`.
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
    # Restate the contiguity the wrap above hides. Runs of ``PAIR_CONTIG`` stay
    # consecutive either way, but the runtime ``%`` loses that for the compiler
    # and every B load degrades into a per-element gather.
    rn = tl.max_contiguous(tl.multiple_of(rn, PAIR_CONTIG), PAIR_CONTIG)
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
# Launch-side resolution
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
    """Everything the pair-tile kernel needs from shape + config resolution."""

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

    Operand dtypes are checked separately by :func:`_check_fp8_operands`.
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


def _check_probs(probs: torch.Tensor, m_total: int) -> None:
    assert probs.ndim == 1 and probs.shape[0] == m_total, (
        f"probs must be [{m_total}], got {tuple(probs.shape)}"
    )
    assert probs.dtype == torch.float32, f"probs must be float32, got {probs.dtype}"


@triton.jit
def _process_grouped_gemm_fp8_glu_tile(
    global_tile_id,
    A,  # [M_total, K] FP8
    B,  # [G, ?, ?]  FP8
    ACT,  # activation out [M_total, I], BF16/FP16
    L1,  # fc1 pre-activation [M_total, 2I], gate||up, BF16/FP16
    PROBS,  # [M_total] fp32 routing probs
    group_offs_ptr,
    scale,  # fp32 a_scale * b_scale, already reduced to a scalar by the caller
    G,
    INTER_N,  # I -- half of the fc1 output width
    K,
    stride_am,
    stride_bg,
    stride_bn,
    stride_actm,
    stride_actn,
    stride_l1m,
    stride_l1n,
    stride_probs,
    num_pid_n,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    N_ALIGNED: tl.constexpr,
    PAIR_CONTIG: tl.constexpr,
    ACTIVATION: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    CACHE_MODIFIER_ACT: tl.constexpr,
    CACHE_MODIFIER_L1: tl.constexpr,
    EPI_CHUNKS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Compute one pair-tile of ``HALF = BLOCK_N // 2`` gate/up columns."""
    HALF: tl.constexpr = BLOCK_SIZE_N // 2

    in_range, m_start_g, M_g, pid_m, pid_n, group_idx = _pair_tile_locate(
        global_tile_id, group_offs_ptr, G, num_pid_n, BLOCK_SIZE_M, GROUP_SIZE_M
    )
    if not in_range:
        return

    acc, rn_g, pair = _pair_tile_dot(
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
        stride_ak=stride_ak,
        stride_bk=stride_bk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        EVEN_K=EVEN_K,
        N_ALIGNED=N_ALIGNED,
        PAIR_CONTIG=PAIR_CONTIG,
        CACHE_MODIFIER_A=CACHE_MODIFIER_A,
        CACHE_MODIFIER_B=CACHE_MODIFIER_B,
        ALLOW_TF32=ALLOW_TF32,
    )
    acc = acc * scale

    # -- Epilogue --
    # Row-chunked: the gate/up/act tiles sit on top of a live accumulator, so a
    # full-width epilogue spills once l1 is also written.
    if N_ALIGNED:
        pair_ok = tl.full((1, HALF), True, tl.int1)
    else:
        pair_ok = (pair < INTER_N)[None, :]

    l1_ty = L1.type.element_ty
    act_ty = ACT.type.element_ty
    R: tl.constexpr = BLOCK_SIZE_M // EPI_CHUNKS

    # Rounded before the split, for the reason _chunk_rows gives, and split once
    # for the whole tile rather than per chunk: the rounded halves hold whole.
    gate_all, up_all = _split_pair_cols(acc.to(l1_ty), BLOCK_SIZE_M, HALF)

    for c in tl.static_range(EPI_CHUNKS):
        gate_s = _chunk_rows(gate_all, c, BLOCK_SIZE_M, HALF, EPI_CHUNKS)
        up_s = _chunk_rows(up_all, c, BLOCK_SIZE_M, HALF, EPI_CHUNKS)
        gate_c, up_c = gate_s.to(tl.float32), up_s.to(tl.float32)
        raw_m = pid_m * BLOCK_SIZE_M + c * R + tl.arange(0, R)
        rm_c = tl.minimum(raw_m, M_g - 1)
        mask_c = (raw_m < M_g)[:, None] & pair_ok

        probs_c = tl.load(
            PROBS + (m_start_g + rm_c) * stride_probs,
            mask=raw_m < M_g,
            other=1.0,
            cache_modifier=".ca",
        ).to(tl.float32)
        act_c = _glu_activation(gate_c, up_c, probs_c, ACTIVATION)

        L1_ = L1 + (m_start_g + rm_c[:, None]) * stride_l1m
        tl.store(
            L1_ + rn_g[None, :] * stride_l1n,
            gate_s,
            mask_c,
            cache_modifier=CACHE_MODIFIER_L1,
        )
        tl.store(
            L1_ + (rn_g + INTER_N)[None, :] * stride_l1n,
            up_s,
            mask_c,
            cache_modifier=CACHE_MODIFIER_L1,
        )
        ACT_ = ACT + (m_start_g + rm_c[:, None]) * stride_actm + rn_g[None, :] * stride_actn
        tl.store(ACT_, act_c.to(act_ty), mask_c, cache_modifier=CACHE_MODIFIER_ACT)


@triton.jit()
def _grouped_fp8_glu_persistent_gemm_kernel(
    # Pointers
    A,  # [M_total, K] FP8
    B,  # [G, ?, ?]  -- (K, 2I) or (2I, K) depending on trans_b, FP8
    ACT,  # [M_total, I]
    L1,  # [M_total, 2I]
    PROBS,  # [M_total] fp32
    A_scale_ptr,  # fp32 scalar
    B_scale_ptr,  # fp32 scalar
    group_offs_ptr,  # [G+1] int64
    # Dimensions
    G,
    INTER_N,  # I
    K,
    # Strides
    stride_am,
    stride_bg,
    stride_bn,
    stride_actm,
    stride_actn,
    stride_l1m,
    stride_l1n,
    stride_probs,
    # Constexpr strides
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    # Tile config
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    N_ALIGNED: tl.constexpr,
    PAIR_CONTIG: tl.constexpr,
    ACTIVATION: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    CACHE_MODIFIER_ACT: tl.constexpr,
    CACHE_MODIFIER_L1: tl.constexpr,
    EPI_CHUNKS: tl.constexpr,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
):
    """Persistent FP8 grouped GEMM + fused GLU activation (CPU-sync-free).

    Tiles are counted over gate/up *pairs*, so ``num_pid_n`` spans I rather
    than the 2I output width. The epilogue writes both ``act`` [M, I] and the
    dequantised pre-activation ``l1`` [M, 2I] (gate||up, before the activation
    and probs), both in the caller's high-precision dtype.
    """
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_n = tl.cdiv(INTER_N, BLOCK_SIZE_N // 2)
    total_tiles = _count_group_tiles(group_offs_ptr, G, num_pid_n, BLOCK_SIZE_M)

    # Hoisted above the tile loop so the tile body sees a plain register.
    scale = tl.load(A_scale_ptr) * tl.load(B_scale_ptr)

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_actm > 0)
    tl.assume(stride_actn > 0)
    tl.assume(stride_l1m > 0)
    tl.assume(stride_l1n > 0)
    tl.assume(stride_probs > 0)

    for global_tile_id in range(pid, total_tiles, NUM_SMS):
        _process_grouped_gemm_fp8_glu_tile(
            global_tile_id,
            A,
            B,
            ACT,
            L1,
            PROBS,
            group_offs_ptr,
            scale,
            G,
            INTER_N,
            K,
            stride_am,
            stride_bg,
            stride_bn,
            stride_actm,
            stride_actn,
            stride_l1m,
            stride_l1n,
            stride_probs,
            num_pid_n,
            stride_ak=stride_ak,
            stride_bk=stride_bk,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            EVEN_K=EVEN_K,
            N_ALIGNED=N_ALIGNED,
            PAIR_CONTIG=PAIR_CONTIG,
            ACTIVATION=ACTIVATION,
            CACHE_MODIFIER_A=CACHE_MODIFIER_A,
            CACHE_MODIFIER_B=CACHE_MODIFIER_B,
            CACHE_MODIFIER_ACT=CACHE_MODIFIER_ACT,
            CACHE_MODIFIER_L1=CACHE_MODIFIER_L1,
            EPI_CHUNKS=EPI_CHUNKS,
            ALLOW_TF32=ALLOW_TF32,
        )


def _resolve_fp8_pair_tile_launch(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_b: bool,
    activation: str,
    out_dtype: torch.dtype,
    num_cu: int | None,
) -> PairTileLaunch:
    """Validate the fc1 operands and pick the tile config for an FP8 pair-tile launch."""
    _check_fp8_operands(a, b)
    _check_out_dtype(out_dtype)
    shape = resolve_pair_tile_shape(a, b, trans_b, activation)
    M_total, G, K, inter_n = shape.M_total, shape.G, shape.K, shape.inter_n
    N = 2 * inter_n

    device_num_cus = get_num_cus()
    num_sms = min(num_cu, device_num_cus) if num_cu is not None and num_cu > 0 else device_num_cus
    avg_m = max(M_total // max(G, 1), 256)
    BLOCK_M, BLOCK_N, BLOCK_K, group_m, cache_a, cache_b, num_stages_val, chunk_size, _grid_sms = (
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
            shape.stride_ak,
            shape.stride_bk,
        )
    )
    # A pair-tile of a given BLOCK_N covers half as many output columns as the
    # unfused GEMM's, but holds an fp32 accumulator of the same size, so the
    # register ceiling caps it at the same 256.
    if BLOCK_N > 256:
        BLOCK_N = 256
    # Three stages at BLOCK_K=128 overflow what the scheduler can hide, so the
    # config helper's occasional 3 is clamped rather than trusted.
    num_stages_val = min(num_stages_val, 2)
    num_warps_val = 8
    waves_per_eu_val, mfma_dim_val, kpack_val = 1, 32, 1
    # act feeds the very next GEMM, so it keeps the default policy; l1 is not
    # read again until the backward pass and only evicts the A/B tiles this
    # kernel re-reads, so it asks not to be kept. Five alternating trials at
    # I=5760 put the streaming hint ahead with no overlap (4.684-4.717 ms
    # against 4.729-4.757). The same hint on act measured no better than .cg.
    cache_out = ".cg"
    cache_l1 = ".cs"
    epi_chunks = 8
    # Halving the effective N width halves the pair-tile grid, so re-derive the
    # swizzle for the actual tile count rather than reusing the unfused one. A
    # pair-tile reads two B column strips ``inter_n`` apart instead of one
    # contiguous strip, so the shorter chunk keeps more of the tiles sharing
    # those strips resident together.
    est_tiles = -(-M_total // BLOCK_M) * -(-inter_n // (BLOCK_N // 2))
    group_m, chunk_size = (2, 16) if est_tiles >= 512 else (4, 64)
    # A pair-tile is BLOCK_N // 2 wide, and tl.dot needs at least 16 columns.
    while BLOCK_N > 32 and inter_n * 2 < BLOCK_N:
        BLOCK_N //= 2

    return PairTileLaunch(
        inter_n=inter_n,
        K=K,
        num_sms=num_sms,
        stride_bg=shape.stride_bg,
        stride_bn=shape.stride_bn,
        stride_ak=shape.stride_ak,
        stride_bk=shape.stride_bk,
        even_k=(K % BLOCK_K == 0),
        n_aligned=(inter_n % (BLOCK_N // 2) == 0),
        pair_contig=math.gcd(BLOCK_N // 2, inter_n),
        grid=dict(
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_K,
            GROUP_SIZE_M=group_m,
            CHUNK_SIZE=chunk_size,
        ),
        knobs=dict(
            num_warps=num_warps_val,
            num_stages=num_stages_val,
            waves_per_eu=waves_per_eu_val,
            matrix_instr_nonkdim=mfma_dim_val,
            kpack=kpack_val,
        ),
        cache_a=cache_a,
        cache_b=cache_b,
        cache_out=cache_out,
        cache_l1=cache_l1,
        epi_chunks=epi_chunks,
    )


def grouped_gemm_fp8_glu_triton_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    probs: torch.Tensor,
    group_offs: torch.Tensor,
    act_out: torch.Tensor,
    intermediate_out: torch.Tensor,
    trans_b: bool = False,
    *,
    activation: str = "silu",
    out_dtype: torch.dtype = torch.bfloat16,
    num_cu: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Persistent FP8 grouped GEMM with a fused GLU-family activation epilogue.

    Computes ``act[g] = f(gate) * up * probs`` where
    ``l1 = [gate|up] = (a[g] @ B_view[g]) * a_scale * b_scale``, in a single
    launch, both in ``out_dtype``. ``l1`` is written because backward's
    :func:`grouped_gemm_fp8_dglu_triton_kernel` needs both halves.

    Args:
        probs: [M_total] float32 routing probabilities. The epilogue scales the
            activation output -- but not ``l1`` -- by these, per row.
        act_out: [M_total, I] buffer receiving the activation.
        intermediate_out: [M_total, 2I] buffer receiving ``l1``. Both are the
            caller's to allocate: every slot is written, so neither needs
            initialising.
        num_cu: Cap the persistent grid at this many CUs. None uses every CU.

    Returns:
        ``(act_out, intermediate_out)``, for call-site convenience.
    """
    _check_scales(a_scale, b_scale)
    plan = _resolve_fp8_pair_tile_launch(a, b, trans_b, activation, out_dtype, num_cu)
    M_total = a.shape[0]
    assert act_out.shape == (M_total, plan.inter_n), (
        f"act_out must be [{M_total}, {plan.inter_n}], got {tuple(act_out.shape)}"
    )
    assert act_out.device == a.device and act_out.dtype == out_dtype
    assert intermediate_out.shape == (M_total, 2 * plan.inter_n), (
        f"intermediate_out must be [{M_total}, {2 * plan.inter_n}], got {tuple(intermediate_out.shape)}"
    )
    assert intermediate_out.device == a.device and intermediate_out.dtype == out_dtype
    _check_probs(probs, M_total)

    act, l1 = act_out, intermediate_out
    _grouped_fp8_glu_persistent_gemm_kernel[(plan.num_sms,)](
        a,
        b,
        act,
        l1,
        probs,
        a_scale,
        b_scale,
        group_offs,
        b.shape[0],
        plan.inter_n,
        plan.K,
        a.stride(0),
        plan.stride_bg,
        plan.stride_bn,
        act.stride(0),
        act.stride(1),
        l1.stride(0),
        l1.stride(1),
        probs.stride(0),
        stride_ak=plan.stride_ak,
        stride_bk=plan.stride_bk,
        NUM_SMS=plan.num_sms,
        NUM_XCDS=NUM_XCDS,
        EVEN_K=plan.even_k,
        N_ALIGNED=plan.n_aligned,
        PAIR_CONTIG=plan.pair_contig,
        ACTIVATION=activation,
        CACHE_MODIFIER_A=plan.cache_a,
        CACHE_MODIFIER_B=plan.cache_b,
        CACHE_MODIFIER_ACT=plan.cache_out,
        CACHE_MODIFIER_L1=plan.cache_l1,
        EPI_CHUNKS=plan.epi_chunks,
        **plan.grid,
        **plan.knobs,
    )
    return act, l1


# ===============================================================================
# fc2 dgrad + GLU-gradient epilogue -- Persistent Kernel (backward, CPU-sync-free)
#
# Computes: dl1[g] = dact_grad(l1[g], (a[g] @ B_view[g]) * a_scale * b_scale)
#
# The activation gradient hangs off the GEMM the backward has to run anyway --
# fc2's dgrad, ``dact = a @ W2^T`` -- and reads the pre-activation the forward
# saved, so ``dact`` never reaches HBM.
#
# Unlike the pair-tile kernel above, the GEMM here is an ordinary grouped GEMM
# over an I-wide output; only the epilogue knows about the gate/up split.
# ===============================================================================


@triton.jit
def _process_grouped_fp8_dglu_tile(
    global_tile_id,
    A,  # [M_total, K] FP8   incoming gradient wrt the fc2 output
    B,  # [G, ?, ?]    FP8   fc2 weights, already oriented so the dot yields dact
    L1,  # [M_total, 2N]      saved fc1 pre-activation, BF16/FP16
    DL1,  # [M_total, 2N]
    GRAD_PROBS_PARTIAL,  # [num_pid_n, M_total] fp32   this tile's slice of the grad_probs sum
    PROBS,  # [M_total] fp32 routing probs
    group_offs_ptr,
    scale,  # fp32 a_scale * b_scale
    G,
    N,  # I
    K,  # fc2 output width
    stride_am,
    stride_bg,
    stride_bn,
    stride_l1m,
    stride_l1n,
    stride_dl1m,
    stride_dl1n,
    stride_grad_probs_partial,
    stride_probs,
    num_pid_n,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    N_EXACT: tl.constexpr,
    ACTIVATION: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    CACHE_MODIFIER_L1: tl.constexpr,
    CACHE_MODIFIER_DL1: tl.constexpr,
    EPI_CHUNKS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """One I-wide dact tile, turned into both halves of dl1 before it leaves registers."""
    group_idx, tile_start, total_tiles = _locate_group(
        global_tile_id, group_offs_ptr, G, num_pid_n, BLOCK_SIZE_M
    )
    if global_tile_id >= total_tiles:
        return

    local_tile = global_tile_id - tile_start
    m_start_g = tl.load(group_offs_ptr + group_idx)  # keep int64 to avoid address overflow
    M_g = (tl.load(group_offs_ptr + group_idx + 1) - tl.load(group_offs_ptr + group_idx)).to(tl.int32)
    pid_m, pid_n = _swizzle_tile(local_tile, tl.cdiv(M_g, BLOCK_SIZE_M), num_pid_n, GROUP_SIZE_M)

    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
    raw_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rn = raw_n % N
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    acc = _gemm_tile_dot(
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
        stride_ak=stride_ak,
        stride_bk=stride_bk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        EVEN_K=EVEN_K,
        CACHE_MODIFIER_A=CACHE_MODIFIER_A,
        CACHE_MODIFIER_B=CACHE_MODIFIER_B,
        ALLOW_TF32=ALLOW_TF32,
    )
    # Dequantise only. probs cannot fold in here: ``dact * act`` is the routing
    # gradient term only if it comes off the gradient *before* probs, so the
    # accumulator has to reach the epilogue unscaled and probs goes on per chunk.
    acc = acc * scale

    # -- Epilogue --
    # Row-blocked: this epilogue needs *two* more input tiles (gate and up) on
    # top of a live accumulator, so a full-width version spills hard. Rows keep
    # every access BLOCK_N-wide.
    out_ty = DL1.type.element_ty
    R: tl.constexpr = BLOCK_SIZE_M // EPI_CHUNKS

    # Rounded before chunking, for the reason _chunk_rows gives.
    acc_r = acc.to(out_ty)

    for c in tl.static_range(EPI_CHUNKS):
        acc_c = _chunk_rows(acc_r, c, BLOCK_SIZE_M, BLOCK_SIZE_N, EPI_CHUNKS).to(tl.float32)

        raw_m = pid_m * BLOCK_SIZE_M + c * R + tl.arange(0, R)
        rm_c = tl.minimum(raw_m, M_g - 1)
        mask_c = (raw_m < M_g)[:, None]

        # gate at l1[:, n], up at l1[:, n + I]: two BLOCK_N-wide reads.
        L1_ = L1 + (m_start_g + rm_c[:, None]) * stride_l1m
        gate = tl.load(
            L1_ + rn[None, :] * stride_l1n,
            mask=mask_c,
            other=0.0,
            cache_modifier=CACHE_MODIFIER_L1,
        )
        up = tl.load(
            L1_ + (rn + N)[None, :] * stride_l1n,
            mask=mask_c,
            other=0.0,
            cache_modifier=CACHE_MODIFIER_L1,
        )

        dgate_c, dup_c, dact_act_c = _glu_activation_grad(
            gate.to(tl.float32), up.to(tl.float32), acc_c, ACTIVATION
        )
        # Reduced in registers, in fp32: beats folding a buffer already rounded
        # to bf16. Folding the whole BLOCK_N is cross-lane but still won -- a
        # sweep of partial widths from 8 up to BLOCK_N was flat to within 3%.
        #
        # ``rn`` wraps modulo N, so when N is not a multiple of BLOCK_N the last
        # tile revisits low columns. The dl1 stores below are idempotent about
        # that, but a sum would count those columns twice.
        if N_EXACT:
            grad_probs_c = tl.sum(dact_act_c, axis=1)
        else:
            grad_probs_c = tl.sum(tl.where((raw_n < N)[None, :], dact_act_c, 0.0), axis=1)
        tl.store(
            GRAD_PROBS_PARTIAL + pid_n.to(tl.int64) * stride_grad_probs_partial + m_start_g + rm_c,
            grad_probs_c,
            mask=raw_m < M_g,
        )

        # acc_c reached the gradient unscaled, so the halves take probs here.
        probs_c = tl.load(
            PROBS + (m_start_g + rm_c) * stride_probs,
            mask=raw_m < M_g,
            other=1.0,
        ).to(tl.float32)[:, None]
        dgate_c = dgate_c * probs_c
        dup_c = dup_c * probs_c

        DL1_ = DL1 + (m_start_g + rm_c[:, None]) * stride_dl1m
        tl.store(
            DL1_ + rn[None, :] * stride_dl1n,
            dgate_c.to(out_ty),
            mask_c,
            cache_modifier=CACHE_MODIFIER_DL1,
        )
        tl.store(
            DL1_ + (rn + N)[None, :] * stride_dl1n,
            dup_c.to(out_ty),
            mask_c,
            cache_modifier=CACHE_MODIFIER_DL1,
        )


@triton.jit()
def _grouped_fp8_dglu_persistent_gemm_kernel(
    A,
    B,
    L1,
    DL1,
    GRAD_PROBS_PARTIAL,
    PROBS,
    A_scale_ptr,
    B_scale_ptr,
    group_offs_ptr,
    G,
    N,
    K,
    stride_am,
    stride_bg,
    stride_bn,
    stride_l1m,
    stride_l1n,
    stride_dl1m,
    stride_dl1n,
    stride_grad_probs_partial,
    stride_probs,
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
    N_EXACT: tl.constexpr,
    ACTIVATION: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    CACHE_MODIFIER_L1: tl.constexpr,
    CACHE_MODIFIER_DL1: tl.constexpr,
    EPI_CHUNKS: tl.constexpr,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
):
    """Persistent FP8 fc2 dgrad with a fused activation-gradient epilogue."""
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = _count_group_tiles(group_offs_ptr, G, num_pid_n, BLOCK_SIZE_M)

    scale = tl.load(A_scale_ptr) * tl.load(B_scale_ptr)

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_l1m > 0)
    tl.assume(stride_l1n > 0)
    tl.assume(stride_dl1m > 0)
    tl.assume(stride_dl1n > 0)
    tl.assume(stride_probs > 0)
    tl.assume(stride_grad_probs_partial > 0)

    for global_tile_id in range(pid, total_tiles, NUM_SMS):
        _process_grouped_fp8_dglu_tile(
            global_tile_id,
            A,
            B,
            L1,
            DL1,
            GRAD_PROBS_PARTIAL,
            PROBS,
            group_offs_ptr,
            scale,
            G,
            N,
            K,
            stride_am,
            stride_bg,
            stride_bn,
            stride_l1m,
            stride_l1n,
            stride_dl1m,
            stride_dl1n,
            stride_grad_probs_partial,
            stride_probs,
            num_pid_n,
            stride_ak=stride_ak,
            stride_bk=stride_bk,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            EVEN_K=EVEN_K,
            N_EXACT=N_EXACT,
            ACTIVATION=ACTIVATION,
            CACHE_MODIFIER_A=CACHE_MODIFIER_A,
            CACHE_MODIFIER_B=CACHE_MODIFIER_B,
            CACHE_MODIFIER_L1=CACHE_MODIFIER_L1,
            CACHE_MODIFIER_DL1=CACHE_MODIFIER_DL1,
            EPI_CHUNKS=EPI_CHUNKS,
            ALLOW_TF32=ALLOW_TF32,
        )


class DgluLaunch(NamedTuple):
    """Everything the fc2-dgrad kernel needs from shape + config resolution."""

    M_total: int
    N: int
    K: int
    G: int
    num_sms: int
    num_pid_n: int
    stride_bn: int
    stride_bk: int
    even_k: bool
    n_exact: bool
    grid: dict
    knobs: dict
    cache_a: str
    cache_b: str
    cache_l1: str
    cache_dl1: str
    epi_chunks: int


def _resolve_fp8_dglu_launch(
    a: torch.Tensor,
    b: torch.Tensor,
    intermediate: torch.Tensor,
    trans_b: bool,
    activation: str,
    num_cu: int | None,
) -> DgluLaunch:
    """Validate the fc2-dgrad operands and pick the tile config.

    ``num_pid_n``, the number of grad_probs partials the epilogue leaves behind,
    follows from ``BLOCK_SIZE_N``. Resolving the tiling here rather than inline
    at the launch is what lets :func:`grouped_gemm_fp8_dglu_grad_probs_partial_spec` report
    a size the kernel is guaranteed to write.
    """
    assert a.ndim == 2, f"a must be 2D, got {a.shape}"
    assert b.ndim == 3, f"b must be 3D, got {b.shape}"
    assert intermediate.ndim == 2, f"intermediate must be 2D, got {intermediate.shape}"
    _check_fp8_operands(a, b)
    _check_out_dtype(intermediate.dtype)
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
    assert intermediate.shape == (M_total, 2 * N), (
        f"intermediate must be [{M_total}, {2 * N}], got {tuple(intermediate.shape)}"
    )
    K = K_a

    device_num_cus = get_num_cus()
    num_sms = min(num_cu, device_num_cus) if num_cu is not None and num_cu > 0 else device_num_cus
    avg_m = max(M_total // max(G, 1), 256)
    BLOCK_M, BLOCK_N, BLOCK_K, group_m, cache_a, cache_b, num_stages_val, chunk_size, _grid_sms = (
        _get_gg_fp8_tw_fwd_config(
            avg_m,
            N,
            K,
            intermediate.dtype,
            a.dtype,
            b.dtype,
            trans_b,
            G,
            num_sms,
            M_total,
            a.stride(1),
            stride_bk,
        )
    )
    # A sharp local optimum: the knob space was swept against the fused total and
    # falls off steeply both ways, for opposite reasons. A bigger tile or fewer
    # epilogue chunks blows the accumulator past the register budget and spills; a
    # smaller tile gives up the arithmetic intensity the GEMM half needs.
    num_warps_val, waves_per_eu_val, mfma_dim_val, kpack_val = 8, 1, 32, 1
    cache_l1 = ".cg"
    # dl1 is streamed once and never read again here, so it wants an evict-first
    # line rather than one that pushes out the GEMM operands, which are reused
    # across tiles. Worth 1.191 -> 1.156 ms of epilogue at I=5760. The load side
    # cannot match it: this Triton rejects ".cs"/".lu" on loads, and ".cv"
    # bypasses the cache outright and costs 40%.
    cache_dl1 = ".cs"
    # Three stages at BLOCK_K=128 overflow what the scheduler can hide. One stage
    # frees the 32 KB an epilogue prefetch would need, but losing the operand
    # double-buffer costs more than such a prefetch could recover.
    num_stages_val = min(num_stages_val, 2)
    # The config helper is tuned for a bare GEMM, which wants the deepest K stage
    # it can hold. This kernel carries a GLU-gradient epilogue on the same
    # accumulator, so the shallower stage pays for itself: half the LDS, spills
    # from 31 to 12, and faster at both production widths.
    BLOCK_K = 64
    if BLOCK_N > 256:
        BLOCK_N = 256
    if M_total >= 65536 and N >= 2048:
        BLOCK_M = 256
        BLOCK_N = 256
    # Narrower row chunks shrink the epilogue's live set, but past a point the
    # extra loop iterations cost more than the spills they save: 16 chunks reach
    # zero spills and still lose to 8 at 12 spills, so spill count is not the
    # thing to minimise. Four is worse than its spill count suggests, because the
    # wider chunk stops the epilogue's loads from covering each other's latency.
    epi_chunks = 8
    est_tiles = -(-M_total // BLOCK_M) * -(-N // BLOCK_N)
    group_m, chunk_size = (2, 16) if est_tiles >= 512 else (4, 64)

    return DgluLaunch(
        M_total=M_total,
        N=N,
        K=K,
        G=G,
        num_sms=num_sms,
        num_pid_n=-(-N // BLOCK_N),
        stride_bn=stride_bn,
        stride_bk=stride_bk,
        even_k=(K % BLOCK_K == 0),
        n_exact=(N % BLOCK_N == 0),
        grid=dict(
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_K,
            GROUP_SIZE_M=group_m,
            CHUNK_SIZE=chunk_size,
        ),
        knobs=dict(
            num_warps=num_warps_val,
            num_stages=num_stages_val,
            waves_per_eu=waves_per_eu_val,
            matrix_instr_nonkdim=mfma_dim_val,
            kpack=kpack_val,
        ),
        cache_a=cache_a,
        cache_b=cache_b,
        cache_l1=cache_l1,
        cache_dl1=cache_dl1,
        epi_chunks=epi_chunks,
    )


class GradProbsPartialSpec(NamedTuple):
    """The grad_probs partial buffer a fused dgrad expects from its caller."""

    shape: tuple[int, int]
    needs_zero: bool


def grouped_gemm_fp8_dglu_grad_probs_partial_spec(
    a: torch.Tensor,
    b: torch.Tensor,
    intermediate: torch.Tensor,
    trans_b: bool = False,
    *,
    activation: str = "silu",
    num_cu: int | None = None,
) -> GradProbsPartialSpec:
    """The ``grad_probs_partial`` buffer :func:`grouped_gemm_fp8_dglu_triton_kernel` expects.

    One partial per tile column; ``grad_probs_partial.sum(0)`` is the gradient
    wrt ``probs``. Allocating and folding stay with the caller, so this module
    owns neither -- pass the same arguments here as to the kernel.

    ``needs_zero`` is False: every row of every group is covered by a tile at
    each tile column, so the kernel writes each slot exactly once. Whether a
    slot is left untouched is a property of the epilogue, hence reported here
    rather than left for the caller to guess.

    Returns:
        ``shape`` is ``(num_tile_cols, M_total)``, float32, M innermost to keep
        the epilogue's partial writes coalesced.
    """
    plan = _resolve_fp8_dglu_launch(a, b, intermediate, trans_b, activation, num_cu)
    return GradProbsPartialSpec(shape=(plan.num_pid_n, plan.M_total), needs_zero=False)


def grouped_gemm_fp8_dglu_triton_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    intermediate: torch.Tensor,
    group_offs: torch.Tensor,
    probs: torch.Tensor,
    out: torch.Tensor,
    grad_probs_partial: torch.Tensor,
    trans_b: bool = False,
    *,
    activation: str = "silu",
    num_cu: int | None = None,
) -> torch.Tensor:
    """FP8 fc2 dgrad with the activation gradient fused into its epilogue.

    Computes ``dact[g] = (a[g] @ B_view[g]) * a_scale * b_scale`` -- the
    gradient wrt the probs-scaled fc2 input -- and consumes it in registers to
    produce ``dl1 = [f'(gate) * up * (dact * probs) | f(gate) * (dact * probs)]``,
    so ``dact`` never reaches HBM. ``b`` is the fc2 weight already oriented for
    the dgrad, i.e. callers pass the ``trans_b`` they used in the forward,
    flipped.

    ``intermediate`` is the ``l1`` that
    :func:`grouped_gemm_fp8_glu_triton_kernel` wrote, so it is always available
    and there is no recompute variant to choose between.

    The gradient wrt ``probs`` is produced here too, but only in partial form:
    its sum runs over the whole of I, which a single tile does not span, so each
    tile folds the columns it owns and writes one partial. The tiling is
    resolved rather than autotuned, so the partial count is known up front and
    no atomics are needed -- the result is bitwise reproducible.

    Args:
        intermediate: [M_total, 2I] saved fc1 pre-activation, gate in [:, :I],
            up in [:, I:].
        out: [M_total, 2I] buffer receiving ``dl1``, in ``intermediate``'s dtype.
        grad_probs_partial: float32 buffer receiving the grad_probs partials,
            sized by :func:`grouped_gemm_fp8_dglu_grad_probs_partial_spec`. Sum
            over dim 0 for the gradient wrt ``probs``.
        num_cu: Cap the persistent grid at this many CUs. None uses every CU.

    Returns:
        ``out``, for call-site convenience.
    """
    _check_scales(a_scale, b_scale)
    plan = _resolve_fp8_dglu_launch(a, b, intermediate, trans_b, activation, num_cu)
    M_total, N, K = plan.M_total, plan.N, plan.K
    _check_probs(probs, M_total)

    assert out.shape == (M_total, 2 * N), f"out must be [{M_total}, {2 * N}], got {tuple(out.shape)}"
    assert out.device == a.device and out.dtype == intermediate.dtype
    assert grad_probs_partial.shape == (plan.num_pid_n, M_total), (
        f"grad_probs_partial must be [{plan.num_pid_n}, {M_total}], got {tuple(grad_probs_partial.shape)}; "
        "size it with grouped_gemm_fp8_dglu_grad_probs_partial_spec"
    )
    assert grad_probs_partial.device == a.device and grad_probs_partial.dtype == torch.float32

    _grouped_fp8_dglu_persistent_gemm_kernel[(plan.num_sms,)](
        a,
        b,
        intermediate,
        out,
        grad_probs_partial,
        probs,
        a_scale,
        b_scale,
        group_offs,
        plan.G,
        N,
        K,
        a.stride(0),
        b.stride(0),
        plan.stride_bn,
        intermediate.stride(0),
        intermediate.stride(1),
        out.stride(0),
        out.stride(1),
        grad_probs_partial.stride(0),
        probs.stride(0),
        stride_ak=a.stride(1),
        stride_bk=plan.stride_bk,
        NUM_SMS=plan.num_sms,
        NUM_XCDS=NUM_XCDS,
        EVEN_K=plan.even_k,
        N_EXACT=plan.n_exact,
        ACTIVATION=activation,
        CACHE_MODIFIER_A=plan.cache_a,
        CACHE_MODIFIER_B=plan.cache_b,
        CACHE_MODIFIER_L1=plan.cache_l1,
        CACHE_MODIFIER_DL1=plan.cache_dl1,
        EPI_CHUNKS=plan.epi_chunks,
        **plan.grid,
        **plan.knobs,
    )
    return out
