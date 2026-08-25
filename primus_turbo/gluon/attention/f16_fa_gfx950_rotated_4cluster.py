###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Vendored from https://github.com/AMD-Triton/gluon-kernels
# Source path: kernels/cdna4/fa/f16_fa_gfx950_rotated_4cluster.py
# Source branch: bangtian/fa-fwd-gfx950-gluon-optimized
# Source commit: 05b349b545ef713cd0ba41a3d89ddf3e3eb6b2c3
#
# Port delta: retained the measured non-persistent/short-causal schedules and
# ordinary AMDGPU function-attribute autotune candidates; removed the standalone
# harness, ABI-locked compiler-plugin loader, and fixed-SM launcher; added the
# validated raw output/LSE boundary, explicit-layout Q scale splats, and the
# non-pipelined short-causal compiler-capability fallback.
###############################################################################

"""Production CDNA4 (gfx950) Flash Attention forward kernel."""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._layouts import (
    DistributedLinearLayout,
    DotOperandLayout,
    PaddedSharedLayout,
)
from triton.experimental.gluon.language.amd import AMDMFMALayout, warp_pipeline_stage
from triton.experimental.gluon.language.amd import slice as amd_slice
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async
from triton.language.core import _aggregate as aggregate

from .f16_fa_gfx950_common import (
    MetaData,
    attn_fwd_inner,
    compute_dot1_qk,
    compute_softmax,
    do_mma,
    get_mma_type_for_arch,
    get_shape_from_layout,
    get_strides_from_layout,
    issue_async_load_k,
    issue_async_load_v,
    nan_propagating_max,
    remap_xcd,
)

# Keep formatting disabled below so the performance-sensitive vendored bodies
# remain byte-attributable to the pinned source identified above.
# fmt: off

_HAS_WARP_PREDICATE = hasattr(gl, "warp_predicate")
HAS_WARP_PREDICATE = tl.constexpr(_HAS_WARP_PREDICATE)
LAZY_RESCALE_THRESHOLD = tl.constexpr(8.0)
DIAGONAL_LAZY_RESCALE_THRESHOLD = tl.constexpr(4.0)
DIAGONAL_LAZY_RESCALE_THRESHOLD_FP16 = tl.constexpr(8.0)
# ---------------------------------------------------------------------------
# Logical sub-cluster primitives for the matched rotated 4-cluster loop.
#
# The hot loop is composed from eight named logical sub-clusters:
#
#   dot_qk (DOT1) -- Q * K^T MFMA -> qk scores
#   dot_pv (DOT2) -- P * V   MFMA -> acc
#   VEC1   -- softmax numerator          (new row-max + exp2 burst -> p, alpha)
#   VEC2   -- softmax denominator + acc  (sum p, acc rescale, l_i, p->fp16 cast)
#   LRK    -- local-read  K  (LDS -> regs)
#   LRV    -- local-read  V  (LDS -> regs)
#   ACK    -- async-copy  K  (global -> LDS)
#   ACV    -- async-copy  V  (global -> LDS)
#
# ---------------------------------------------------------------------------

@gluon.jit
def sc_vec1(qk, m_run, start_n, start_m,
            qk_scale: gl.constexpr,
            MASK_STEPS: gl.constexpr, IS_CAUSAL: gl.constexpr,
            MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
            BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
            BALANCE_CAUSAL_WAVES: gl.constexpr,
            mma_layout: gl.constexpr,
            mma_offs_n_col: gl.constexpr, mma_offs_m_row: gl.constexpr):
    """VEC1: softmax numerator -- (mask +) new row-max + exp2 burst (DOT2 cluster).

    From the qk scores produced by DOT1 this iteration, computes the new running
    max m_new = max(m_run, rowmax(qk)*scale), the unnormalized probabilities
    p = exp2(qk*scale - m_new), and the rescale factor alpha = exp2(m_run - m_new).
    p and alpha are carried to the next iteration (consumed by VEC2 and DOT2).
    This is the expensive transcendental group, paired with the P*V MFMA so the
    exp throughput overlaps the matrix engine.

    When MASK_STEPS the scores are scaled and masked (causal + K-bound) before
    the max/exp2 (mirroring ``compute_softmax``). The unmasked branch keeps the
    FMA-friendly form (scale folded into the max and exp2 inputs); MASK_STEPS is
    constexpr so that branch is identical to the unmasked-only schedule after DCE.
    """
    if MASK_STEPS:
        POSITIVE_SCALE: gl.constexpr = qk_scale > 0.0
        qk_sm = qk if POSITIVE_SCALE else qk * qk_scale
        if IS_CAUSAL:
            causal_offs_n = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
            local_m = gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
            if BALANCE_CAUSAL_WAVES:
                wave_m = local_m // 32
                wave_m = wave_m ^ ((wave_m // 4) * 3)
                local_m = wave_m * 32 + local_m % 32
            causal_offs_m = start_m * BLOCK_M + local_m
            causal_boundary = causal_offs_n[None, :] + MAX_SEQLENS_Q - MAX_SEQLENS_K
            causal_mask = causal_offs_m[:, None] >= causal_boundary
            qk_sm = gl.where(causal_mask, qk_sm, gl.full([BLOCK_M, BLOCK_N], float("-inf"),
                                                         dtype=gl.float32, layout=mma_layout))
        CHECK_K_BOUNDS: gl.constexpr = MAX_SEQLENS_K % BLOCK_N != 0
        if CHECK_K_BOUNDS:
            bound_offs = start_n + gl.arange(
                0, BLOCK_N, layout=mma_offs_n_col)
            bound_mask = bound_offs[None, :] < MAX_SEQLENS_K
            qk_sm = gl.where(
                bound_mask, qk_sm,
                gl.full([BLOCK_M, BLOCK_N], float("-inf"),
                        dtype=gl.float32, layout=mma_layout))
        m_ij = nan_propagating_max(qk_sm, axis=1)
        if POSITIVE_SCALE:
            m_ij = m_ij * qk_scale
        m_new = gl.maximum(m_run, m_ij, propagate_nan=tl.PropagateNan.ALL)
        if POSITIVE_SCALE:
            p = gl.exp2(gl.fma(qk_sm, qk_scale, -m_new[:, None]))
        else:
            p = gl.exp2(qk_sm - m_new[:, None])
        alpha = gl.exp2(m_run - m_new)
    else:
        m_ij = nan_propagating_max(qk, axis=1) * qk_scale
        m_new = gl.maximum(m_run, m_ij, propagate_nan=tl.PropagateNan.ALL)
        p = gl.exp2(qk * qk_scale - m_new[:, None])
        alpha = gl.exp2(m_run - m_new)
    return m_new, p, alpha


@gluon.jit
def sc_vec2(acc, l_i, p, alpha, p_dot_layout: gl.constexpr,
            out_dtype: gl.constexpr, CAST_P_FIRST: gl.constexpr):
    """VEC2: softmax denominator + accumulator correction (DOT1 cluster).

    Updates the running denominator (l_i = l_i*alpha + sum p) with one cross-lane
    sum reduction, rescales the accumulator (acc *= alpha), and casts p to fp16
    with the layout convert that prepares the operand for the immediately-
    following DOT2. p and alpha were produced by VEC1 in the *previous* iteration
    (the carried previous-tile probabilities).

    CAST_P_FIRST selects between two equivalent dependency orders. The cast-first
    schedule increases MFMA/VALU overlap for D128 BM128/256 x BN64 full-attention
    tiles, while the reduction-first schedule avoids regressions for causal,
    D64, D256, and smaller-BM tiles.
    """
    if CAST_P_FIRST:
        p_dot = gl.convert_layout(p.to(out_dtype), p_dot_layout)
        l_ij = gl.sum(p, axis=1)
    else:
        l_ij = gl.sum(p, axis=1)
        p_dot = gl.convert_layout(p.to(out_dtype), p_dot_layout)
    acc = acc * alpha[:, None]
    l_i = l_i * alpha + l_ij
    return acc, l_i, p_dot


@gluon.jit
def sc_vec2_split_acc(acc0, acc1, l_i, p, alpha,
                      p_dot_layout: gl.constexpr, out_dtype: gl.constexpr):
    """VEC2 with the output accumulator kept as two persistent D subtiles.

    Present the accumulator correction first and the P conversion last. This
    gives the scheduler one contiguous rescale chain and keeps the conversion
    next to the following P*V consumer.
    """
    acc0 = acc0 * alpha[:, None]
    acc1 = acc1 * alpha[:, None]
    l_ij = gl.sum(p, axis=1)
    l_i = l_i * alpha + l_ij
    p_dot = gl.convert_layout(p.to(out_dtype), p_dot_layout)
    return acc0, acc1, l_i, p_dot


@gluon.jit
def sc_dot_pv(acc, p_dot, v_dot):
    """dot_pv: P @ V -> acc (p already cast in VEC2, V already in registers)."""
    return do_mma("mfma_cdna4", p_dot, v_dot, acc)


@gluon.jit
def sc_causal_wave_active(start_n, start_m,
                          MAX_SEQLENS_Q: gl.constexpr,
                          MAX_SEQLENS_K: gl.constexpr,
                          BLOCK_M: gl.constexpr,
                          BALANCE_CAUSAL_WAVES: gl.constexpr,
                          mma_offs_m_row: gl.constexpr):
    """Wave-uniform predicate for causal score tiles with 32 M rows per wave.

    The MFMA layout assigns one contiguous 32-row band to every wave.  A wave
    whose last query row precedes the tile's first key column is wholly above
    the causal diagonal, so both of its MFMAs are provable no-ops.
    """
    local_m = gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
    if BALANCE_CAUSAL_WAVES:
        wave_m = local_m // 32
        wave_m = wave_m ^ ((wave_m // 4) * 3)
        local_m = wave_m * 32 + local_m % 32
    wave_m_last = start_m * BLOCK_M + (local_m // 32) * 32 + 31
    causal_n_first = start_n + MAX_SEQLENS_Q - MAX_SEQLENS_K
    return wave_m_last >= causal_n_first


@gluon.jit
def sc_split_cols(x):
    """Split a matrix into two zero-copy column subtiles."""
    layout: gl.constexpr = x.type.layout
    x0, x1 = x.reshape(
        [x.shape[0], 2, x.shape[1] // 2]).permute(0, 2, 1).split()
    x0 = gl.convert_layout(x0, layout, assert_trivial=True)
    x1 = gl.convert_layout(x1, layout, assert_trivial=True)
    return x0, x1


@gluon.jit
def sc_concat_cols(x0, x1):
    """Reassemble two column subtiles without moving data."""
    gl.static_assert(x0.type.layout == x1.type.layout)
    layout: gl.constexpr = x0.type.layout
    shape: gl.constexpr = [x0.shape[0], x0.shape[1] + x1.shape[1]]
    x = gl.join(x0, x1).permute(0, 2, 1).reshape(shape)
    return gl.convert_layout(x, layout, assert_trivial=True)


@gluon.jit
def sc_sum_rows_chain4(x, ROTATE_FINAL: gl.constexpr):
    """Reduce four zero-copy column slices through one dependency chain."""
    x_01, x_23 = sc_split_cols(x)
    x_0, x_1 = sc_split_cols(x_01)
    x_2, x_3 = sc_split_cols(x_23)
    partial = x_0 + x_1
    if ROTATE_FINAL:
        # Stage-four BF16 schedules place the high-half slice first. This keeps
        # the same three-add dependency chain while shortening the live range
        # that overlaps the following MFMA group.
        partial = partial + x_3
        partial = partial + x_2
    else:
        partial = partial + x_2
        partial = partial + x_3
    return gl.sum(partial, axis=1)


@gluon.jit
def sc_vec1_lazy(qk, m_run, qk_scale: gl.constexpr,
                 SCALE_ON_Q: gl.constexpr, VEC1_SPLIT: gl.constexpr):
    """Lazy VEC1 with five eighths of the exp2 work kept in DOT2.

    The running max advances only when the new tile exceeds its current frame
    by more than 8 in log2 units. Accumulator and denominator remain in that
    same frame, so their final ratio is unchanged while most rescale operations
    become skippable. Three raw score slices are carried to VEC2 to balance the
    two MFMA clusters.
    """
    m_new, score_delta, advance = sc_vec1_lazy_max(
        qk, m_run, qk_scale, SCALE_ON_Q)
    p_0123, carried_4, carried_5, carried_6, carried_7 = sc_vec1_lazy_exp(
        qk, m_new, qk_scale, SCALE_ON_Q, VEC1_SPLIT)
    # Carry the already-computed score delta and its update predicate. Stable
    # waves skip the correction; updating waves form exp2(-score_delta) inside
    # the predicated body without repeating either comparison or subtraction.
    return (m_new, p_0123, carried_4, carried_5, carried_6, carried_7,
            score_delta, advance)


@gluon.jit
def sc_vec1_lazy_max(qk, m_run, qk_scale: gl.constexpr,
                     SCALE_ON_Q: gl.constexpr):
    """Prepare the lazy softmax frame while retaining the score tile."""
    if SCALE_ON_Q:
        m_ij = nan_propagating_max(qk, axis=1)
    else:
        m_ij = nan_propagating_max(qk, axis=1) * qk_scale
    score_delta = m_ij - m_run
    advance = score_delta > LAZY_RESCALE_THRESHOLD
    m_new = gl.where(advance, m_ij, m_run)
    return m_new, score_delta, advance


@gluon.jit
def sc_vec1_lazy_exp(qk, m_new, qk_scale: gl.constexpr,
                     SCALE_ON_Q: gl.constexpr,
                     VEC1_SPLIT: gl.constexpr):
    """Finish VEC1 after an interleaved P-by-V reduction window."""
    qk_lo, qk_hi = sc_split_cols(qk)
    qk_a, qk_b = sc_split_cols(qk_lo)
    qk_c, qk_d = sc_split_cols(qk_hi)
    qk_0, qk_1 = sc_split_cols(qk_a)
    qk_2, qk_3 = sc_split_cols(qk_b)
    qk_4, qk_5 = sc_split_cols(qk_c)
    qk_6, qk_7 = sc_split_cols(qk_d)

    if SCALE_ON_Q:
        p_0 = gl.exp2(qk_0 - m_new[:, None])
        p_1 = gl.exp2(qk_1 - m_new[:, None])
        p_2 = gl.exp2(qk_2 - m_new[:, None])
        p_3 = gl.exp2(qk_3 - m_new[:, None])
        if VEC1_SPLIT == 5:
            p_4 = gl.exp2(qk_4 - m_new[:, None])
    else:
        p_0 = gl.exp2(gl.fma(qk_0, qk_scale, -m_new[:, None]))
        p_1 = gl.exp2(gl.fma(qk_1, qk_scale, -m_new[:, None]))
        p_2 = gl.exp2(gl.fma(qk_2, qk_scale, -m_new[:, None]))
        p_3 = gl.exp2(gl.fma(qk_3, qk_scale, -m_new[:, None]))
        if VEC1_SPLIT == 5:
            p_4 = gl.exp2(gl.fma(qk_4, qk_scale, -m_new[:, None]))

    p_01 = sc_concat_cols(p_0, p_1)
    p_23 = sc_concat_cols(p_2, p_3)
    p_0123 = sc_concat_cols(p_01, p_23)
    if VEC1_SPLIT == 5:
        carried_4 = p_4
        carried_5 = qk_5
        carried_6 = qk_6
        carried_7 = qk_7
    else:
        carried_4 = qk_4
        carried_5 = qk_5
        carried_6 = qk_6
        carried_7 = qk_7
    return p_0123, carried_4, carried_5, carried_6, carried_7


@gluon.jit
def sc_vec2_lazy(l_i, p_0123, p_4, qk_5, qk_6, qk_7, m_new,
                 p_dot_layout: gl.constexpr, out_dtype: gl.constexpr,
                 qk_scale: gl.constexpr, SCALE_ON_Q: gl.constexpr,
                 VEC1_SPLIT: gl.constexpr,
                 CHAIN_BF16_ROWSUM: gl.constexpr):
    """Finish the deferred three eighths of exp2 work in DOT1."""
    if VEC1_SPLIT == 5:
        if SCALE_ON_Q:
            p_5 = gl.exp2(qk_5 - m_new[:, None])
            p_6 = gl.exp2(qk_6 - m_new[:, None])
            p_7 = gl.exp2(qk_7 - m_new[:, None])
        else:
            p_5 = gl.exp2(gl.fma(qk_5, qk_scale, -m_new[:, None]))
            p_6 = gl.exp2(gl.fma(qk_6, qk_scale, -m_new[:, None]))
            p_7 = gl.exp2(gl.fma(qk_7, qk_scale, -m_new[:, None]))
        p_45 = sc_concat_cols(p_4, p_5)
        p_67 = sc_concat_cols(p_6, p_7)
        p = sc_concat_cols(p_0123, sc_concat_cols(p_45, p_67))
    else:
        if SCALE_ON_Q:
            p_4 = gl.exp2(p_4 - m_new[:, None])
            p_5 = gl.exp2(qk_5 - m_new[:, None])
            p_6 = gl.exp2(qk_6 - m_new[:, None])
            p_7 = gl.exp2(qk_7 - m_new[:, None])
        else:
            p_4 = gl.exp2(gl.fma(p_4, qk_scale, -m_new[:, None]))
            p_5 = gl.exp2(gl.fma(qk_5, qk_scale, -m_new[:, None]))
            p_6 = gl.exp2(gl.fma(qk_6, qk_scale, -m_new[:, None]))
            p_7 = gl.exp2(gl.fma(qk_7, qk_scale, -m_new[:, None]))
        p_45 = sc_concat_cols(p_4, p_5)
        p_67 = sc_concat_cols(p_6, p_7)
        p = sc_concat_cols(p_0123, sc_concat_cols(p_45, p_67))
    p_cast = p.to(out_dtype)
    # FP16 P is already the operand consumed by P*V. Reducing that same tensor
    # makes the denominator consistent with the computed numerator, shortens
    # FP32 probability liveness, and lets the backend use packed half adds.
    # BF16 reduction lowering is substantially slower on gfx950, so it keeps
    # the original FP32 denominator. Non-lazy/ragged paths are unchanged.
    if out_dtype == gl.float16:
        l_ij = gl.sum(p_cast, axis=1)
    elif CHAIN_BF16_ROWSUM:
        # BM256 exposes enough independent MFMA work to hide this longer,
        # lower-liveness chain. The BM128 schedule retains the reduction tree.
        l_ij = sc_sum_rows_chain4(
            p, CHAIN_BF16_ROWSUM == 2)
    else:
        l_ij = gl.sum(p, axis=1)
    p_dot = gl.convert_layout(p_cast, p_dot_layout)
    l_i = l_i + l_ij
    return l_i, p_dot


@gluon.jit
def sc_vec2_lazy_exp_fragment(qk_fragment, m_new,
                              qk_scale: gl.constexpr,
                              SCALE_ON_Q: gl.constexpr):
    """Form one deferred probability fragment for the QK-step weave."""
    if SCALE_ON_Q:
        return gl.exp2(qk_fragment - m_new[:, None])
    return gl.exp2(gl.fma(qk_fragment, qk_scale, -m_new[:, None]))


@gluon.jit
def sc_vec2_lazy_fp16_finalize(
    l_i, p_0123, p_4, p_5, p_6, p_7,
    p_dot_layout: gl.constexpr,
):
    """Finish the FP16 denominator and P operand after woven exponentials."""
    p_45 = sc_concat_cols(p_4, p_5)
    p_67 = sc_concat_cols(p_6, p_7)
    p = sc_concat_cols(p_0123, sc_concat_cols(p_45, p_67))
    p_cast = p.to(gl.float16)
    l_i = l_i + gl.sum(p_cast, axis=1)
    p_dot = gl.convert_layout(p_cast, p_dot_layout)
    return l_i, p_dot


@gluon.jit
def sc_dot_pv_step4_vec1(
    acc, p_dot, v_dot, qk, m_run,
    qk_scale: gl.constexpr, SCALE_ON_Q: gl.constexpr,
    VEC1_SPLIT: gl.constexpr, ROWMAX_AFTER_TWO: gl.constexpr,
):
    """Interleave row-max with four native N16 P-by-V reduction steps."""
    gl.static_assert(p_dot.shape[1] == 64)
    gl.static_assert(v_dot.shape[0] == 64)
    p0 = amd_slice(p_dot, [p_dot.shape[0], 16], [0, 0])
    p1 = amd_slice(p_dot, [p_dot.shape[0], 16], [0, 16])
    p2 = amd_slice(p_dot, [p_dot.shape[0], 16], [0, 32])
    p3 = amd_slice(p_dot, [p_dot.shape[0], 16], [0, 48])
    v0 = amd_slice(v_dot, [16, v_dot.shape[1]], [0, 0])
    v1 = amd_slice(v_dot, [16, v_dot.shape[1]], [16, 0])
    v2 = amd_slice(v_dot, [16, v_dot.shape[1]], [32, 0])
    v3 = amd_slice(v_dot, [16, v_dot.shape[1]], [48, 0])

    acc = do_mma("mfma_cdna4", p0, v0, acc)
    if ROWMAX_AFTER_TWO:
        acc = do_mma("mfma_cdna4", p1, v1, acc)
        m_new, score_delta, advance = sc_vec1_lazy_max(
            qk, m_run, qk_scale, SCALE_ON_Q)
    else:
        m_new, score_delta, advance = sc_vec1_lazy_max(
            qk, m_run, qk_scale, SCALE_ON_Q)
        acc = do_mma("mfma_cdna4", p1, v1, acc)
    acc = do_mma("mfma_cdna4", p2, v2, acc)
    acc = do_mma("mfma_cdna4", p3, v3, acc)
    p_0123, carried_4, carried_5, carried_6, carried_7 = sc_vec1_lazy_exp(
        qk, m_new, qk_scale, SCALE_ON_Q, VEC1_SPLIT)
    return (acc, m_new, p_0123, carried_4, carried_5, carried_6,
            carried_7, score_delta, advance)


@gluon.jit
def sc_dot_qk_k16(q_dot, kt_dot, qk, K_OFFSET: gl.constexpr):
    """Issue one native K16 step of a D128 Q-by-K reduction."""
    q_fragment = amd_slice(
        q_dot, [q_dot.shape[0], 16], [0, K_OFFSET])
    k_fragment = amd_slice(
        kt_dot, [16, kt_dot.shape[1]], [K_OFFSET, 0])
    return do_mma("mfma_cdna4", q_fragment, k_fragment, qk)


@gluon.jit
def sc_dot_qk_step8_vec2(
    q_dot, kt_dot, l_i,
    p_0123, p_4, qk_5, qk_6, qk_7, m_new,
    p_dot_layout: gl.constexpr,
    qk_scale: gl.constexpr, SCALE_ON_Q: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    mma_layout: gl.constexpr,
):
    """Interleave deferred VEC2 with eight native K16 Q-by-K steps."""
    gl.static_assert(q_dot.shape[1] == 128)
    gl.static_assert(kt_dot.shape[0] == 128)

    qk = gl.zeros(
        [q_dot.shape[0], kt_dot.shape[1]],
        dtype=gl.float32, layout=mma_layout)
    qk = sc_dot_qk_k16(q_dot, kt_dot, qk, 0)
    qk = sc_dot_qk_k16(q_dot, kt_dot, qk, 16)
    qk = sc_dot_qk_k16(q_dot, kt_dot, qk, 32)
    qk = sc_dot_qk_k16(q_dot, kt_dot, qk, 48)
    qk = sc_dot_qk_k16(q_dot, kt_dot, qk, 64)
    p_5 = sc_vec2_lazy_exp_fragment(
        qk_5, m_new, qk_scale, SCALE_ON_Q)
    qk = sc_dot_qk_k16(q_dot, kt_dot, qk, 80)
    p_6 = sc_vec2_lazy_exp_fragment(
        qk_6, m_new, qk_scale, SCALE_ON_Q)
    qk = sc_dot_qk_k16(q_dot, kt_dot, qk, 96)
    p_7 = sc_vec2_lazy_exp_fragment(
        qk_7, m_new, qk_scale, SCALE_ON_Q)
    if IS_CAUSAL:
        l_i, p_dot = sc_vec2_lazy_fp16_finalize(
            l_i, p_0123, p_4, p_5, p_6, p_7, p_dot_layout)
        qk = sc_dot_qk_k16(q_dot, kt_dot, qk, 112)
    else:
        p_45 = sc_concat_cols(p_4, p_5)
        p_67 = sc_concat_cols(p_6, p_7)
        p = sc_concat_cols(p_0123, sc_concat_cols(p_45, p_67))
        p_cast = p.to(gl.float16)
        qk = sc_dot_qk_k16(q_dot, kt_dot, qk, 112)
        l_i = l_i + gl.sum(p_cast, axis=1)
        p_dot = gl.convert_layout(p_cast, p_dot_layout)
    return qk, l_i, p_dot


@gluon.jit
def _sc_rescale(acc, l_i, score_delta):
    alpha = gl.exp2(-score_delta)
    return acc * alpha[:, None], l_i * alpha


if _HAS_WARP_PREDICATE:
    @gluon.jit
    def sc_rescale_lazy(acc, l_i, score_delta, need):
        """Skip both factor formation and correction for stable waves."""
        return gl.warp_predicate(
            need, (acc, l_i), _sc_rescale, args=(score_delta, ))

else:
    @gluon.jit
    def sc_rescale_lazy(acc, l_i, score_delta, need):
        """Compatibility definition; the old-compiler path never calls it."""
        return _sc_rescale(acc, l_i, score_delta)


_SC_SCALE_ACC_PACK64_ASM = tl.constexpr("\n".join(
    [
        "v_cmp_ne_u32_e32 vcc, 0, $192",
        "s_cbranch_vccz 1f",
        "s_and_saveexec_b64 vcc, vcc",
    ]
    + [
        f"v_mul_f32_e32 ${i}, ${64 + i}, ${128 + i}"
        for i in range(64)
    ]
    + ["s_mov_b64 exec, vcc", "1:"]
))
_SC_SCALE_ACC_PACK64_CONSTRAINTS = tl.constexpr(",".join(
    ["=v"] * 64
    + [str(i) for i in range(64)]
    + ["v"] * 128
    + ["~{vcc}"]
))

_SC_SCALE_ACC_PACK32X2_ASM = tl.constexpr("\n".join(
    [
        "v_cmp_ne_u32_e32 vcc, 0, $96",
        "s_cbranch_vccz 1f",
        "s_and_saveexec_b64 vcc, vcc",
    ]
    + [f"v_pk_mul_f32 ${i}, ${i}, $64" for i in range(32)]
    + ["s_mov_b64 exec, vcc", "1:"]
))
_SC_SCALE_ACC_PACK32X2_CONSTRAINTS = tl.constexpr(",".join(
    ["=v"] * 32
    + [str(i) for i in range(32)]
    + ["v"] * 64
    + ["~{vcc}"]
))

@gluon.jit
def _sc_scale_acc_pack64(acc, alpha, need):
    """Conditionally rescale all 64 thread-local accumulator values."""
    return gl.inline_asm_elementwise(
        _SC_SCALE_ACC_PACK64_ASM,
        _SC_SCALE_ACC_PACK64_CONSTRAINTS,
        [acc, alpha, need],
        dtype=gl.float32,
        is_pure=False,
        pack=64,
    )


@gluon.jit
def _sc_scale_acc_pack32x2(acc, alpha, need):
    """Conditionally rescale 32 adjacent FP32 pairs with packed VALU."""
    acc_layout: gl.constexpr = acc.type.layout
    # Pair adjacent columns so the i64 containers follow the MFMA accumulator's
    # native register order.  Pairing the two D64 halves instead forces copies
    # and raises allocation; this form remains copy-coalescible and spill-free.
    acc0, acc1 = acc.reshape(
        [acc.shape[0], acc.shape[1] // 2, 2]).split()
    acc0_bits = acc0.to(gl.uint32, bitcast=True).to(gl.uint64)
    acc1_bits = acc1.to(gl.uint32, bitcast=True).to(gl.uint64)
    acc_packed = acc0_bits | (acc1_bits << 32)
    alpha = gl.convert_layout(alpha, acc0.type.layout)
    need = gl.convert_layout(need, acc0.type.layout)
    alpha_bits = alpha.to(gl.uint32, bitcast=True).to(gl.uint64)
    alpha_packed = alpha_bits | (alpha_bits << 32)
    acc_packed = gl.inline_asm_elementwise(
        _SC_SCALE_ACC_PACK32X2_ASM,
        _SC_SCALE_ACC_PACK32X2_CONSTRAINTS,
        [acc_packed, alpha_packed, need],
        dtype=gl.uint64,
        is_pure=False,
        pack=32,
    )
    acc0 = acc_packed.to(gl.uint32).to(gl.float32, bitcast=True)
    acc1 = (acc_packed >> 32).to(gl.uint32).to(
        gl.float32, bitcast=True)
    acc = gl.join(acc0, acc1).reshape(acc.shape)
    return gl.convert_layout(acc, acc_layout)


@gluon.jit
def sc_qk_war_barrier_pack32(qk):
    """Anchor a K-slot WAR handoff after all thread-local QK fragments."""
    # Each N32 hardware-MFMA result is defined by one final instruction.  A
    # zero-copy N8 view supplies four values per lane from each result, which is
    # sufficient to anchor the scalar barrier without tying all 32 fragments
    # through an opaque inline-assembly region.
    qk0, qk1 = sc_split_cols(qk)
    qk0, _ = sc_split_cols(qk0)
    qk0, _ = sc_split_cols(qk0)
    qk1, _ = sc_split_cols(qk1)
    qk1, _ = sc_split_cols(qk1)
    gl.inline_asm_elementwise(
        "s_waitcnt lgkmcnt(0)\ns_barrier",
        "=s,=s,=s,=s,v,v,v,v,v,v,v,v",
        [qk0, qk1],
        dtype=gl.int32,
        is_pure=False,
        pack=4,
    )
    return qk


@gluon.jit
def sc_qk_war_barrier_relaxed(qk):
    """Anchor both QK results before a K-slot WAR rendezvous."""
    qk0, qk1 = sc_split_cols(qk)
    qk0, _ = sc_split_cols(qk0)
    qk0, _ = sc_split_cols(qk0)
    qk1, _ = sc_split_cols(qk1)
    qk1, _ = sc_split_cols(qk1)
    gl.inline_asm_elementwise(
        "s_barrier",
        "=s,=s,=s,=s,v,v,v,v,v,v,v,v",
        [qk0, qk1],
        dtype=gl.int32,
        is_pure=False,
        pack=4,
    )
    return qk


@gluon.jit
def sc_pv_war_barrier_relaxed(acc):
    """Anchor every D32 P-by-V result before a V-slot WAR rendezvous."""
    # D128 P-by-V is emitted as four independent D32 accumulator groups. One
    # zero-copy D8 view from each group makes the scalar barrier depend on the
    # final MFMA that defines every group, after all V LDS reads are consumed.
    acc_lo, acc_hi = sc_split_cols(acc)
    acc_0, acc_1 = sc_split_cols(acc_lo)
    acc_2, acc_3 = sc_split_cols(acc_hi)
    acc_0, _ = sc_split_cols(acc_0)
    acc_0, _ = sc_split_cols(acc_0)
    acc_1, _ = sc_split_cols(acc_1)
    acc_1, _ = sc_split_cols(acc_1)
    acc_2, _ = sc_split_cols(acc_2)
    acc_2, _ = sc_split_cols(acc_2)
    acc_3, _ = sc_split_cols(acc_3)
    acc_3, _ = sc_split_cols(acc_3)
    gl.inline_asm_elementwise(
        "s_barrier",
        "=s,=s,=s,=s," + ",".join(["v"] * 16),
        [acc_0, acc_1, acc_2, acc_3],
        dtype=gl.int32,
        is_pure=False,
        pack=4,
    )
    return acc


@gluon.jit
def _sc_prepare_alpha_deferred(score_delta, need):
    """Form the correction factor only on waves that actually advance."""
    return gl.inline_asm_elementwise(
        """
        v_mov_b32_e32 $0, 0x3f800000
        v_cmp_ne_u32_e32 vcc, 0, $2
        s_cbranch_vccz 1f
        s_and_saveexec_b64 vcc, vcc
        v_exp_f32_e64 $0, -$1
        s_mov_b64 exec, vcc
        1:
        """,
        "=&v,v,v,~{vcc}",
        [score_delta, need.to(gl.int32)],
        dtype=gl.float32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def sc_rescale_lazy_inline_pack64(acc, l_i, score_delta, need):
    """Skip the D128 correction when the existing base-2 frame is valid."""
    alpha = gl.exp2(-score_delta)
    acc = _sc_scale_acc_pack64(
        acc, alpha[:, None], need.to(gl.int32)[:, None])
    l_i = gl.where(need, l_i * alpha, l_i)
    return acc, l_i


@gluon.jit
def sc_rescale_lazy_inline_pack32x2(acc, l_i, score_delta, need):
    """Use packed FP32 correction for the measured short FP16 diagonal."""
    alpha = gl.exp2(-score_delta)
    acc = _sc_scale_acc_pack32x2(
        acc, alpha[:, None], need.to(gl.int32)[:, None])
    l_i = gl.where(need, l_i * alpha, l_i)
    return acc, l_i


@gluon.jit
def sc_rescale_lazy_inline_pack64_deferred(
    acc, l_i, score_delta, need,
):
    """Skip both correction-factor formation and D128 scaling when stable."""
    alpha = _sc_prepare_alpha_deferred(score_delta, need)
    acc = _sc_scale_acc_pack64(
        acc, alpha[:, None], need.to(gl.int32)[:, None])
    l_i = gl.where(need, l_i * alpha, l_i)
    return acc, l_i


@gluon.jit
def sc_softmax_causal_lazy_pack64(
    acc, l_i, m_i, qk, start_n, start_m,
    qk_scale: gl.constexpr,
    MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    BALANCE_CAUSAL_WAVES: gl.constexpr,
    RESCALE_THRESHOLD: gl.constexpr,
    EMPTY_SAFE: gl.constexpr,
    DEFER_ALPHA: gl.constexpr,
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr, mma_offs_m_row: gl.constexpr,
):
    """Thresholded causal softmax using one grouped source-only correction."""
    local_n = gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
    local_m = gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
    if BALANCE_CAUSAL_WAVES:
        wave_m = local_m // 32
        wave_m = wave_m ^ ((wave_m // 4) * 3)
        local_m = wave_m * 32 + local_m % 32
    USE_LOCAL_CAUSAL_COORDS: gl.constexpr = (
        MAX_SEQLENS_K > 4 * BLOCK_M)
    if USE_LOCAL_CAUSAL_COORDS:
        # This helper is entered only for an aligned BM128 causal diagonal.
        # Its tile base is start_m*BM plus either zero or BN64, so XOR extracts
        # that local offset without forming two global coordinate vectors.
        diagonal_offset = start_n ^ (start_m * BLOCK_M)
        causal_mask = local_m[:, None] >= (
            local_n[None, :] + diagonal_offset)
    else:
        causal_offs_n = start_n + local_n
        causal_offs_m = start_m * BLOCK_M + local_m
        causal_boundary = causal_offs_n[None, :] + (
            MAX_SEQLENS_Q - MAX_SEQLENS_K)
        causal_mask = causal_offs_m[:, None] >= causal_boundary
    qk_masked = gl.where(
        causal_mask, qk,
        gl.full([BLOCK_M, BLOCK_N], float("-inf"),
                dtype=gl.float32, layout=mma_layout),
    )

    SCALE_ON_Q: gl.constexpr = qk_scale == 1.0
    m_tile = nan_propagating_max(qk_masked, axis=1)
    if not SCALE_ON_Q:
        m_tile = m_tile * qk_scale
    score_delta = m_tile - m_i
    if EMPTY_SAFE:
        has_prior = m_i != float("-inf")
        advance = (score_delta > RESCALE_THRESHOLD) & has_prior
        m_new = gl.where(
            has_prior, gl.where(advance, m_tile, m_i), m_tile)
        need_correction = advance | ~has_prior
    else:
        advance = score_delta > RESCALE_THRESHOLD
        m_new = gl.where(advance, m_tile, m_i)
        need_correction = advance
    if SCALE_ON_Q:
        p = gl.exp2(qk_masked - m_new[:, None])
    else:
        p = gl.exp2(gl.fma(qk_masked, qk_scale, -m_new[:, None]))
    if DEFER_ALPHA:
        acc, l_i = sc_rescale_lazy_inline_pack64_deferred(
            acc, l_i, score_delta, need_correction)
    elif RESCALE_THRESHOLD == DIAGONAL_LAZY_RESCALE_THRESHOLD_FP16:
        # This is the non-deferred FP16 N512 route.  N1024 loses from the
        # changed schedule and BF16 is neutral, so both retain scalar packing.
        acc, l_i = sc_rescale_lazy_inline_pack32x2(
            acc, l_i, score_delta, need_correction)
    else:
        acc, l_i = sc_rescale_lazy_inline_pack64(
            acc, l_i, score_delta, need_correction)
    l_i = l_i + gl.sum(p, axis=1)
    return acc, l_i, m_new, p


@gluon.jit
def sc_lr(smem_slot, dot_layout: gl.constexpr):
    """LRK / LRV: local-read a tile from LDS into registers."""
    return cdna4_async.load_shared_relaxed(smem_slot, dot_layout)


@gluon.jit
def sc_issue_async_unmasked(smem_slot, base, offsets):
    """Issue one full-tile async copy using a loop-invariant offset pattern."""
    cdna4_async.buffer_load_to_shared(smem_slot, base, offsets)
    cdna4_async.commit_group()


@gluon.jit
def sc_execution_barrier():
    """Rendezvous all waves without adding an LDS visibility fence."""
    return gl.inline_asm_elementwise(
        "s_waitcnt lgkmcnt(0)\ns_barrier", "=s", [], dtype=gl.int32,
        is_pure=False, pack=1)


@gluon.jit
def sc_war_barrier(LIGHTWEIGHT: gl.constexpr):
    """Protect an LDS slot overwrite with the selected barrier semantics."""
    if LIGHTWEIGHT:
        sc_execution_barrier()
    else:
        gl.barrier()


# ---------------------------------------------------------------------------
# Two-slot preload path for short ranges and the pruned causal diagonal
# ---------------------------------------------------------------------------

@gluon.jit
def sc_predicated_causal_tile(
    acc, l_i, m_i,
    q_dot, kt_slot, v_slot, start_n, start_m,
    qk_scale: gl.constexpr,
    MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    kt_dot_layout: gl.constexpr, p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr, mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr, mma_offs_m_row: gl.constexpr,
    BALANCE_CAUSAL_WAVES: gl.constexpr,
    ENABLE_CLASS_DIAGONAL_LAZY: gl.constexpr,
):
    """Compute one causal tile inside a wave-uniform predicated region."""
    kt_dot = sc_lr(kt_slot, kt_dot_layout)
    qk = compute_dot1_qk(q_dot, kt_dot, BLOCK_M, BLOCK_N, mma_layout)
    USE_CLASS_DIAGONAL_LAZY: gl.constexpr = (
        ENABLE_CLASS_DIAGONAL_LAZY
        and MAX_SEQLENS_Q == MAX_SEQLENS_K
        and MAX_SEQLENS_Q <= 1024
        and MAX_SEQLENS_Q % BLOCK_M == 0
        and MAX_SEQLENS_K % BLOCK_N == 0
        and BLOCK_M == 128 and BLOCK_N == 64
        and q_dot.shape[1] == 128 and gl.num_warps() == 4
    )
    if USE_CLASS_DIAGONAL_LAZY:
        if MAX_SEQLENS_Q <= 512 and start_m > 0:
            acc, l_i, m_i, p = sc_softmax_causal_lazy_pack64(
                acc, l_i, m_i, qk, start_n, start_m, qk_scale,
                MAX_SEQLENS_Q, MAX_SEQLENS_K, BLOCK_M, BLOCK_N,
                BALANCE_CAUSAL_WAVES,
                (DIAGONAL_LAZY_RESCALE_THRESHOLD_FP16
                 if q_dot.dtype == gl.float16
                 else DIAGONAL_LAZY_RESCALE_THRESHOLD),
                False, False,
                mma_layout, mma_offs_n_col, mma_offs_m_row,
            )
        elif MAX_SEQLENS_Q <= 512:
            acc, l_i, m_i, p = compute_softmax(
                acc, l_i, m_i, qk, start_n, start_m,
                MAX_SEQLENS_Q, MAX_SEQLENS_K, qk_scale,
                MAX_SEQLENS_Q, MAX_SEQLENS_K,
                BLOCK_M, BLOCK_N, True, True, False,
                mma_layout, mma_offs_n_col, mma_offs_m_row,
                BALANCE_CAUSAL_WAVES,
            )
        else:
            acc, l_i, m_i, p = sc_softmax_causal_lazy_pack64(
                acc, l_i, m_i, qk, start_n, start_m, qk_scale,
                MAX_SEQLENS_Q, MAX_SEQLENS_K, BLOCK_M, BLOCK_N,
                BALANCE_CAUSAL_WAVES,
                DIAGONAL_LAZY_RESCALE_THRESHOLD_FP16,
                True, True,
                mma_layout, mma_offs_n_col, mma_offs_m_row,
            )
    else:
        acc, l_i, m_i, p = compute_softmax(
            acc, l_i, m_i, qk, start_n, start_m,
            MAX_SEQLENS_Q, MAX_SEQLENS_K, qk_scale,
            MAX_SEQLENS_Q, MAX_SEQLENS_K,
            BLOCK_M, BLOCK_N, True, True, False,
            mma_layout, mma_offs_n_col, mma_offs_m_row,
            BALANCE_CAUSAL_WAVES,
        )
    p_dot = gl.convert_layout(p.to(q_dot.dtype), p_dot_layout)
    v_dot = sc_lr(v_slot, v_dot_layout)
    acc = sc_dot_pv(acc, p_dot, v_dot)
    return acc, l_i, m_i


@gluon.jit
def sc_prepare_final_recip(_unused_recip, l_i):
    """Prepare a wave's final reciprocal while sibling waves keep computing."""
    return gl.extra.libdevice.fast_dividef(1.0, l_i)


@gluon.jit
def sc_predicated_causal_tile_final(
    acc, l_i, m_i, _unused_recip,
    q_dot, kt_slot, v_slot, start_n, start_m,
    qk_scale: gl.constexpr,
    MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    kt_dot_layout: gl.constexpr, p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr, mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr, mma_offs_m_row: gl.constexpr,
    BALANCE_CAUSAL_WAVES: gl.constexpr,
    ENABLE_CLASS_DIAGONAL_LAZY: gl.constexpr,
):
    """Compute a wave's final tile and expose reciprocal work beside P*V."""
    kt_dot = sc_lr(kt_slot, kt_dot_layout)
    qk = compute_dot1_qk(q_dot, kt_dot, BLOCK_M, BLOCK_N, mma_layout)
    USE_CLASS_DIAGONAL_LAZY: gl.constexpr = (
        ENABLE_CLASS_DIAGONAL_LAZY
        and MAX_SEQLENS_Q == MAX_SEQLENS_K
        and MAX_SEQLENS_Q <= 1024
        and MAX_SEQLENS_Q % BLOCK_M == 0
        and MAX_SEQLENS_K % BLOCK_N == 0
        and BLOCK_M == 128 and BLOCK_N == 64
        and q_dot.shape[1] == 128 and gl.num_warps() == 4
    )
    if USE_CLASS_DIAGONAL_LAZY:
        if MAX_SEQLENS_Q <= 512:
            acc, l_i, m_i, p = sc_softmax_causal_lazy_pack64(
                acc, l_i, m_i, qk, start_n, start_m, qk_scale,
                MAX_SEQLENS_Q, MAX_SEQLENS_K, BLOCK_M, BLOCK_N,
                BALANCE_CAUSAL_WAVES,
                (DIAGONAL_LAZY_RESCALE_THRESHOLD_FP16
                 if q_dot.dtype == gl.float16
                 else DIAGONAL_LAZY_RESCALE_THRESHOLD),
                False, False,
                mma_layout, mma_offs_n_col, mma_offs_m_row,
            )
        else:
            acc, l_i, m_i, p = sc_softmax_causal_lazy_pack64(
                acc, l_i, m_i, qk, start_n, start_m, qk_scale,
                MAX_SEQLENS_Q, MAX_SEQLENS_K, BLOCK_M, BLOCK_N,
                BALANCE_CAUSAL_WAVES,
                DIAGONAL_LAZY_RESCALE_THRESHOLD_FP16,
                False, True,
                mma_layout, mma_offs_n_col, mma_offs_m_row,
            )
    else:
        acc, l_i, m_i, p = compute_softmax(
            acc, l_i, m_i, qk, start_n, start_m,
            MAX_SEQLENS_Q, MAX_SEQLENS_K, qk_scale,
            MAX_SEQLENS_Q, MAX_SEQLENS_K,
            BLOCK_M, BLOCK_N, True, True, False,
            mma_layout, mma_offs_n_col, mma_offs_m_row,
            BALANCE_CAUSAL_WAVES,
        )
    p_dot = gl.convert_layout(p.to(q_dot.dtype), p_dot_layout)
    l_recip = gl.extra.libdevice.fast_dividef(1.0, l_i)
    v_dot = sc_lr(v_slot, v_dot_layout)
    acc = sc_dot_pv(acc, p_dot, v_dot)
    return acc, l_i, m_i, l_recip


@gluon.jit
def sc_predicated_causal_tile_from_state(
    _unused_acc, _unused_l, _unused_m, acc, l_i, m_i,
    q_dot, kt_slot, v_slot, start_n, start_m,
    qk_scale: gl.constexpr,
    MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    kt_dot_layout: gl.constexpr, p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr, mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr, mma_offs_m_row: gl.constexpr,
    BALANCE_CAUSAL_WAVES: gl.constexpr,
    ENABLE_CLASS_DIAGONAL_LAZY: gl.constexpr,
):
    """Predicated tile whose carried merge state is supplied read-only."""
    return sc_predicated_causal_tile(
        acc, l_i, m_i,
        q_dot, kt_slot, v_slot, start_n, start_m,
        qk_scale, MAX_SEQLENS_Q, MAX_SEQLENS_K,
        BLOCK_M, BLOCK_N,
        kt_dot_layout, p_dot_layout, v_dot_layout,
        mma_layout, mma_offs_n_col, mma_offs_m_row,
        BALANCE_CAUSAL_WAVES,
        ENABLE_CLASS_DIAGONAL_LAZY,
    )


@gluon.jit
def sc_predicated_causal_tile_final_from_state(
    _unused_acc, _unused_l, _unused_m, _unused_recip,
    acc, l_i, m_i, recip,
    q_dot, kt_slot, v_slot, start_n, start_m,
    qk_scale: gl.constexpr,
    MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    kt_dot_layout: gl.constexpr, p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr, mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr, mma_offs_m_row: gl.constexpr,
    BALANCE_CAUSAL_WAVES: gl.constexpr,
    ENABLE_CLASS_DIAGONAL_LAZY: gl.constexpr,
):
    """Final-tile variant whose carried merge state is supplied read-only."""
    return sc_predicated_causal_tile_final(
        acc, l_i, m_i, recip,
        q_dot, kt_slot, v_slot, start_n, start_m,
        qk_scale, MAX_SEQLENS_Q, MAX_SEQLENS_K,
        BLOCK_M, BLOCK_N,
        kt_dot_layout, p_dot_layout, v_dot_layout,
        mma_layout, mma_offs_n_col, mma_offs_m_row,
        BALANCE_CAUSAL_WAVES,
        ENABLE_CLASS_DIAGONAL_LAZY,
    )


@gluon.jit
def sc_predicated_unmasked_tile_from_state(
    _unused_acc, _unused_l, _unused_m, acc, l_i, m_i,
    q_dot, kt_slot, v_slot, start_n, start_m,
    qk_scale: gl.constexpr,
    MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    kt_dot_layout: gl.constexpr, p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr, mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr, mma_offs_m_row: gl.constexpr,
):
    """Predicated tile for a wave whose complete causal tile is valid."""
    kt_dot = sc_lr(kt_slot, kt_dot_layout)
    qk = compute_dot1_qk(q_dot, kt_dot, BLOCK_M, BLOCK_N, mma_layout)
    acc, l_i, m_i, p = compute_softmax(
        acc, l_i, m_i, qk, start_n, start_m,
        MAX_SEQLENS_Q, MAX_SEQLENS_K, qk_scale,
        MAX_SEQLENS_Q, MAX_SEQLENS_K,
        BLOCK_M, BLOCK_N, False, False, False,
        mma_layout, mma_offs_n_col, mma_offs_m_row,
    )
    p_dot = gl.convert_layout(p.to(q_dot.dtype), p_dot_layout)
    v_dot = sc_lr(v_slot, v_dot_layout)
    acc = sc_dot_pv(acc, p_dot, v_dot)
    return acc, l_i, m_i


@gluon.jit
def sc_causal_wave_tile_fully_valid(
    start_n, start_m,
    MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    BALANCE_CAUSAL_WAVES: gl.constexpr,
    mma_offs_m_row: gl.constexpr,
):
    """Whether every row in this wave can consume the complete K tile."""
    local_m = gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
    if BALANCE_CAUSAL_WAVES:
        wave_m = local_m // 32
        wave_m = wave_m ^ ((wave_m // 4) * 3)
        local_m = wave_m * 32 + local_m % 32
    wave_m_first = start_m * BLOCK_M + (local_m // 32) * 32
    causal_last_valid = wave_m_first + MAX_SEQLENS_K - MAX_SEQLENS_Q
    return start_n + BLOCK_N - 1 <= causal_last_valid


@gluon.jit
def attn_fwd_inner_short(
    acc, l_i, m_i, q_dot, k_base, v_base, start_m,
    stride_kn, stride_kk, stride_vk, stride_vn,
    block_start, n_blocks,
    kt_smem, v_smem,
    qk_scale: gl.constexpr,
    MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr, ACTUAL_BLOCK_DMODEL: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    BALANCE_CAUSAL_WAVES: gl.constexpr,
    DIAGONAL_PREFETCHED: gl.constexpr,
    SKIP_ENTRY_WAIT: gl.constexpr,
    kt_async_layout: gl.constexpr, v_async_layout: gl.constexpr,
    kt_dot_layout: gl.constexpr, p_dot_layout: gl.constexpr, v_dot_layout: gl.constexpr,
    mma_layout: gl.constexpr, mma_offs_n_col: gl.constexpr, mma_offs_m_row: gl.constexpr,
    ENABLE_CLASS_DIAGONAL_LAZY: gl.constexpr,
):
    """Two-slot fallback when remaining blocks can't fill the pipeline.

    The matched pipeline uses two LDS slots even though its pipeline depth is four,
    so short tails are processed in chunks that fit the same K/V ring.

    Scores are always masked. K/V DMA bounds are emitted only for a ragged K
    tile or a padded head dimension.
    """
    if DIAGONAL_PREFETCHED:
        # Prefix handoff order is K1,K0,V0,V1. Drain through V0 while
        # diagonal tile 0 covers the final V1 transfer.
        cdna4_async.wait_group(1)
    elif not SKIP_ENTRY_WAIT:
        cdna4_async.wait_group(0)
    num_fallback = n_blocks - block_start
    kt_slot0 = kt_smem.index(0)
    kt_slot1 = kt_smem.index(1)
    v_slot0 = v_smem.index(0)
    v_slot1 = v_smem.index(1)
    WAVE_SKIP_CAUSAL: gl.constexpr = (
        HAS_WARP_PREDICATE and IS_CAUSAL
        and BLOCK_M == 32 * gl.num_warps()
    )
    MASK_LOADS: gl.constexpr = (
        MAX_SEQLENS_K % BLOCK_N != 0
        or ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL
    )
    USE_FOUR_SLOT_DIAGONAL: gl.constexpr = (
        WAVE_SKIP_CAUSAL and MAX_SEQLENS_K % BLOCK_N == 0
        and MAX_SEQLENS_Q == MAX_SEQLENS_K
        and BLOCK_M == 256 and BLOCK_N == 64 and gl.num_warps() == 8
    )
    if USE_FOUR_SLOT_DIAGONAL:
        # Equal-length aligned BM256/BN64 self-attention has exactly four
        # diagonal tiles, so each can retain a unique LDS slot. A nonzero
        # block_start means the full-prefix drain already prefetched all four.
        if BLOCK_DMODEL == 128 and MAX_SEQLENS_K // BLOCK_N <= 32:
            need_diagonal_load = block_start == 0
        else:
            need_diagonal_load: gl.constexpr = True
        if need_diagonal_load:
            for slot in gl.static_range(4):
                if slot < num_fallback:
                    start_n = (block_start + slot) * BLOCK_N
                    issue_async_load_k(
                        kt_smem.index(slot), k_base, start_n,
                        stride_kn, stride_kk,
                        MAX_SEQLENS_K, MASK_LOADS, MAX_SEQLENS_K, False,
                        BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
                        kt_async_layout,
                    )
                    issue_async_load_v(
                        v_smem.index(slot), v_base, start_n,
                        stride_vk, stride_vn,
                        MAX_SEQLENS_K, MASK_LOADS, MAX_SEQLENS_K, False,
                        BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
                        v_async_layout,
                    )

        cdna4_async.wait_group(0)
        UNMASK_FULL_DIAGONAL_WAVES: gl.constexpr = (
            BLOCK_DMODEL == 128 and MAX_SEQLENS_K // BLOCK_N <= 32)
        for i in tl.range(0, num_fallback):
            slot_idx = i.to(tl.int32)
            start_n = (block_start + i) * BLOCK_N
            half_n: gl.constexpr = BLOCK_N // 2
            active_full = sc_causal_wave_active(
                start_n + half_n, start_m,
                MAX_SEQLENS_Q, MAX_SEQLENS_K,
                BLOCK_M, BALANCE_CAUSAL_WAVES, mma_offs_m_row)
            if UNMASK_FULL_DIAGONAL_WAVES:
                fully_valid = sc_causal_wave_tile_fully_valid(
                    start_n, start_m,
                    MAX_SEQLENS_Q, MAX_SEQLENS_K,
                    BLOCK_M, BLOCK_N, BALANCE_CAUSAL_WAVES,
                    mma_offs_m_row)
                partial_full = active_full & ~fully_valid
                next_acc, next_l, next_m = gl.warp_predicate(
                    fully_valid,
                    (gl.zeros_like(acc), gl.zeros_like(l_i), gl.zeros_like(m_i)),
                    sc_predicated_unmasked_tile_from_state,
                    args=(
                        acc, l_i, m_i,
                        q_dot, kt_smem.index(slot_idx), v_smem.index(slot_idx),
                        start_n, start_m, qk_scale,
                        MAX_SEQLENS_Q, MAX_SEQLENS_K,
                        BLOCK_M, BLOCK_N,
                        kt_dot_layout, p_dot_layout, v_dot_layout,
                        mma_layout, mma_offs_n_col, mma_offs_m_row,
                    ),
                )
                acc = gl.where(fully_valid[:, None], next_acc, acc)
                l_i = gl.where(fully_valid, next_l, l_i)
                m_i = gl.where(fully_valid, next_m, m_i)
                next_acc, next_l, next_m = gl.warp_predicate(
                    partial_full,
                    (gl.zeros_like(acc), gl.zeros_like(l_i), gl.zeros_like(m_i)),
                    sc_predicated_causal_tile_from_state,
                    args=(
                        acc, l_i, m_i,
                        q_dot, kt_smem.index(slot_idx), v_smem.index(slot_idx),
                        start_n, start_m, qk_scale,
                        MAX_SEQLENS_Q, MAX_SEQLENS_K,
                        BLOCK_M, BLOCK_N,
                        kt_dot_layout, p_dot_layout, v_dot_layout,
                        mma_layout, mma_offs_n_col, mma_offs_m_row,
                        BALANCE_CAUSAL_WAVES,
                        ENABLE_CLASS_DIAGONAL_LAZY,
                    ),
                )
                acc = gl.where(partial_full[:, None], next_acc, acc)
                l_i = gl.where(partial_full, next_l, l_i)
                m_i = gl.where(partial_full, next_m, m_i)
            else:
                acc, l_i, m_i = gl.warp_predicate(
                    active_full, (acc, l_i, m_i), sc_predicated_causal_tile,
                    args=(
                        q_dot, kt_smem.index(slot_idx), v_smem.index(slot_idx),
                        start_n, start_m, qk_scale,
                        MAX_SEQLENS_Q, MAX_SEQLENS_K,
                        BLOCK_M, BLOCK_N,
                        kt_dot_layout, p_dot_layout, v_dot_layout,
                        mma_layout, mma_offs_n_col, mma_offs_m_row,
                        BALANCE_CAUSAL_WAVES,
                        ENABLE_CLASS_DIAGONAL_LAZY,
                    ),
                )

            # The next lower wave has valid scores only in this tile's low
            # N32 half.  All higher waves already consumed the full BN64 tile.
            active_lo = sc_causal_wave_active(
                start_n, start_m,
                MAX_SEQLENS_Q, MAX_SEQLENS_K,
                BLOCK_M, BALANCE_CAUSAL_WAVES, mma_offs_m_row)
            active_lo_only = active_lo & ~active_full
            kt_lo = kt_smem.index(slot_idx).slice(0, half_n, dim=1)
            v_lo = v_smem.index(slot_idx).slice(0, half_n, dim=0)
            next_acc, next_l, next_m = gl.warp_predicate(
                active_lo_only,
                (gl.zeros_like(acc), gl.zeros_like(l_i), gl.zeros_like(m_i)),
                sc_predicated_causal_tile_from_state,
                args=(
                    acc, l_i, m_i,
                    q_dot, kt_lo, v_lo, start_n, start_m,
                    qk_scale,
                    MAX_SEQLENS_Q, MAX_SEQLENS_K,
                    BLOCK_M, half_n,
                    kt_dot_layout, p_dot_layout, v_dot_layout,
                    mma_layout, mma_offs_n_col, mma_offs_m_row,
                    BALANCE_CAUSAL_WAVES,
                    ENABLE_CLASS_DIAGONAL_LAZY,
                ),
            )
            acc = gl.where(active_lo_only[:, None], next_acc, acc)
            l_i = gl.where(active_lo_only, next_l, l_i)
            m_i = gl.where(active_lo_only, next_m, m_i)
        return acc, l_i, m_i

    SUBTILE_FIRST_DIAGONAL: gl.constexpr = (
        WAVE_SKIP_CAUSAL
        and MAX_SEQLENS_Q == MAX_SEQLENS_K
        and MAX_SEQLENS_Q % BLOCK_M == 0
        and MAX_SEQLENS_K % BLOCK_N == 0
        and BLOCK_M == 128 and BLOCK_N == 64
        and BLOCK_DMODEL == 128 and gl.num_warps() == 4
    )
    STAGGER_FINAL_NORMALIZE: gl.constexpr = (
        SUBTILE_FIRST_DIAGONAL
        and MAX_SEQLENS_K <= 16 * BLOCK_N
    )
    if SUBTILE_FIRST_DIAGONAL:
        if not DIAGONAL_PREFETCHED:
            issue_async_load_k(
                kt_slot0, k_base, block_start * BLOCK_N,
                stride_kn, stride_kk,
                MAX_SEQLENS_K, False, MAX_SEQLENS_K, False,
                BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
                kt_async_layout,
            )
            issue_async_load_v(
                v_slot0, v_base, block_start * BLOCK_N,
                stride_vk, stride_vn,
                MAX_SEQLENS_K, False, MAX_SEQLENS_K, False,
                BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
                v_async_layout,
            )
            start_n_1 = (block_start + 1) * BLOCK_N
            issue_async_load_k(
                kt_slot1, k_base, start_n_1,
                stride_kn, stride_kk,
                MAX_SEQLENS_K, False, MAX_SEQLENS_K, False,
                BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
                kt_async_layout,
            )
            issue_async_load_v(
                v_slot1, v_base, start_n_1,
                stride_vk, stride_vn,
                MAX_SEQLENS_K, False, MAX_SEQLENS_K, False,
                BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
                v_async_layout,
            )

        cdna4_async.wait_group(2)
        start_n = block_start * BLOCK_N
        half_n: gl.constexpr = BLOCK_N // 2
        active_full = sc_causal_wave_active(
            start_n + half_n, start_m,
            MAX_SEQLENS_Q, MAX_SEQLENS_K,
            BLOCK_M, BALANCE_CAUSAL_WAVES, mma_offs_m_row)
        active_lo_only = ~active_full
        kt_lo = kt_slot0.slice(0, half_n, dim=1)
        v_lo = v_slot0.slice(0, half_n, dim=0)
        acc, l_i, m_i = gl.warp_predicate(
            active_full, (acc, l_i, m_i), sc_predicated_causal_tile,
            args=(
                q_dot, kt_slot0, v_slot0,
                start_n, start_m, qk_scale,
                MAX_SEQLENS_Q, MAX_SEQLENS_K, BLOCK_M, BLOCK_N,
                kt_dot_layout, p_dot_layout, v_dot_layout,
                mma_layout, mma_offs_n_col, mma_offs_m_row,
                BALANCE_CAUSAL_WAVES,
                ENABLE_CLASS_DIAGONAL_LAZY,
            ),
        )
        next_acc, next_l, next_m = gl.warp_predicate(
            active_lo_only,
            (gl.zeros_like(acc), gl.zeros_like(l_i), gl.zeros_like(m_i)),
            sc_predicated_causal_tile_from_state,
            args=(
                acc, l_i, m_i,
                q_dot, kt_lo, v_lo, start_n, start_m,
                qk_scale,
                MAX_SEQLENS_Q, MAX_SEQLENS_K, BLOCK_M, half_n,
                kt_dot_layout, p_dot_layout, v_dot_layout, mma_layout,
                mma_offs_n_col, mma_offs_m_row,
                BALANCE_CAUSAL_WAVES,
                ENABLE_CLASS_DIAGONAL_LAZY,
            ),
        )
        acc = gl.where(active_lo_only[:, None], next_acc, acc)
        l_i = gl.where(active_lo_only, next_l, l_i)
        m_i = gl.where(active_lo_only, next_m, m_i)

        cdna4_async.wait_group(0)
        start_n = (block_start + 1) * BLOCK_N
        active_full = sc_causal_wave_active(
            start_n + half_n, start_m,
            MAX_SEQLENS_Q, MAX_SEQLENS_K,
            BLOCK_M, BALANCE_CAUSAL_WAVES, mma_offs_m_row)
        active_lo = sc_causal_wave_active(
            start_n, start_m,
            MAX_SEQLENS_Q, MAX_SEQLENS_K,
            BLOCK_M, BALANCE_CAUSAL_WAVES, mma_offs_m_row)
        active_lo_only = active_lo & ~active_full
        if STAGGER_FINAL_NORMALIZE:
            # These waves have no valid key in tile 1.  Compute their final
            # reciprocal while the other two SIMDs execute that tile.  Keep
            # the accumulator scaling uniform below so it remains packed.
            final_recip = gl.full_like(l_i, 1.0)
            recip_before_early = final_recip
            early_recip = gl.warp_predicate(
                ~active_lo, final_recip, sc_prepare_final_recip, args=(l_i,))
            # Break the direct predicate-result edge before entering the next
            # sibling region; the stock AMD lowering folds this false-path
            # merge back into the region phi.
            final_recip = gl.where(
                ~active_lo, early_recip, recip_before_early)
        kt_lo_1 = kt_slot1.slice(0, half_n, dim=1)
        v_lo_1 = v_slot1.slice(0, half_n, dim=0)
        if STAGGER_FINAL_NORMALIZE:
            acc, l_i, m_i, final_recip = gl.warp_predicate(
                active_full, (acc, l_i, m_i, final_recip),
                sc_predicated_causal_tile_final,
                args=(
                    q_dot, kt_slot1, v_slot1,
                    start_n, start_m, qk_scale,
                    MAX_SEQLENS_Q, MAX_SEQLENS_K, BLOCK_M, BLOCK_N,
                    kt_dot_layout, p_dot_layout, v_dot_layout,
                    mma_layout, mma_offs_n_col, mma_offs_m_row,
                    BALANCE_CAUSAL_WAVES,
                    ENABLE_CLASS_DIAGONAL_LAZY,
                ),
            )
            next_acc, next_l, next_m, next_recip = gl.warp_predicate(
                active_lo_only,
                (
                    gl.zeros_like(acc), gl.zeros_like(l_i),
                    gl.zeros_like(m_i), gl.zeros_like(final_recip),
                ),
                sc_predicated_causal_tile_final_from_state,
                args=(
                    acc, l_i, m_i, final_recip,
                    q_dot, kt_lo_1, v_lo_1, start_n, start_m,
                    qk_scale,
                    MAX_SEQLENS_Q, MAX_SEQLENS_K, BLOCK_M, half_n,
                    kt_dot_layout, p_dot_layout, v_dot_layout, mma_layout,
                    mma_offs_n_col, mma_offs_m_row,
                    BALANCE_CAUSAL_WAVES,
                    ENABLE_CLASS_DIAGONAL_LAZY,
                ),
            )
            acc = gl.where(active_lo_only[:, None], next_acc, acc)
            l_i = gl.where(active_lo_only, next_l, l_i)
            m_i = gl.where(active_lo_only, next_m, m_i)
            final_recip = gl.where(
                active_lo_only, next_recip, final_recip)
            acc = acc * final_recip[:, None]
        else:
            acc, l_i, m_i = gl.warp_predicate(
                active_full, (acc, l_i, m_i),
                sc_predicated_causal_tile,
                args=(
                    q_dot, kt_slot1, v_slot1,
                    start_n, start_m, qk_scale,
                    MAX_SEQLENS_Q, MAX_SEQLENS_K, BLOCK_M, BLOCK_N,
                    kt_dot_layout, p_dot_layout, v_dot_layout,
                    mma_layout, mma_offs_n_col, mma_offs_m_row,
                    BALANCE_CAUSAL_WAVES,
                    ENABLE_CLASS_DIAGONAL_LAZY,
                ),
            )
            next_acc, next_l, next_m = gl.warp_predicate(
                active_lo_only,
                (gl.zeros_like(acc), gl.zeros_like(l_i), gl.zeros_like(m_i)),
                sc_predicated_causal_tile_from_state,
                args=(
                    acc, l_i, m_i,
                    q_dot, kt_lo_1, v_lo_1, start_n, start_m,
                    qk_scale,
                    MAX_SEQLENS_Q, MAX_SEQLENS_K, BLOCK_M, half_n,
                    kt_dot_layout, p_dot_layout, v_dot_layout, mma_layout,
                    mma_offs_n_col, mma_offs_m_row,
                    BALANCE_CAUSAL_WAVES,
                    ENABLE_CLASS_DIAGONAL_LAZY,
                ),
            )
            acc = gl.where(active_lo_only[:, None], next_acc, acc)
            l_i = gl.where(active_lo_only, next_l, l_i)
            m_i = gl.where(active_lo_only, next_m, m_i)
        return acc, l_i, m_i

    for chunk_start in tl.range(0, num_fallback, 2):
        for slot in gl.static_range(2):
            i = chunk_start + slot
            if i < num_fallback:
                start_n = (block_start + i) * BLOCK_N
                issue_async_load_k(
                    kt_smem.index(slot), k_base, start_n,
                    stride_kn, stride_kk,
                    MAX_SEQLENS_K, MASK_LOADS, MAX_SEQLENS_K, False,
                    BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
                    kt_async_layout,
                )
                issue_async_load_v(
                    v_smem.index(slot), v_base, start_n,
                    stride_vk, stride_vn,
                    MAX_SEQLENS_K, MASK_LOADS, MAX_SEQLENS_K, False,
                    BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
                    v_async_layout,
                )

        if chunk_start + 1 < num_fallback:
            # Four commits are ordered K0,V0,K1,V1. Waiting down to two makes
            # the first pair readable while the second pair remains in flight.
            cdna4_async.wait_group(2)
        else:
            cdna4_async.wait_group(0)

        for slot in gl.static_range(2):
            i = chunk_start + slot
            if i < num_fallback:
                if slot == 1:
                    cdna4_async.wait_group(0)
                start_n = (block_start + i) * BLOCK_N

                if WAVE_SKIP_CAUSAL:
                    active = sc_causal_wave_active(
                        start_n, start_m,
                        MAX_SEQLENS_Q, MAX_SEQLENS_K,
                        BLOCK_M, BALANCE_CAUSAL_WAVES,
                        mma_offs_m_row)
                    acc, l_i, m_i = gl.warp_predicate(
                        active, (acc, l_i, m_i),
                        sc_predicated_causal_tile,
                        args=(
                            q_dot,
                            kt_smem.index(slot), v_smem.index(slot),
                            start_n, start_m, qk_scale,
                            MAX_SEQLENS_Q, MAX_SEQLENS_K,
                            BLOCK_M, BLOCK_N,
                            kt_dot_layout, p_dot_layout, v_dot_layout,
                            mma_layout, mma_offs_n_col, mma_offs_m_row,
                            BALANCE_CAUSAL_WAVES,
                            ENABLE_CLASS_DIAGONAL_LAZY,
                        ),
                    )
                else:
                    kt_dot = cdna4_async.load_shared_relaxed(
                        kt_smem.index(slot), kt_dot_layout)
                    qk = compute_dot1_qk(
                        q_dot, kt_dot, BLOCK_M, BLOCK_N, mma_layout)
                    acc, l_i, m_i, p = compute_softmax(
                        acc, l_i, m_i, qk, start_n, start_m,
                        MAX_SEQLENS_Q, MAX_SEQLENS_K,
                        qk_scale,
                        MAX_SEQLENS_Q, MAX_SEQLENS_K,
                        BLOCK_M, BLOCK_N, True, IS_CAUSAL, False,
                        mma_layout, mma_offs_n_col, mma_offs_m_row,
                        BALANCE_CAUSAL_WAVES,
                    )
                    p_dot = gl.convert_layout(
                        p.to(q_dot.dtype), p_dot_layout)
                    v_dot = sc_lr(v_smem.index(slot), v_dot_layout)
                    acc = sc_dot_pv(acc, p_dot, v_dot)

        if WAVE_SKIP_CAUSAL and chunk_start + 2 < num_fallback:
            # Predicated waves can finish a chunk at different times.  Do not
            # let a skipped wave reuse either LDS slot while another wave is
            # still consuming the old K/V tile.
            gl.barrier()

    return acc, l_i, m_i


@gluon.jit
def attn_fwd_inner_full2_lazy(
    acc, l_i, m_i, q_dot, k_base, v_base,
    stride_kn, stride_kk, stride_vk, stride_vn,
    block_start, kt_smem, v_smem,
    qk_scale: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    kt_async_layout: gl.constexpr, v_async_layout: gl.constexpr,
    kt_dot_layout: gl.constexpr, p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr, mma_layout: gl.constexpr,
    mma_offs_m_row: gl.constexpr,
    SKIP_ENTRY_WAIT: gl.constexpr,
):
    """Process an aligned two-block full prefix with the BM128 lazy frame."""
    SCALE_ON_Q: gl.constexpr = qk_scale == 1.0
    if not SKIP_ENTRY_WAIT:
        cdna4_async.wait_group(0)
    kt_slot0 = kt_smem.index(0)
    kt_slot1 = kt_smem.index(1)
    v_slot0 = v_smem.index(0)
    v_slot1 = v_smem.index(1)
    kt_ad: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_an: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    kt_off = (
        gl.arange(0, BLOCK_DMODEL, layout=kt_ad)[:, None] * stride_kk
        + gl.arange(0, BLOCK_N, layout=kt_an)[None, :] * stride_kn
    )
    v_an: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    v_ad: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_off = (
        gl.arange(0, BLOCK_N, layout=v_an)[:, None] * stride_vk
        + gl.arange(0, BLOCK_DMODEL, layout=v_ad)[None, :] * stride_vn
    )
    kt_step = BLOCK_N * stride_kn
    v_step = BLOCK_N * stride_vk
    sc_issue_async_unmasked(
        kt_slot0, k_base + block_start * kt_step, kt_off)
    sc_issue_async_unmasked(
        v_slot0, v_base + block_start * v_step, v_off)
    sc_issue_async_unmasked(
        kt_slot1, k_base + (block_start + 1) * kt_step, kt_off)
    sc_issue_async_unmasked(
        v_slot1, v_base + (block_start + 1) * v_step, v_off)

    # The first K/V pair is readable while the second remains in flight.
    cdna4_async.wait_group(2)
    kt_dot = sc_lr(kt_slot0, kt_dot_layout)
    qk = compute_dot1_qk(q_dot, kt_dot, BLOCK_M, BLOCK_N, mma_layout)
    m_run, p_0123, p_4, qk_5, qk_6, qk_7, _, _ = sc_vec1_lazy(
        qk, m_i, qk_scale, SCALE_ON_Q, 4)
    # The prefix starts at m=-inf with an empty accumulator, so its first
    # correction is exactly zero and can be materialized directly.
    l_i = gl.zeros([BLOCK_M], dtype=gl.float32, layout=mma_offs_m_row)
    l_i, p_dot = sc_vec2_lazy(
        l_i, p_0123, p_4, qk_5, qk_6, qk_7, m_run,
        p_dot_layout, q_dot.dtype, qk_scale, SCALE_ON_Q, 4,
        BLOCK_M == 256)
    v_dot = sc_lr(v_slot0, v_dot_layout)
    acc = sc_dot_pv(acc, p_dot, v_dot)

    cdna4_async.wait_group(0)
    kt_dot = sc_lr(kt_slot1, kt_dot_layout)
    qk = compute_dot1_qk(q_dot, kt_dot, BLOCK_M, BLOCK_N, mma_layout)
    m_run, p_0123, p_4, qk_5, qk_6, qk_7, delta, advance = sc_vec1_lazy(
        qk, m_run, qk_scale, SCALE_ON_Q, 4)
    acc, l_i = sc_rescale_lazy(acc, l_i, delta, advance)
    l_i, p_dot = sc_vec2_lazy(
        l_i, p_0123, p_4, qk_5, qk_6, qk_7, m_run,
        p_dot_layout, q_dot.dtype, qk_scale, SCALE_ON_Q, 4,
        BLOCK_M == 256)
    v_dot = sc_lr(v_slot1, v_dot_layout)
    acc = sc_dot_pv(acc, p_dot, v_dot)
    return acc, l_i, m_run


# ---------------------------------------------------------------------------
# matched rotated 4-cluster pipelined inner loop
# ---------------------------------------------------------------------------

@gluon.jit
def attn_fwd_inner_pipelined(
    acc, l_i, m_i, q_dot, k_base, v_base, start_m,
    stride_kn, stride_kk, stride_vk, stride_vn,
    block_start, block_end,
    kt_smem, kt_smem1, v_smem,
    qk_scale: gl.constexpr,
    MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,
    MASK_STEPS: gl.constexpr, IS_CAUSAL: gl.constexpr,
    BALANCE_CAUSAL_WAVES: gl.constexpr,
    SKIP_ENTRY_WAIT: gl.constexpr,
    SEPARATE_K_SLOTS: gl.constexpr,
    CHAIN_BF16_ROWSUM: gl.constexpr,
    kt_async_layout: gl.constexpr, v_async_layout: gl.constexpr,
    kt_dot_layout: gl.constexpr, p_dot_layout: gl.constexpr, v_dot_layout: gl.constexpr,
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr, mma_offs_m_row: gl.constexpr,
):
    """
    4-cluster pipelined inner attention loop.

    ``MASK_STEPS`` (constexpr) selects masking: when False the loop is the exact
    unmasked schedule (the FMA-friendly softmax via the unmasked branch of
    sc_vec1, and unmasked global loads), so its LLIR is byte-identical to the
    unmasked-only version after dead-code elimination. When True the per-tile
    Q*K^T scores are
    scaled and masked (causal + K-bound) before the softmax max/exp2, and the
    global K/V loads are masked, exactly mirroring ``compute_softmax`` -- the
    schedule (cluster shape, prefetch depth, iglp hints) is unchanged.

    Pipeline (4 stages, 0..3; stage 0 = work producing THIS iteration's output).
    The whole softmax numerator (new max + the big exp2 burst, VEC1) is rotated
    one stage ahead so it is emitted AFTER the P*V MFMA and feeds the *next*
    iteration's P*V across the loop back-edge (exactly like the reference), instead of
    feeding the same iteration:

      dot_pv  s0   VEC2 s0   LRV  s0       (this tile's output: sum/acc/cast + PV)
      dot_qk  s1   VEC1 s1                 (next tile's QK, new max + exp burst)
      LRK     s2   ACV  s2                 (K read + V prefetch, 2 ahead)
      ACK     s3                           (K prefetch, 3 ahead)

    Pipeline DEPTH is 4 but N-buffering is only 2 (double-buffered LDS for both
    K and V): the other stages are carried in registers and in-flight global
    loads (ACK).

    Loop-carried state into iteration i:
      kt_dot  = K regs for tile i+1   (from LRK[i+1] last iter)
      m_run   = m_new[i]              (running max through tile i; from VEC1 last iter)
      generic path: p_c=p[i], alpha_c=alpha[i]
      lazy path: split p/qk fragments plus score delta and advance predicate
      acc, l_i = running accumulators
    """
    if not SKIP_ENTRY_WAIT:
        cdna4_async.wait_group(0)
    if SEPARATE_K_SLOTS:
        kt_slot0 = kt_smem.index(0)
        kt_slot1 = kt_smem1.index(0)
    else:
        kt_slot0 = kt_smem.index(0)
        kt_slot1 = kt_smem.index(1)
    v_slot0 = v_smem.index(0)
    v_slot1 = v_smem.index(1)

    BUF_DEPTH: gl.constexpr = 2
    # Intended steady-state async depth: keep 2*BUF_DEPTH-2 == 2 commit groups in
    # flight, so a wait_group(2) before each LDS read drains exactly the tile being
    # read (the oldest of 3 outstanding). The two loop reads use WAIT_LOOP-1 though:
    # the LLVM backend derives a too-loose s_waitcnt vmcnt from wait_group(2) under
    # this kernel's register pressure, letting an LDS ds_read race ahead of its
    # global->LDS async copy. Waiting for one fewer group forces a tight enough vmcnt
    # (the extra-drained group is not yet needed) and costs no measured performance.
    WAIT_LOOP: gl.constexpr = 2 * BUF_DEPTH - 2  # == 2
    CAST_P_FIRST: gl.constexpr = (
        (not IS_CAUSAL or MASK_STEPS)
        and BLOCK_DMODEL == 128 and BLOCK_M >= 128 and BLOCK_N == 64
    )
    USE_LAZY_RESCALE: gl.constexpr = (
        HAS_WARP_PREDICATE and not MASK_STEPS
        and BLOCK_N == 64 and BLOCK_DMODEL == 128
        and (
            (BLOCK_M == 256 and gl.num_warps() == 8)
            or (IS_CAUSAL
                and BLOCK_M == 128 and gl.num_warps() == 4)
        )
        # The paired lazy loop derives its drain parity from the static shape.
        # Keep ragged tails on the general exact-rescale path.
        and MAX_SEQLENS_Q % BLOCK_M == 0
        and MAX_SEQLENS_K % BLOCK_N == 0
    )
    USE_ONE_TILE_PREFIX_DRAIN: gl.constexpr = (
        USE_LAZY_RESCALE and IS_CAUSAL
        and MAX_SEQLENS_Q == MAX_SEQLENS_K
        and (
            MAX_SEQLENS_K == 8 * BLOCK_N
            or MAX_SEQLENS_K == 16 * BLOCK_N
        )
        and BLOCK_M == 128 and BLOCK_N == 64
        and BLOCK_DMODEL == 128 and gl.num_warps() == 4
    )
    USE_ONE_TILE_FOUR_SLOT_PREFIX_DRAIN: gl.constexpr = (
        USE_LAZY_RESCALE and IS_CAUSAL
        and MAX_SEQLENS_Q == MAX_SEQLENS_K
        and MAX_SEQLENS_K == 2048
        and BLOCK_M == 256 and BLOCK_N == 64
        and BLOCK_DMODEL == 128 and gl.num_warps() == 8
    )
    # Selected BF16 N512 handoffs only protect completed LDS reads from a
    # subsequent overwrite.  Wait for those reads and rendezvous the waves
    # without imposing the visibility-fence schedule needed at true
    # producer/consumer handoffs.  FP16 and longer rows retain gl.barrier():
    # direct A/B shows that their compiler schedules are faster that way.
    BF16_LIGHTWEIGHT_WAR_BARRIER: gl.constexpr = (
        IS_CAUSAL and MAX_SEQLENS_Q == MAX_SEQLENS_K
        and MAX_SEQLENS_K == 8 * BLOCK_N
        and BLOCK_M == 128 and BLOCK_N == 64
        and BLOCK_DMODEL == 128 and gl.num_warps() == 4
        and q_dot.dtype == gl.bfloat16
    )
    if USE_LAZY_RESCALE:
        # This path is only entered for the initial unmasked prefix
        # (block_start=0, m_i=-inf).  Its first correction would multiply both
        # state tensors by zero, so materialize that equivalent initial state
        # directly and omit the correction from the prologue below.
        l_i = gl.zeros(
            [BLOCK_M], dtype=gl.float32, layout=mma_offs_m_row)
    LAZY_VEC1_SPLIT: gl.constexpr = (
        4 if IS_CAUSAL and BLOCK_M == 128 and gl.num_warps() == 4
        else 5
    )
    SCALE_ON_Q: gl.constexpr = qk_scale == 1.0
    STEP_PV_VEC1: gl.constexpr = (
        USE_LAZY_RESCALE and BLOCK_N == 64 and BLOCK_DMODEL == 128
        and (
            (
                BLOCK_M == 256 and gl.num_warps() == 8
                # BF16 causal rows need at least 64 K/V tiles to amortize the
                # extra score lifetime. FP16 and non-causal rows win from the
                # first BM256 schedule onward.
                and (
                    not IS_CAUSAL or q_dot.dtype == gl.float16
                    or MAX_SEQLENS_K // BLOCK_N >= 64
                )
            )
            or (
                IS_CAUSAL and BLOCK_M == 128 and gl.num_warps() == 4
                and MAX_SEQLENS_Q == MAX_SEQLENS_K
                and MAX_SEQLENS_K == 8 * BLOCK_N
            )
        )
    )
    STEP_QK_VEC2: gl.constexpr = (
        USE_LAZY_RESCALE and BLOCK_M == 256 and BLOCK_N == 64
        and BLOCK_DMODEL == 128 and gl.num_warps() == 8
        and q_dot.dtype == gl.float16
        and (
            (not IS_CAUSAL and MAX_SEQLENS_K >= 2048)
            or (IS_CAUSAL and MAX_SEQLENS_K >= 4096)
        )
    )
    # Keep the P*V output as two persistent half-width accumulators across the
    # hot loop (D32 subtiles for D64, D64 subtiles for D128). This exposes one
    # softmax VEC1 group between independent MFMA halves and removes repeated
    # full-width accumulator handoffs. It benefits the 8-warp (two waves/SIMD)
    # schedules; the 4-warp variant has no second wave to overlap.
    SPLIT_PV: gl.constexpr = (
        not MASK_STEPS and (BLOCK_DMODEL == 64 or BLOCK_DMODEL == 128)
        and BLOCK_M >= 128 and gl.num_warps() == 8 and not USE_LAZY_RESCALE
    )
    # Score masking and copy bounds are independent. An aligned causal tail
    # still needs its triangular score mask, but every selected K/V tile and D
    # lane is in bounds, so its DMA can use the full unmasked copy path.
    MASK_LOADS: gl.constexpr = (
        MASK_STEPS
        and (MAX_SEQLENS_K % BLOCK_N != 0 or ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL)
    )

    # Full aligned copies have a fixed intra-tile pattern. Hoist it once and
    # advance only the base pointer in the loop; expressing start_n as part of
    # every lane's offset otherwise leaves address-vector arithmetic in each
    # ACK/ACV stage after inlining.
    if USE_LAZY_RESCALE:
        kt_ad: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
        kt_an: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
        kt_off = (
            gl.arange(0, BLOCK_DMODEL, layout=kt_ad)[:, None] * stride_kk
            + gl.arange(0, BLOCK_N, layout=kt_an)[None, :] * stride_kn
        )
        v_an: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
        v_ad: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
        v_off = (
            gl.arange(0, BLOCK_N, layout=v_an)[:, None] * stride_vk
            + gl.arange(0, BLOCK_DMODEL, layout=v_ad)[None, :] * stride_vn
        )
        kt_step = BLOCK_N * stride_kn
        v_step = BLOCK_N * stride_vk

    # -- Prologue ----------------------------------------------------------
    # Prime the rotated pipeline for output tile 0: compute the FULL ahead-work
    # for tile 0 (qk[0], m_new[0], and the exp2 burst p[0]/alpha[0]) and the K
    # regs for tile 1, plus stage K[0..2] / V[0..1] into LDS. K is prefetched
    # 3-ahead so three K tiles (0,1,2) must be staged into the 2 K slots -- slot
    # 0 is reused for K[2] after LRK[0] reads K[0] (guarded by a barrier).
    #
    # Commit order: K0, V0, K1, (barrier) K2, V1  ->  end pending {K2, V1},
    # matching the loop's steady-state entry condition.
    b0 = block_start
    if USE_LAZY_RESCALE:
        sc_issue_async_unmasked(kt_slot0, k_base + (b0 + 0) * kt_step, kt_off)  # ACK[0]
        sc_issue_async_unmasked(v_slot0, v_base + (b0 + 0) * v_step, v_off)     # ACV[0]
        sc_issue_async_unmasked(kt_slot1, k_base + (b0 + 1) * kt_step, kt_off)  # ACK[1]
    else:
        issue_async_load_k(kt_slot0, k_base, (b0 + 0) * BLOCK_N,
                           stride_kn, stride_kk, MAX_SEQLENS_K, MASK_LOADS, MAX_SEQLENS_K, False,
                           BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL, kt_async_layout)  # ACK[0]
        issue_async_load_v(v_slot0, v_base, (b0 + 0) * BLOCK_N,
                           stride_vk, stride_vn, MAX_SEQLENS_K, MASK_LOADS, MAX_SEQLENS_K, False,
                           BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL, v_async_layout)   # ACV[0]
        issue_async_load_k(kt_slot1, k_base, (b0 + 1) * BLOCK_N,
                           stride_kn, stride_kk, MAX_SEQLENS_K, MASK_LOADS, MAX_SEQLENS_K, False,
                           BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL, kt_async_layout)  # ACK[1]

    if MASK_STEPS:
        n0 = ((b0 + 0) * BLOCK_N).to(tl.int32)
    else:
        n0 = 0
    cdna4_async.wait_group(2)                                       # K[0] complete
    kt0 = sc_lr(kt_slot0, kt_dot_layout)                            # LRK[0] -> K regs tile 0
    qk = compute_dot1_qk(q_dot, kt0, BLOCK_M, BLOCK_N, mma_layout)  # dot_qk[0] -> qk[0]
    if USE_LAZY_RESCALE:
        m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, delta_c, advance_c = sc_vec1_lazy(
            qk, m_i, qk_scale, SCALE_ON_Q, LAZY_VEC1_SPLIT)
    else:
        m_run, p_c, alpha_c = sc_vec1(
            qk, m_i, n0, start_m, qk_scale,
            MASK_STEPS, IS_CAUSAL, MAX_SEQLENS_Q, MAX_SEQLENS_K,
            BLOCK_M, BLOCK_N, BALANCE_CAUSAL_WAVES,
            mma_layout, mma_offs_n_col, mma_offs_m_row)

    # Keep the K-slot WAR handoff after every QK fragment has consumed LRK[0].
    # A plain source barrier has no SSA edge to qk, so the backend can otherwise
    # place it between the QK MFMAs and expose the LDS-read drain.  The anchor
    # repays its scheduling cost once the scheduled K range spans at least
    # sixteen tiles; the shorter class route keeps the stock schedule.
    ANCHOR_QK_WAR: gl.constexpr = (
        USE_LAZY_RESCALE
        and BLOCK_M == 128 and gl.num_warps() == 4
        and MAX_SEQLENS_K // BLOCK_N >= 16)
    RELAXED_QK_WAR: gl.constexpr = (
        USE_LAZY_RESCALE and IS_CAUSAL
        and MAX_SEQLENS_Q == MAX_SEQLENS_K
        and MAX_SEQLENS_K == 8 * BLOCK_N
        and BLOCK_M == 128 and gl.num_warps() == 4
        and q_dot.dtype == gl.float16)
    if RELAXED_QK_WAR:
        # The QK result is the last consumer of the old K0 LDS slot.  Tie both
        # N32 hardware-MFMA results to an execution rendezvous so K2 cannot
        # overwrite the slot early, while avoiding the broader memory-fence
        # schedule of gl.barrier().
        qk = sc_qk_war_barrier_relaxed(qk)                           # WAR: LRK[0] vs K[2]
    elif ANCHOR_QK_WAR:
        if q_dot.dtype == gl.float16:
            qk = sc_qk_war_barrier_relaxed(qk)                     # WAR: QK[0] result vs K[2] write
        else:
            qk = sc_qk_war_barrier_pack32(qk)                      # WAR: LRK[0] ds_read vs K[2] write
    else:
        gl.barrier()
    if USE_LAZY_RESCALE:
        sc_issue_async_unmasked(kt_slot0, k_base + (b0 + 2) * kt_step, kt_off)  # ACK[2]
    else:
        issue_async_load_k(kt_slot0, k_base, (b0 + 2) * BLOCK_N,
                           stride_kn, stride_kk, MAX_SEQLENS_K, MASK_LOADS, MAX_SEQLENS_K, False,
                           BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL, kt_async_layout)  # ACK[2] (slot0 reuse)
    cdna4_async.wait_group(1)                                       # K[1] complete
    kt_dot = sc_lr(kt_slot1, kt_dot_layout)                         # LRK[1] -> K regs tile 1
    if USE_LAZY_RESCALE:
        sc_issue_async_unmasked(v_slot1, v_base + (b0 + 1) * v_step, v_off)  # ACV[1]
    else:
        issue_async_load_v(v_slot1, v_base, (b0 + 1) * BLOCK_N,
                           stride_vk, stride_vn, MAX_SEQLENS_K, MASK_LOADS, MAX_SEQLENS_K, False,
                           BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL, v_async_layout)   # ACV[1]

    if SPLIT_PV:
        acc_sub0, acc_sub1 = sc_split_cols(acc)

    # -- Main loop (full rotated body) -------------------------------------
    # Runs output tiles [block_start, block_end-3): the last full iteration
    # whose K prefetch (ACK[i+3]) is still in bounds. The final three tiles are
    # drained below without out-of-bounds global prefetch.
    if USE_LAZY_RESCALE:
        # Pair adjacent iterations so the ping-pong LDS slots become compile-time
        # constants. Besides removing modulo/slot selection from the hot loop,
        # the wider body gives the LLIR scheduler both halves of the K/V cadence
        # at once. MASK_STEPS=False implies an exact, full K range starting at 0.
        NUM_BLOCKS: gl.constexpr = MAX_SEQLENS_K // BLOCK_N
        DRAIN_TILES: gl.constexpr = (
            1 if (USE_ONE_TILE_PREFIX_DRAIN
                  or USE_ONE_TILE_FOUR_SLOT_PREFIX_DRAIN)
            else 3)
        ODD_TAIL: gl.constexpr = (NUM_BLOCKS - DRAIN_TILES) % 2 == 1
        main_loop_pairs = (block_end - DRAIN_TILES - block_start) // 2
        # The heavy short-causal BF16 class has exactly two pair iterations.
        # Exposing both at once removes the loop backedge and lets max-ILP
        # eliminate three barriers without changing the FP16 or longer-row
        # instruction streams.
        PAIR_LOOP_UNROLL: gl.constexpr = (
            2 if (IS_CAUSAL and MAX_SEQLENS_Q == 512
                  and q_dot.dtype == gl.bfloat16)
            else 1)
        for pair_idx in tl.range(
                0, main_loop_pairs, loop_unroll_factor=PAIR_LOOP_UNROLL):
            block_n = block_start + pair_idx * 2

            # Even tile: cur=0, next=1.
            with warp_pipeline_stage("dot1"):
                if STEP_QK_VEC2:
                    qk, l_i, p_dot = sc_dot_qk_step8_vec2(
                        q_dot, kt_dot, l_i,
                        p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, m_run,
                        p_dot_layout, qk_scale, SCALE_ON_Q, IS_CAUSAL,
                        mma_layout)
                else:
                    qk = compute_dot1_qk(
                        q_dot, kt_dot, BLOCK_M, BLOCK_N, mma_layout)
                    l_i, p_dot = sc_vec2_lazy(
                        l_i, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, m_run,
                        p_dot_layout, q_dot.dtype, qk_scale, SCALE_ON_Q,
                        LAZY_VEC1_SPLIT, CHAIN_BF16_ROWSUM)
            cdna4_async.wait_group(WAIT_LOOP - 1)
            with warp_pipeline_stage("mem1"):
                v_dot = sc_lr(v_slot0, v_dot_layout)
                sc_issue_async_unmasked(
                    kt_slot1, k_base + (block_n + 3) * kt_step, kt_off)
            with warp_pipeline_stage("dot2"):
                if STEP_PV_VEC1:
                    (acc, m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6,
                     qk_c_7, delta_c, advance_c) = sc_dot_pv_step4_vec1(
                        acc, p_dot, v_dot, qk, m_run, qk_scale,
                        SCALE_ON_Q, LAZY_VEC1_SPLIT, BLOCK_M == 128)
                else:
                    acc = sc_dot_pv(acc, p_dot, v_dot)
                    (m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6,
                     qk_c_7, delta_c, advance_c) = sc_vec1_lazy(
                        qk, m_run, qk_scale, SCALE_ON_Q,
                        LAZY_VEC1_SPLIT)
            cdna4_async.wait_group(WAIT_LOOP - 1)
            with warp_pipeline_stage("mem2"):
                kt_dot = sc_lr(kt_slot0, kt_dot_layout)
                sc_issue_async_unmasked(
                    v_slot0, v_base + (block_n + 2) * v_step, v_off)
                acc, l_i = sc_rescale_lazy(
                    acc, l_i, delta_c, advance_c)

            # Odd tile: cur=1, next=0.
            with warp_pipeline_stage("dot1"):
                if STEP_QK_VEC2:
                    qk, l_i, p_dot = sc_dot_qk_step8_vec2(
                        q_dot, kt_dot, l_i,
                        p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, m_run,
                        p_dot_layout, qk_scale, SCALE_ON_Q, IS_CAUSAL,
                        mma_layout)
                else:
                    qk = compute_dot1_qk(
                        q_dot, kt_dot, BLOCK_M, BLOCK_N, mma_layout)
                    l_i, p_dot = sc_vec2_lazy(
                        l_i, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, m_run,
                        p_dot_layout, q_dot.dtype, qk_scale, SCALE_ON_Q,
                        LAZY_VEC1_SPLIT, CHAIN_BF16_ROWSUM)
            cdna4_async.wait_group(WAIT_LOOP - 1)
            with warp_pipeline_stage("mem1"):
                v_dot = sc_lr(v_slot1, v_dot_layout)
                sc_issue_async_unmasked(
                    kt_slot0, k_base + (block_n + 4) * kt_step, kt_off)
            with warp_pipeline_stage("dot2"):
                if STEP_PV_VEC1:
                    (acc, m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6,
                     qk_c_7, delta_c, advance_c) = sc_dot_pv_step4_vec1(
                        acc, p_dot, v_dot, qk, m_run, qk_scale,
                        SCALE_ON_Q, LAZY_VEC1_SPLIT, BLOCK_M == 128)
                else:
                    acc = sc_dot_pv(acc, p_dot, v_dot)
                    (m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6,
                     qk_c_7, delta_c, advance_c) = sc_vec1_lazy(
                        qk, m_run, qk_scale, SCALE_ON_Q,
                        LAZY_VEC1_SPLIT)
            cdna4_async.wait_group(WAIT_LOOP - 1)
            with warp_pipeline_stage("mem2"):
                kt_dot = sc_lr(kt_slot1, kt_dot_layout)
                sc_issue_async_unmasked(
                    v_slot1, v_base + (block_n + 3) * v_step, v_off)
                acc, l_i = sc_rescale_lazy(
                    acc, l_i, delta_c, advance_c)

        # The paired loop leaves one even-slot tile when (NUM_BLOCKS - 3) is odd.
        if ODD_TAIL:
            block_n = block_start + main_loop_pairs * 2
            with warp_pipeline_stage("dot1"):
                if STEP_QK_VEC2:
                    qk, l_i, p_dot = sc_dot_qk_step8_vec2(
                        q_dot, kt_dot, l_i,
                        p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, m_run,
                        p_dot_layout, qk_scale, SCALE_ON_Q, IS_CAUSAL,
                        mma_layout)
                else:
                    qk = compute_dot1_qk(
                        q_dot, kt_dot, BLOCK_M, BLOCK_N, mma_layout)
                    l_i, p_dot = sc_vec2_lazy(
                        l_i, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, m_run,
                        p_dot_layout, q_dot.dtype, qk_scale, SCALE_ON_Q,
                        LAZY_VEC1_SPLIT, CHAIN_BF16_ROWSUM)
            cdna4_async.wait_group(WAIT_LOOP - 1)
            with warp_pipeline_stage("mem1"):
                v_dot = sc_lr(v_slot0, v_dot_layout)
                sc_issue_async_unmasked(
                    kt_slot1, k_base + (block_n + 3) * kt_step, kt_off)
            with warp_pipeline_stage("dot2"):
                if STEP_PV_VEC1:
                    (acc, m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6,
                     qk_c_7, delta_c, advance_c) = sc_dot_pv_step4_vec1(
                        acc, p_dot, v_dot, qk, m_run, qk_scale,
                        SCALE_ON_Q, LAZY_VEC1_SPLIT, BLOCK_M == 128)
                else:
                    acc = sc_dot_pv(acc, p_dot, v_dot)
                    (m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6,
                     qk_c_7, delta_c, advance_c) = sc_vec1_lazy(
                        qk, m_run, qk_scale, SCALE_ON_Q,
                        LAZY_VEC1_SPLIT)
            cdna4_async.wait_group(WAIT_LOOP - 1)
            with warp_pipeline_stage("mem2"):
                kt_dot = sc_lr(kt_slot0, kt_dot_layout)
                sc_issue_async_unmasked(
                    v_slot0, v_base + (block_n + 2) * v_step, v_off)
                acc, l_i = sc_rescale_lazy(
                    acc, l_i, delta_c, advance_c)
    else:
        for block_n in tl.range(block_start, block_end - 3):
            cur_slot = ((block_n - block_start) % BUF_DEPTH).to(tl.int32)
            nxt_slot = ((block_n + 1 - block_start) % BUF_DEPTH).to(tl.int32)
            ack_n = ((block_n + 3) * BLOCK_N).to(tl.int32)
            acv_n = ((block_n + 2) * BLOCK_N).to(tl.int32)
            if MASK_STEPS:
                ahead_n = ((block_n + 1) * BLOCK_N).to(tl.int32)
            else:
                ahead_n = 0

            with warp_pipeline_stage("dot1"):
                qk = compute_dot1_qk(q_dot, kt_dot, BLOCK_M, BLOCK_N, mma_layout)
                if SPLIT_PV:
                    acc_sub0, acc_sub1, l_i, p_dot = sc_vec2_split_acc(
                        acc_sub0, acc_sub1, l_i, p_c, alpha_c,
                        p_dot_layout, q_dot.dtype)
                else:
                    acc, l_i, p_dot = sc_vec2(
                        acc, l_i, p_c, alpha_c, p_dot_layout,
                        q_dot.dtype, CAST_P_FIRST)

            cdna4_async.wait_group(WAIT_LOOP - 1)
            with warp_pipeline_stage("mem1"):
                v_dot = sc_lr(v_smem.index(cur_slot), v_dot_layout)
                issue_async_load_k(
                    kt_smem.index(nxt_slot), k_base, ack_n,
                    stride_kn, stride_kk, MAX_SEQLENS_K, MASK_LOADS, MAX_SEQLENS_K, False,
                    BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL, kt_async_layout)

            with warp_pipeline_stage("dot2"):
                if SPLIT_PV:
                    v_sub0, v_sub1 = sc_split_cols(v_dot)
                    acc_sub0 = do_mma("mfma_cdna4", p_dot, v_sub0, acc_sub0)
                    m_run, p_c, alpha_c = sc_vec1(
                        qk, m_run, ahead_n, start_m, qk_scale,
                        MASK_STEPS, IS_CAUSAL, MAX_SEQLENS_Q, MAX_SEQLENS_K,
                        BLOCK_M, BLOCK_N, BALANCE_CAUSAL_WAVES,
                        mma_layout, mma_offs_n_col, mma_offs_m_row)
                    acc_sub1 = do_mma("mfma_cdna4", p_dot, v_sub1, acc_sub1)
                else:
                    acc = sc_dot_pv(acc, p_dot, v_dot)
                    m_run, p_c, alpha_c = sc_vec1(
                        qk, m_run, ahead_n, start_m, qk_scale,
                        MASK_STEPS, IS_CAUSAL, MAX_SEQLENS_Q, MAX_SEQLENS_K,
                        BLOCK_M, BLOCK_N, BALANCE_CAUSAL_WAVES, mma_layout,
                        mma_offs_n_col, mma_offs_m_row)

            cdna4_async.wait_group(WAIT_LOOP - 1)
            with warp_pipeline_stage("mem2"):
                if gl.num_warps() == 8:
                    # With two waves per SIMD, V can start its independent
                    # global-to-LDS copy while the current K operand is read.
                    issue_async_load_v(
                        v_smem.index(cur_slot), v_base, acv_n,
                        stride_vk, stride_vn, MAX_SEQLENS_K, MASK_LOADS,
                        MAX_SEQLENS_K, False, BLOCK_N, BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL, v_async_layout)
                    kt_dot = sc_lr(kt_smem.index(cur_slot), kt_dot_layout)
                else:
                    # Four-warp schedules have no second wave to cover the
                    # reordered K read and retain the original cadence.
                    kt_dot = sc_lr(kt_smem.index(cur_slot), kt_dot_layout)
                    issue_async_load_v(
                        v_smem.index(cur_slot), v_base, acv_n,
                        stride_vk, stride_vn, MAX_SEQLENS_K, MASK_LOADS,
                        MAX_SEQLENS_K, False, BLOCK_N, BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL, v_async_layout)

        if SPLIT_PV:
            acc = sc_concat_cols(acc_sub0, acc_sub1)

    if (USE_ONE_TILE_PREFIX_DRAIN
            or USE_ONE_TILE_FOUR_SLOT_PREFIX_DRAIN):
        # The extended odd tail has completed every prefix tile except n-1.
        # Its final mem2 leaves diagonal K0 resident in slot 0, starts
        # diagonal K1 in slot 1, and starts diagonal V0 in slot 0. Consume
        # p[n-1] with the now-ready V[n-1], then reuse that dead V slot for
        # diagonal V1. The short helper drains K1/V0 while V1 is in flight,
        # so no prefix/diagonal copies are reissued.
        if USE_ONE_TILE_FOUR_SLOT_PREFIX_DRAIN:
            # Prefix slots 0/1 already carry diagonal K0/K1 and V0 will be
            # committed by the final loop mem2 stage.  Start the two diagonal
            # pairs that own the otherwise-unused slots 2/3 while the final
            # prefix softmax/PV still has independent work to execute.
            for diagonal_slot in gl.static_range(2):
                sc_issue_async_unmasked(
                    kt_smem.index(diagonal_slot + 2),
                    k_base + (block_end + diagonal_slot + 2) * kt_step,
                    kt_off)
                sc_issue_async_unmasked(
                    v_smem.index(diagonal_slot + 2),
                    v_base + (block_end + diagonal_slot + 2) * v_step,
                    v_off)
        l_i, p_dot = sc_vec2_lazy(
            l_i, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, m_run,
            p_dot_layout, q_dot.dtype, qk_scale, SCALE_ON_Q,
            LAZY_VEC1_SPLIT, CHAIN_BF16_ROWSUM)
        v_dot = sc_lr(v_slot1, v_dot_layout)
        acc = sc_dot_pv(acc, p_dot, v_dot)
        RELAXED_PV_WAR: gl.constexpr = (
            IS_CAUSAL and MAX_SEQLENS_Q == MAX_SEQLENS_K
            and MAX_SEQLENS_K == 8 * BLOCK_N
            and BLOCK_M == 128 and BLOCK_N == 64
            and BLOCK_DMODEL == 128 and gl.num_warps() == 4
            and q_dot.dtype == gl.float16
        )
        if RELAXED_PV_WAR:
            acc = sc_pv_war_barrier_relaxed(acc)
        else:
            sc_war_barrier(BF16_LIGHTWEIGHT_WAR_BARRIER)
        sc_issue_async_unmasked(
            v_slot1, v_base + (block_end + 1) * v_step, v_off)
        return acc, l_i, m_run

    # -- Drain (last 3 output tiles, no OOB global prefetch) ---------------
    # After the loop: outputs [.., n-4] done; K[0..n-1] and V[0..n-2] in LDS
    # (V[n-1] still to load); carried kt_dot=K regs tile n-2 and
    # m_run=m_new[n-3]. The generic path carries p/alpha for n-3; the lazy path
    # carries split p/qk fragments plus its delta/predicate. Pending async work
    # is {V[n-3],K[n-1],V[n-2]}.
    PREFETCH_FOUR_SLOT_CAUSAL_DIAGONAL: gl.constexpr = (
        USE_LAZY_RESCALE and IS_CAUSAL
        and MAX_SEQLENS_Q == MAX_SEQLENS_K
        and BLOCK_M == 256 and BLOCK_N == 64
        and BLOCK_DMODEL == 128 and gl.num_warps() == 8
        and MAX_SEQLENS_K // BLOCK_N <= 32
    )
    PREFETCH_TWO_SLOT_CAUSAL_DIAGONAL: gl.constexpr = (
        USE_LAZY_RESCALE and IS_CAUSAL and not MASK_STEPS
        and MAX_SEQLENS_Q == MAX_SEQLENS_K
        and BLOCK_M == 128 and BLOCK_N == 64
        and BLOCK_DMODEL == 128 and gl.num_warps() == 4
        and MAX_SEQLENS_K // BLOCK_N >= 8
    )
    DRAIN_STEP_PV_VEC1: gl.constexpr = (
        STEP_PV_VEC1 and not IS_CAUSAL
        and MAX_SEQLENS_K >= 8 * BLOCK_N
        and BLOCK_M == 256 and BLOCK_N == 64
        and BLOCK_DMODEL == 128 and gl.num_warps() == 8
    )
    nm3 = block_end - 3
    nm2 = block_end - 2
    nm1 = block_end - 1
    s_nm3 = ((nm3 - block_start) % BUF_DEPTH).to(tl.int32)
    s_nm2 = ((nm2 - block_start) % BUF_DEPTH).to(tl.int32)
    s_nm1 = ((nm1 - block_start) % BUF_DEPTH).to(tl.int32)
    v_nm3 = v_smem.index(s_nm3)
    v_nm2 = v_smem.index(s_nm2)
    v_nm1 = v_smem.index(s_nm1)
    if SEPARATE_K_SLOTS:
        # The aligned non-causal D128 split path always has an even number of
        # BN64 blocks, so n-1 resides in the odd slot.
        kt_nm1 = kt_slot1
    else:
        kt_nm1 = kt_smem.index(s_nm1)
    if MASK_STEPS:
        nm2_n = (nm2 * BLOCK_N).to(tl.int32)
        nm1_n = (nm1 * BLOCK_N).to(tl.int32)
    else:
        nm2_n = 0
        nm1_n = 0

    if PREFETCH_FOUR_SLOT_CAUSAL_DIAGONAL:
        # Slots 2/3 are outside the two-slot prefix ring.  Prefetch diagonal
        # tiles 2/3 across the complete three-tile drain; tiles 0/1 enter
        # slots 0/1 only after their final prefix reads below.
        for diagonal_slot in gl.static_range(2):
            sc_issue_async_unmasked(
                kt_smem.index(diagonal_slot + 2),
                k_base + (block_end + diagonal_slot + 2) * kt_step, kt_off)
            sc_issue_async_unmasked(
                v_smem.index(diagonal_slot + 2),
                v_base + (block_end + diagonal_slot + 2) * v_step, v_off)

    # output tile n-3 (also issues the final V prefetch, ACV[n-1])
    qk = compute_dot1_qk(q_dot, kt_dot, BLOCK_M, BLOCK_N, mma_layout)       # dot_qk[n-2]
    if PREFETCH_FOUR_SLOT_CAUSAL_DIAGONAL:
        cdna4_async.wait_group(6)
    else:
        cdna4_async.wait_group(2)                                       # V[n-3] complete
    if USE_LAZY_RESCALE:
        l_i, p_dot = sc_vec2_lazy(
            l_i, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, m_run,
            p_dot_layout, q_dot.dtype, qk_scale, SCALE_ON_Q,
            LAZY_VEC1_SPLIT, CHAIN_BF16_ROWSUM)
    else:
        acc, l_i, p_dot = sc_vec2(
            acc, l_i, p_c, alpha_c, p_dot_layout,
            q_dot.dtype, CAST_P_FIRST)  # VEC2[n-3]
    v_dot = sc_lr(v_nm3, v_dot_layout)                                  # LRV[n-3]
    if DRAIN_STEP_PV_VEC1:
        (acc, m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6,
         qk_c_7, delta_c, advance_c) = sc_dot_pv_step4_vec1(
            acc, p_dot, v_dot, qk, m_run, qk_scale,
            SCALE_ON_Q, LAZY_VEC1_SPLIT, BLOCK_M == 128)
    elif USE_LAZY_RESCALE:
        acc = sc_dot_pv(acc, p_dot, v_dot)
    else:
        acc = sc_dot_pv(acc, p_dot, v_dot)                              # dot_pv[n-3]
    EARLY_FINAL_V_RESULT_WAR: gl.constexpr = (
        USE_LAZY_RESCALE and IS_CAUSAL
        and q_dot.dtype == gl.float16
        and MAX_SEQLENS_Q == MAX_SEQLENS_K
        and MAX_SEQLENS_K // BLOCK_N >= 128
        and BLOCK_M == 256 and BLOCK_N == 64
        and BLOCK_DMODEL == 128 and gl.num_warps() == 8)
    if EARLY_FINAL_V_RESULT_WAR:
        # The completed P-by-V result proves that every old V-slot LDS read
        # has issued. Release the slot and start V[n-1] while the independent
        # softmax/rescale tail below is still executing.
        acc = sc_pv_war_barrier_relaxed(acc)
        sc_issue_async_unmasked(
            v_nm1, v_base + nm1 * v_step, v_off)                       # ACV[n-1]
    if USE_LAZY_RESCALE:
        if not DRAIN_STEP_PV_VEC1:
            (m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7,
             delta_c, advance_c) = sc_vec1_lazy(
                qk, m_run, qk_scale, SCALE_ON_Q, LAZY_VEC1_SPLIT)
    else:
        m_run, p_c, alpha_c = sc_vec1(
            qk, m_run, nm2_n, start_m, qk_scale,
            MASK_STEPS, IS_CAUSAL, MAX_SEQLENS_Q, MAX_SEQLENS_K,
            BLOCK_M, BLOCK_N, BALANCE_CAUSAL_WAVES,
            mma_layout, mma_offs_n_col, mma_offs_m_row)
    if USE_LAZY_RESCALE:
        acc, l_i = sc_rescale_lazy(
            acc, l_i, delta_c, advance_c)
    if not EARLY_FINAL_V_RESULT_WAR:
        sc_war_barrier(BF16_LIGHTWEIGHT_WAR_BARRIER)                  # WAR: LRV[n-3] vs V[n-1] write
        if USE_LAZY_RESCALE:
            sc_issue_async_unmasked(v_nm1, v_base + nm1 * v_step, v_off)  # ACV[n-1]
        else:
            issue_async_load_v(v_nm1, v_base, (nm1 * BLOCK_N).to(tl.int32),
                               stride_vk, stride_vn, MAX_SEQLENS_K, MASK_LOADS, MAX_SEQLENS_K, False,
                               BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL, v_async_layout)   # ACV[n-1]
    if PREFETCH_FOUR_SLOT_CAUSAL_DIAGONAL:
        cdna4_async.wait_group(6)
    else:
        cdna4_async.wait_group(2)                                       # K[n-1] complete
    kt_dot = sc_lr(kt_nm1, kt_dot_layout)                               # LRK[n-1] -> K regs tile n-1
    if PREFETCH_TWO_SLOT_CAUSAL_DIAGONAL:
        # K[n-1] was the final prefix read from slot 1. Reuse that K slot for
        # diagonal K1 while V[n-1] remains independently live in its V slot.
        sc_war_barrier(BF16_LIGHTWEIGHT_WAR_BARRIER)
        sc_issue_async_unmasked(
            kt_slot1, k_base + (block_end + 1) * kt_step, kt_off)

    # output tile n-2
    qk = compute_dot1_qk(q_dot, kt_dot, BLOCK_M, BLOCK_N, mma_layout)       # dot_qk[n-1]
    if PREFETCH_FOUR_SLOT_CAUSAL_DIAGONAL:
        cdna4_async.wait_group(5)
    elif PREFETCH_TWO_SLOT_CAUSAL_DIAGONAL:
        cdna4_async.wait_group(2)
    else:
        cdna4_async.wait_group(1)                                       # V[n-2] complete
    if USE_LAZY_RESCALE:
        l_i, p_dot = sc_vec2_lazy(
            l_i, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, m_run,
            p_dot_layout, q_dot.dtype, qk_scale, SCALE_ON_Q,
            LAZY_VEC1_SPLIT, CHAIN_BF16_ROWSUM)
    else:
        acc, l_i, p_dot = sc_vec2(
            acc, l_i, p_c, alpha_c, p_dot_layout,
            q_dot.dtype, CAST_P_FIRST)  # VEC2[n-2]
    v_dot = sc_lr(v_nm2, v_dot_layout)                                  # LRV[n-2]
    if PREFETCH_TWO_SLOT_CAUSAL_DIAGONAL:
        # Slot 0 is dead for both prefix operands after LRV[n-2]. Start the
        # complete diagonal tile 0 pair before the penultimate P*V chain.
        gl.barrier()
        sc_issue_async_unmasked(
            kt_slot0, k_base + block_end * kt_step, kt_off)
        sc_issue_async_unmasked(
            v_slot0, v_base + block_end * v_step, v_off)
    if USE_LAZY_RESCALE:
        if DRAIN_STEP_PV_VEC1:
            (acc, m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6,
             qk_c_7, delta_c, advance_c) = sc_dot_pv_step4_vec1(
                acc, p_dot, v_dot, qk, m_run, qk_scale,
                SCALE_ON_Q, LAZY_VEC1_SPLIT, BLOCK_M == 128)
        else:
            acc = sc_dot_pv(acc, p_dot, v_dot)
    else:
        acc = sc_dot_pv(acc, p_dot, v_dot)                              # dot_pv[n-2]
    if USE_LAZY_RESCALE:
        if not DRAIN_STEP_PV_VEC1:
            (m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7,
             delta_c, advance_c) = sc_vec1_lazy(
                qk, m_run, qk_scale, SCALE_ON_Q, LAZY_VEC1_SPLIT)
    else:
        m_run, p_c, alpha_c = sc_vec1(
            qk, m_run, nm1_n, start_m, qk_scale,
            MASK_STEPS, IS_CAUSAL, MAX_SEQLENS_Q, MAX_SEQLENS_K,
            BLOCK_M, BLOCK_N, BALANCE_CAUSAL_WAVES,
            mma_layout, mma_offs_n_col, mma_offs_m_row)
    if USE_LAZY_RESCALE:
        acc, l_i = sc_rescale_lazy(
            acc, l_i, delta_c, advance_c)

    # output tile n-1 (final full-prefix tile).  For the aligned causal BM256
    # path, slots 2/3 are unused by the two-slot prefix ring.  Start the first
    # two diagonal K/V pairs there after the prefix VMEM drain so their latency
    # overlaps this final softmax/PV.
    if PREFETCH_TWO_SLOT_CAUSAL_DIAGONAL:
        cdna4_async.wait_group(3)
    else:
        cdna4_async.wait_group(0)                                       # V[n-1] complete
    if USE_LAZY_RESCALE:
        l_i, p_dot = sc_vec2_lazy(
            l_i, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, m_run,
            p_dot_layout, q_dot.dtype, qk_scale, SCALE_ON_Q,
            LAZY_VEC1_SPLIT, CHAIN_BF16_ROWSUM)
    else:
        acc, l_i, p_dot = sc_vec2(
            acc, l_i, p_c, alpha_c, p_dot_layout,
            q_dot.dtype, CAST_P_FIRST)  # VEC2[n-1]
    v_dot = sc_lr(v_nm1, v_dot_layout)                                  # LRV[n-1]
    if PREFETCH_FOUR_SLOT_CAUSAL_DIAGONAL:
        gl.barrier()
        for diagonal_slot in gl.static_range(2):
            sc_issue_async_unmasked(
                kt_smem.index(diagonal_slot),
                k_base + (block_end + diagonal_slot) * kt_step, kt_off)
            sc_issue_async_unmasked(
                v_smem.index(diagonal_slot),
                v_base + (block_end + diagonal_slot) * v_step, v_off)
    if PREFETCH_TWO_SLOT_CAUSAL_DIAGONAL:
        # Complete slot 1 only after the prefix's final V read. The four
        # diagonal copies now overlap the last prefix P*V below.
        sc_war_barrier(BF16_LIGHTWEIGHT_WAR_BARRIER)
        sc_issue_async_unmasked(
            v_slot1, v_base + (block_end + 1) * v_step, v_off)
    if USE_LAZY_RESCALE:
        acc = sc_dot_pv(acc, p_dot, v_dot)
    else:
        acc = sc_dot_pv(acc, p_dot, v_dot)                              # dot_pv[n-1]

    return acc, l_i, m_run


@aggregate
class AttentionInnerContext:
    q_dot: gl.tensor
    k_base: gl.tensor
    v_base: gl.tensor
    offs_n: gl.tensor
    offs_d: gl.tensor
    kt_offs_d: gl.tensor
    kt_offs_n: gl.tensor
    start_m: gl.tensor
    stride_kn: gl.tensor | gl.constexpr
    stride_kk: gl.tensor | gl.constexpr
    stride_vk: gl.tensor | gl.constexpr
    stride_vn: gl.tensor | gl.constexpr
    kt_smem: gl.shared_memory_descriptor
    kt_smem1: gl.shared_memory_descriptor
    v_smem: gl.shared_memory_descriptor
    seqlen_q: gl.tensor | gl.constexpr
    seqlen_k: gl.tensor | gl.constexpr
    qk_scale: gl.constexpr
    MAX_SEQLENS_Q: gl.constexpr
    MAX_SEQLENS_K: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_DMODEL: gl.constexpr
    ACTUAL_BLOCK_DMODEL: gl.constexpr
    NUM_STAGES: gl.constexpr
    IS_CAUSAL: gl.constexpr
    BALANCE_CAUSAL_WAVES: gl.constexpr
    PRE_LOAD_V: gl.constexpr
    VARLEN: gl.constexpr
    MMA_TYPE: gl.constexpr
    kt_blocked_layout: gl.constexpr
    blocked_layout: gl.constexpr
    kt_async_layout: gl.constexpr
    v_async_layout: gl.constexpr
    kt_dot_layout: gl.constexpr
    p_dot_layout: gl.constexpr
    v_dot_layout: gl.constexpr
    mma_layout: gl.constexpr
    mma_offs_n_col: gl.constexpr
    mma_offs_m_row: gl.constexpr
    ENABLE_CLASS_DIAGONAL_LAZY: gl.constexpr
    SEPARATE_K_SLOTS: gl.constexpr

    @gluon.jit
    def pipelined(self, acc, l_i, m_i, block_start, block_end,
                  MASK_STEPS: gl.constexpr,
                  SKIP_ENTRY_WAIT: gl.constexpr = False):
        return attn_fwd_inner_pipelined(
            acc, l_i, m_i, self.q_dot, self.k_base, self.v_base, self.start_m,
            self.stride_kn, self.stride_kk, self.stride_vk, self.stride_vn,
            block_start, block_end,
            self.kt_smem, self.kt_smem1, self.v_smem,
            self.qk_scale,
            self.MAX_SEQLENS_Q, self.MAX_SEQLENS_K,
            self.BLOCK_M, self.BLOCK_N, self.BLOCK_DMODEL, self.ACTUAL_BLOCK_DMODEL,
            MASK_STEPS, self.IS_CAUSAL, self.BALANCE_CAUSAL_WAVES,
            SKIP_ENTRY_WAIT,
            self.SEPARATE_K_SLOTS,
            (
                2
                if self.BLOCK_M == 256
                and self.NUM_STAGES == 4
                and (
                    self.IS_CAUSAL
                    or self.MAX_SEQLENS_K // self.BLOCK_N < 256
                )
                else 1
                if self.BLOCK_M == 256
                and (
                    self.NUM_STAGES == 4
                    or self.MAX_SEQLENS_K // self.BLOCK_N >= 64
                )
                else 0
            ),
            self.kt_async_layout, self.v_async_layout,
            self.kt_dot_layout, self.p_dot_layout, self.v_dot_layout,
            self.mma_layout, self.mma_offs_n_col, self.mma_offs_m_row,
        )

    @gluon.jit
    def short(self, acc, l_i, m_i, block_start, block_end,
              DIAGONAL_PREFETCHED: gl.constexpr = False,
              SKIP_ENTRY_WAIT: gl.constexpr = False):
        return attn_fwd_inner_short(
            acc, l_i, m_i, self.q_dot, self.k_base, self.v_base,
            self.start_m,
            self.stride_kn, self.stride_kk, self.stride_vk, self.stride_vn,
            block_start, block_end,
            self.kt_smem, self.v_smem,
            self.qk_scale,
            self.MAX_SEQLENS_Q, self.MAX_SEQLENS_K,
            self.BLOCK_M, self.BLOCK_N, self.BLOCK_DMODEL, self.ACTUAL_BLOCK_DMODEL,
            self.IS_CAUSAL, self.BALANCE_CAUSAL_WAVES,
            DIAGONAL_PREFETCHED, SKIP_ENTRY_WAIT,
            self.kt_async_layout, self.v_async_layout,
            self.kt_dot_layout, self.p_dot_layout, self.v_dot_layout,
            self.mma_layout, self.mma_offs_n_col, self.mma_offs_m_row,
            self.ENABLE_CLASS_DIAGONAL_LAZY,
        )

    @gluon.jit
    def full2_lazy(self, acc, l_i, m_i, block_start,
                   SKIP_ENTRY_WAIT: gl.constexpr = False):
        return attn_fwd_inner_full2_lazy(
            acc, l_i, m_i, self.q_dot, self.k_base, self.v_base,
            self.stride_kn, self.stride_kk,
            self.stride_vk, self.stride_vn,
            block_start, self.kt_smem, self.v_smem,
            self.qk_scale,
            self.BLOCK_M, self.BLOCK_N,
            self.BLOCK_DMODEL,
            self.kt_async_layout, self.v_async_layout,
            self.kt_dot_layout, self.p_dot_layout, self.v_dot_layout,
            self.mma_layout, self.mma_offs_m_row, SKIP_ENTRY_WAIT,
        )

    @gluon.jit
    def non_pipelined_fallback(
        self, acc, l_i, m_i, kt_ptrs, v_ptrs,
        block_start, block_end,
        MASK_STEPS: gl.constexpr, IS_CAUSAL: gl.constexpr,
    ):
        return attn_fwd_inner(
            acc, l_i, m_i, self.q_dot, kt_ptrs, v_ptrs, self.offs_n, self.offs_d,
            self.kt_offs_d, self.kt_offs_n, self.start_m,
            self.stride_kn, self.stride_vk,
            block_start, block_end,
            self.kt_smem, self.v_smem,
            self.seqlen_q, self.seqlen_k, self.qk_scale,
            self.MAX_SEQLENS_Q, self.MAX_SEQLENS_K,
            self.BLOCK_M, self.BLOCK_N, self.BLOCK_DMODEL, self.ACTUAL_BLOCK_DMODEL,
            self.PRE_LOAD_V, MASK_STEPS, IS_CAUSAL, self.VARLEN,
            self.MMA_TYPE, self.kt_blocked_layout, self.blocked_layout,
            self.kt_dot_layout, self.p_dot_layout, self.v_dot_layout,
            self.mma_layout, self.mma_offs_n_col, self.mma_offs_m_row,
            self.BALANCE_CAUSAL_WAVES,
        )

# ---------------------------------------------------------------------------
# Autotune configs
# ---------------------------------------------------------------------------

VGPR_ONLY_FN_ATTRS = (("amdgpu-agpr-alloc", "0,0"),)
NO_DISPATCH_ID_FN_ATTRS = (("amdgpu-no-dispatch-id", ""),)
VGPR_ONLY_PRELOAD_FN_ATTRS = (
    VGPR_ONLY_FN_ATTRS + NO_DISPATCH_ID_FN_ATTRS
)
VGPR_ONLY_PRELOAD_MAX_ILP_FN_ATTRS = (
    VGPR_ONLY_PRELOAD_FN_ATTRS
    + (("amdgpu-sched-strategy", "max-ilp"),)
)

def get_gluon_cdna_autotune_configs():
    return [
        # These are ordinary AMD backend function attributes, not an external
        # LLVM pass. Keep the optimized candidates as autotune peers so a
        # compiler that supports them can select the pinned source schedule.
        triton.Config({
            'BLOCK_M': 256,
            'BLOCK_N': 32,
            'PRE_LOAD_V': False,
            'NUM_STAGES': 2,
            'waves_per_eu': 2,
            'llvm_fn_attrs': VGPR_ONLY_FN_ATTRS,
        }, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'PRE_LOAD_V': False, 'NUM_STAGES': 4, 'waves_per_eu': 2}, num_warps=8),
        triton.Config({
            'BLOCK_M': 256,
            'BLOCK_N': 64,
            'PRE_LOAD_V': False,
            'NUM_STAGES': 3,
            'waves_per_eu': 2,
            'llvm_fn_attrs': VGPR_ONLY_FN_ATTRS,
        }, num_warps=8),
        triton.Config({
            'BLOCK_M': 256,
            'BLOCK_N': 64,
            'PRE_LOAD_V': False,
            'NUM_STAGES': 4,
            'waves_per_eu': 2,
            'llvm_fn_attrs': VGPR_ONLY_PRELOAD_FN_ATTRS,
        }, num_warps=8),
        triton.Config({
            'BLOCK_M': 256,
            'BLOCK_N': 64,
            'PRE_LOAD_V': False,
            'NUM_STAGES': 4,
            'waves_per_eu': 2,
            'llvm_fn_attrs': VGPR_ONLY_PRELOAD_MAX_ILP_FN_ATTRS,
        }, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 4, 'waves_per_eu': 0}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 4, 'waves_per_eu': 3}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 2, 'waves_per_eu': 2}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 2, 'waves_per_eu': 0}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 2, 'waves_per_eu': 3}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 1, 'waves_per_eu': 2}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'PRE_LOAD_V': True,  'NUM_STAGES': 1, 'waves_per_eu': 2}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': True,  'NUM_STAGES': 1, 'waves_per_eu': 2}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 1, 'waves_per_eu': 2}, num_warps=8),
        # BM=128, BN=64, NW=8 is excluded at D=128. NS=4 needs LLVM PR
        # https://github.com/llvm/llvm-project/pull/193499; NS=2 can race under
        # a full-size launch even when a one-CTA smoke test happens to pass.
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 4, 'waves_per_eu': 2}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 4, 'waves_per_eu': 1}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 4, 'waves_per_eu': 0}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 4, 'waves_per_eu': 3}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 2, 'waves_per_eu': 2}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 2, 'waves_per_eu': 0}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 2, 'waves_per_eu': 3}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 1, 'waves_per_eu': 2}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 1, 'waves_per_eu': 1}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 2, 'waves_per_eu': 0}, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 2, 'waves_per_eu': 2}, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 4, 'waves_per_eu': 2}, num_warps=2),
        # BLOCK_N=32 configs (narrower tiles for reduced MFMA work and register pressure)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'PRE_LOAD_V': False, 'NUM_STAGES': 4, 'waves_per_eu': 2}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'PRE_LOAD_V': False, 'NUM_STAGES': 4, 'waves_per_eu': 0}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'PRE_LOAD_V': False, 'NUM_STAGES': 2, 'waves_per_eu': 2}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'PRE_LOAD_V': False, 'NUM_STAGES': 2, 'waves_per_eu': 0}, num_warps=4),
        # D=256 pipelined configs: must use BLOCK_N=32 (BN=64 exceeds LDS capacity at D=256)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'PRE_LOAD_V': False, 'NUM_STAGES': 4, 'waves_per_eu': 2}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'PRE_LOAD_V': False, 'NUM_STAGES': 4, 'waves_per_eu': 0}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'PRE_LOAD_V': False, 'NUM_STAGES': 2, 'waves_per_eu': 2}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'PRE_LOAD_V': False, 'NUM_STAGES': 2, 'waves_per_eu': 0}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'PRE_LOAD_V': False, 'NUM_STAGES': 1, 'waves_per_eu': 2}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'PRE_LOAD_V': False, 'NUM_STAGES': 1, 'waves_per_eu': 2}, num_warps=4),
    ]


def get_gluon_autotune_configs():
    return get_gluon_cdna_autotune_configs()


def prune_unsafe_configs(configs, named_args, **kwargs):
    """Exclude async-pipeline configurations unsupported by the launch."""
    is_causal = kwargs.get("IS_CAUSAL", named_args.get("IS_CAUSAL", False))
    actual_block_dmodel = kwargs.get(
        "ACTUAL_BLOCK_DMODEL",
        named_args.get("ACTUAL_BLOCK_DMODEL"),
    )
    unsafe_causal_stage2 = not _HAS_WARP_PREDICATE and is_causal
    unsafe_partial_head_dim = (
        actual_block_dmodel is not None and actual_block_dmodel % 16 != 0
    )
    padded_block_dmodel = (
        max(1 << (actual_block_dmodel - 1).bit_length(), 16)
        if unsafe_partial_head_dim
        else None
    )
    if not unsafe_causal_stage2 and not unsafe_partial_head_dim:
        return configs
    # Partial 16-element head tails do not lower through the multi-stage
    # shared-memory path. Mirror USE_PIPELINED below so non-pipelined
    # candidates remain available in every padded head-dimension bucket.
    return [
        config
        for config in configs
        if not (unsafe_causal_stage2 and config.kwargs.get("NUM_STAGES") == 2)
        and not (
            unsafe_partial_head_dim
            and config.kwargs.get("NUM_STAGES", 1) > 1
            and padded_block_dmodel >= 64
            and not (
                padded_block_dmodel >= 256
                and config.kwargs.get("BLOCK_N") >= 64
            )
            and not (
                padded_block_dmodel < 128
                and config.kwargs.get("BLOCK_N") < 64
                and config.num_warps >= 8
            )
        )
    ]


GLUON_AUTOTUNE_KEYS = ['IS_CAUSAL', 'MAX_SEQLENS_Q', 'MAX_SEQLENS_K', 'ACTUAL_BLOCK_DMODEL', 'HQ', 'HK']


# ---------------------------------------------------------------------------
# Main Gluon kernel
# ---------------------------------------------------------------------------

@gluon.jit
def _compute_attention_tile(
    Q, K, V, SM_SCALE: gl.constexpr, L, Out,
    off_z, off_h_q, off_h_k, start_m, DIAG_OFFSET,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    HQ: gl.constexpr, HK: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,
    MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_DMODEL: gl.constexpr, BLOCK_N: gl.constexpr,
    PRE_LOAD_V: gl.constexpr,
    MMA_TYPE: gl.constexpr, NUM_STAGES: gl.constexpr,
    ENABLE_CLASS_DIAGONAL_LAZY: gl.constexpr,
):
    """Compute one rotated-4cluster output tile for a fixed batch/head/query block."""
    num_warps: gl.constexpr = gl.num_warps()

    gl.assume(stride_qz >= 0); gl.assume(stride_qh >= 0)
    gl.assume(stride_qm >= 0); gl.assume(stride_qk >= 0)
    gl.assume(stride_kz >= 0); gl.assume(stride_kh >= 0)
    gl.assume(stride_kn >= 0); gl.assume(stride_kk >= 0)
    gl.assume(stride_vz >= 0); gl.assume(stride_vh >= 0)
    gl.assume(stride_vk >= 0); gl.assume(stride_vn >= 0)
    gl.assume(stride_oz >= 0); gl.assume(stride_oh >= 0)
    gl.assume(stride_om >= 0); gl.assume(stride_on >= 0)

    mma_layout: gl.constexpr = AMDMFMALayout(
        version=4, instr_shape=[32, 32, 16], transposed=True,
        warps_per_cta=[num_warps, 1])
    k_width:          gl.constexpr = 8
    threads_per_warp: gl.constexpr = 64
    pv_k_width:       gl.constexpr = 4

    q_dot_layout:  gl.constexpr = DotOperandLayout(operand_index=0, parent=mma_layout, k_width=k_width)
    kt_dot_layout: gl.constexpr = DotOperandLayout(operand_index=1, parent=mma_layout, k_width=k_width)
    p_dot_layout:  gl.constexpr = DotOperandLayout(operand_index=0, parent=mma_layout, k_width=pv_k_width)
    v_dot_layout:  gl.constexpr = DotOperandLayout(operand_index=1, parent=mma_layout, k_width=pv_k_width)

    # D128 stores eight adjacent elements per thread.  The 8-warp BM256
    # schedule uses 16 D lanes; the 4-warp BM128 short-causal schedule uses
    # eight D lanes.  Both choices shorten Q staging and the final
    # accumulator-to-output redistribution for their respective tile shapes.
    # Keep LSE on the original M-major layout: its reduction vector has no D
    # dimension and performs better with 16 lanes distributed over M.
    use_d128_lane_layout: gl.constexpr = (
        BLOCK_M == 256 and BLOCK_N == 64
        and BLOCK_DMODEL == 128 and num_warps == 8
    )
    is_bm128_d128: gl.constexpr = (
        BLOCK_M == 128 and (BLOCK_N == 32 or BLOCK_N == 64)
        and BLOCK_DMODEL == 128 and num_warps == 4
    )
    if use_d128_lane_layout:
        blocked_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8], threads_per_warp=[4, threads_per_warp // 4],
            warps_per_cta=[num_warps, 1], order=[1, 0])
    elif is_bm128_d128:
        blocked_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8], threads_per_warp=[8, 8],
            warps_per_cta=[num_warps, 1], order=[1, 0])
    else:
        blocked_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8], threads_per_warp=[threads_per_warp // 4, 4],
            warps_per_cta=[num_warps, 1], order=[1, 0])
    lse_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[threads_per_warp // 4, 4],
        warps_per_cta=[num_warps, 1], order=[1, 0])
    kt_blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1], threads_per_warp=[1, threads_per_warp],
        warps_per_cta=[1, num_warps], order=[0, 1])

    offs_m_layout:    gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    offs_d_layout:    gl.constexpr = gl.SliceLayout(dim=0, parent=blocked_layout)
    offs_n_layout:    gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    offs_m_lse_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=lse_layout)
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_blocked_layout)
    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_blocked_layout)
    mma_offs_n_col:   gl.constexpr = gl.SliceLayout(dim=0, parent=mma_layout)
    mma_offs_m_row:   gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)
    mma_m_layout:     gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)

    BALANCE_CAUSAL_WAVES: gl.constexpr = (
        IS_CAUSAL and MAX_SEQLENS_Q == MAX_SEQLENS_K
        and MAX_SEQLENS_Q % BLOCK_M == 0
        and BLOCK_DMODEL == 128
        and BLOCK_M == 256 and BLOCK_N <= 64 and num_warps == 8
    )
    local_m = gl.arange(0, BLOCK_M, layout=offs_m_layout)
    local_m_lse = gl.arange(0, BLOCK_M, layout=offs_m_lse_layout)
    if BALANCE_CAUSAL_WAVES:
        wave_m = local_m // 32
        wave_m = wave_m ^ ((wave_m // 4) * 3)
        local_m = wave_m * 32 + local_m % 32
        wave_m_lse = local_m_lse // 32
        wave_m_lse = wave_m_lse ^ ((wave_m_lse // 4) * 3)
        local_m_lse = wave_m_lse * 32 + local_m_lse % 32
    offs_m = start_m * BLOCK_M + local_m
    offs_m_lse = start_m * BLOCK_M + local_m_lse
    offs_d    = gl.arange(0, BLOCK_DMODEL, layout=offs_d_layout)
    offs_n    = gl.arange(0, BLOCK_N,      layout=offs_n_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kt_offs_n = gl.arange(0, BLOCK_N,      layout=kt_offs_n_layout)

    ALIGNED_QD: gl.constexpr = (
        MAX_SEQLENS_Q % BLOCK_M == 0
        and ACTUAL_BLOCK_DMODEL == BLOCK_DMODEL
    )
    # Prescaling removes one score-matrix multiply per K/V tile. The regular
    # non-causal path keeps the measured FP16-only gate. The aligned D128 lazy
    # schedule enables it for both dtypes and for a causal full prefix because
    # subtract-only normalization is part of its balanced DOT1/DOT2 budget.
    lazy_rescale_candidate: gl.constexpr = (
        HAS_WARP_PREDICATE
        and BLOCK_N == 64 and BLOCK_DMODEL == 128
        and (
            (BLOCK_M == 256 and num_warps == 8)
            or (IS_CAUSAL
                and BLOCK_M == 128 and num_warps == 4)
        )
        and MAX_SEQLENS_Q % BLOCK_M == 0
        and MAX_SEQLENS_K % BLOCK_N == 0
    )
    SCORE_SCALE_BM256: gl.constexpr = (
        BLOCK_M == 256 and BLOCK_N == 64
        and BLOCK_DMODEL == 128 and num_warps == 8
        and (
            (IS_CAUSAL and MAX_SEQLENS_K <= 4096)
            or (
                not IS_CAUSAL
                and (
                    MAX_SEQLENS_K <= 512
                    or (Q.dtype.element_ty == gl.float16
                        and MAX_SEQLENS_K <= 1024)
                )
            )
        )
    )
    prescale_q: gl.constexpr = (
        BLOCK_DMODEL <= 128
        and (
            (not IS_CAUSAL and Q.dtype.element_ty == gl.float16)
            or lazy_rescale_candidate
        )
        and (
            not IS_CAUSAL
            or not (BLOCK_M == 128 and BLOCK_N == 64
                    and BLOCK_DMODEL == 128 and num_warps == 4)
        )
        and not SCORE_SCALE_BM256
    )
    Q_EARLY_DMA: gl.constexpr = (
        IS_CAUSAL and ALIGNED_QD
        and BLOCK_M == 128 and BLOCK_N == 64
        and BLOCK_DMODEL == 128 and num_warps == 4
    )
    Q_REGISTER_CG: gl.constexpr = (
        ALIGNED_QD
        and BLOCK_M == 256 and BLOCK_N == 64
        and BLOCK_DMODEL == 128 and num_warps == 8
    )
    # The Q DMA above is completely drained before the first inner helper.
    # On very short rows, avoid immediately issuing a second empty async-copy
    # drain. Longer rows retain their instruction-identical established path.
    SKIP_EMPTY_INITIAL_WAIT: gl.constexpr = (
        Q_EARLY_DMA and MAX_SEQLENS_K <= 8 * BLOCK_N)
    q_base = Q + off_z * stride_qz + off_h_q * stride_qh
    k_base = K + off_z * stride_kz + off_h_k * stride_kh
    v_base = V + off_z * stride_vz + off_h_k * stride_vh
    if Q_EARLY_DMA:
        # A 16-byte shift after each 2 KiB logical interval spreads the eight
        # MFMA operand reads across all LDS banks.  The former 1 KiB interval
        # repeated four bank groups within a wave and generated a fixed bank-
        # conflict penalty on every query tile.
        q_smem_layout: gl.constexpr = PaddedSharedLayout(
            interval_padding_pairs=[[1024, 8]],
            offset_bases=[
                [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32],
                [0, 64], [16, 0], [32, 0], [64, 0],
                [1, 0], [2, 0], [4, 0], [8, 0],
            ],
            cga_layout=[], shape=[BLOCK_M, BLOCK_DMODEL])
        q_async_layout: gl.constexpr = DistributedLinearLayout(
            reg_bases=[
                [0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [64, 0],
            ],
            lane_bases=[
                [0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0],
            ],
            warp_bases=[[1, 0], [2, 0]],
            block_bases=[], shape=[BLOCK_M, BLOCK_DMODEL])
        q_smem = gl.allocate_shared_memory(
            Q.dtype.element_ty, [BLOCK_M, BLOCK_DMODEL],
            layout=q_smem_layout)
        q_am: gl.constexpr = gl.SliceLayout(dim=1, parent=q_async_layout)
        q_ad: gl.constexpr = gl.SliceLayout(dim=0, parent=q_async_layout)
        q_local_m = gl.arange(0, BLOCK_M, layout=q_am)
        q_off = (
            q_local_m[:, None] * stride_qm
            + gl.arange(0, BLOCK_DMODEL, layout=q_ad)[None, :] * stride_qk
        )
        # Q is consumed once by this CTA.  Bypass the local cache so this
        # one-shot transfer does not displace K/V lines reused by neighboring
        # query tiles; on gfx950 `.cg` lowers the short-causal Q drain latency.
        cdna4_async.buffer_load_to_shared(
            q_smem, q_base + start_m * BLOCK_M * stride_qm, q_off,
            cache_modifier=".cg")
        cdna4_async.commit_group()
        cdna4_async.wait_group(0)
        q_dot = q_smem.load(q_dot_layout)
    else:
        q_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
            vec=8, per_phase=1, max_phase=16, order=[1, 0])
        q_smem = gl.allocate_shared_memory(
            Q.dtype.element_ty, [BLOCK_M, BLOCK_DMODEL], layout=q_smem_layout)
        q_ptrs = (
            q_base + offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qk
        )
        if ALIGNED_QD:
            if Q_REGISTER_CG:
                q = gl.load(q_ptrs, cache_modifier=".cg")
            else:
                q = gl.load(q_ptrs)
        else:
            q_mask = offs_m[:, None] < MAX_SEQLENS_Q
            if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
                q_mask = q_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
            q = gl.load(q_ptrs, mask=q_mask, other=0.0)
        q_smem.store(q)
        q_dot = q_smem.load(q_dot_layout)

    if prescale_q:
        dot_q_scale: gl.constexpr = SM_SCALE * 1.44269504089
        q_dot_f32 = q_dot.to(gl.float32)
        # Keep the constexpr scale in the operand's distributed layout.
        # Some Gluon revisions do not infer an encoding for scalar splats.
        q_dot_scale = gl.full(
            [BLOCK_M, BLOCK_DMODEL], dot_q_scale,
            dtype=gl.float32, layout=q_dot_layout)
        q_dot = (q_dot_f32 * q_dot_scale).to(Q.dtype.element_ty)

    m_i = gl.full(
        [BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
    l_i = gl.full(
        [BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout)
    acc = gl.zeros(
        [BLOCK_M, BLOCK_DMODEL], dtype=gl.float32, layout=mma_layout)

    qk_scale: gl.constexpr = 1.0 if prescale_q else SM_SCALE * 1.44269504089

    n_blocks_total:  gl.constexpr = (MAX_SEQLENS_K + BLOCK_N - 1) // BLOCK_N
    n_extra_tokens:  gl.constexpr = MAX_SEQLENS_K % BLOCK_N
    padded_block_k:  gl.constexpr = n_extra_tokens != 0
    is_modulo_mn:    gl.constexpr = not padded_block_k and (MAX_SEQLENS_Q % BLOCK_M == 0)
    ALIGNED_SELF_CAUSAL: gl.constexpr = (
        IS_CAUSAL and MAX_SEQLENS_Q == MAX_SEQLENS_K
        and MAX_SEQLENS_Q % BLOCK_M == 0
        and MAX_SEQLENS_K % BLOCK_N == 0
        and BLOCK_M % BLOCK_N == 0
    )

    if IS_CAUSAL:
        if ALIGNED_SELF_CAUSAL:
            # The launch grid constrains start_m to [0, N/BM), so the causal
            # limit is already in range and all divisions are exact.
            n_blocks = (start_m + 1) * (BLOCK_M // BLOCK_N)
        else:
            causal_block_limit = (start_m + 1) * BLOCK_M + DIAG_OFFSET
            n_blocks = gl.minimum(
                n_blocks_total,
                (causal_block_limit + BLOCK_N - 1) // BLOCK_N,
            )
        masked_blocks: gl.constexpr = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        n_blocks = n_blocks_total
        masked_blocks: gl.constexpr = 1 if padded_block_k else 0

    if ALIGNED_SELF_CAUSAL:
        masked_blocks_clamped = masked_blocks
    else:
        masked_blocks_clamped = gl.minimum(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks_clamped

    kt_ptrs = k_base + kt_offs_d[:, None] * stride_kk + kt_offs_n[None, :] * stride_kn
    v_ptrs  = v_base + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn

    UNSAFE_CAUSAL_STAGE2: gl.constexpr = (
        not HAS_WARP_PREDICATE and IS_CAUSAL and NUM_STAGES == 2
    )
    USE_PIPELINED: gl.constexpr = (
        NUM_STAGES > 1
        and not UNSAFE_CAUSAL_STAGE2
        and BLOCK_DMODEL >= 64
        and not (BLOCK_DMODEL >= 256 and BLOCK_N >= 64)
        and not (BLOCK_DMODEL < 128 and BLOCK_N < 64 and num_warps >= 8)
    )

    if USE_PIPELINED:
        if BLOCK_DMODEL >= 256 and BLOCK_N >= 32 and num_warps == 8:
            # D=256, BN=32, 8 warps: [128,0] in lane (not reg) to satisfy DMA constraints.
            # Layout matches the extend attention kernel's _kt_dll_bases_8w(D=256) pattern.
            kt_offset_bases: gl.constexpr = [
                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0],
                [0, 16],
                [0, 1], [0, 2], [0, 4], [0, 8]
            ]
            v_offset_bases: gl.constexpr = [
                [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128],
                [16, 0],
                [1, 0], [2, 0], [4, 0], [8, 0]
            ]
            kt_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8]],
                lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [0, 16]],
                warp_bases=[[0, 1], [0, 2], [0, 4]],
                block_bases=[],
                shape=[BLOCK_DMODEL, BLOCK_N])
            v_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
                lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [16, 0]],
                warp_bases=[[1, 0], [2, 0], [4, 0]],
                block_bases=[],
                shape=[BLOCK_N, BLOCK_DMODEL])
        elif BLOCK_DMODEL >= 256 and BLOCK_N >= 32 and num_warps == 4:
            # D=256, BN=32, 4 warps: 5 reg bases, [128,0] in lane.
            # Matches _kt_dll_bases_4w(D=256) from extend attention.
            kt_offset_bases: gl.constexpr = [
                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0],
                [0, 16],
                [0, 1], [0, 2], [0, 4], [0, 8]
            ]
            v_offset_bases: gl.constexpr = [
                [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128],
                [16, 0],
                [1, 0], [2, 0], [4, 0], [8, 0]
            ]
            kt_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]],
                lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [0, 16]],
                warp_bases=[[0, 1], [0, 2]],
                block_bases=[],
                shape=[BLOCK_DMODEL, BLOCK_N])
            v_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]],
                lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [16, 0]],
                warp_bases=[[1, 0], [2, 0]],
                block_bases=[],
                shape=[BLOCK_N, BLOCK_DMODEL])
        elif BLOCK_DMODEL >= 128 and BLOCK_N >= 64 and num_warps == 8:
            kt_offset_bases: gl.constexpr = [
                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0],
                [0, 16], [0, 32],
                [0, 1], [0, 2], [0, 4], [0, 8]
            ]
            v_offset_bases: gl.constexpr = [
                [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
                [16, 0], [32, 0],
                [1, 0], [2, 0], [4, 0], [8, 0]
            ]
            kt_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8]],
                lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
                warp_bases=[[0, 1], [0, 2], [0, 4]],
                block_bases=[],
                shape=[BLOCK_DMODEL, BLOCK_N])
            v_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
                lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                warp_bases=[[1, 0], [2, 0], [4, 0]],
                block_bases=[],
                shape=[BLOCK_N, BLOCK_DMODEL])
        elif BLOCK_DMODEL >= 128 and BLOCK_N >= 64 and num_warps == 4:
            kt_offset_bases: gl.constexpr = [
                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0],
                [0, 16], [0, 32],
                [0, 1], [0, 2], [0, 4], [0, 8]
            ]
            v_offset_bases: gl.constexpr = [
                [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
                [16, 0], [32, 0],
                [1, 0], [2, 0], [4, 0], [8, 0]
            ]
            kt_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8], [0, 4]],
                lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
                warp_bases=[[0, 1], [0, 2]],
                block_bases=[],
                shape=[BLOCK_DMODEL, BLOCK_N])
            v_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0], [4, 0]],
                lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                warp_bases=[[1, 0], [2, 0]],
                block_bases=[],
                shape=[BLOCK_N, BLOCK_DMODEL])
        elif BLOCK_DMODEL >= 128 and BLOCK_N >= 32 and num_warps == 4:
            # D=128, BN=32, 4 warps: D-fast layout for both KT and V
            kt_offset_bases: gl.constexpr = [
                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0],
                [0, 16], [0, 8],
                [0, 1], [0, 2], [0, 4]
            ]
            v_offset_bases: gl.constexpr = [
                [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
                [16, 0], [8, 0],
                [1, 0], [2, 0], [4, 0]
            ]
            kt_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4]],
                lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 8]],
                warp_bases=[[0, 1], [0, 2]],
                block_bases=[],
                shape=[BLOCK_DMODEL, BLOCK_N])
            v_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0]],
                lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [8, 0]],
                warp_bases=[[1, 0], [2, 0]],
                block_bases=[],
                shape=[BLOCK_N, BLOCK_DMODEL])
        elif BLOCK_DMODEL >= 128 and BLOCK_N >= 32 and num_warps == 8:
            # D=128, BN=32, 8 warps: D-fast layout for both KT and V
            kt_offset_bases: gl.constexpr = [
                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0],
                [0, 16], [0, 8],
                [0, 1], [0, 2], [0, 4]
            ]
            v_offset_bases: gl.constexpr = [
                [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
                [16, 0], [8, 0],
                [1, 0], [2, 0], [4, 0]
            ]
            kt_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[1, 0], [2, 0], [4, 0]],
                lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 8]],
                warp_bases=[[0, 1], [0, 2], [0, 4]],
                block_bases=[],
                shape=[BLOCK_DMODEL, BLOCK_N])
            v_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4]],
                lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [8, 0]],
                warp_bases=[[1, 0], [2, 0], [4, 0]],
                block_bases=[],
                shape=[BLOCK_N, BLOCK_DMODEL])
        elif BLOCK_DMODEL >= 64 and BLOCK_N >= 64 and num_warps == 8:
            # D=64, BN=64, 8 warps
            kt_offset_bases: gl.constexpr = [
                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
                [0, 16], [0, 32],
                [0, 1], [0, 2], [0, 4], [0, 8]
            ]
            v_offset_bases: gl.constexpr = [
                [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32],
                [16, 0], [32, 0],
                [1, 0], [2, 0], [4, 0], [8, 0]
            ]
            kt_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[1, 0], [2, 0], [4, 0]],
                lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 1]],
                warp_bases=[[0, 2], [0, 4], [0, 8]],
                block_bases=[],
                shape=[BLOCK_DMODEL, BLOCK_N])
            v_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4]],
                lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [1, 0]],
                warp_bases=[[2, 0], [4, 0], [8, 0]],
                block_bases=[],
                shape=[BLOCK_N, BLOCK_DMODEL])
        elif BLOCK_DMODEL >= 128 and num_warps == 2:
            # D=128, 2 warps (for BLOCK_M=64)
            kt_offset_bases: gl.constexpr = [
                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0],
                [0, 16], [0, 32],
                [0, 1], [0, 2], [0, 4], [0, 8]
            ]
            v_offset_bases: gl.constexpr = [
                [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
                [16, 0], [32, 0],
                [1, 0], [2, 0], [4, 0], [8, 0]
            ]
            kt_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8], [0, 4], [0, 2]],
                lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
                warp_bases=[[0, 1]],
                block_bases=[],
                shape=[BLOCK_DMODEL, BLOCK_N])
            v_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0], [4, 0], [2, 0]],
                lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                warp_bases=[[1, 0]],
                block_bases=[],
                shape=[BLOCK_N, BLOCK_DMODEL])
        elif BLOCK_DMODEL >= 64 and BLOCK_N >= 64 and num_warps == 4:
            # D=64, BN=64, 4 warps
            kt_offset_bases: gl.constexpr = [
                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
                [0, 16], [0, 32],
                [0, 1], [0, 2], [0, 4], [0, 8]
            ]
            v_offset_bases: gl.constexpr = [
                [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32],
                [16, 0], [32, 0],
                [1, 0], [2, 0], [4, 0], [8, 0]
            ]
            kt_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8]],
                lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 1]],
                warp_bases=[[0, 2], [0, 4]],
                block_bases=[],
                shape=[BLOCK_DMODEL, BLOCK_N])
            v_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
                lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [1, 0]],
                warp_bases=[[2, 0], [4, 0]],
                block_bases=[],
                shape=[BLOCK_N, BLOCK_DMODEL])
        elif BLOCK_DMODEL >= 64 and BLOCK_N >= 32 and num_warps == 4:
            # D=64, BN=32, 4 warps: D-fast layout for both KT and V
            kt_offset_bases: gl.constexpr = [
                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
                [0, 16], [0, 8], [0, 4],
                [0, 1], [0, 2]
            ]
            v_offset_bases: gl.constexpr = [
                [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32],
                [16, 0], [8, 0], [4, 0],
                [1, 0], [2, 0]
            ]
            kt_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[1, 0], [2, 0], [4, 0]],
                lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 8], [0, 4]],
                warp_bases=[[0, 1], [0, 2]],
                block_bases=[],
                shape=[BLOCK_DMODEL, BLOCK_N])
            v_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4]],
                lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [8, 0], [4, 0]],
                warp_bases=[[1, 0], [2, 0]],
                block_bases=[],
                shape=[BLOCK_N, BLOCK_DMODEL])
        else:
            # D=64, 2 warps (for BLOCK_M=64)
            kt_offset_bases: gl.constexpr = [
                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
                [0, 16], [0, 32],
                [0, 1], [0, 2], [0, 4], [0, 8]
            ]
            v_offset_bases: gl.constexpr = [
                [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32],
                [16, 0], [32, 0],
                [1, 0], [2, 0], [4, 0], [8, 0]
            ]
            kt_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8], [0, 4]],
                lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 1]],
                warp_bases=[[0, 2]],
                block_bases=[],
                shape=[BLOCK_DMODEL, BLOCK_N])
            v_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0], [4, 0]],
                lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [1, 0]],
                warp_bases=[[2, 0]],
                block_bases=[],
                shape=[BLOCK_N, BLOCK_DMODEL])

        kt_async_smem_layout: gl.constexpr = PaddedSharedLayout(
            interval_padding_pairs=[[512, 8]],
            offset_bases=kt_offset_bases,
            cga_layout=[],
            shape=[BLOCK_DMODEL, BLOCK_N])
        v_async_smem_layout: gl.constexpr = PaddedSharedLayout(
            interval_padding_pairs=[[512, 32]],
            offset_bases=v_offset_bases,
            cga_layout=[],
            shape=[BLOCK_N, BLOCK_DMODEL])

        # The long-causal BM256/BN64 winner already has one CTA resident from
        # its register footprint. Four slots therefore do not lower occupancy
        # and let its four-tile diagonal avoid divergent slot reuse entirely.
        BUF_DEPTH: gl.constexpr = (
            4 if HAS_WARP_PREDICATE and IS_CAUSAL
            and MAX_SEQLENS_Q == MAX_SEQLENS_K
            and MAX_SEQLENS_K % BLOCK_N == 0
            and BLOCK_M == 256 and BLOCK_N == 64 and num_warps == 8
            else 2
        )
        SEPARATE_K_SLOTS: gl.constexpr = (
            HAS_WARP_PREDICATE
            and not IS_CAUSAL and HQ == HK
            and MAX_SEQLENS_Q == MAX_SEQLENS_K
            and MAX_SEQLENS_Q % BLOCK_M == 0
            and MAX_SEQLENS_K % BLOCK_N == 0
            and BLOCK_M == 256 and BLOCK_N == 64
            and BLOCK_DMODEL == 128 and num_warps == 8
            and (
                (Q.dtype.element_ty == gl.bfloat16
                 and MAX_SEQLENS_K >= 1024)
                or (Q.dtype.element_ty == gl.float16
                    and MAX_SEQLENS_K >= 2048)
            )
        )
        if SEPARATE_K_SLOTS:
            # Distinct allocation IDs reshape the backend placement of the
            # ping-pong K operands without increasing the total LDS footprint.
            kt_smem = gl.allocate_shared_memory(
                Q.dtype.element_ty, [1, BLOCK_DMODEL, BLOCK_N],
                layout=kt_async_smem_layout)
            kt_smem1 = gl.allocate_shared_memory(
                Q.dtype.element_ty, [1, BLOCK_DMODEL, BLOCK_N],
                layout=kt_async_smem_layout)
        else:
            kt_smem = gl.allocate_shared_memory(
                Q.dtype.element_ty, [BUF_DEPTH, BLOCK_DMODEL, BLOCK_N],
                layout=kt_async_smem_layout)
            kt_smem1 = kt_smem
        v_smem = gl.allocate_shared_memory(
            Q.dtype.element_ty, [BUF_DEPTH, BLOCK_N, BLOCK_DMODEL],
            layout=v_async_smem_layout)
        inner_ctx: gl.constexpr = AttentionInnerContext(
            q_dot, k_base, v_base,
            offs_n, offs_d, kt_offs_d, kt_offs_n, start_m,
            stride_kn, stride_kk, stride_vk, stride_vn,
            kt_smem, kt_smem1, v_smem,
            MAX_SEQLENS_Q, MAX_SEQLENS_K, qk_scale,
            MAX_SEQLENS_Q, MAX_SEQLENS_K,
            BLOCK_M, BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
            NUM_STAGES, IS_CAUSAL, BALANCE_CAUSAL_WAVES, PRE_LOAD_V, False,
            MMA_TYPE, kt_blocked_layout, blocked_layout,
            kt_async_layout, v_async_layout,
            kt_dot_layout, p_dot_layout, v_dot_layout,
            mma_layout, mma_offs_n_col, mma_offs_m_row,
            ENABLE_CLASS_DIAGONAL_LAZY,
            SEPARATE_K_SLOTS,
        )
        USE_FULL2_LAZY: gl.constexpr = (
            HAS_WARP_PREDICATE and IS_CAUSAL
            and MAX_SEQLENS_Q == MAX_SEQLENS_K
            and MAX_SEQLENS_Q % BLOCK_M == 0
            and MAX_SEQLENS_K % BLOCK_N == 0
            and BLOCK_M == 128 and BLOCK_N == 64
            and BLOCK_DMODEL == 128 and num_warps == 4
        )
        if (n_blocks > NUM_STAGES
                and (n_blocks - n_full_blocks) < NUM_STAGES
                and n_full_blocks != n_blocks):
            # Small masked region with some full blocks: run the whole range on the
            # rotated loop with masking (the causal/bound mask is a no-op on
            # the full blocks).
            acc, l_i, m_i = inner_ctx.pipelined(
                acc, l_i, m_i, 0, n_blocks, True, SKIP_EMPTY_INITIAL_WAIT)
        elif n_blocks > NUM_STAGES:
            if n_full_blocks >= 4:
                # Fully-unmasked block region: matched rotated 4-cluster loop
                # (MASK_STEPS=False). Shared by causal and non-causal: for causal
                # these are the below-diagonal full blocks; the masked diagonal tail
                # is handled by the same rotated loop with masking (or the short
                # preload-all path when the tail is too small to fill the pipeline).
                acc, l_i, m_i = inner_ctx.pipelined(
                    acc, l_i, m_i, 0, n_full_blocks, False,
                    SKIP_EMPTY_INITIAL_WAIT)
                masked_start = n_full_blocks
                if (USE_FULL2_LAZY
                        and MAX_SEQLENS_K // BLOCK_N >= 8):
                    # The BM128 prefix drain has already handed both diagonal
                    # K/V pairs into slots 0/1. Consume them without reissuing
                    # the copies in the short-tail helper.
                    acc, l_i, m_i = inner_ctx.short(
                        acc, l_i, m_i, masked_start, n_blocks, True)
                    masked_start = n_blocks
            elif USE_FULL2_LAZY and n_full_blocks == 2:
                acc, l_i, m_i = inner_ctx.full2_lazy(
                    acc, l_i, m_i, 0, SKIP_EMPTY_INITIAL_WAIT)
                masked_start = n_full_blocks
            else:
                masked_start = 0
            remaining_blocks = n_blocks - masked_start
            USE_PREDICATED_DIAGONAL: gl.constexpr = (
                HAS_WARP_PREDICATE and IS_CAUSAL and is_modulo_mn
                and BLOCK_M == 32 * gl.num_warps()
            )
            if USE_PREDICATED_DIAGONAL and remaining_blocks > 0:
                # All waves collaborate on the K/V preloads, then waves whose
                # complete 32-row band is above the diagonal skip the complete
                # QK -> softmax -> PV tile body.
                acc, l_i, m_i = inner_ctx.short(
                    acc, l_i, m_i, masked_start, n_blocks)
            elif remaining_blocks >= NUM_STAGES:
                # Masked diagonal / K-bound tail large enough to fill the pipeline:
                # run it on the same rotated loop with masking.
                acc, l_i, m_i = inner_ctx.pipelined(
                    acc, l_i, m_i, masked_start, n_blocks, True)
            elif remaining_blocks > 0:
                acc, l_i, m_i = inner_ctx.short(
                    acc, l_i, m_i, masked_start, n_blocks)
        elif n_blocks > 0:
            acc, l_i, m_i = inner_ctx.short(
                acc, l_i, m_i, 0, n_blocks, False,
                SKIP_EMPTY_INITIAL_WAIT)
    else:
        kt_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=16, order=[0, 1])
        v_smem_layout:  gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=16, order=[1, 0])
        kt_smem = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_DMODEL, BLOCK_N], layout=kt_smem_layout)
        v_smem  = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_N, BLOCK_DMODEL], layout=v_smem_layout)
        inner_ctx: gl.constexpr = AttentionInnerContext(
            q_dot, k_base, v_base,
            offs_n, offs_d, kt_offs_d, kt_offs_n, start_m,
            stride_kn, stride_kk, stride_vk, stride_vn,
            kt_smem, kt_smem, v_smem,
            MAX_SEQLENS_Q, MAX_SEQLENS_K, qk_scale,
            MAX_SEQLENS_Q, MAX_SEQLENS_K,
            BLOCK_M, BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
            NUM_STAGES, IS_CAUSAL, BALANCE_CAUSAL_WAVES, PRE_LOAD_V, False,
            MMA_TYPE, kt_blocked_layout, blocked_layout,
            kt_blocked_layout, blocked_layout,
            kt_dot_layout, p_dot_layout, v_dot_layout,
            mma_layout, mma_offs_n_col, mma_offs_m_row,
            ENABLE_CLASS_DIAGONAL_LAZY,
            False,
        )

        if n_full_blocks > 0:
            acc, l_i, m_i, kt_ptrs, v_ptrs = inner_ctx.non_pipelined_fallback(
                acc, l_i, m_i, kt_ptrs, v_ptrs,
                0, n_full_blocks, False, False,
            )

        if masked_blocks > 0:
            acc, l_i, m_i, kt_ptrs, v_ptrs = inner_ctx.non_pipelined_fallback(
                acc, l_i, m_i, kt_ptrs, v_ptrs,
                n_full_blocks, n_blocks, True, IS_CAUSAL,
            )

    # The two IEEE divisions in the epilogue are a visible fixed cost when a
    # CTA covers only a short K range.  Output is stored as FP16/BF16, so the
    # native reciprocal's error is below the output quantization step.  Keep
    # the precisely rounded path once the fixed epilogue is well amortized.
    USE_FAST_SHORT_EPILOGUE: gl.constexpr = (
        MAX_SEQLENS_K <= 32 * BLOCK_N)
    NORMALIZED_IN_PRUNED_SHORT: gl.constexpr = (
        # Only the predicated pipelined short path normalizes its accumulator
        # before this epilogue.  NUM_STAGES=1 uses attn_fwd_inner instead.
        USE_PIPELINED and HAS_WARP_PREDICATE and IS_CAUSAL
        and MAX_SEQLENS_Q == MAX_SEQLENS_K
        and MAX_SEQLENS_Q % BLOCK_M == 0
        and MAX_SEQLENS_K % BLOCK_N == 0
        and MAX_SEQLENS_K <= 16 * BLOCK_N
        and BLOCK_M == 128 and BLOCK_N == 64
        and BLOCK_DMODEL == 128 and num_warps == 4
    )
    if not NORMALIZED_IN_PRUNED_SHORT:
        if USE_FAST_SHORT_EPILOGUE:
            l_recip = gl.extra.libdevice.fast_dividef(1.0, l_i)
        else:
            l_recip = 1.0 / l_i
        acc = acc * l_recip[:, None]

    o_base  = Out + off_z * stride_oz + off_h_q * stride_oh
    acc_out = acc.to(Out.dtype.element_ty)
    USE_WARP_LOCAL_EPILOGUE: gl.constexpr = (
        IS_CAUSAL and ACTUAL_BLOCK_DMODEL == 128
        and BLOCK_DMODEL == 128
        and (
            (MAX_SEQLENS_K + BLOCK_N - 1) // BLOCK_N <= 64
            or (
                (Out.dtype.element_ty == gl.bfloat16
                 or Out.dtype.element_ty == gl.float16)
                and (MAX_SEQLENS_K + BLOCK_N - 1) // BLOCK_N <= 256
            )
        )
        and (
            (BLOCK_M == 128 and num_warps == 4)
            or (BLOCK_M == 256 and num_warps == 8)
        )
    )
    # Short non-causal rows can keep the scalar LSE result in the native
    # per-wave MFMA row layout while their vector output retains the established
    # CTA-wide blocked conversion. BF16 wins through sixteen BN64 tiles; FP16
    # crosses over only above twelve tiles. Longer rows better amortize the
    # existing conversion and retain its faster store schedule.
    USE_WARP_LOCAL_LSE: gl.constexpr = (
        USE_WARP_LOCAL_EPILOGUE
        or (
            not IS_CAUSAL and ALIGNED_QD
            and (
                q_dot.dtype == gl.bfloat16
                or MAX_SEQLENS_K > 12 * BLOCK_N
            )
            and ACTUAL_BLOCK_DMODEL == 128
            and BLOCK_DMODEL == 128
            and BLOCK_M == 256 and num_warps == 8
            and MAX_SEQLENS_K <= 16 * BLOCK_N
        )
    )
    if USE_WARP_LOCAL_EPILOGUE:
        # Swap the native MFMA N-coordinate bits 4 and 8 in-wave. This gives
        # each lane eight adjacent FP16/BF16 values, enabling 128-bit stores
        # without the CTA-wide LDS round trip used by BlockedLayout.
        if BLOCK_M == 256:
            store_warp_bases: gl.constexpr = [
                [32, 0], [64, 0], [128, 0],
            ]
        else:
            store_warp_bases: gl.constexpr = [[32, 0], [64, 0]]
        mma_store_layout: gl.constexpr = DistributedLinearLayout(
            reg_bases=[
                [0, 1], [0, 2], [0, 4], [0, 16], [0, 32], [0, 64],
            ],
            lane_bases=[
                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8],
            ],
            warp_bases=store_warp_bases,
            block_bases=[], shape=[BLOCK_M, BLOCK_DMODEL])
        store_offs_m_layout: gl.constexpr = gl.SliceLayout(
            dim=1, parent=mma_store_layout)
        store_offs_d_layout: gl.constexpr = gl.SliceLayout(
            dim=0, parent=mma_store_layout)
        offs_m_o = (
            start_m * BLOCK_M
            + gl.arange(0, BLOCK_M, layout=store_offs_m_layout)
        )
        if BALANCE_CAUSAL_WAVES:
            local_m_o = offs_m_o - start_m * BLOCK_M
            wave_m_o = local_m_o // 32
            wave_m_o = wave_m_o ^ ((wave_m_o // 4) * 3)
            offs_m_o = (
                start_m * BLOCK_M
                + wave_m_o * 32 + local_m_o % 32
            )
        offs_d_o = gl.arange(
            0, BLOCK_DMODEL, layout=store_offs_d_layout)
        acc_store = gl.convert_layout(acc_out, mma_store_layout)
        o_ptrs = (
            o_base + offs_m_o[:, None] * stride_om
            + offs_d_o[None, :] * stride_on
        )
        if ALIGNED_QD:
            gl.store(o_ptrs, acc_store)
        else:
            o_mask = offs_m_o[:, None] < MAX_SEQLENS_Q
            if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
                o_mask = o_mask & (
                    offs_d_o[None, :] < ACTUAL_BLOCK_DMODEL)
            gl.store(o_ptrs, acc_store, mask=o_mask)
    else:
        o_ptrs = (
            o_base + offs_m[:, None] * stride_om
            + offs_d[None, :] * stride_on
        )
        acc_blocked = gl.convert_layout(acc_out, blocked_layout)
        if ALIGNED_QD:
            gl.store(o_ptrs, acc_blocked)
        else:
            o_mask = offs_m[:, None] < MAX_SEQLENS_Q
            if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
                o_mask = o_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
            gl.store(o_ptrs, acc_blocked, mask=o_mask)

    # m_i and log2(l_i) share the same base-2 frame.  Combine them before the
    # single log2(e) conversion instead of scaling both vectors separately.
    if USE_FAST_SHORT_EPILOGUE:
        # Multiplication by ln(2) is the same base conversion as division by
        # log2(e), without invoking the generic IEEE division sequence.
        lse = (m_i + gl.log2(l_i)) * 0.6931471805599453
    else:
        lse = (m_i + gl.log2(l_i)) / 1.44269504089
    l_base = L + off_z * HQ * MAX_SEQLENS_Q + off_h_q * MAX_SEQLENS_Q
    if USE_WARP_LOCAL_LSE:
        offs_m_lse_mma = (
            start_m * BLOCK_M
            + gl.arange(0, BLOCK_M, layout=mma_m_layout)
        )
        if BALANCE_CAUSAL_WAVES:
            local_m_lse_mma = offs_m_lse_mma - start_m * BLOCK_M
            wave_m_lse_mma = local_m_lse_mma // 32
            wave_m_lse_mma = (
                wave_m_lse_mma ^ ((wave_m_lse_mma // 4) * 3)
            )
            offs_m_lse_mma = (
                start_m * BLOCK_M
                + wave_m_lse_mma * 32 + local_m_lse_mma % 32
            )
        l_ptrs = l_base + offs_m_lse_mma
        if MAX_SEQLENS_Q % BLOCK_M == 0:
            gl.store(l_ptrs, lse)
        else:
            l_mask = offs_m_lse_mma < MAX_SEQLENS_Q
            gl.store(l_ptrs, lse, mask=l_mask)
    else:
        l_ptrs = l_base + offs_m_lse
        lse_blocked = gl.convert_layout(lse, offs_m_lse_layout)
        if MAX_SEQLENS_Q % BLOCK_M == 0:
            gl.store(l_ptrs, lse_blocked)
        else:
            l_mask = offs_m_lse < MAX_SEQLENS_Q
            gl.store(l_ptrs, lse_blocked, mask=l_mask)



@triton.autotune(
    configs=get_gluon_autotune_configs(),
    key=GLUON_AUTOTUNE_KEYS,
    prune_configs_by={"early_config_prune": prune_unsafe_configs},
)
@gluon.jit
def gluon_attn_fwd(Q, K, V, SM_SCALE: gl.constexpr, L, Out,
                   stride_qz: gl.constexpr, stride_qh: gl.constexpr,
                   stride_qm, stride_qk,
                   stride_kz: gl.constexpr, stride_kh: gl.constexpr,
                   stride_kn, stride_kk,
                   stride_vz: gl.constexpr, stride_vh: gl.constexpr,
                   stride_vk, stride_vn,
                   stride_oz: gl.constexpr, stride_oh: gl.constexpr,
                   stride_om, stride_on,
                   HQ: gl.constexpr, HK: gl.constexpr,
                   ACTUAL_BLOCK_DMODEL: gl.constexpr,
                   MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
                   IS_CAUSAL: gl.constexpr,
                   BLOCK_M: gl.constexpr, BLOCK_DMODEL: gl.constexpr, BLOCK_N: gl.constexpr,
                   PRE_LOAD_V: gl.constexpr,
                   MMA_TYPE: gl.constexpr, NUM_STAGES: gl.constexpr,
                   STATIC_STRIDE_KN: gl.constexpr = -1,
                   STATIC_STRIDE_QM: gl.constexpr = -1):
    """
    Gluon Flash Attention Forward Kernel (AMD CDNA4 / gfx950).
    Grid: (num_heads_q, num_m_blocks, batch)
    """
    start_m = gl.program_id(1)
    off_h_q = gl.program_id(0)
    # For MHA, transpose query/head order inside a bounded scheduling window.
    # Non-causal rows retain a fixed 128-CTA window.  Causal BM256 rows use a
    # 32-head window: this keeps enough uniformly phased CTAs in flight while
    # revisiting each head's common K/V stream sooner than the global
    # heavy-query-first grid.  Long H64 rows have a second measured phase
    # point at 24 heads: it wins FP16 from 32 query tiles and BF16 from 64
    # query tiles.  The final 16-head tail is decoded explicitly, so this is
    # still a bijection over the original work.  GQA already reuses K/V across
    # adjacent query heads and keeps the earlier long-row mapping below.
    num_m_blocks: gl.constexpr = (
        (MAX_SEQLENS_Q + BLOCK_M - 1) // BLOCK_M
    )
    use_causal_24_head_window: gl.constexpr = (
        IS_CAUSAL and HQ == 64
        and (
            (Q.dtype.element_ty == gl.float16 and num_m_blocks >= 32)
            or (Q.dtype.element_ty == gl.bfloat16 and num_m_blocks >= 64)
        )
    )
    xcd_window_heads: gl.constexpr = (
        (24 if use_causal_24_head_window else 32) if IS_CAUSAL
        else (128 // num_m_blocks if num_m_blocks < 128 else 1)
    )
    use_xcd_query_window: gl.constexpr = (
        BLOCK_M == 256
        and HQ == HK
        and xcd_window_heads < HQ
        and (
            use_causal_24_head_window
            or HQ % xcd_window_heads == 0
        )
        and (not IS_CAUSAL or num_m_blocks >= 8)
    )
    # The BF16 split-3 Q16 object has a reproducible address/XCD phase point:
    # traversing each proven 32-head locality set in reverse keeps all work and
    # cache-reuse distance unchanged, but reduces LDS FIFO pressure and improves
    # MFMA/VALU coexecution.  Deeper Q32 rows need the original direction.
    reverse_causal_bf16_q16_window: gl.constexpr = (
        use_xcd_query_window
        and IS_CAUSAL
        and Q.dtype.element_ty == gl.bfloat16
        and HQ == 64 and HK == 64
        and num_m_blocks == 16
        and BLOCK_N == 64
        and NUM_STAGES == 3
        and gl.num_warps() == 8
        and xcd_window_heads == 32
    )
    if use_xcd_query_window:
        linear_wg = off_h_q + start_m * HQ
        group_span: gl.constexpr = xcd_window_heads * num_m_blocks
        if HQ % xcd_window_heads == 0:
            within_group = linear_wg % group_span
            off_h_q = (
                (linear_wg // group_span) * xcd_window_heads
                + within_group % xcd_window_heads
            )
            start_m = within_group // xcd_window_heads
        else:
            tail_heads: gl.constexpr = HQ - 2 * xcd_window_heads
            if (Q.dtype.element_ty == gl.float16
                    and num_m_blocks % 8 == 0):
                # For H64/G24 and Q divisible by eight, each 24*Q segment
                # begins on a raw query-row boundary. Decode the same
                # 24/24/16 permutation locally, factoring 24 as 8*3 to avoid
                # carrying the large-span quotient through the hot kernel.
                full_group_rows: gl.constexpr = 3 * num_m_blocks // 8
                tail_row_begin: gl.constexpr = 2 * full_group_rows
                if start_m < full_group_rows:
                    packed = off_h_q + start_m * HQ
                    packed8 = packed // 8
                    mapped_m = packed8 // 3
                    off_h_q = (
                        packed % 8 + (packed8 - mapped_m * 3) * 8
                    )
                    start_m = mapped_m
                elif start_m < tail_row_begin:
                    packed = (
                        off_h_q + (start_m - full_group_rows) * HQ
                    )
                    packed8 = packed // 8
                    mapped_m = packed8 // 3
                    off_h_q = (
                        xcd_window_heads + packed % 8
                        + (packed8 - mapped_m * 3) * 8
                    )
                    start_m = mapped_m
                else:
                    packed = (
                        off_h_q + (start_m - tail_row_begin) * HQ
                    )
                    off_h_q = (
                        2 * xcd_window_heads + packed % tail_heads
                    )
                    start_m = packed // tail_heads
            else:
                full_heads: gl.constexpr = (
                    (HQ // xcd_window_heads) * xcd_window_heads
                )
                full_span: gl.constexpr = full_heads * num_m_blocks
                if linear_wg < full_span:
                    within_group = linear_wg % group_span
                    off_h_q = (
                        (linear_wg // group_span) * xcd_window_heads
                        + within_group % xcd_window_heads
                    )
                    start_m = within_group // xcd_window_heads
                else:
                    within_tail = linear_wg - full_span
                    off_h_q = full_heads + within_tail % tail_heads
                    start_m = within_tail // tail_heads
        if reverse_causal_bf16_q16_window:
            group_base = (off_h_q // 32) * 32
            off_h_q = group_base + 31 - off_h_q % 32
    elif (not IS_CAUSAL and BLOCK_M == 256
          and MAX_SEQLENS_Q >= 32 * BLOCK_M and HQ % 8 == 0):
        # Preserve the prior long-row query-fast mapping for GQA and for MHA
        # head counts that cannot use the bounded-window bijection.
        linear_wg = off_h_q + start_m * HQ
        off_h_q = linear_wg // num_m_blocks
        start_m = linear_wg % num_m_blocks
    else:
        # Causal work is already launched heavy-query-first.  For the short
        # four-wave schedule and the long split-4 schedule, keeping heads linear
        # gives each XCD a more even sequence of triangular CTA costs.  The
        # split-3 BM256 schedule still benefits from the established head remap.
        causal_linear_heads: gl.constexpr = (
            IS_CAUSAL and BLOCK_N == 64
            and (
                (BLOCK_M == 128 and NUM_STAGES == 2 and gl.num_warps() == 4)
                or (BLOCK_M == 256 and NUM_STAGES == 4 and gl.num_warps() == 8)
            )
        )
        if not causal_linear_heads:
            off_h_q = remap_xcd(off_h_q, HQ)
    off_z = gl.program_id(2)
    if IS_CAUSAL:
        # Monotonically increasing triangular work leaves a long tail of heavy
        # query tiles. Launch the heavy end first so the final partial CU wave
        # contains only cheap tiles.
        start_m = num_m_blocks - 1 - start_m
    off_h_k = off_h_q * HK // HQ
    _compute_attention_tile(
        Q, K, V, SM_SCALE, L, Out,
        off_z, off_h_q, off_h_k, start_m,
        MAX_SEQLENS_K - MAX_SEQLENS_Q,
        stride_qz, stride_qh,
        STATIC_STRIDE_QM if STATIC_STRIDE_QM >= 0 else stride_qm,
        stride_qk,
        stride_kz, stride_kh,
        STATIC_STRIDE_KN if STATIC_STRIDE_KN >= 0 else stride_kn,
        stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        stride_oz, stride_oh, stride_om, stride_on,
        HQ=HQ, HK=HK, ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
        MAX_SEQLENS_Q=MAX_SEQLENS_Q, MAX_SEQLENS_K=MAX_SEQLENS_K,
        IS_CAUSAL=IS_CAUSAL,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N=BLOCK_N,
        PRE_LOAD_V=PRE_LOAD_V, MMA_TYPE=MMA_TYPE, NUM_STAGES=NUM_STAGES,
        ENABLE_CLASS_DIAGONAL_LAZY=(
            IS_CAUSAL and MAX_SEQLENS_Q == 1024
            and Q.dtype.element_ty == gl.float16
        ),
    )


@gluon.jit
def gluon_attn_fwd_short_causal_classes(
    Q, K, V, SM_SCALE: gl.constexpr, L, Out,
    stride_qz: gl.constexpr, stride_qh: gl.constexpr,
    stride_qm: gl.constexpr, stride_qk: gl.constexpr,
    stride_kz: gl.constexpr, stride_kh: gl.constexpr,
    stride_kn: gl.constexpr, stride_kk: gl.constexpr,
    stride_vz: gl.constexpr, stride_vh: gl.constexpr,
    stride_vk: gl.constexpr, stride_vn: gl.constexpr,
    stride_oz: gl.constexpr, stride_oh: gl.constexpr,
    stride_om: gl.constexpr, stride_on: gl.constexpr,
    HQ: gl.constexpr, HK: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,
    MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    BLOCK_N: gl.constexpr, PRE_LOAD_V: gl.constexpr,
    MMA_TYPE: gl.constexpr, NUM_STAGES: gl.constexpr,
):
    """Constant-fold each query position for aligned short causal rows."""
    num_m_blocks: gl.constexpr = MAX_SEQLENS_Q // BLOCK_M
    off_h_q = remap_xcd(gl.program_id(0), HQ)
    query_class = gl.program_id(1)
    off_z = gl.program_id(2)
    off_h_k = off_h_q * HK // HQ

    # Keep all classes in one launch so heavy and light CTAs remain mixed.
    # Each inlined branch still sees a constant query position and deletes
    # causal routing and mask work that cannot execute for that position.
    for i in gl.static_range(num_m_blocks):
        if query_class == i:
            start_m = off_z * 0 + (num_m_blocks - 1 - i)
            _compute_attention_tile(
                Q, K, V, SM_SCALE, L, Out,
                off_z, off_h_q, off_h_k, start_m,
                MAX_SEQLENS_K - MAX_SEQLENS_Q,
                stride_qz, stride_qh, stride_qm, stride_qk,
                stride_kz, stride_kh, stride_kn, stride_kk,
                stride_vz, stride_vh, stride_vk, stride_vn,
                stride_oz, stride_oh, stride_om, stride_on,
                HQ=HQ, HK=HK,
                ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
                MAX_SEQLENS_Q=MAX_SEQLENS_Q,
                MAX_SEQLENS_K=MAX_SEQLENS_K,
                IS_CAUSAL=True,
                BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL,
                BLOCK_N=BLOCK_N, PRE_LOAD_V=PRE_LOAD_V,
                MMA_TYPE=MMA_TYPE, NUM_STAGES=NUM_STAGES,
                ENABLE_CLASS_DIAGONAL_LAZY=True,
            )

# ---------------------------------------------------------------------------
# Metadata / input helpers (adapted from flash_attention.py)
# ---------------------------------------------------------------------------

def _next_power_of_2_at_least_16(n: int) -> int:
    """Return the next power of two greater than or equal to n, with minimum 16."""
    return max(1 << (n - 1).bit_length(), 16)


def run_gluon_attention(q, k, v, o, metadata: MetaData):
    """Run the non-persistent forward kernel and return natural-log FP32 LSE."""
    batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, metadata)
    q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(q, k, v, o, metadata)

    padded_d_model = _next_power_of_2_at_least_16(head_size)

    M = torch.empty((batch, nheads_q, metadata.max_seqlens_q), device=q.device, dtype=torch.float32)

    arch = triton.runtime.driver.active.get_current_target().arch
    mma_type = get_mma_type_for_arch(arch)

    def grid(META):
        return (nheads_q, triton.cdiv(metadata.max_seqlens_q, META['BLOCK_M']), batch)

    use_short_causal_classes = (
        metadata.causal
        and head_size == 128
        and padded_d_model == 128
        and metadata.max_seqlens_q == metadata.max_seqlens_k
        and metadata.max_seqlens_q <= 512
        and metadata.max_seqlens_q % 128 == 0
        and metadata.max_seqlens_k % 64 == 0
    )
    if use_short_causal_classes:
        # Without predicated wave pruning, causal stage-2 lowering is not
        # portable across Gluon compilers. Keep the specialized short route,
        # but use its proven non-pipelined schedule for every short class when
        # the compiler does not expose that capability.
        use_nonpipelined_short_causal = not _HAS_WARP_PREDICATE
        short_llvm_fn_attrs = ()
        if not use_nonpipelined_short_causal and metadata.max_seqlens_q == 512:
            short_llvm_fn_attrs = (
                ("amdgpu-sched-strategy", "max-ilp"),
                *NO_DISPATCH_ID_FN_ATTRS,
            )
        gluon_attn_fwd_short_causal_classes[
            (nheads_q, metadata.max_seqlens_q // 128, batch)
        ](
            q, k, v, metadata.sm_scale, M, o,
            *q_strides, *k_strides, *v_strides, *o_strides,
            HQ=nheads_q, HK=nheads_k, ACTUAL_BLOCK_DMODEL=head_size,
            MAX_SEQLENS_Q=metadata.max_seqlens_q,
            MAX_SEQLENS_K=metadata.max_seqlens_k,
            BLOCK_M=128, BLOCK_DMODEL=128, BLOCK_N=64,
            PRE_LOAD_V=False, MMA_TYPE=mma_type,
            NUM_STAGES=1 if use_nonpipelined_short_causal else 2,
            num_warps=4,
            num_stages=1 if use_nonpipelined_short_causal else 3,
            waves_per_eu=0,
            llvm_fn_attrs=short_llvm_fn_attrs,
        )
        return M

    # The aligned D128 BM128/BM256 schedules benefit when K's actual row
    # stride reaches address lowering as a compile-time value.  Keep the
    # dynamic argument and full autotuner for every other schedule family;
    # this specializes arbitrary dense BHSD/BSHD physical strides rather than
    # assuming contiguous storage.
    use_causal_krow_specialization = (
        metadata.causal
        and not metadata.varlen
        and metadata.layout in ("bhsd", "bshd")
        and head_size == 128
        and padded_d_model == 128
        and metadata.max_seqlens_q == metadata.max_seqlens_k
        and metadata.max_seqlens_q % 256 == 0
        and metadata.max_seqlens_k % 64 == 0
        and 16 <= metadata.max_seqlens_k // 64 <= 64
    )
    use_noncausal_split_krow_specialization = (
        not metadata.causal
        and nheads_q == nheads_k
        and not metadata.varlen
        and metadata.layout in ("bhsd", "bshd")
        and head_size == 128
        and padded_d_model == 128
        and metadata.max_seqlens_q == metadata.max_seqlens_k
        and metadata.max_seqlens_q % 256 == 0
        and metadata.max_seqlens_k % 64 == 0
        and (
            (q.dtype == torch.bfloat16
             and metadata.max_seqlens_k >= 1024)
            or (q.dtype == torch.float16
                and metadata.max_seqlens_k >= 2048)
        )
    )
    use_krow_specialization = (
        use_causal_krow_specialization
        or use_noncausal_split_krow_specialization
    )
    use_short_qrow_specialization = (
        not metadata.varlen
        and metadata.layout in ("bhsd", "bshd")
        and (
            metadata.max_seqlens_k <= 512
            or (metadata.causal and metadata.max_seqlens_k <= 1024)
        )
    )

    gluon_attn_fwd[grid](
        q, k, v, metadata.sm_scale, M, o,
        *q_strides, *k_strides, *v_strides, *o_strides,
        HQ=nheads_q, HK=nheads_k, ACTUAL_BLOCK_DMODEL=head_size,
        MAX_SEQLENS_Q=metadata.max_seqlens_q, MAX_SEQLENS_K=metadata.max_seqlens_k,
        IS_CAUSAL=metadata.causal,
        BLOCK_DMODEL=padded_d_model,
        MMA_TYPE=mma_type,
        STATIC_STRIDE_KN=(
            k_strides[2] if use_krow_specialization else -1
        ),
        STATIC_STRIDE_QM=(
            q_strides[2]
            if use_short_qrow_specialization else -1
        ),
    )
    return M


def flash_attn_gluon_raw(
    q,
    k,
    v,
    *,
    softmax_scale: float,
    causal: bool,
    qkv_format: str,
):
    """Return (out, lse) for raw BSHD or BHSD tensors.

    q, k, and v are contiguous in their declared raw shapes with unit inner
    stride. out has the same raw shape and layout as q. lse is natural-log
    FP32 with shape [B, Hq, Sq].
    """
    if qkv_format not in ("bshd", "bhsd"):
        raise ValueError(
            f"qkv_format must be 'bshd' or 'bhsd', got {qkv_format!r}"
        )
    if any(tensor.ndim != 4 for tensor in (q, k, v)):
        raise ValueError("q, k, and v must all be rank-4 tensors")
    if not all(tensor.is_cuda for tensor in (q, k, v)):
        raise ValueError("q, k, and v must all be CUDA tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q, k, and v must use float16 or bfloat16")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")
    if any(size <= 0 for tensor in (q, k, v) for size in tensor.shape):
        raise ValueError("q, k, and v dimensions must all be positive")
    if any(tensor.stride(-1) != 1 for tensor in (q, k, v)):
        raise ValueError("q, k, and v must have unit stride in the head dimension")
    layout_is_valid = all(tensor.is_contiguous() for tensor in (q, k, v))
    if not layout_is_valid:
        raise ValueError(
            f"q, k, and v storage does not match qkv_format={qkv_format!r}"
        )

    if qkv_format == "bshd":
        batch, seqlen_q, nheads_q, head_size_q = q.shape
        batch_kv, seqlen_k, nheads_k, head_size_k = k.shape
    else:
        batch, nheads_q, seqlen_q, head_size_q = q.shape
        batch_kv, nheads_k, seqlen_k, head_size_k = k.shape

    if batch != batch_kv:
        raise ValueError("q, k, and v must have the same batch size")
    if head_size_q != head_size_k:
        raise ValueError("q, k, and v must have the same head dimension")
    if head_size_q > 256:
        raise ValueError(
            f"head dimension must be at most 256, got {head_size_q}"
        )
    if nheads_q % nheads_k != 0:
        raise ValueError(
            f"query heads ({nheads_q}) must be divisible by KV heads ({nheads_k})"
        )
    if causal and seqlen_q > seqlen_k:
        raise ValueError(
            "causal attention requires query sequence length to be no greater "
            f"than KV sequence length (got {seqlen_q} > {seqlen_k})"
        )

    metadata = MetaData(sm_scale=float(softmax_scale))
    metadata.layout = qkv_format
    metadata.max_seqlens_q = seqlen_q
    metadata.max_seqlens_k = seqlen_k
    if causal:
        metadata.need_causal()

    out = torch.empty_like(q)
    lse = run_gluon_attention(q, k, v, out, metadata)
    return out, lse


# fmt: on
