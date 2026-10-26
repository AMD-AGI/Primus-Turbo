###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Standalone SwiGLU forward/backward kernels (FlyDSL).

The gated activation is parametrized by :class:`~primus_turbo.flydsl.utils.glu_activation.GLUActivation`
(SiLU-SwiGLU by default, ``swigluoai`` for MiniMax-M3), baked in at compile time.
"""

from __future__ import annotations

import itertools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.math as fmath
import torch
from flydsl._mlir.dialects import vector as _vector
from flydsl.expr.primitive import get_dyn_shared
from flydsl.expr.primitive import ptrtoint as _fly_ptrtoint

from primus_turbo.flydsl.mega.tune_utils import (
    Config,
    autotune,
)
from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
)
from primus_turbo.flydsl.utils.glu_activation import (
    GLUActivation,
    activation_constexpr,
)
from primus_turbo.flydsl.utils.prims import ld, st

_VEC = 8
_WARP = 64
_POOL_BLOCK_M = 256  # pool-block granularity (fixed policy; matches symm_buffer.BLOCK_M)


def _bound(value, vtype):
    """Vector constant for a clamp bound, or None for an infinite (absent) one."""
    return None if math.isinf(value) else fx.arith.constant_vector(value, vtype)


def _clampv(x, lo, hi):
    if lo is not None:
        x = fx.arith.maximumf(x, lo)
    if hi is not None:
        x = fx.arith.minimumf(x, hi)
    return x


def _mul_const(x, c, vtype):
    # c == 1 emits nothing, so the default spec's IR is the historical one.
    return x if c == 1.0 else fx.arith.mulf(x, fx.arith.constant_vector(c, vtype))


def _add_const(x, c, vtype):
    # Skipping c == 0 is not only cheaper: -0.0 + 0.0 is +0.0, which would flip the sign of a zero.
    return x if c == 0.0 else fx.arith.addf(x, fx.arith.constant_vector(c, vtype))


def _load_gate_up(acc_rsrc, bias_rsrc, t2e_rsrc, m, row_base, col, I: int, BM: int, f32v):
    """The row's fc1 output halves in f32; with an expert bias (``bias_rsrc``) added before any clamp,
    the order Megatron's fc1 (TE GroupedLinear with ``skip_bias_add=False``) applies it in."""
    gate = buffer_load(acc_rsrc, row_base + col, vec_width=_VEC, dtype=fx.T.bf16)
    up = buffer_load(acc_rsrc, row_base + fx.Int32(I) + col, vec_width=_VEC, dtype=fx.T.bf16)
    gate, up = fx.arith.extf(f32v, gate), fx.arith.extf(f32v, up)
    if bias_rsrc is not None:
        # every pool block belongs to one expert
        bias_base = buffer_load(t2e_rsrc, m // fx.Int32(BM), vec_width=1, dtype=fx.T.i32) * fx.Int32(2 * I)
        gate = fx.arith.addf(
            gate,
            fx.arith.extf(f32v, buffer_load(bias_rsrc, bias_base + col, vec_width=_VEC, dtype=fx.T.bf16)),
        )
        up = fx.arith.addf(
            up,
            fx.arith.extf(
                f32v, buffer_load(bias_rsrc, bias_base + fx.Int32(I) + col, vec_width=_VEC, dtype=fx.T.bf16)
            ),
        )
    return gate, up


def _make_swiglu(
    I: int, with_scale: bool, BM: int, grid_x: int, block_threads: int, act: tuple, with_bias: bool = False
):
    alpha, glu_offset, gate_lo, gate_hi, up_clamp = act
    two_I = 2 * I
    cols_per_block = _VEC * block_threads
    assert I % _VEC == 0, f"I={I} not divisible by vec width {_VEC}"

    partial_tail = I % cols_per_block != 0

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def swiglu_kernel(
        ACC1: fx.Tensor,
        ACT: fx.Tensor,
        SCALE: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        BIAS: fx.Tensor,
        TILE_TO_EXPERT: fx.Tensor,
    ):
        f32v = fx.T.vec(_VEC, fx.T.f32)
        bf16v = fx.T.vec(_VEC, fx.T.bf16)
        thread_index = fx.thread_idx.x
        block_index_x, block_index_y, _ = fx.block_idx
        col = block_index_y * fx.Int32(cols_per_block) + thread_index * fx.Int32(_VEC)

        acc_rsrc = create_buffer_resource(ACC1, max_size=True)
        act_rsrc = create_buffer_resource(ACT, max_size=True)
        scale_rsrc = create_buffer_resource(SCALE, max_size=True) if with_scale else None
        bias_rsrc = create_buffer_resource(BIAS, max_size=True) if with_bias else None
        t2e_rsrc = create_buffer_resource(TILE_TO_EXPERT, max_size=True) if with_bias else None
        g_lo, g_hi = _bound(gate_lo, f32v), _bound(gate_hi, f32v)
        u_lo, u_hi = _bound(-up_clamp, f32v), _bound(up_clamp, f32v)
        one = fx.arith.constant_vector(1.0, f32v)
        neg1 = fx.arith.constant_vector(-1.0, f32v)

        def compute_row(m):
            # i32 offset: assumes M*two_I < 2^31 elements (holds for all EP pool sizes).
            row_base = m * fx.Int32(two_I)
            if not partial_tail or col < fx.Int32(I):
                gate, up = _load_gate_up(acc_rsrc, bias_rsrc, t2e_rsrc, m, row_base, col, I, BM, f32v)
                g = _clampv(gate, g_lo, g_hi)
                u = _add_const(_clampv(up, u_lo, u_hi), glu_offset, f32v)
                denom = fx.arith.addf(one, fmath.exp(fx.arith.mulf(_mul_const(g, alpha, f32v), neg1)))
                act = fx.arith.mulf(fx.arith.divf(g, denom), u)
                if with_scale:
                    sc = buffer_load(scale_rsrc, m, vec_width=1, dtype=fx.T.f32)
                    act = fx.arith.mulf(act, _vector.broadcast(f32v, sc))
                buffer_store(fx.arith.trunc_f(bf16v, act), act_rsrc, m * fx.Int32(I) + col)

        # grid-stride over real pool rows only (NUM_TILE_BLOCKS * BM); unused tail skipped.
        m_real = buffer_load(
            create_buffer_resource(NUM_TILE_BLOCKS, max_size=True), 0, vec_width=1, dtype=fx.T.i32
        ) * fx.Int32(BM)
        m = block_index_x
        while m < m_real:
            compute_row(m)
            m = m + fx.Int32(grid_x)

    return swiglu_kernel


def _make_swiglu_bwd(
    I: int,
    with_scale: bool,
    BM: int,
    grid_x: int,
    block_threads: int,
    with_gate: bool,
    with_act_w: bool,
    act: tuple,
    with_bias: bool = False,
):
    alpha, glu_offset, gate_lo, gate_hi, up_clamp = act
    two_I = 2 * I
    # Gate reduction folds all I columns per row; block spans multiple warps.
    if with_gate:
        assert block_threads % _WARP == 0, f"block_threads must be a multiple of {_WARP}, got {block_threads}"
    cols_per_block = _VEC * block_threads
    assert I % _VEC == 0, f"I={I} not divisible by vec width {_VEC}"
    # constexpr: ceil-tile columns; last tile is partial and needs lane guards.
    n_col_tiles = (I + cols_per_block - 1) // cols_per_block
    partial_tail = I % cols_per_block != 0
    n_warps = block_threads // _WARP

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def swiglu_bwd_kernel(
        DACT: fx.Tensor,
        ACC1: fx.Tensor,
        DACC1: fx.Tensor,
        SCALE: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        GRAD_GATE: fx.Tensor,
        ACT_W: fx.Tensor,
        BIAS: fx.Tensor,
        TILE_TO_EXPERT: fx.Tensor,
    ):
        f32v = fx.T.vec(_VEC, fx.T.f32)
        bf16v = fx.T.vec(_VEC, fx.T.bf16)
        thread_index = fx.thread_idx.x
        block_index_x, block_index_y, _ = fx.block_idx
        # LDS scratch for cross-warp gate reduction (one f32 slot per warp).
        lds_base = fx.as_ir_value(_fly_ptrtoint(get_dyn_shared())) if with_gate else None

        dact_rsrc = create_buffer_resource(DACT, max_size=True)
        acc_rsrc = create_buffer_resource(ACC1, max_size=True)
        dacc_rsrc = create_buffer_resource(DACC1, max_size=True)
        scale_rsrc = create_buffer_resource(SCALE, max_size=True) if with_scale else None
        grad_gate_rsrc = create_buffer_resource(GRAD_GATE, max_size=True) if with_gate else None
        act_w_rsrc = create_buffer_resource(ACT_W, max_size=True) if with_act_w else None
        bias_rsrc = create_buffer_resource(BIAS, max_size=True) if with_bias else None
        t2e_rsrc = create_buffer_resource(TILE_TO_EXPERT, max_size=True) if with_bias else None
        g_lo, g_hi = _bound(gate_lo, f32v), _bound(gate_hi, f32v)
        u_lo, u_hi = _bound(-up_clamp, f32v), _bound(up_clamp, f32v)
        one = fx.arith.constant_vector(1.0, f32v)
        zero = fx.arith.constant_vector(0.0, f32v)
        neg1 = fx.arith.constant_vector(-1.0, f32v)

        def compute_tile(m, col, gate_parts=None, guard=False):
            row_base = m * fx.Int32(two_I)
            # fx predicate for the partial tail (None when the tile is fully in bounds).
            in_bounds = (col < fx.Int32(I)) if guard else None
            # The resources are sized to the tensors, so a tail lane's load has to be
            # folded back inside them: on the last row there is no next row to spill
            # into. The value is discarded either way -- the stores below are guarded
            # and the gate contribution is zeroed.
            ld_col = fx.arith.select(in_bounds, col, fx.Int32(0)) if guard else col
            gate, up = _load_gate_up(acc_rsrc, bias_rsrc, t2e_rsrc, m, row_base, ld_col, I, BM, f32v)
            d = fx.arith.extf(
                f32v, buffer_load(dact_rsrc, m * fx.Int32(I) + ld_col, vec_width=_VEC, dtype=fx.T.bf16)
            )
            d_raw = d
            if with_scale:
                sc = buffer_load(scale_rsrc, m, vec_width=1, dtype=fx.T.f32)
                d = fx.arith.mulf(d, _vector.broadcast(f32v, sc))

            gc = _clampv(gate, g_lo, g_hi)
            uc = _clampv(up, u_lo, u_hi)
            # d(u + offset)/du = 1: only the gate side sees the offset.
            uo = _add_const(uc, glu_offset, f32v)
            agc = _mul_const(gc, alpha, f32v)
            sig = fx.arith.divf(one, fx.arith.addf(one, fmath.exp(fx.arith.mulf(agc, neg1))))
            s = fx.arith.mulf(gc, sig)
            duc = fx.arith.mulf(d, s)
            # d/dg [g * sigmoid(alpha g)] = sig * (1 + alpha g (1 - sig))
            dsilu = fx.arith.mulf(sig, fx.arith.addf(one, fx.arith.mulf(agc, fx.arith.subf(one, sig))))
            dgc = fx.arith.mulf(fx.arith.mulf(d, uo), dsilu)
            mg = fx.arith.select(fx.arith.cmpf(fx.arith.CmpFPredicate.OEQ, gate, gc), one, zero)
            mu = fx.arith.select(fx.arith.cmpf(fx.arith.CmpFPredicate.OEQ, up, uc), one, zero)
            dgate = fx.arith.trunc_f(bf16v, fx.arith.mulf(dgc, mg))
            dup = fx.arith.trunc_f(bf16v, fx.arith.mulf(duc, mu))

            if not guard or in_bounds:
                # Padding rows write garbage dx here; upstream must zero Xpad to keep dW1 clean.
                buffer_store(dgate, dacc_rsrc, row_base + col)
                buffer_store(dup, dacc_rsrc, row_base + fx.Int32(I) + col)
                if with_act_w:
                    sc_w = buffer_load(scale_rsrc, m, vec_width=1, dtype=fx.T.f32)
                    act_w_v = fx.arith.mulf(fx.arith.mulf(s, uo), _vector.broadcast(f32v, sc_w))
                    buffer_store(fx.arith.trunc_f(bf16v, act_w_v), act_w_rsrc, m * fx.Int32(I) + col)

            if with_gate:
                contrib = fx.arith.mulf(d_raw, fx.arith.mulf(s, uo))
                if guard:
                    # OOB lanes read garbage; zero them so the row sum stays clean.
                    contrib = fx.arith.select(in_bounds, contrib, zero)
                gate_parts.append(
                    fx.arith.ArithValue(_vector.reduction(fx.T.f32, _vector.CombiningKind.ADD, contrib))
                )

        def compute_row(m):
            if with_gate:
                col0 = thread_index * fx.Int32(_VEC)
                gate_parts = []
                for ct in fx.range_constexpr(n_col_tiles):
                    guard = partial_tail and (ct == n_col_tiles - 1)
                    compute_tile(m, fx.Int32(ct * cols_per_block) + col0, gate_parts, guard=guard)
                partial = gate_parts[0]
                for p in gate_parts[1:]:
                    partial = partial.addf(p)
                # Butterfly warp reduce: every lane ends with its warp's sum.
                wave_off = 1
                while wave_off < 64:
                    partial = partial.addf(fx.arith.ArithValue(partial.shuffle_xor(wave_off, 64)))
                    wave_off = wave_off * 2
                # Each warp leader publishes its sum; thread 0 folds across warps.
                if thread_index % fx.Int32(_WARP) == fx.Int32(0):
                    st(lds_base, thread_index // fx.Int32(_WARP), partial, scope="workgroup", space=3)
                fx.gpu.barrier()
                if thread_index == fx.Int32(0):
                    total = fx.arith.ArithValue(
                        ld(lds_base, fx.Int32(0), scope="workgroup", space=3, dtype=fx.T.f32)
                    )
                    for w in range(1, n_warps):
                        total = total.addf(
                            fx.arith.ArithValue(
                                ld(lds_base, fx.Int32(w), scope="workgroup", space=3, dtype=fx.T.f32)
                            )
                        )
                    buffer_store(total, grad_gate_rsrc, m)
                # Guard LDS before next loop iteration overwrites it (WAR).
                fx.gpu.barrier()
            else:
                compute_tile(
                    m,
                    block_index_y * fx.Int32(cols_per_block) + thread_index * fx.Int32(_VEC),
                    guard=partial_tail,
                )

        # grid-stride over real pool rows only (NUM_TILE_BLOCKS * BM); skip the unused tail.
        m_real = buffer_load(
            create_buffer_resource(NUM_TILE_BLOCKS, max_size=True), 0, vec_width=1, dtype=fx.T.i32
        ) * fx.Int32(BM)
        m = block_index_x
        while m < m_real:
            compute_row(m)
            m = m + fx.Int32(grid_x)

    return swiglu_bwd_kernel


@autotune(
    configs=[
        Config(grid_x=grid_x, block_threads=block_threads)
        for grid_x, block_threads in itertools.product((1024, 2048, 4096), (256, 512))
    ],
    key=["I", "with_scale", "BM", "with_bias"],
)
@flyc.jit
def _compiled_swiglu(
    ACC1,
    ACT,
    SCALE,
    NUM_TILE_BLOCKS,
    BIAS,
    TILE_TO_EXPERT,
    I: fx.Constexpr[int],
    with_scale: fx.Constexpr[int],
    BM: fx.Constexpr[int],
    activation: fx.Constexpr[tuple],
    with_bias: fx.Constexpr[int],
    grid_x: fx.Constexpr[int],
    block_threads: fx.Constexpr[int],
    stream: fx.Stream,
):
    cols_per_block = _VEC * block_threads
    n_col_tiles = (I + cols_per_block - 1) // cols_per_block
    kernel = _make_swiglu(I, bool(with_scale), BM, grid_x, block_threads, activation, bool(with_bias))
    kernel(ACC1, ACT, SCALE, NUM_TILE_BLOCKS, BIAS, TILE_TO_EXPERT).launch(
        grid=(grid_x, n_col_tiles, 1),
        block=(block_threads, 1, 1),
        stream=stream,
    )


@autotune(
    configs=[
        Config(grid_x=grid_x, block_threads=block_threads)
        for grid_x, block_threads in itertools.product((2048, 4096), (256, 512))
    ],
    rep=5,
    key=["I", "with_scale", "BM", "with_gate", "with_act_w", "with_bias"],
)
@flyc.jit
def _compiled_swiglu_bwd(
    DACT,
    ACC1,
    DACC1,
    SCALE,
    NUM_TILE_BLOCKS,
    GRAD_GATE,
    ACT_W,
    BIAS,
    TILE_TO_EXPERT,
    I: fx.Constexpr[int],
    with_scale: fx.Constexpr[int],
    BM: fx.Constexpr[int],
    with_gate: fx.Constexpr[int],
    with_act_w: fx.Constexpr[int],
    activation: fx.Constexpr[tuple],
    with_bias: fx.Constexpr[int],
    grid_x: fx.Constexpr[int],
    block_threads: fx.Constexpr[int],
    stream: fx.Stream,
):
    cols_per_block = _VEC * block_threads
    n_col_tiles = (I + cols_per_block - 1) // cols_per_block
    kernel = _make_swiglu_bwd(
        I,
        bool(with_scale),
        BM,
        grid_x,
        block_threads,
        bool(with_gate),
        bool(with_act_w),
        activation,
        bool(with_bias),
    )
    grid_y = 1 if with_gate else n_col_tiles
    # One f32 LDS slot per warp for cross-warp gate reduction.
    smem = (block_threads // _WARP) * 4 if with_gate else 0
    kernel(DACT, ACC1, DACC1, SCALE, NUM_TILE_BLOCKS, GRAD_GATE, ACT_W, BIAS, TILE_TO_EXPERT).launch(
        grid=(grid_x, grid_y, 1),
        block=(block_threads, 1, 1),
        stream=stream,
        smem=smem,
    )


def swiglu_backward_flydsl_kernel(
    dact: torch.Tensor,
    x: torch.Tensor,
    num_tile_blocks: torch.Tensor,
    scale: torch.Tensor | None = None,
    return_gate: bool = False,
    return_act_w: bool = False,
    activation: GLUActivation | None = None,
    bias: torch.Tensor | None = None,
    tile_to_expert: torch.Tensor | None = None,
):
    """``bias`` [E_local, 2I] is the experts' fc1 bias, added to ``x`` before the activation; row m
    takes the bias of expert ``tile_to_expert[m // 256]``. ``dx`` is then also the bias gradient."""
    M, two_I = x.shape
    assert two_I % 2 == 0, f"x last dim must be even (gate||up), got {two_I}"
    I = two_I // 2
    assert dact.size(1) == I, f"dact[...,I] vs x[...,2I] mismatch: {dact.shape} {x.shape}"
    if return_act_w:
        assert scale is not None, "return_act_w needs scale (the per-row routing weight)"
    with_bias = _check_bias(bias, tile_to_expert, two_I)
    x = x.contiguous()
    dact = dact.contiguous()
    dx = torch.empty((M, two_I), dtype=torch.bfloat16, device=x.device)
    with_scale = scale is not None
    scale_d = scale if with_scale else x
    ntb_d = num_tile_blocks
    grad_gate = torch.empty((M,), dtype=torch.float32, device=x.device) if return_gate else x
    act_w = torch.empty((M, I), dtype=torch.bfloat16, device=x.device) if return_act_w else x
    _compiled_swiglu_bwd(
        DACT=dact,
        ACC1=x,
        DACC1=dx,
        SCALE=scale_d,
        NUM_TILE_BLOCKS=ntb_d,
        GRAD_GATE=grad_gate,
        ACT_W=act_w,
        BIAS=bias.contiguous() if with_bias else x,
        TILE_TO_EXPERT=tile_to_expert if with_bias else ntb_d,
        I=I,
        with_scale=int(with_scale),
        BM=_POOL_BLOCK_M,
        with_gate=int(return_gate),
        with_act_w=int(return_act_w),
        activation=activation_constexpr(activation),
        with_bias=int(with_bias),
        stream=torch.cuda.current_stream(),
    )
    if return_gate and return_act_w:
        return dx, grad_gate, act_w
    if return_gate:
        return dx, grad_gate
    if return_act_w:
        return dx, act_w
    return dx


def swiglu_flydsl_kernel(
    x: torch.Tensor,
    num_tile_blocks: torch.Tensor,
    scale: torch.Tensor | None = None,
    stream=None,
    activation: GLUActivation | None = None,
    bias: torch.Tensor | None = None,
    tile_to_expert: torch.Tensor | None = None,
) -> torch.Tensor:
    """``bias`` / ``tile_to_expert``: the experts' fc1 bias, see :func:`swiglu_backward_flydsl_kernel`."""
    if stream is None:
        stream = torch.cuda.current_stream()
    M, two_I = x.shape
    assert two_I % 2 == 0, f"x last dim must be even (gate||up), got {two_I}"
    I = two_I // 2
    with_bias = _check_bias(bias, tile_to_expert, two_I)
    act = torch.empty((M, I), dtype=torch.bfloat16, device=x.device)
    with_scale = scale is not None
    scale_d = scale if with_scale else x
    ntb_d = num_tile_blocks
    _compiled_swiglu(
        ACC1=x,
        ACT=act,
        SCALE=scale_d,
        NUM_TILE_BLOCKS=ntb_d,
        BIAS=bias.contiguous() if with_bias else x,
        TILE_TO_EXPERT=tile_to_expert if with_bias else ntb_d,
        I=I,
        with_scale=int(with_scale),
        BM=_POOL_BLOCK_M,
        activation=activation_constexpr(activation),
        with_bias=int(with_bias),
        stream=stream,
    )
    return act


def _check_bias(bias, tile_to_expert, two_I) -> bool:
    if bias is None:
        return False
    assert tile_to_expert is not None, "an expert bias needs the handle's tile_to_expert"
    assert bias.dim() == 2 and bias.size(1) == two_I, (
        f"bias must be [E_local, {two_I}], got {tuple(bias.shape)}"
    )
    assert bias.dtype == torch.bfloat16, f"bias must be bf16, got {bias.dtype}"
    assert tile_to_expert.dtype == torch.int32, f"tile_to_expert must be int32, got {tile_to_expert.dtype}"
    return True
