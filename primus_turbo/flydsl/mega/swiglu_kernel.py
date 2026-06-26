###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Mega-MoE SwiGLU epilogue kernels (FlyDSL).

Fused SwiGLU forward/backward in one FlyDSL pass.  Optional features:
  - with_scale: per-row routing-weight scale (act[m,:] *= SCALE[m])
  - with_bound: no-sync grid-stride over a device-bounded real-row count (NUM_TILE_BLOCKS)
  - with_gate (bwd): gate gradient (d w.r.t. the per-row scale); 1 wave/block, single store/row (no zero-init)

Depends only on ``flydsl`` + ``torch`` (no upstream ``kernels`` package).
"""
from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.math as fmath
import torch
from flydsl._mlir.dialects import vector as _vector
from flydsl.expr.buffer_ops import buffer_load, buffer_store, create_buffer_resource

from primus_turbo.flydsl.mega.symm_buffer import get_symm_buffer_for_mega_moe

# SwiGLU clamp contract: baked into the kernels; must match the reference.
ACTIVATION_CLAMP = 10.0

_VEC = 8
_WARP = 64  # gfx950 wavefront (gate-grad variant runs 1 wave/block for a single store)
_BLOCK_THREADS = 128
_COLS_PER_BLOCK = _VEC * _BLOCK_THREADS  # 1024


def _clampv(x, lo, hi):
    """Vectorized clamp(x, lo, hi) (pure expression; inlined into the kernels)."""
    return fx.arith.minimumf(fx.arith.maximumf(x, lo), hi)


def _make_swiglu(I: int, with_scale: bool = False, with_bound: bool = False, BM: int = 0, GRID_X: int = 0):
    two_I = 2 * I
    assert I % _COLS_PER_BLOCK == 0, f"I={I} not divisible by {_COLS_PER_BLOCK}"

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def swiglu_kernel(ACC1: fx.Tensor, ACT: fx.Tensor, SCALE: fx.Tensor, NUM_TILE_BLOCKS: fx.Tensor):
        f32v = fx.T.VectorType.get([_VEC], fx.T.f32())
        bf16v = fx.T.VectorType.get([_VEC], fx.T.bf16())
        thread_index = fx.thread_idx.x
        # M-major grid: M can exceed HIP's grid.y cap
        block_index_x, block_index_y, _ = fx.block_idx
        col = block_index_y * fx.Int32(_COLS_PER_BLOCK) + thread_index * fx.Int32(_VEC)

        acc_rsrc = create_buffer_resource(ACC1, max_size=True)
        act_rsrc = create_buffer_resource(ACT, max_size=True)
        scale_rsrc = create_buffer_resource(SCALE, max_size=True) if with_scale else None
        # loop-invariant constants (hoisted out of the per-row body)
        lo = fx.arith.constant_vector(-ACTIVATION_CLAMP, f32v)
        hi = fx.arith.constant_vector(ACTIVATION_CLAMP, f32v)
        one = fx.arith.constant_vector(1.0, f32v)
        neg1 = fx.arith.constant_vector(-1.0, f32v)

        def compute_row(m):
            row_base = m * fx.Int32(two_I)
            gate = buffer_load(acc_rsrc, row_base + col, vec_width=_VEC, dtype=fx.T.bf16())
            up = buffer_load(acc_rsrc, row_base + fx.Int32(I) + col, vec_width=_VEC, dtype=fx.T.bf16())
            g = _clampv(fx.arith.extf(f32v, gate), lo, hi)
            u = _clampv(fx.arith.extf(f32v, up), lo, hi)
            # silu(g) = g / (1 + exp(-g))
            denom = fx.arith.addf(one, fmath.exp(fx.arith.mulf(g, neg1)))
            act = fx.arith.mulf(fx.arith.divf(g, denom), u)
            if with_scale:
                # per-row routing-weight scale: act[m,:] *= SCALE[m]
                sc = buffer_load(scale_rsrc, m, vec_width=1, dtype=fx.T.f32())
                act = fx.arith.mulf(act, _vector.broadcast(f32v, sc))
            buffer_store(fx.arith.trunc_f(bf16v, act), act_rsrc, m * fx.Int32(I) + col)

        if with_bound:
            # no-sync grid-stride over device-bounded real rows
            m_real = buffer_load(
                create_buffer_resource(NUM_TILE_BLOCKS, max_size=True), 0, vec_width=1, dtype=fx.T.i32()
            ) * fx.Int32(BM)
            m = block_index_x
            while m < m_real:
                compute_row(m)
                m = m + fx.Int32(GRID_X)
        else:
            compute_row(block_index_x)

    # bounded: GRID_X persistent blocks; unbounded: M blocks
    grid_x = GRID_X if with_bound else None

    @flyc.jit
    def launch(
        ACC1: fx.Tensor,
        ACT: fx.Tensor,
        SCALE: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        M: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        swiglu_kernel(ACC1, ACT, SCALE, NUM_TILE_BLOCKS).launch(
            grid=(grid_x if with_bound else M, I // _COLS_PER_BLOCK, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


def _make_swiglu_bwd(
    I: int,
    with_scale: bool = False,
    with_bound: bool = False,
    BM: int = 0,
    GRID_X: int = 0,
    with_gate: bool = False,
    with_act_w: bool = False,
):
    two_I = 2 * I
    # gate: 1 wave/block loops all tiles -> single store (empty-safe); non-gate: 2-wave grid_y-tiled
    block_threads = _WARP if with_gate else _BLOCK_THREADS
    cols_per_block = _VEC * block_threads
    assert I % cols_per_block == 0, f"I={I} not divisible by {cols_per_block}"
    n_col_tiles = I // cols_per_block

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def swiglu_bwd_kernel(
        DACT: fx.Tensor,
        ACC1: fx.Tensor,
        DACC1: fx.Tensor,
        SCALE: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        GRAD_GATE: fx.Tensor,
        ACT_W: fx.Tensor,
    ):
        f32v = fx.T.VectorType.get([_VEC], fx.T.f32())
        bf16v = fx.T.VectorType.get([_VEC], fx.T.bf16())
        thread_index = fx.thread_idx.x
        # M-major grid: M can exceed HIP's grid.y cap
        block_index_x, block_index_y, _ = fx.block_idx

        dact_rsrc = create_buffer_resource(DACT, max_size=True)
        acc_rsrc = create_buffer_resource(ACC1, max_size=True)
        dacc_rsrc = create_buffer_resource(DACC1, max_size=True)
        scale_rsrc = create_buffer_resource(SCALE, max_size=True) if with_scale else None
        grad_gate_rsrc = create_buffer_resource(GRAD_GATE, max_size=True) if with_gate else None
        act_w_rsrc = create_buffer_resource(ACT_W, max_size=True) if with_act_w else None
        # loop-invariant constants (hoisted out of the per-tile body)
        lo = fx.arith.constant_vector(-ACTIVATION_CLAMP, f32v)
        hi = fx.arith.constant_vector(ACTIVATION_CLAMP, f32v)
        one = fx.arith.constant_vector(1.0, f32v)
        zero = fx.arith.constant_vector(0.0, f32v)
        neg1 = fx.arith.constant_vector(-1.0, f32v)

        # store dgate/dup for one col-tile; with_gate appends the tile's gate-grad partial (list: AST rewriter strips nested returns)
        def compute_tile(m, col, gate_parts=None):
            row_base = m * fx.Int32(two_I)
            gate = fx.arith.extf(
                f32v, buffer_load(acc_rsrc, row_base + col, vec_width=_VEC, dtype=fx.T.bf16())
            )
            up = fx.arith.extf(
                f32v, buffer_load(acc_rsrc, row_base + fx.Int32(I) + col, vec_width=_VEC, dtype=fx.T.bf16())
            )
            d = fx.arith.extf(
                f32v, buffer_load(dact_rsrc, m * fx.Int32(I) + col, vec_width=_VEC, dtype=fx.T.bf16())
            )
            d_raw = d  # unscaled dact, for the gate gradient
            if with_scale:
                # per-row routing-weight scale: d *= SCALE[m]
                sc = buffer_load(scale_rsrc, m, vec_width=1, dtype=fx.T.f32())
                d = fx.arith.mulf(d, _vector.broadcast(f32v, sc))

            gc = _clampv(gate, lo, hi)
            uc = _clampv(up, lo, hi)
            sig = fx.arith.divf(one, fx.arith.addf(one, fmath.exp(fx.arith.mulf(gc, neg1))))
            s = fx.arith.mulf(gc, sig)
            # duc = d * s ;  dgc = d * uc * sig*(1 + gc*(1-sig))
            duc = fx.arith.mulf(d, s)
            dsilu = fx.arith.mulf(sig, fx.arith.addf(one, fx.arith.mulf(gc, fx.arith.subf(one, sig))))
            dgc = fx.arith.mulf(fx.arith.mulf(d, uc), dsilu)
            # clamp-mask: in-range iff clamp(x)==x
            mg = fx.arith.select(fx.arith.cmpf(fx.arith.CmpFPredicate.OEQ, gate, gc), one, zero)
            mu = fx.arith.select(fx.arith.cmpf(fx.arith.CmpFPredicate.OEQ, up, uc), one, zero)
            dgate = fx.arith.trunc_f(bf16v, fx.arith.mulf(dgc, mg))
            dup = fx.arith.trunc_f(bf16v, fx.arith.mulf(duc, mu))
            buffer_store(dgate, dacc_rsrc, row_base + col)
            buffer_store(dup, dacc_rsrc, row_base + fx.Int32(I) + col)

            if with_act_w:
                # weighted fwd act = silu(gc)*uc*SCALE[m] -> dW2 B operand (folds the host act_w mul).
                # re-load scale: rewriter scopes per-if-branch, so the with_scale `sc` isn't visible here.
                sc_w = buffer_load(scale_rsrc, m, vec_width=1, dtype=fx.T.f32())
                act_w_v = fx.arith.mulf(fx.arith.mulf(s, uc), _vector.broadcast(f32v, sc_w))
                buffer_store(fx.arith.trunc_f(bf16v, act_w_v), act_w_rsrc, m * fx.Int32(I) + col)

            if with_gate:
                # gate-grad partial = sum over this tile's columns (per thread)
                contrib = fx.arith.mulf(d_raw, fx.arith.mulf(s, uc))
                gate_parts.append(
                    fx.arith.ArithValue(_vector.reduction(fx.T.f32(), _vector.CombiningKind.ADD, contrib))
                )

        def compute_row(m):
            if with_gate:
                # loop tiles -> wave all-reduce -> lane 0 stores the full-row gate grad once
                col0 = thread_index * fx.Int32(_VEC)
                gate_parts = []  # range_constexpr -> Python-level unroll (no scf.for)
                for ct in fx.range_constexpr(n_col_tiles):
                    compute_tile(m, fx.Int32(ct * cols_per_block) + col0, gate_parts)
                partial = gate_parts[0]
                for p in gate_parts[1:]:
                    partial = partial.addf(p)
                wave_off = 1
                while wave_off < 64:
                    partial = partial.addf(fx.arith.ArithValue(partial.shuffle_xor(wave_off, 64)))
                    wave_off = wave_off * 2
                if thread_index == fx.Int32(0):
                    buffer_store(partial, grad_gate_rsrc, m)
            else:
                compute_tile(m, block_index_y * fx.Int32(cols_per_block) + thread_index * fx.Int32(_VEC))

        if with_bound:
            # no-sync grid-stride over device-bounded real rows
            m_real = buffer_load(
                create_buffer_resource(NUM_TILE_BLOCKS, max_size=True), 0, vec_width=1, dtype=fx.T.i32()
            ) * fx.Int32(BM)
            m = block_index_x
            while m < m_real:
                compute_row(m)
                m = m + fx.Int32(GRID_X)
        else:
            compute_row(block_index_x)

    grid_x = GRID_X if with_bound else None
    # gate variant loops I internally -> grid_y = 1; else tile I across grid_y.
    grid_y = 1 if with_gate else (I // cols_per_block)

    @flyc.jit
    def launch(
        DACT: fx.Tensor,
        ACC1: fx.Tensor,
        DACC1: fx.Tensor,
        SCALE: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        GRAD_GATE: fx.Tensor,
        ACT_W: fx.Tensor,
        M: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        swiglu_bwd_kernel(DACT, ACC1, DACC1, SCALE, NUM_TILE_BLOCKS, GRAD_GATE, ACT_W).launch(
            grid=(grid_x if with_bound else M, grid_y, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


# grid-stride block count for the no-sync bounded swiglu
_SWIGLU_GRID_X = 8192


def _active_symm_bounds():
    """BM + device real-tile count from the active mega-MoE symm workspace.

    Returns ``(symm.block_m, symm.meta_scalars[1:2])`` so the epilogue rides the same
    device-bounded real-row count as the fused pipeline; ``(0, None)`` (unbounded over
    all rows) when no workspace is active (standalone use)."""
    try:
        symm = get_symm_buffer_for_mega_moe()
    except RuntimeError:
        return 0, None
    return symm.block_m, symm.meta_scalars[1:2]


def _resolve_opts(scale, num_tile_blocks, dummy):
    """Resolve scale/bound flags + device tensors (dummy binds off paths) -> (with_scale, scale_d, with_bound, ntb_d, grid_x)."""
    with_scale = scale is not None
    with_bound = num_tile_blocks is not None
    return (
        with_scale,
        scale if with_scale else dummy,
        with_bound,
        num_tile_blocks if with_bound else dummy,
        _SWIGLU_GRID_X if with_bound else 0,
    )


@functools.lru_cache(maxsize=32)
def _compile(I: int, with_scale: bool = False, with_bound: bool = False, BM: int = 0, GRID_X: int = 0):
    return _make_swiglu(I, with_scale, with_bound, BM, GRID_X)


@functools.lru_cache(maxsize=32)
def _compile_bwd(
    I: int,
    with_scale: bool = False,
    with_bound: bool = False,
    BM: int = 0,
    GRID_X: int = 0,
    with_gate: bool = False,
    with_act_w: bool = False,
):
    return _make_swiglu_bwd(I, with_scale, with_bound, BM, GRID_X, with_gate, with_act_w)


def swiglu_backward(
    dact: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor | None = None,
    return_gate: bool = False,
    return_act_w: bool = False,
):
    """Backward SwiGLU over x[M,2I] given dact[M,I] (optional per-row scale); allocates + returns
    dx[M,2I]. The device-bounded real-row count rides the active symm workspace. return_gate ->
    also grad_gate (empty, single store/row = triton_dist dscale). return_act_w -> also act_w =
    (recomputed fwd act) * scale [M, I] bf16 (the dW2 B operand), folding the host
    ``saved_act * weight`` mul; requires scale (with_gate recommended)."""
    M, two_I = x.shape
    assert two_I % 2 == 0, f"x last dim must be even (gate||up), got {two_I}"
    I = two_I // 2
    assert dact.size(1) == I, f"dact[...,I] vs x[...,2I] mismatch: {dact.shape} {x.shape}"
    if return_act_w:
        assert scale is not None, "return_act_w needs scale (the per-row routing weight)"
    x = x.contiguous()
    dact = dact.contiguous()
    dx = torch.empty((M, two_I), dtype=torch.bfloat16, device=x.device)
    BM, num_tile_blocks = _active_symm_bounds()
    with_scale, scale_d, with_bound, ntb_d, grid_x = _resolve_opts(scale, num_tile_blocks, x)
    # single-store gate grad -> empty (no zeros) is safe; dummy binds the off path.
    grad_gate = torch.empty((M,), dtype=torch.float32, device=x.device) if return_gate else x
    act_w = torch.empty((M, I), dtype=torch.bfloat16, device=x.device) if return_act_w else x
    _compile_bwd(I, with_scale, with_bound, BM, grid_x, return_gate, return_act_w)(
        dact, x, dx, scale_d, ntb_d, grad_gate, act_w, M, stream=torch.cuda.current_stream()
    )
    if return_gate and return_act_w:
        return dx, grad_gate, act_w
    if return_gate:
        return dx, grad_gate
    if return_act_w:
        return dx, act_w
    return dx


def swiglu(
    x: torch.Tensor,
    scale: torch.Tensor | None = None,
    stream=None,
) -> torch.Tensor:
    """Fused SwiGLU over the rows of x[M,2I] in one FlyDSL pass; allocates + returns act[M,I].
    Optional per-row scale; the device-bounded real-row count rides the active symm workspace."""
    if stream is None:
        stream = torch.cuda.current_stream()
    M, two_I = x.shape
    assert two_I % 2 == 0, f"x last dim must be even (gate||up), got {two_I}"
    I = two_I // 2
    act = torch.empty((M, I), dtype=torch.bfloat16, device=x.device)
    BM, num_tile_blocks = _active_symm_bounds()
    with_scale, scale_d, with_bound, ntb_d, grid_x = _resolve_opts(scale, num_tile_blocks, x)
    _compile(I, with_scale, with_bound, BM, grid_x)(x, act, scale_d, ntb_d, M, stream=stream)
    return act
