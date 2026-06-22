###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Mega-MoE SwiGLU epilogue (FlyDSL), moved from MegaKernelFlyDSL/kernels/swiglu_flydsl.py.

Fused SwiGLU forward/backward in one FlyDSL pass.  Optional features:
  - with_scale: per-row routing-weight scale (act[m,:] *= SCALE[m])
  - with_bound: no-sync grid-stride over a device-bounded real-row count (NUM_TILE_BLOCKS)
  - with_gate (bwd): accumulate the gate gradient via wave all-reduce + 1 atomic/wave

Self-contained: the ``atomic_add_f32`` prim (used only by the backward
gate-gradient path) is vendored here, so this module depends only on
``flydsl`` + ``torch`` (no upstream ``kernels`` package).
"""
from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.buffer_ops as bo
import flydsl.expr.math as fmath
import torch
from flydsl._mlir.dialects import vector as _vector
from flydsl.expr.buffer_ops import (
    _unwrap_value,
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_llvm_ptr,
    extract_base_index,
    get_element_ptr,
)

# SwiGLU clamp contract: must match the reference; override via the `clamp` arg.
ACTIVATION_CLAMP = 10.0

_VEC = 8
_BLOCK_THREADS = 128
_COLS_PER_BLOCK = _VEC * _BLOCK_THREADS  # 1024

# --------------------------------------------------------------------------- #
# Vendored atomic prims (was kernels.prims)
# --------------------------------------------------------------------------- #
_llvm = bo.llvm
_ORD = _llvm.AtomicOrdering
_SCOPE = "agent"  # device-wide scope (Triton scope="gpu" lowers to this)
_I4 = 4  # int32 byte stride / atomic alignment


def _scope(scope):
    """Map scope name to LLVM syncscope ('sys' -> None = system default)."""
    if scope == "sys":
        return None
    return scope  # 'agent' (or any explicit syncscope string)


def _elem_ptr_i32(tensor, idx):
    """LLVM ptr to int32 element ``tensor[idx]`` (idx an fx/i32 value or int)."""
    base = create_llvm_ptr(extract_base_index(tensor, address_space=1), 1)
    byte_off = _unwrap_value(idx * fx.Int32(_I4))
    return get_element_ptr(base, byte_offset=byte_off, elem_type=fx.T.i8())


def atomic_add_f32(tensor, idx, val, *, scope=_SCOPE):
    """Atomic f32 add into ``tensor[idx]``, returns OLD value; relaxed, for reductions."""
    ptr = _elem_ptr_i32(tensor, idx)  # byte-addressed GEP; elem type from `val`
    res = _llvm.atomicrmw(
        _llvm.AtomicBinOp.fadd,
        ptr,
        _unwrap_value(val),
        _ORD.monotonic,
        syncscope=_scope(scope),
        alignment=_I4,
    )
    return fx.arith.ArithValue(res)


def _make_swiglu(
    I: int, clamp: float, with_scale: bool = False, with_bound: bool = False, BM: int = 0, GRID_X: int = 0
):
    two_I = 2 * I
    assert I % _COLS_PER_BLOCK == 0, f"I={I} not divisible by {_COLS_PER_BLOCK}"

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def swiglu_k(ACC1: fx.Tensor, ACT: fx.Tensor, SCALE: fx.Tensor, NUM_TILE_BLOCKS: fx.Tensor):
        f32v = fx.T.VectorType.get([_VEC], fx.T.f32())
        bf16v = fx.T.VectorType.get([_VEC], fx.T.bf16())
        thread_index = fx.thread_idx.x
        # M-major grid: M can exceed HIP's grid.y cap
        block_index_x, block_index_y, _ = fx.block_idx
        col = block_index_y * fx.Int32(_COLS_PER_BLOCK) + thread_index * fx.Int32(_VEC)

        acc_rsrc = create_buffer_resource(ACC1, max_size=True)
        act_rsrc = create_buffer_resource(ACT, max_size=True)
        scale_rsrc = create_buffer_resource(SCALE, max_size=True) if with_scale else None

        def compute_row(m):
            row_base = m * fx.Int32(two_I)
            gate = buffer_load(acc_rsrc, row_base + col, vec_width=_VEC, dtype=fx.T.bf16())
            up = buffer_load(acc_rsrc, row_base + fx.Int32(I) + col, vec_width=_VEC, dtype=fx.T.bf16())
            g = fx.arith.extf(f32v, gate)
            u = fx.arith.extf(f32v, up)
            lo = fx.arith.constant_vector(-float(clamp), f32v)
            hi = fx.arith.constant_vector(float(clamp), f32v)
            g = fx.arith.minimumf(fx.arith.maximumf(g, lo), hi)
            u = fx.arith.minimumf(fx.arith.maximumf(u, lo), hi)
            # silu(g) = g / (1 + exp(-g))
            neg_g = fx.arith.mulf(g, fx.arith.constant_vector(-1.0, f32v))
            denom = fx.arith.addf(fx.arith.constant_vector(1.0, f32v), fmath.exp(neg_g))
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
        swiglu_k(ACC1, ACT, SCALE, NUM_TILE_BLOCKS).launch(
            grid=(grid_x if with_bound else M, I // _COLS_PER_BLOCK, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


def _make_swiglu_bwd(
    I: int,
    clamp: float,
    with_scale: bool = False,
    with_bound: bool = False,
    BM: int = 0,
    GRID_X: int = 0,
    with_gate: bool = False,
):
    two_I = 2 * I
    assert I % _COLS_PER_BLOCK == 0, f"I={I} not divisible by {_COLS_PER_BLOCK}"

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def swiglu_bwd_k(
        DACT: fx.Tensor,
        ACC1: fx.Tensor,
        DACC1: fx.Tensor,
        SCALE: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        GRAD_GATE: fx.Tensor,
    ):
        f32v = fx.T.VectorType.get([_VEC], fx.T.f32())
        bf16v = fx.T.VectorType.get([_VEC], fx.T.bf16())
        thread_index = fx.thread_idx.x
        # M-major grid: M can exceed HIP's grid.y cap
        block_index_x, block_index_y, _ = fx.block_idx
        col = block_index_y * fx.Int32(_COLS_PER_BLOCK) + thread_index * fx.Int32(_VEC)

        dact_rsrc = create_buffer_resource(DACT, max_size=True)
        acc_rsrc = create_buffer_resource(ACC1, max_size=True)
        dacc_rsrc = create_buffer_resource(DACC1, max_size=True)
        scale_rsrc = create_buffer_resource(SCALE, max_size=True) if with_scale else None

        def compute_row(m):
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

            lo = fx.arith.constant_vector(-float(clamp), f32v)
            hi = fx.arith.constant_vector(float(clamp), f32v)
            one = fx.arith.constant_vector(1.0, f32v)
            zero = fx.arith.constant_vector(0.0, f32v)
            neg1 = fx.arith.constant_vector(-1.0, f32v)
            gc = fx.arith.minimumf(fx.arith.maximumf(gate, lo), hi)
            uc = fx.arith.minimumf(fx.arith.maximumf(up, lo), hi)
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

            if with_gate:
                # gate gradient: per-thread sum -> wave all-reduce -> 1 atomic/wave
                contrib = fx.arith.mulf(d_raw, fx.arith.mulf(s, uc))
                partial = fx.arith.ArithValue(
                    _vector.reduction(fx.T.f32(), _vector.CombiningKind.ADD, contrib)
                )
                wave_off = 1
                while wave_off < 64:
                    partial = partial.addf(fx.arith.ArithValue(partial.shuffle_xor(wave_off, 64)))
                    wave_off = wave_off * 2
                if (thread_index % fx.Int32(64)) == fx.Int32(0):
                    atomic_add_f32(GRAD_GATE, m, partial)

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

    @flyc.jit
    def launch(
        DACT: fx.Tensor,
        ACC1: fx.Tensor,
        DACC1: fx.Tensor,
        SCALE: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        GRAD_GATE: fx.Tensor,
        M: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        swiglu_bwd_k(DACT, ACC1, DACC1, SCALE, NUM_TILE_BLOCKS, GRAD_GATE).launch(
            grid=(grid_x if with_bound else M, I // _COLS_PER_BLOCK, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


# grid-stride block count for the no-sync bounded swiglu
_SWIGLU_GRID_X = 8192


@functools.lru_cache(maxsize=32)
def _compile(
    I: int, clamp_milli: int, with_scale: bool = False, with_bound: bool = False, BM: int = 0, GRID_X: int = 0
):
    return _make_swiglu(I, clamp_milli / 1000.0, with_scale, with_bound, BM, GRID_X)


@functools.lru_cache(maxsize=32)
def _compile_bwd(
    I: int,
    clamp_milli: int,
    with_scale: bool = False,
    with_bound: bool = False,
    BM: int = 0,
    GRID_X: int = 0,
    with_gate: bool = False,
):
    return _make_swiglu_bwd(I, clamp_milli / 1000.0, with_scale, with_bound, BM, GRID_X, with_gate)


def swiglu_backward(
    dact: torch.Tensor,
    acc1: torch.Tensor,
    I: int,
    clamp: float = ACTIVATION_CLAMP,
    scale: torch.Tensor | None = None,
    num_tile_blocks: torch.Tensor | None = None,
    BM: int = 0,
    grad_gate: torch.Tensor | None = None,
):
    """Backward SwiGLU in one FlyDSL pass; optional scale/bound/gate-gradient. Returns dacc1 (and grad_gate if requested)."""
    assert acc1.size(1) == 2 * I and dact.size(1) == I
    acc1 = acc1.contiguous()
    dact = dact.contiguous()
    M = acc1.size(0)
    dacc1 = torch.empty((M, 2 * I), dtype=torch.bfloat16, device=acc1.device)
    # off-feature args are never read (gated by with_* constexpr); bind acc1 to
    # satisfy the fixed kernel signature -- no dummy allocation needed.
    with_scale = scale is not None
    scale = scale if with_scale else acc1
    with_bound = num_tile_blocks is not None
    num_tile_blocks_d = num_tile_blocks if with_bound else acc1
    grid_x = _SWIGLU_GRID_X if with_bound else 0
    with_gate = grad_gate is not None
    gg = grad_gate if with_gate else acc1
    _compile_bwd(I, int(round(clamp * 1000)), with_scale, with_bound, BM, grid_x, with_gate)(
        dact, acc1, dacc1, scale, num_tile_blocks_d, gg, M, stream=torch.cuda.current_stream()
    )
    return (dacc1, grad_gate) if with_gate else dacc1


def swiglu(
    acc1: torch.Tensor,
    act: torch.Tensor | None,
    I: int,
    M: int,
    scale: torch.Tensor | None = None,
    clamp: float = ACTIVATION_CLAMP,
    stream=None,
    num_tile_blocks: torch.Tensor | None = None,
    BM: int = 0,
) -> torch.Tensor:
    """Fused SwiGLU over the first M rows in one FlyDSL pass; allocates `act` if None, optional scale/bound."""
    if stream is None:
        stream = torch.cuda.current_stream()
    if act is None:
        act = torch.empty((acc1.size(0), I), dtype=torch.bfloat16, device=acc1.device)
    # off-feature args are never read (gated by with_* constexpr); bind acc1 to
    # satisfy the fixed kernel signature -- no dummy allocation needed.
    with_scale = scale is not None
    scale = scale if with_scale else acc1
    with_bound = num_tile_blocks is not None
    num_tile_blocks_d = num_tile_blocks if with_bound else acc1
    grid_x = _SWIGLU_GRID_X if with_bound else 0
    _compile(I, int(round(clamp * 1000)), with_scale, with_bound, BM, grid_x)(
        acc1, act, scale, num_tile_blocks_d, M, stream=stream
    )
    return act
