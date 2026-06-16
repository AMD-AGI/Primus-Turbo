###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared per-tile emit API for the dense FP8 GEMM kernels (FlyDSL).

Decouples the per-tile compute (prelude -> uniform-K pipeline -> epilog -> store)
from the per-kernel scheduler/launch so the GEMM tile is written once and reused
by the NT, NN and TN layouts. Design follows ``docs/TILE_API.md``:

  - ``TileSpec`` is a ``Protocol`` (the launcher contract: ``cache_tag``,
    ``kernel_name``, ``build_launch``); custom dtype/epilogue gemm specs satisfy
    it and own their own kernel signature. ``DenseFp8TileSpec`` is the built-in
    impl for dense FP8 per-tensor.
  - ``make_tile_spec(...)`` is host-side (``@lru_cache``); it resolves the
    compile-time geometry + the ``@fx.struct`` shared storage + the layout policy.
  - ``DenseFp8TileSpec.emit(...)`` runs at trace time inside ``@flyc.kernel`` and
    splices the exact op sequence inline -> bit-identical IR vs the standalone
    kernels (perf-neutral by construction; see ``docs/TILE_API.md`` section 5/7).
  - ``schedule_tile`` + ``run_uniform_k_pipeline`` are reusable free functions so
    a custom spec composes them instead of reimplementing the pipeline.

The op sequence here is moved verbatim from ``gemm_fp8_kernel.py``'s
``_compile_dense_{nt,nn,tn}``; the only seams are compile-time layout/dtype
policy (loaders, swizzle, base offsets, k-step unit, LDS layout, inplace MFMA,
K-tail mask, end-of-iter drain). This module must NOT contain
``from __future__ import annotations`` (it would stringify the ``@fx.struct``
field annotations and break the LDS layout computation)."""

import functools
from collections import namedtuple
from typing import Callable, Protocol, runtime_checkable

# isort: off
from primus_turbo.flydsl.utils.gemm_helper import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    S2RLoaderTr,
    StoreCPerTensor,
    asm_mma_do,
    ceildiv,
    compute_global_swizzle,
    compute_global_swizzle_nn,
    make_fp8_buffer_tensor,
    make_value_attrs,
    mask_a_tail,
    wait_barrier,
    xcd_remap_pid,
)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ReplaceIfWithDispatch
from flydsl.expr import arith
from flydsl.expr import range_constexpr, rocdl

# isort: on


def _emit_if_then(cond, then_fn):
    """Emit a runtime (dynamic-cond) ``if cond: then_fn()`` from a free function.

    The ``@flyc.kernel`` AST transform rewrites a lexical ``if`` on a dynamic
    value into ``scf_if_dispatch(cond, then_fn)``; that rewrite is body-only, so
    inside this shared free function we must call the same primitive directly.
    Emits the identical scf.if -> bit-identical IR vs the standalone kernels."""
    ReplaceIfWithDispatch.scf_if_dispatch(cond, then_fn)


BLOCK_K = 128
# TN bank-spread LDS chunk stride: 1056 (=1024+32) un-aligns the per-wave chunk
# base across LDS banks to remove the transpose-read bank conflict; the G2S
# writer and the S2R reader must use the same value.
_LDS_CS = 1056


# Reusable uniform-K pipeline geometry: the compile-time scalars the shared
# software pipeline reads. Passing this (not ``self``) lets custom specs reuse
# ``run_uniform_k_pipeline`` without subclassing.
PipelineGeometry = namedtuple(
    "PipelineGeometry",
    [
        "N_LDS_STEPS_A",
        "N_LDS_STEPS_B",
        "N_TILES_A",
        "N_TILES_B",
        "N_ACCUMS",
        "LDS_BLOCK_M",
        "LDS_BLOCK_N",
        "BLOCK_M",
        "BLOCK_N",
        "K_ITERS",
        "K_TAIL",
        "end_iter_drain",
        "main_b0_no_drain",
        "mask_a_in_tail",
    ],
)


# ──────────────────────────────────────────────────────────────────────
# TileSpec contract. Two layers, both optional to override:
#   1. Launcher contract: ``cache_tag``, ``kernel_name``, ``BLOCK_M/N``,
#      ``shared_storage``, ``pipeline_geom``, ``materialize_tid``,
#      ``build_launch`` -- what the launcher + emit template need.
#   2. Per-stage emit hooks: ``schedule``, ``base_offsets``, ``make_buffers``,
#      ``global_swizzle``, ``build_mfma``, ``build_loaders``, ``build_store`` --
#      each a single seam of ``emit_uniform_k_tile``. Override only the stage(s)
#      you need (e.g. a custom epilogue = override ``build_store`` alone), reuse
#      the rest. Or override ``emit`` wholesale for a fully bespoke tile.
# A concrete spec owns its own kernel signature (dtype/epilogue/scale form) via
# ``build_launch`` + ``emit``, so different gemm families carry different I/O.
# ──────────────────────────────────────────────────────────────────────
@runtime_checkable
class TileSpec(Protocol):
    """Host-side tile spec contract: launcher surface + overridable emit hooks."""

    # --- launcher contract ---
    cache_tag: tuple  # JIT cache-key discriminator
    kernel_name: str  # emitted symbol name
    BLOCK_M: int  # tile M (grid sizing + n_blocks)
    BLOCK_N: int  # tile N
    shared_storage: type  # @fx.struct LDS class -> SharedAllocator().allocate(...)
    pipeline_geom: PipelineGeometry  # scalars run_uniform_k_pipeline reads
    materialize_tid: bool  # str(thread_idx.x) before lazy tr16 S2R use

    def build_launch(self, *, waves_per_eu: int = 2, agpr_alloc: int = 0) -> Callable:
        """Build the ``@flyc.kernel`` + ``@flyc.jit`` launcher for this spec."""
        ...

    def emit(self, *, A, B, C, A_scale, B_scale, c_m, c_n, group_base=0, lds=None) -> None:
        """Splice the tile inline (trace-time, inside ``@flyc.kernel``). ``group_base``
        = per-expert B slab scalar (0 = dense); ``lds`` = optional caller storage."""
        ...

    # --- per-stage emit hooks (each an overridable seam) ---
    def schedule(self, *, c_m, c_n, n_blocks):
        """tile-id -> (block_m, block_n)."""
        ...

    def base_offsets(self, *, block_m, block_n, c_m, c_n, group_base=0):
        """-> (A0, A1, B0, B1, a_k_mult, b_k_mult) global bases + k-step units.
        ``group_base`` adds the per-expert B slab (0 = dense)."""
        ...

    def make_buffers(self, *, A, B):
        """raw A/B tensors -> (a_div, b_div) buffer views."""
        ...

    def global_swizzle(self, *, lane_id, wave_id, c_m, c_n):
        """-> (gl_off_a, gl_off_b) global-load swizzle offsets."""
        ...

    def build_mfma(self):
        """-> the MFMA op-emitter (dtype/inplace policy)."""
        ...

    def build_loaders(self, *, a_div, b_div, gl_off_a, gl_off_b, wave_id, wave_m, wave_n):
        """-> (a_g2s, b_g2s, a_s2r, b_s2r) G2S/S2R loaders."""
        ...

    def build_store(self, *, A_scale, B_scale, C, c_m, c_n, mfma):
        """-> the C epilogue/store op-emitter."""
        ...


# ──────────────────────────────────────────────────────────────────────
# Scheduler (caller-side tile-id mapping; trace-time expr builder). One helper
# for all layouts: group_n==0 is the 1D GROUP_M swizzle used by NT/NN (and by
# TN's default); group_n>0 is TN's 2D band. The emitted block_m/block_n
# arithmetic matches the standalone kernels.
# ──────────────────────────────────────────────────────────────────────
def schedule_tile(c_m, n_blocks, block_m_size, GROUP_M, group_n, num_xcd):
    """tile-id -> (block_m, block_n) after the XCD-aware PID remap. group_n==0:
    1D GROUP_M super-row swizzle for L2 reuse; group_n>0: 2D band (N split into
    width-group_n bands, GROUP_M inside each). group_size clamps the last band so
    any GROUP_M/group_n >= 1 is correct (arith.select = integer min). Bijection."""
    num_pid_m = ceildiv(c_m, block_m_size)
    pid = xcd_remap_pid(fx.block_idx.x, num_pid_m * n_blocks, num_xcd)
    if group_n > 0:
        band_tiles = num_pid_m * group_n
        band = pid // band_tiles
        pid_in_band = pid % band_tiles
        band_n0 = band * group_n
        rem_n = n_blocks - band_n0
        band_w = arith.select(rem_n < group_n, rem_n, fx.Int32(group_n))
        nig = GROUP_M * band_w
        gid = pid_in_band // nig
        pig = pid_in_band % nig
        fpm = gid * GROUP_M
        rem_m = num_pid_m - fpm
        gsm = arith.select(rem_m < GROUP_M, rem_m, fx.Int32(GROUP_M))
        return fpm + (pig % gsm), band_n0 + (pig // gsm)
    nig = GROUP_M * n_blocks
    gid = pid // nig
    pig = pid % nig
    fpm = gid * GROUP_M
    rem_m = num_pid_m - fpm
    gsm = arith.select(rem_m < GROUP_M, rem_m, fx.Int32(GROUP_M))
    return fpm + (pig % gsm), pig // gsm


# ──────────────────────────────────────────────────────────────────────
# Scheduler: tile-id -> (block_m, block_n) strategy, HELD by the spec. CUTLASS
# makes this a first-class TileScheduler type (Persistent / StreamK); here the
# two strategies are the standalone-dense XCD swizzle (L2 reuse) and the fused
# grouped LINEAR no-sync map (front-loaded past the comm blocks; preserves the
# runtime over-launch self-bound). A spec composes a scheduler instead of
# subclassing to override ``schedule`` -- the dense-vs-grouped tile-map HARD
# divergence is now one swapped object, not an override.
# ──────────────────────────────────────────────────────────────────────
class _Scheduler:
    """Tile-id mapping strategy base. ``map`` returns (block_m, block_n)."""

    def map(self, spec, *, c_m, c_n, n_blocks):
        ...


class XcdSwizzleScheduler(_Scheduler):
    """Standalone dense map: XCD-aware PID remap + 1D GROUP_M (or 2D band, TN)
    swizzle for per-XCD L2 reuse."""

    def __init__(self, *, GROUP_M, group_n, num_xcd):
        self.GROUP_M = GROUP_M
        self.group_n = group_n
        self.num_xcd = num_xcd

    def map(self, spec, *, c_m, c_n, n_blocks):
        return schedule_tile(c_m, n_blocks, spec.BLOCK_M, self.GROUP_M, self.group_n, self.num_xcd)


class LinearNoSyncScheduler(_Scheduler):
    """Fused grouped map: LINEAR row-major tile-id, offset past the front
    comm/comb blocks (no xcd_remap/GROUP_M -> preserves the runtime
    ``num_tile_blocks`` over-launch self-bound)."""

    def __init__(self, *, num_comm_blocks):
        self.num_comm_blocks = num_comm_blocks

    def map(self, spec, *, c_m, c_n, n_blocks):
        tile_index = fx.block_idx.x - fx.Int32(self.num_comm_blocks)
        return tile_index // n_blocks, tile_index % n_blocks


# ──────────────────────────────────────────────────────────────────────
# LayoutPolicy: all per-layout knowledge in ONE object (NT/NN/TN). CUTLASS
# selects loaders/offsets by layout-tag specialization; CK by tensor-descriptor
# coordinate transforms. Here each policy concentrates the LDS geometry, the
# per-stage emit hooks (base_offsets / global_swizzle / build_loaders) and the
# race-fix booleans, so a new layout is a NEW CLASS -- not edits in 4 places.
# DenseFp8TileSpec holds one policy and forwards. Stateless singletons; methods
# take the spec for geometry, so emission is bit-identical to the old branches.
# The dtype/scale policy (build_mfma atom, build_store) stays on the spec --
# orthogonal to layout. ``base_offsets`` carries the optional ``group_base``
# scalar seam (per-expert B slab; 0 = dense -> identical IR).
# ──────────────────────────────────────────────────────────────────────
class _LayoutPolicy:
    """Base layout policy (NT behavior). Subclasses override the seams."""

    name = "nt"
    materialize_tid = False      # str(thread_idx.x) before lazy tr16 S2R use
    mask_a_in_tail = True        # A [M,K] tail K-cols in-bounds -> mask
    main_b0_no_drain = False     # TN-only: b0 covered by following a0 drain
    inplace_mma = False          # TN-only: asm-inplace MFMA (accum in AGPR)
    wants_end_iter_drain = True  # NT-only: end-of-iter G2S drain (race fix)

    def lds_geometry(self, *, LDS_BLOCK_M, LDS_BLOCK_N, BLOCK_K):
        """-> (N_LDS_STEPS_A, N_LDS_STEPS_B, chunk_stride, a_lds_size, b_lds_size)."""
        N_LDS_STEPS_A = LDS_BLOCK_M // 64
        N_LDS_STEPS_B = LDS_BLOCK_N // 64
        return N_LDS_STEPS_A, N_LDS_STEPS_B, 1024, LDS_BLOCK_M * BLOCK_K, LDS_BLOCK_N * BLOCK_K

    def base_offsets(self, spec, *, block_m, block_n, c_m, c_n, group_base=0):
        """NT: A [M,K] & B [N,K] both row-major K-contig -> *K bases, unit step."""
        BLOCK_M, BLOCK_N, K = spec.BLOCK_M, spec.BLOCK_N, spec.K
        LDS_BLOCK_M, LDS_BLOCK_N = spec.LDS_BLOCK_M, spec.LDS_BLOCK_N
        A0 = (block_m * BLOCK_M) * K
        A1 = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        B0 = (block_n * BLOCK_N) * K
        B1 = (block_n * BLOCK_N + LDS_BLOCK_N) * K
        return A0, A1, B0 + group_base, B1 + group_base, 1, 1

    def global_swizzle(self, spec, *, lane_id, wave_id, c_m, c_n):
        """NT: both A and B row-major K-contig."""
        K, R = spec.K, spec.N_LDS_ROUNDS
        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, R, preshuffled=False)
        gl_off_b = compute_global_swizzle(lane_id, wave_id, K, R, preshuffled=False)
        return gl_off_a, gl_off_b

    def build_loaders(self, spec, *, a_div, b_div, gl_off_a, gl_off_b, wave_id, wave_m, wave_n):
        """NT: plain G2S + plain S2R on both operands."""
        F8 = fx.Float8E4M3FN.ir_type
        a_g2s = G2SLoader(a_div, gl_off_a, spec.N_LDS_STEPS_A, F8, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, spec.N_LDS_STEPS_B, F8, wave_id)
        a_s2r = S2RLoader(wave_m, spec.N_TILES_A)
        b_s2r = S2RLoader(wave_n, spec.N_TILES_B)
        return a_g2s, b_g2s, a_s2r, b_s2r


class _NTLayoutPolicy(_LayoutPolicy):
    """NT: A [M,K] K-contig, B [N,K] K-contig (= B^T of [K,N]). Base = NT."""

    name = "nt"


class _NNLayoutPolicy(_LayoutPolicy):
    """NN: A [M,K], B [K,N]. B is K-row strided (b_k_mult = c_n) and read via the
    tr16 transpose S2R; A path is NT-identical."""

    name = "nn"
    materialize_tid = True
    mask_a_in_tail = True
    wants_end_iter_drain = False

    def base_offsets(self, spec, *, block_m, block_n, c_m, c_n, group_base=0):
        BLOCK_M, BLOCK_N, K = spec.BLOCK_M, spec.BLOCK_N, spec.K
        LDS_BLOCK_M, LDS_BLOCK_N = spec.LDS_BLOCK_M, spec.LDS_BLOCK_N
        A0 = (block_m * BLOCK_M) * K
        A1 = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        B0 = block_n * BLOCK_N + 0
        B1 = block_n * BLOCK_N + LDS_BLOCK_N
        return A0, A1, B0 + group_base, B1 + group_base, 1, c_n

    def global_swizzle(self, spec, *, lane_id, wave_id, c_m, c_n):
        K, R = spec.K, spec.N_LDS_ROUNDS
        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, R, preshuffled=False)
        gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, c_n, R)
        return gl_off_a, gl_off_b

    def build_loaders(self, spec, *, a_div, b_div, gl_off_a, gl_off_b, wave_id, wave_m, wave_n):
        F8 = fx.Float8E4M3FN.ir_type
        a_g2s = G2SLoader(a_div, gl_off_a, spec.N_LDS_STEPS_A, F8, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, spec.N_LDS_STEPS_B, F8, wave_id)
        a_s2r = S2RLoader(wave_m, spec.N_TILES_A)
        b_s2r = S2RLoaderTr(
            wave_n, spec.N_TILES_B, 32, inline_asm=spec.b_inline_asm_load, vmcnt_hint=spec.vmcnt_hint
        )
        return a_g2s, b_g2s, a_s2r, b_s2r


class _TNLayoutPolicy(_LayoutPolicy):
    """TN: A [K,M], B [K,N], C = A^T @ B. Both operands K-row strided + tr8
    transpose load + asm-inplace MFMA + bank-spread LDS chunk stride."""

    name = "tn"
    materialize_tid = True
    mask_a_in_tail = False
    main_b0_no_drain = True
    inplace_mma = True
    wants_end_iter_drain = False

    def lds_geometry(self, *, LDS_BLOCK_M, LDS_BLOCK_N, BLOCK_K):
        # tr8 transpose load spans K_log [0,128) -> >= 2 G2S rounds / 16K slot.
        N_LDS_STEPS_A = max(LDS_BLOCK_M // 64, 2)
        N_LDS_STEPS_B = LDS_BLOCK_N // 64
        a_lds_size = max(LDS_BLOCK_M * BLOCK_K, 2 * 8 * 1024) // 1024 * _LDS_CS
        b_lds_size = (LDS_BLOCK_N * BLOCK_K) // 1024 * _LDS_CS
        return N_LDS_STEPS_A, N_LDS_STEPS_B, _LDS_CS, a_lds_size, b_lds_size

    def base_offsets(self, spec, *, block_m, block_n, c_m, c_n, group_base=0):
        BLOCK_M, BLOCK_N = spec.BLOCK_M, spec.BLOCK_N
        LDS_BLOCK_M, LDS_BLOCK_N = spec.LDS_BLOCK_M, spec.LDS_BLOCK_N
        A0 = block_m * BLOCK_M + 0
        A1 = block_m * BLOCK_M + LDS_BLOCK_M
        B0 = block_n * BLOCK_N + 0
        B1 = block_n * BLOCK_N + LDS_BLOCK_N
        return A0, A1, B0 + group_base, B1 + group_base, c_m, c_n

    def global_swizzle(self, spec, *, lane_id, wave_id, c_m, c_n):
        R = spec.N_LDS_ROUNDS
        gl_off_a = compute_global_swizzle_nn(lane_id, wave_id, c_m, R)
        gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, c_n, R)
        return gl_off_a, gl_off_b

    def build_loaders(self, spec, *, a_div, b_div, gl_off_a, gl_off_b, wave_id, wave_m, wave_n):
        F8 = fx.Float8E4M3FN.ir_type
        cs = spec.chunk_stride
        a_g2s = G2SLoader(a_div, gl_off_a, spec.N_LDS_STEPS_A, F8, wave_id, chunk_stride=cs)
        b_g2s = G2SLoader(b_div, gl_off_b, spec.N_LDS_STEPS_B, F8, wave_id, chunk_stride=cs)
        a_s2r = S2RLoaderTr(
            wave_m, spec.N_TILES_A, spec.LDS_BLOCK_M // 2, inline_asm=True,
            vmcnt_hint=spec.vmcnt_hint, chunk_stride=cs,
        )
        b_s2r = S2RLoaderTr(
            wave_n, spec.N_TILES_B, 32, inline_asm=True, vmcnt_hint=spec.vmcnt_hint, chunk_stride=cs
        )
        return a_g2s, b_g2s, a_s2r, b_s2r


_LAYOUT_POLICIES = {"nt": _NTLayoutPolicy(), "nn": _NNLayoutPolicy(), "tn": _TNLayoutPolicy()}


# ──────────────────────────────────────────────────────────────────────
# Epilogue: the C-store strategy, HELD by the spec (mirrors LayoutPolicy /
# Scheduler). CUTLASS makes this a composable visitor tree (EVT: scale -> bias ->
# act -> store). Here the per-tensor scaled store carries an optional chain of
# element-wise f32 nodes applied post-scale / pre-cast (bias, activation), each a
# trace-time ``f32 -> f32`` emit callable. Empty chain = plain scaled store ->
# identical IR. A combine scatter-add node (the perf prize) is a future node.
# ──────────────────────────────────────────────────────────────────────
def _epi_relu(v):
    """ReLU epilogue node: max(v, 0) on the post-scale f32 accumulator. Uses
    ``maximumf`` so the numeric wrapper (and its ``.to`` cast) is preserved."""
    return v.maximumf(fx.Float32(0.0))


# Activation registry for the epilogue node chain (name -> f32->f32 emit fn).
_EPILOGUE_ACTS = {"relu": _epi_relu}


class PerTensorEpilogue:
    """Per-tensor-scaled store (out bf16/fp16) with an optional EVT-style chain of
    element-wise f32 nodes (``nodes``: each ``f32 -> f32``). ``nodes=()`` -> the
    plain scaled store (bit-identical IR)."""

    def __init__(self, *, out_fp16, nodes=(), cache_key=()):
        self.out_fp16 = out_fp16
        self.nodes = tuple(nodes)
        # JIT cache discriminator for the node chain (lambdas have no stable hash;
        # the caller names the chain). () when no nodes -> spec cache_tag unchanged.
        self.cache_key = tuple(cache_key) if nodes else ()

    def _elem_fn(self):
        """Fold the node chain into one ``f32 -> f32`` callable (None if empty)."""
        if not self.nodes:
            return None
        nodes = self.nodes

        def fn(v):
            for node in nodes:
                v = node(v)
            return v

        return fn

    def build(self, spec, *, A_scale, B_scale, C, c_m, c_n, mfma):
        out_ty = fx.Float16 if self.out_fp16 else fx.BFloat16
        return StoreCPerTensor(
            A_scale, B_scale, C, c_m, c_n, mfma.idx, spec.N_TILES_A, spec.N_TILES_B, out_ty,
            elem_fn=self._elem_fn(),
        )


# ──────────────────────────────────────────────────────────────────────
# Tile spec: compile-time geometry + layout policy + shared storage.
# ──────────────────────────────────────────────────────────────────────
class DenseFp8TileSpec:
    """Dense FP8 per-tensor tile spec: compile-time geometry + layout/dtype
    policy + ``@fx.struct`` shared storage. Built host-side by
    ``make_tile_spec``; ``emit`` splices the tile inline inside ``@flyc.kernel``.
    Satisfies the ``TileSpec`` protocol."""

    def __init__(
        self,
        *,
        layout,
        K,
        BLOCK_M,
        BLOCK_N,
        GROUP_M,
        num_xcd,
        group_n,
        nt_vmcnt,
        vmcnt_hint,
        b_inline_asm_load,
        cbsz,
        blgp,
        out_fp16,
        act=None,
    ):
        assert layout in ("nt", "nn", "tn")
        assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
        # Odd-K native K-tail: ceil(K/128) iters; last iter length K_TAIL (0 = exact
        # multiple). NT/NN mask the tail's invalid A K-columns; TN's K-rows are OOB.
        K_ITERS = (K + BLOCK_K - 1) // BLOCK_K
        K_TAIL = K % BLOCK_K
        assert K_ITERS >= 2, f"K_ITERS={K_ITERS} too small; need K >= 129 (ceil(K/128) >= 2)"

        N_TILES_A = BLOCK_M // 64
        N_TILES_B = BLOCK_N // 128
        N_ACCUMS = N_TILES_A * N_TILES_B
        assert N_ACCUMS > 0
        LDS_BLOCK_M = BLOCK_M // 2
        LDS_BLOCK_N = BLOCK_N // 2

        # LDS geometry is layout-specific (TN's tr8 transpose load forces >= 2 G2S
        # rounds + a bank-spread chunk stride). The policy owns it.
        layout_policy = _LAYOUT_POLICIES[layout]
        N_LDS_STEPS_A, N_LDS_STEPS_B, chunk_stride, a_lds_size, b_lds_size = layout_policy.lds_geometry(
            LDS_BLOCK_M=LDS_BLOCK_M, LDS_BLOCK_N=LDS_BLOCK_N, BLOCK_K=BLOCK_K
        )
        N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)

        @fx.struct
        class SharedStorage:
            A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
            A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
            A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
            A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
            B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
            B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
            B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
            B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

        self.layout = layout
        self.K = K
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.GROUP_M = GROUP_M
        self.num_xcd = num_xcd
        self.group_n = group_n
        # Tile-id scheduler (held, not subclassed). Dense = XCD swizzle; grouped
        # specs swap in LinearNoSyncScheduler in their own __init__.
        self.scheduler = XcdSwizzleScheduler(GROUP_M=GROUP_M, group_n=group_n, num_xcd=num_xcd)
        # C-store epilogue (held). Per-tensor scale + optional activation node
        # (EVT chain); act=None -> plain scaled store (identical IR).
        _nodes = () if act is None else (_EPILOGUE_ACTS[act],)
        self.epilogue = PerTensorEpilogue(
            out_fp16=out_fp16, nodes=_nodes, cache_key=() if act is None else (act,)
        )
        self.nt_vmcnt = nt_vmcnt
        self.vmcnt_hint = vmcnt_hint
        self.b_inline_asm_load = b_inline_asm_load
        self.cbsz = cbsz
        self.blgp = blgp
        self.out_fp16 = out_fp16

        self.K_ITERS = K_ITERS
        self.K_TAIL = K_TAIL
        self.N_TILES_A = N_TILES_A
        self.N_TILES_B = N_TILES_B
        self.N_ACCUMS = N_ACCUMS
        self.LDS_BLOCK_M = LDS_BLOCK_M
        self.LDS_BLOCK_N = LDS_BLOCK_N
        self.N_LDS_STEPS_A = N_LDS_STEPS_A
        self.N_LDS_STEPS_B = N_LDS_STEPS_B
        self.N_LDS_ROUNDS = N_LDS_ROUNDS
        self.chunk_stride = chunk_stride
        self.shared_storage = SharedStorage

        # Layout policy: the spec HOLDS one and forwards the per-stage hooks +
        # the race-fix booleans to it (all per-layout knowledge lives in there).
        self.layout_policy = layout_policy
        self.materialize_tid = layout_policy.materialize_tid  # tr16 S2R load-order fix
        self.mask_a_in_tail = layout_policy.mask_a_in_tail  # A [M,K] tail K-cols in-bounds
        # NT G2S race fix: the drain depth is nt_vmcnt only when the layout wants it.
        self.end_iter_drain = nt_vmcnt if layout_policy.wants_end_iter_drain else None
        self.main_b0_no_drain = layout_policy.main_b0_no_drain  # TN b0 covered by following a0 drain
        self.inplace_mma = layout_policy.inplace_mma  # asm-inplace MFMA (accum in AGPR)

        # JIT cache-key discriminator: all geometry lives in this one opaque spec,
        # which FlyDSL's cache-key collector cannot hash; the kernel closure
        # references this scalar tuple so distinct specs hash distinctly. See
        # flydsl jit_function._collect_closure_scalar_vals.
        self.cache_tag = (
            layout,
            K,
            BLOCK_M,
            BLOCK_N,
            GROUP_M,
            num_xcd,
            group_n,
            nt_vmcnt,
            vmcnt_hint,
            b_inline_asm_load,
            cbsz,
            blgp,
            out_fp16,
        )
        # Fold the epilogue node-chain discriminator (() when no nodes -> unchanged).
        self.cache_tag = self.cache_tag + self.epilogue.cache_key

        # Emitted symbol name (per TileSpec protocol).
        self.kernel_name = "kernel_dense_" + layout

        # Scalars the shared uniform-K pipeline reads (reuse seam).
        self.pipeline_geom = PipelineGeometry(
            N_LDS_STEPS_A=N_LDS_STEPS_A,
            N_LDS_STEPS_B=N_LDS_STEPS_B,
            N_TILES_A=N_TILES_A,
            N_TILES_B=N_TILES_B,
            N_ACCUMS=N_ACCUMS,
            LDS_BLOCK_M=LDS_BLOCK_M,
            LDS_BLOCK_N=LDS_BLOCK_N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            K_ITERS=K_ITERS,
            K_TAIL=K_TAIL,
            end_iter_drain=self.end_iter_drain,
            main_b0_no_drain=self.main_b0_no_drain,
            mask_a_in_tail=self.mask_a_in_tail,
        )

    # ──────────────────────────────────────────────────────────────────
    def build_launch(self, *, waves_per_eu=2, agpr_alloc=0):
        """Build the ``@flyc.kernel`` + ``@flyc.jit`` launcher for this spec.
        The kernel references ``self.cache_tag`` so the JIT cache distinguishes
        configs; its __name__ matches the standalone so the emitted symbol is
        identical."""
        spec = self
        tag = self.cache_tag

        def kernel(A, B, C, A_scale, B_scale, c_m: fx.Int32, c_n: fx.Int32):
            _ = tag  # JIT cache-key discriminator; emits no IR
            spec.emit(A=A, B=B, C=C, A_scale=A_scale, B_scale=B_scale, c_m=c_m, c_n=c_n)

        kernel.__name__ = self.kernel_name
        kernel.__qualname__ = kernel.__name__
        kernel = flyc.kernel(kernel, known_block_size=[512, 1, 1])

        @flyc.jit
        def launch(A, B, C, A_scale, B_scale, c_m: fx.Int32, c_n: fx.Int32, stream: fx.Stream):
            grid_x = ceildiv(c_m, spec.BLOCK_M) * ceildiv(c_n, spec.BLOCK_N)
            kernel(
                A,
                B,
                C,
                A_scale,
                B_scale,
                c_m,
                c_n,
                value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
            ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

        return launch

    # ──────────────────────────────────────────────────────────────────
    def emit(self, *, A, B, C, A_scale, B_scale, c_m, c_n, group_base=0, lds=None):
        """Splice the whole tile inline via the shared uniform-K template. The
        template drives the per-stage hooks below; override a hook to customize a
        single stage, or override this method for a fully bespoke tile. ``group_base``
        (per-expert B slab; 0 = dense) and ``lds`` (caller-allocated shared storage;
        None = allocate inside) are the grouped/fused seams -- dense passes neither
        -> bit-identical IR."""
        emit_uniform_k_tile(self, A=A, B=B, C=C, A_scale=A_scale, B_scale=B_scale,
                            c_m=c_m, c_n=c_n, group_base=group_base, lds=lds)

    # ── per-stage emit hooks (each an overridable seam) ─────────────────
    def schedule(self, *, c_m, c_n, n_blocks):
        """tile-id -> (block_m, block_n) via the held scheduler (dense = XCD
        swizzle; grouped = LINEAR no-sync map)."""
        return self.scheduler.map(self, c_m=c_m, c_n=c_n, n_blocks=n_blocks)

    def base_offsets(self, *, block_m, block_n, c_m, c_n, group_base=0):
        """Per-tile global base offsets + k-step unit (forwarded to the layout
        policy). ``group_base`` is the optional per-expert B slab scalar (0 =
        dense -> identical IR); grouped specs pass ``g_idx * c_n * k_unit``."""
        return self.layout_policy.base_offsets(
            self, block_m=block_m, block_n=block_n, c_m=c_m, c_n=c_n, group_base=group_base
        )

    def make_buffers(self, *, A, B):
        """raw A/B fp8 tensors -> (a_div, b_div) buffer views."""
        F8_IR_t = fx.Float8E4M3FN.ir_type
        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
        return a_div, b_div

    def global_swizzle(self, *, lane_id, wave_id, c_m, c_n):
        """Global-load swizzle (forwarded to the layout policy)."""
        return self.layout_policy.global_swizzle(
            self, lane_id=lane_id, wave_id=wave_id, c_m=c_m, c_n=c_n
        )

    def build_mfma(self):
        """The MFMA op-emitter (dtype/inplace policy)."""
        mfma = Mfma16x16x128(self.N_TILES_A, self.N_TILES_B)
        if self.inplace_mma:
            # asm-inplace MFMA (accum in AGPR); cbsz/blgp select srcA/srcB fp8 fmt.
            _mm = "2"
            _cbsz = self.cbsz
            _blgp = self.blgp
            mfma._do_mma = lambda _a, _b, _c, _m=_mm: asm_mma_do(_a, _b, _c, mode=_m, cbsz=_cbsz, blgp=_blgp)
        elif self.cbsz or self.blgp:
            # E5M2 / hybrid: rebuild the MFMA atom with per-operand fp8 fmt
            # (cbsz->srcA, blgp->srcB). Same instruction family / frag layout.
            _ea = fx.Float8E5M2 if self.cbsz else fx.Float8E4M3FN
            _eb = fx.Float8E5M2 if self.blgp else fx.Float8E4M3FN
            mfma.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, _ea, _eb))
        return mfma

    def build_loaders(self, *, a_div, b_div, gl_off_a, gl_off_b, wave_id, wave_m, wave_n):
        """The G2S/S2R loaders (forwarded to the layout policy)."""
        return self.layout_policy.build_loaders(
            self, a_div=a_div, b_div=b_div, gl_off_a=gl_off_a, gl_off_b=gl_off_b,
            wave_id=wave_id, wave_m=wave_m, wave_n=wave_n,
        )

    def build_store(self, *, A_scale, B_scale, C, c_m, c_n, mfma):
        """The C epilogue/store op-emitter (forwarded to the held epilogue)."""
        return self.epilogue.build(self, A_scale=A_scale, B_scale=B_scale, C=C, c_m=c_m, c_n=c_n, mfma=mfma)


# ──────────────────────────────────────────────────────────────────────
# Shared uniform-K tile-emit template (reusable free function). Drives any
# ``TileSpec`` through its stage hooks then runs the pipeline; emission order is
# identical to the standalone emit (perf-neutral). A custom spec overrides
# individual hooks for a single-stage change, or supplies its own ``emit``.
# ──────────────────────────────────────────────────────────────────────
def emit_uniform_k_tile(spec, *, A, B, C, A_scale, B_scale, c_m, c_n, group_base=0, lds=None):
    """Splice the whole tile inline: setup -> stage hooks -> pipeline. Call
    inside ``@flyc.kernel`` with the (flattened) tensors and dims. ``group_base``
    is the optional per-expert B slab scalar (0 = dense); ``lds`` is the optional
    caller-allocated shared storage (None -> allocate here, e.g. NN under a guard)."""
    # Materialize thread_idx.x before any lazy tr16 S2R use (nn/tn).
    if spec.materialize_tid:
        _ = str(fx.thread_idx.x)

    n_blocks = ceildiv(c_n, spec.BLOCK_N)

    if lds is None:
        lds = fx.SharedAllocator().allocate(spec.shared_storage).peek()
    a_cur0 = lds.A_lds_cur_0
    a_cur1 = lds.A_lds_cur_1
    a_next0 = lds.A_lds_next_0
    a_next1 = lds.A_lds_next_1
    b_cur0 = lds.B_lds_cur_0
    b_cur1 = lds.B_lds_cur_1
    b_next0 = lds.B_lds_next_0
    b_next1 = lds.B_lds_next_1

    lane_id = fx.thread_idx.x % 64
    wave_id = fx.thread_idx.x // 64
    wave_m = wave_id // 4
    wave_n = wave_id % 4

    block_m, block_n = spec.schedule(c_m=c_m, c_n=c_n, n_blocks=n_blocks)

    (
        A0_gl_offset,
        A1_gl_offset,
        B0_gl_offset,
        B1_gl_offset,
        a_k_mult,
        b_k_mult,
    ) = spec.base_offsets(block_m=block_m, block_n=block_n, c_m=c_m, c_n=c_n, group_base=group_base)

    a_div, b_div = spec.make_buffers(A=A, B=B)
    gl_off_a, gl_off_b = spec.global_swizzle(lane_id=lane_id, wave_id=wave_id, c_m=c_m, c_n=c_n)
    mfma = spec.build_mfma()
    a_g2s, b_g2s, a_s2r, b_s2r = spec.build_loaders(
        a_div=a_div,
        b_div=b_div,
        gl_off_a=gl_off_a,
        gl_off_b=gl_off_b,
        wave_id=wave_id,
        wave_m=wave_m,
        wave_n=wave_n,
    )
    store_c = spec.build_store(A_scale=A_scale, B_scale=B_scale, C=C, c_m=c_m, c_n=c_n, mfma=mfma)

    run_uniform_k_pipeline(
        spec.pipeline_geom,
        a_g2s=a_g2s,
        b_g2s=b_g2s,
        a_s2r=a_s2r,
        b_s2r=b_s2r,
        mfma=mfma,
        store_c=store_c,
        a_cur0=a_cur0,
        a_cur1=a_cur1,
        a_next0=a_next0,
        a_next1=a_next1,
        b_cur0=b_cur0,
        b_cur1=b_cur1,
        b_next0=b_next0,
        b_next1=b_next1,
        A0_gl_offset=A0_gl_offset,
        A1_gl_offset=A1_gl_offset,
        B0_gl_offset=B0_gl_offset,
        B1_gl_offset=B1_gl_offset,
        a_k_mult=a_k_mult,
        b_k_mult=b_k_mult,
        lane_id=lane_id,
        wave_m=wave_m,
        wave_n=wave_n,
        block_m=block_m,
        block_n=block_n,
    )


# ──────────────────────────────────────────────────────────────────────
# Shared uniform-K software pipeline (reusable free function). Takes a
# ``PipelineGeometry`` (not ``self``) so custom specs can reuse it directly.
# ──────────────────────────────────────────────────────────────────────
def run_uniform_k_pipeline(
    geom,
    *,
    a_g2s,
        b_g2s,
        a_s2r,
        b_s2r,
        mfma,
        store_c,
        a_cur0,
        a_cur1,
        a_next0,
        a_next1,
        b_cur0,
        b_cur1,
        b_next0,
        b_next1,
        A0_gl_offset,
        A1_gl_offset,
        B0_gl_offset,
        B1_gl_offset,
        a_k_mult,
        b_k_mult,
        lane_id,
        wave_m,
        wave_n,
        block_m,
        block_n,
    ):
        """The uniform-K software pipeline: prelude -> main K-loop -> epilog1 ->
        epilog2 (K-tail) -> scale+store. Op sequence is identical across NT/NN/TN;
        the seams are (a_k_mult, b_k_mult) k-step units, the K-tail A mask, the
        TN main-loop b0 ``drain=False``, and the NT end-of-iter G2S drain."""
        N_LDS_STEPS_A = geom.N_LDS_STEPS_A
        N_LDS_STEPS_B = geom.N_LDS_STEPS_B
        N_TILES_A = geom.N_TILES_A
        N_TILES_B = geom.N_TILES_B
        N_ACCUMS = geom.N_ACCUMS
        LDS_BLOCK_M = geom.LDS_BLOCK_M
        LDS_BLOCK_N = geom.LDS_BLOCK_N
        BLOCK_M = geom.BLOCK_M
        BLOCK_N = geom.BLOCK_N
        K_ITERS = geom.K_ITERS
        K_TAIL = geom.K_TAIL
        end_iter_drain = geom.end_iter_drain
        main_b0_no_drain = geom.main_b0_no_drain
        mask_a = geom.mask_a_in_tail

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        # Prelude: k=0 -> cur, k=1 -> next (a_next1 lazily on first main iter).
        b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K * b_k_mult)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K * a_k_mult)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K * b_k_mult)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K * a_k_mult)

        _emit_if_then(wave_m == 1, lambda: rocdl.s_barrier())

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K * b_k_mult)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K * a_k_mult)
        b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K * b_k_mult)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        # Main K-loop. Each iter: s2r {a0,b0,b1,a1} -> 4 mma (c00->c01->c10->c11)
        # interleaved with k+1 (a_next1) and k+2 (a_cur0, b_cur0, b_cur1) prefetches.
        for k in range_constexpr(K_ITERS - 2):
            if main_b0_no_drain:
                # b0 drain=False: covered by the following a0 load's lgkmcnt(0)
                # before c00 consumes b0 (TN).
                b0_frag = b_s2r.load(b_cur0, drain=False)
            else:
                b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K * a_k_mult)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K * b_k_mult)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K * a_k_mult)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K * b_k_mult)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            rocdl.s_setprio(1)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            if end_iter_drain is not None and end_iter_drain >= 0:
                _llvm.inline_asm(
                    res=None,
                    operands_=[],
                    asm_string=f"s_waitcnt vmcnt({end_iter_drain})",
                    constraints="",
                    has_side_effects=True,
                )  # end-of-iter G2S drain (race fix)
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 1 (k = K_ITERS - 2). The a_g2s.load(a_next1, A1 + (k+1)*step)
        # line is the c10/c11 stale-a1 pipeline fix.
        k = K_ITERS - 2
        b0_frag = b_s2r.load(b_cur0)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b0_frag = b_s2r.load(b_next0)
        a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K * a_k_mult)  # stale-a1 fix
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 2 (k = K_ITERS - 1) -- the K-tail block. NT/NN mask the A operand
        # so invalid K-columns (>= K_TAIL) contribute 0 (no-op when K_TAIL==0);
        # TN needs no mask (its A K-rows are fully OOB and clamp to 0).
        a0_frag = a_s2r.load(a_cur0)
        if mask_a:
            a0_frag = mask_a_tail(a0_frag, lane_id, K_TAIL)
        wait_barrier(0)

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        if mask_a:
            a1_frag = mask_a_tail(a1_frag, lane_id, K_TAIL)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        # Scale + store.
        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)


@functools.lru_cache(maxsize=256)
def make_tile_spec(
    *,
    layout,
    K,
    BLOCK_M=256,
    BLOCK_N=256,
    GROUP_M=1,
    num_xcd=8,
    group_n=0,
    nt_vmcnt=3,
    vmcnt_hint=2,
    b_inline_asm_load=False,
    cbsz=0,
    blgp=0,
    out_fp16=False,
    act=None,
):
    """Host-side tile spec factory (cached). ``layout`` in {nt, nn, tn};
    ``act`` is an optional epilogue activation name (e.g. "relu")."""
    return DenseFp8TileSpec(
        layout=layout,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUP_M=GROUP_M,
        num_xcd=num_xcd,
        group_n=group_n,
        nt_vmcnt=nt_vmcnt,
        vmcnt_hint=vmcnt_hint,
        b_inline_asm_load=b_inline_asm_load,
        cbsz=cbsz,
        blgp=blgp,
        out_fp16=out_fp16,
        act=act,
    )


# ──────────────────────────────────────────────────────────────────────
# Tiled kernel builders. Drop-in replacements for the standalone
# ``_compile_dense_{nt,nn,tn}`` (same signature / returned launcher) that build
# the kernel from ``TileSpec.build_launch``.
# ──────────────────────────────────────────────────────────────────────
def _make_tiled_launch(spec, waves_per_eu, agpr_alloc):
    """Thin shim: defer to the spec's own ``build_launch`` (the launch policy now
    lives on the spec so custom dtype/epilogue specs own their kernel signature)."""
    return spec.build_launch(waves_per_eu=waves_per_eu, agpr_alloc=agpr_alloc)


@functools.lru_cache(maxsize=256)
def compile_dense_nt_tiled(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 1,
    waves_per_eu: int = 2,
    agpr_alloc: int = 0,
    nt_vmcnt: int = 3,
    num_xcd: int = 8,
    cbsz: int = 0,
    blgp: int = 0,
    out_fp16: bool = False,
):
    """NT: A [M,K] K-contig, B_T [N,K] K-contig (= B^T of [K,N]), C [M,N]."""
    spec = make_tile_spec(
        layout="nt",
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUP_M=GROUP_M,
        num_xcd=num_xcd,
        nt_vmcnt=nt_vmcnt,
        cbsz=cbsz,
        blgp=blgp,
        out_fp16=out_fp16,
    )
    return _make_tiled_launch(spec, waves_per_eu, agpr_alloc)


@functools.lru_cache(maxsize=128)
def compile_dense_nn_tiled(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 4,
    num_xcd: int = 8,
    waves_per_eu: int = 2,
    agpr_alloc: int = 0,
    b_inline_asm_load: bool = False,
    vmcnt_hint: int = 2,
    cbsz: int = 0,
    blgp: int = 0,
    out_fp16: bool = False,
):
    """NN: A [M,K], B [K,N], C [M,N]."""
    if b_inline_asm_load and agpr_alloc == 0:
        raise ValueError(
            "b_inline_asm_load=True requires agpr_alloc > 0 (a compiler-decided "
            "AGPR count conflicts with the inline-asm operand constraints); "
            "pin AGPR to a nonzero value such as 32."
        )
    spec = make_tile_spec(
        layout="nn",
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUP_M=GROUP_M,
        num_xcd=num_xcd,
        vmcnt_hint=vmcnt_hint,
        b_inline_asm_load=b_inline_asm_load,
        cbsz=cbsz,
        blgp=blgp,
        out_fp16=out_fp16,
    )
    return _make_tiled_launch(spec, waves_per_eu, agpr_alloc)


@functools.lru_cache(maxsize=128)
def compile_dense_tn_tiled(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 4,
    waves_per_eu: int = 2,
    vmcnt_hint: int = 3,
    group_n: int = 0,
    num_xcd: int = 8,
    cbsz: int = 0,
    blgp: int = 0,
    out_fp16: bool = False,
):
    """TN: A [K,M], B [K,N], C [M,N] = A^T @ B. Forces 128 AGPRs (inplace MFMA)."""
    spec = make_tile_spec(
        layout="tn",
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUP_M=GROUP_M,
        num_xcd=num_xcd,
        group_n=group_n,
        vmcnt_hint=vmcnt_hint,
        cbsz=cbsz,
        blgp=blgp,
        out_fp16=out_fp16,
    )
    return _make_tiled_launch(spec, waves_per_eu, agpr_alloc=128)
