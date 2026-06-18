###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus-Turbo dense FP8 GEMM kernel (FlyDSL) v2: NT, NN and TN layouts.

Same public surface as ``gemm_fp8_kernel`` (the ``gemm_fp8_tensorwise_flydsl_kernel``
entry + the per-shape autotune dispatch), but the kernels are NO LONGER the
standalone hand-written ``_compile_dense_*`` bodies -- they are built HERE from the
composable ``common.tile_spec`` TileSpec policies. The kernel-builder layer
(``make_tile_spec`` + ``_make_tiled_launch`` + ``compile_dense_{nt,nn,tn}_tiled``)
lives in THIS module; ``common.tile_spec`` is now the pure spec/policy library
(the held policies + ``DenseFp8TileSpec`` + ``validate_tile_config``).

The tiled compilers emit IR that is bit-identical to the standalone kernels
(verified by ``.gate/compare_v2.py``), so v2 is a drop-in replacement:
same dispatch, same autotune tables, same emitted symbols, same JIT cache shape.
The host-side glue (byte-flat views, scalar-scale cast, compiled-launcher cache,
autotune candidate tables) is reused from ``gemm_fp8_kernel`` so the two modules
cannot drift."""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

from primus_turbo.flydsl.utils.gemm_helper import make_value_attrs

# The spec/policy library (held policies + the spec class + the legality gate).
# Kernel construction (the builders below) consumes these but lives in THIS module.
from primus_turbo.flydsl.common.tile_spec import (
    BLOCK_K,
    DenseFp8TileSpec,
    validate_tile_config,
)

# Pure host-side glue + autotune tables reused from v1 (no kernel construction
# here): shared so v2 and v1 tune over the same candidates and cannot drift.
from primus_turbo.flydsl.gemm.gemm_fp8_kernel import (
    _NN_CANDIDATES,
    _NT_CANDIDATES,
    _as_i8_flat,
    _get_compiled_dense,
    _scalar_scale,
)


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
        block_tile=(BLOCK_M, BLOCK_N, BLOCK_K),
        # Per-wave tile 64x32 -> 2x4 wave grid (quadrant 128x128 / warp); LDS half fixed 2x2.
        warp_tile=(BLOCK_M // 4, BLOCK_N // 8, BLOCK_K),
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
# ``_compile_dense_{nt,nn,tn}`` (same signature / returned launcher). The dense
# concrete kernel + launch live HERE (the spec only provides ``emit`` + hooks +
# ``scheduler.grid``).
# ──────────────────────────────────────────────────────────────────────
def _make_tiled_launch(spec, waves_per_eu, agpr_alloc):
    """Build the dense ``@flyc.kernel`` + ``@flyc.jit`` launcher for ``spec``. The
    kernel references ``spec.cache_tag`` so the JIT cache distinguishes configs; its
    __name__ matches the standalone so the emitted symbol is identical."""
    tag = spec.cache_tag

    def kernel(A, B, C, A_scale, B_scale, c_m: fx.Int32, c_n: fx.Int32):
        _ = tag  # JIT cache-key discriminator; emits no IR
        spec.emit(A=A, B=B, C=C, A_scale=A_scale,
                  B_scale=B_scale, c_m=c_m, c_n=c_n)

    kernel.__name__ = spec.kernel_name
    kernel.__qualname__ = kernel.__name__
    kernel = flyc.kernel(kernel, known_block_size=[512, 1, 1])

    def launch(A, B, C, A_scale, B_scale, c_m: fx.Int32, c_n: fx.Int32, stream: fx.Stream):
        grid_x = spec.scheduler_spec.grid(spec.geom, c_m, c_n)
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

    # Match the standalone host-launcher symbol (``launch_dense_{layout}``).
    launch.__name__ = "launch_dense_" + spec.layout
    launch.__qualname__ = launch.__name__
    return flyc.jit(launch)


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
    # agpr rule lives in the central validator (the spec ctor can't see agpr_alloc).
    validate_tile_config(layout="nn", K=K, block_tile=(BLOCK_M, BLOCK_N, BLOCK_K),
                         b_inline_asm_load=b_inline_asm_load, agpr_alloc=agpr_alloc)
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


# Standalone-name aliases so the autotune dispatch bodies below (and external
# diff harnesses) read identically -- the builders above ARE the tiled compilers.
_compile_dense_nt = compile_dense_nt_tiled
_compile_dense_nn = compile_dense_nn_tiled
_compile_dense_tn = compile_dense_tn_tiled

# Per-shape autotune caches (v2-local; keyed by (M,N,K,cbsz,blgp,out_fp16)).
_NN_AUTOTUNE_CACHE: dict = {}
_NT_AUTOTUNE_CACHE: dict = {}
_TN_AUTOTUNE_CACHE: dict = {}


def _autotune_nn_dispatch(args, M, N, K, cbsz=0, blgp=0, out_fp16=False):
    """First-call bench NN candidates, cache best (launch, cfg) by (M,N,K).

    Identical to v1's NN autotune; only the compiler is the tiled (TileSpec) one.
    Runtime micro-benches each (BM, GROUP_M, num_xcd, AG) candidate, finite-checks
    the output, times 2-warmup + 20-iter, and caches the fastest by shape.
    """
    import torch as _torch

    key = (M, N, K, cbsz, blgp, out_fp16)
    if key in _NN_AUTOTUNE_CACHE:
        return _NN_AUTOTUNE_CACHE[key]
    out_view = args[2]
    best_us = float("inf")
    best = None
    for bm, gm, xcd, ag in _NN_CANDIDATES:
        # odd-M (M % bm != 0) is fine: the partial last M-tile is bounded by c_m
        # (StoreCPerTensor clamp) and the global SRD (HW OOB clamp on the A G2S
        # load), so no even-tiling filter is needed.
        try:
            # inline-asm ds_read_b64_tr_b8 on by default (drops the per-K-iter
            # compiler-auto vmcnt(0) drains).
            launch = _compile_dense_nn(
                K=K,
                BLOCK_M=bm,
                BLOCK_N=256,
                GROUP_M=gm,
                num_xcd=xcd,
                agpr_alloc=ag,
                b_inline_asm_load=True,
                vmcnt_hint=2,
                cbsz=cbsz,
                blgp=blgp,
                out_fp16=out_fp16,
            )
            c = _get_compiled_dense(launch, args)
            c(*args)
            _torch.cuda.synchronize()
            sample = out_view.view(-1)[:1024].float()
            if not _torch.isfinite(sample).all().item():
                continue
            for _ in range(2):
                c(*args)
            _torch.cuda.synchronize()
            e0 = _torch.cuda.Event(enable_timing=True)
            e1 = _torch.cuda.Event(enable_timing=True)
            _torch.cuda.synchronize()
            e0.record()
            for _ in range(20):
                c(*args)
            e1.record()
            _torch.cuda.synchronize()
            us = e0.elapsed_time(e1) * 1000.0 / 20
            if us < best_us:
                best_us = us
                best = (launch, (bm, gm, xcd, ag))
        except Exception:
            continue
    if best is None:
        raise RuntimeError(f"NN autotune found no working cfg for ({M},{N},{K})")
    _NN_AUTOTUNE_CACHE[key] = best
    return best


def _autotune_nt_dispatch(args, M, N, K, cbsz=0, blgp=0, out_fp16=False):
    """First-call bench NT candidates, cache best (launch, cfg) by (M,N,K).

    Identical to v1's NT autotune; only the compiler is the tiled (TileSpec) one.
    """
    import torch as _torch

    key = (M, N, K, cbsz, blgp, out_fp16)
    if key in _NT_AUTOTUNE_CACHE:
        return _NT_AUTOTUNE_CACHE[key]
    out_view = args[2]
    best_us = float("inf")
    best = None
    for bm, gm, xcd, ag in _NT_CANDIDATES:
        # odd-M (M % bm != 0) is fine: the partial last M-tile is bounded by c_m
        # (StoreCPerTensor clamp) and the global SRD (HW OOB clamp on the A G2S
        # load), so no even-tiling filter is needed.
        try:
            launch = _compile_dense_nt(
                K=K,
                BLOCK_M=bm,
                BLOCK_N=256,
                GROUP_M=gm,
                agpr_alloc=ag,
                num_xcd=xcd,
                cbsz=cbsz,
                blgp=blgp,
                out_fp16=out_fp16,
            )
            c = _get_compiled_dense(launch, args)
            c(*args)
            _torch.cuda.synchronize()
            sample = out_view.view(-1)[:1024].float()
            if not _torch.isfinite(sample).all().item():
                continue
            for _ in range(2):
                c(*args)
            _torch.cuda.synchronize()
            e0 = _torch.cuda.Event(enable_timing=True)
            e1 = _torch.cuda.Event(enable_timing=True)
            _torch.cuda.synchronize()
            e0.record()
            for _ in range(20):
                c(*args)
            e1.record()
            _torch.cuda.synchronize()
            us = e0.elapsed_time(e1) * 1000.0 / 20
            if us < best_us:
                best_us = us
                best = (launch, (bm, gm, xcd, ag))
        except Exception:
            continue
    if best is None:
        raise RuntimeError(f"NT autotune found no working cfg for ({M},{N},{K})")
    _NT_AUTOTUNE_CACHE[key] = best
    return best


def _autotune_tn_dispatch(args, M, N, K, cbsz=0, blgp=0, out_fp16=False):
    """First-call bench TN candidates, cache best (launch, cfg) by (M,N,K).

    Identical to v1's TN autotune; only the compiler is the tiled (TileSpec) one.
    1D GROUP_M=4 with num_xcd 8 vs 1 (XCD-aware PID remap).
    """
    import torch as _torch

    key = (M, N, K, cbsz, blgp, out_fp16)
    if key in _TN_AUTOTUNE_CACHE:
        return _TN_AUTOTUNE_CACHE[key]
    # Occupancy routing: BLOCK_M=BLOCK_N=256 yields ceil(M/256)*ceil(N/256) tiles;
    # below NUM_CUS the grid can't fill every CU, so BLOCK_M=128 doubles the M-tile
    # count. Above it the smaller block's per-tile overhead dominates.
    NUM_CUS = 256
    tiles_256 = ((M + 255) // 256) * ((N + 255) // 256)
    bm = 128 if tiles_256 < NUM_CUS else 256
    out_view = args[2]
    best_us = float("inf")
    best = None
    for xcd in (8, 1):
        try:
            launch = _compile_dense_tn(
                K=K,
                BLOCK_M=bm,
                BLOCK_N=256,
                GROUP_M=4,
                vmcnt_hint=3,
                group_n=0,
                num_xcd=xcd,
                cbsz=cbsz,
                blgp=blgp,
                out_fp16=out_fp16,
            )
            c = _get_compiled_dense(launch, args)
            c(*args)
            _torch.cuda.synchronize()
            sample = out_view.view(-1)[:1024].float()
            if not _torch.isfinite(sample).all().item():
                continue
            for _ in range(2):
                c(*args)
            _torch.cuda.synchronize()
            e0 = _torch.cuda.Event(enable_timing=True)
            e1 = _torch.cuda.Event(enable_timing=True)
            _torch.cuda.synchronize()
            e0.record()
            for _ in range(20):
                c(*args)
            e1.record()
            _torch.cuda.synchronize()
            us = e0.elapsed_time(e1) * 1000.0 / 20
            if us < best_us:
                best_us = us
                best = (launch, (bm, 4, 0, xcd))
        except Exception:
            continue
    if best is None:
        raise RuntimeError(f"TN autotune found no working cfg for ({M},{N},{K})")
    _TN_AUTOTUNE_CACHE[key] = best
    return best


def gemm_fp8_tensorwise_flydsl_kernel(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
) -> torch.Tensor:
    """Dense FP8 GEMM, per-tensor scaling (TileSpec-built kernels). Inputs
    E4M3/E5M2/hybrid, out bf16/fp16, arbitrary K (native K-tail). Dispatch by
    (trans_a, trans_b): NT (F,T), NN (F,F, dgrad), TN (T,F) run native; TT (T,T)
    unsupported. trans_c=True returns out.t().contiguous()."""
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise NotImplementedError(f"FlyDSL wrapper emits bf16 or fp16. Got {out_dtype}.")
    assert a.dim() == 2 and b.dim() == 2
    # Per-operand fp8 format -> MFMA cbsz(srcA)/blgp(srcB): 0=E4M3, 1=E5M2.
    cbsz = 1 if a.dtype == torch.float8_e5m2 else 0
    blgp = 1 if b.dtype == torch.float8_e5m2 else 0
    # fp16 vs bf16 output dtype for StoreCPerTensor (both from the f32 accumulator).
    out_fp16 = out_dtype == torch.float16

    if trans_a and (not trans_b):
        # TN native: A [K, M], B [K, N]. Math C = A^T @ B.
        K_a, M = a.shape
        K_b, N = b.shape
        assert K_a == K_b, f"TN K mismatch: a {a.shape}, b {b.shape}"
        K = K_a
        a_scale_v = _scalar_scale(a_scale_inv, a.device)
        b_scale_v = _scalar_scale(b_scale_inv, a.device)
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
        # TN: per-shape autotune picks the best candidate cfg, cached by (M,N,K).
        args = (
            _as_i8_flat(a),
            _as_i8_flat(b),
            out.contiguous().view(-1),
            a_scale_v,
            b_scale_v,
            M,
            N,
            torch.cuda.current_stream(),
        )
        launch, _cfg = _autotune_tn_dispatch(args, M, N, K, cbsz, blgp, out_fp16)
        _get_compiled_dense(launch, args)(*args)
        if trans_c:
            return out.t().contiguous()
        return out

    # Dispatch by layout.
    if (not trans_a) and (not trans_b):
        # NN native: A [M, K], B [K, N].
        M, K_a = a.shape
        K_b, N = b.shape
        assert K_a == K_b, f"NN K mismatch: a {a.shape}, b {b.shape}"
        K = K_a
        a_scale_v = _scalar_scale(a_scale_inv, a.device)
        b_scale_v = _scalar_scale(b_scale_inv, a.device)
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
        # NN: per-shape runtime autotune over the candidate tiles, caches by
        # (M,N,K). Build args before autotune (it benches against them).
        args = (
            _as_i8_flat(a),
            _as_i8_flat(b),
            out.contiguous().view(-1),
            a_scale_v,
            b_scale_v,
            M,
            N,
            torch.cuda.current_stream(),
        )
        launch, _cfg = _autotune_nn_dispatch(args, M, N, K, cbsz, blgp, out_fp16)
        _get_compiled_dense(launch, args)(*args)
    elif (not trans_a) and trans_b:
        # NT native: A [M, K], B [N, K] (B^T storage of [K, N]).
        M, K_a = a.shape
        N, K_b = b.shape
        assert K_a == K_b, f"NT K mismatch: a {a.shape}, b {b.shape}"
        K = K_a
        a_scale_v = _scalar_scale(a_scale_inv, a.device)
        b_scale_v = _scalar_scale(b_scale_inv, a.device)
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
        # NT: per-shape runtime autotune over the 8w/v3 candidate tiles, caches
        # by (M,N,K). Build args before autotune (it benches against them).
        args = (
            _as_i8_flat(a),
            _as_i8_flat(b),
            out.contiguous().view(-1),
            a_scale_v,
            b_scale_v,
            M,
            N,
            torch.cuda.current_stream(),
        )
        launch, _cfg = _autotune_nt_dispatch(args, M, N, K, cbsz, blgp, out_fp16)
        _get_compiled_dense(launch, args)(*args)
    else:
        raise NotImplementedError(
            f"FlyDSL fp8 GEMM does not support the TT layout " f"(trans_a={trans_a}, trans_b={trans_b})."
        )
    if trans_c:
        return out.t().contiguous()
    return out
