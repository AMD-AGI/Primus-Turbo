###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Python launcher for the FlyDSL blockscale (blockwise FP8) GEMM.

Drives three independent vendored kernels (one per GEMM layout) so they can each
be optimized separately, all with per-block scaling (ScaleBlockM=1,
ScaleBlockN=128, ScaleBlockK=128):

    kernels/blockscale_fwd_gemm.py    forward / NT  -> gemm_fp8_blockwise_flydsl
    kernels/blockscale_dgrad_gemm.py  dgrad   / NN  -> gemm_fp8_blockwise_flydsl_dgrad
    kernels/blockscale_wgrad_gemm.py  wgrad   / TN  -> gemm_fp8_blockwise_flydsl_wgrad

The canonical forward / NT path computes:

    out[M, N] = (a_fp8 * a_scale) @ (b_fp8 * b_scale)^T

Layout contract (matches Primus-Turbo's BLOCKWISE quant outputs):
    a_fp8:        [M, K]            fp8  (row-major)
    b_fp8:        [N, K]            fp8  (row-major; pre-shuffled here)
    a_scale_inv:  [M, K // 128]     fp32 (per-row, per-K-block)
    b_scale_inv:  [N // 128, K // 128] fp32 (per-2D-block)
    out:          [M, N]            bf16 / fp16

The kernel itself expects the activation scale transposed to [K // 128, M] and
the weight pre-shuffled with layout (16, 16); both transforms are applied here.

All FlyDSL imports are lazy so that importing this module (and therefore
``primus_turbo``) never fails when FlyDSL is not installed. The compiled kernel
is cached per (M, N, K, tile, out_dtype): ``flyc.compile`` binds the first set
of tensor pointers but the returned handle accepts fresh pointers on later
calls, so caching is keyed only on shape/dtype (never on tensor identity).
"""
from __future__ import annotations

import functools
import os
import sys
from typing import Optional, Tuple

import torch

__all__ = [
    "gemm_fp8_blockwise_flydsl",
    "gemm_fp8_blockwise_flydsl_dgrad",
    "gemm_fp8_blockwise_flydsl_wgrad",
    "flydsl_blockwise_gemm_supported",
    "flydsl_blockwise_wgrad_supported",
    "is_flydsl_available",
    "shuffle_b",
]

# Per-block scale geometry of the FlyDSL blockscale kernel.
_SCALE_BLOCK = 128
# Tile candidates (tile_m, tile_n, tile_k) supported by the kernel. Mirrors the
# search space in FlyDSL's blockscale preshuffle GEMM test harness.
_TILE_CANDIDATES = (
    (16, 64, 256),
    (16, 128, 256),
    (32, 64, 128),
    (32, 64, 256),
    (32, 128, 128),
    (32, 128, 256),
    (64, 64, 128),
    (64, 64, 256),
    (64, 128, 128),
    (64, 128, 256),
    (64, 256, 128),
)
_SUPPORTED_ARCHS = ("gfx942", "gfx950")

# Compiled-kernel cache: key=(M, N, K, tile_m, tile_n, tile_k, out_dtype_str).
_compiled_cache: dict = {}


def _flydsl_root_candidates() -> Tuple[str, ...]:
    """Filesystem locations to probe for the FlyDSL source ``kernels`` package.

    The blockscale kernel lives in FlyDSL's ``kernels`` package, which is not
    shipped inside the ``flydsl`` wheel. Allow an explicit override via env and
    fall back to a couple of conventional checkout locations.
    """
    roots = []
    for env_var in ("PRIMUS_TURBO_FLYDSL_ROOT", "FLYDSL_ROOT"):
        val = os.environ.get(env_var, "").strip()
        if val:
            roots.append(val)
    roots.extend(
        (
            "/apps/tas/yaoc/work/learn/FlyDSL",
            os.path.expanduser("~/FlyDSL"),
        )
    )
    return tuple(dict.fromkeys(roots))  # dedup, keep order


@functools.lru_cache(maxsize=1)
def _ensure_flydsl_kernels_on_path() -> bool:
    """Make FlyDSL's source ``kernels`` package importable; True on success.

    The ``kernels`` package (mfma_epilogues, mfma_preshuffle_pipeline, ...) ships
    with the FlyDSL source tree, not the wheel, so it is resolved on demand. The
    vendored wgrad kernel also imports those helpers via ``from kernels...``.
    """
    try:
        import flydsl.compiler  # noqa: F401  (ensures flydsl is importable)
    except Exception:
        return False

    try:
        import kernels.mfma_preshuffle_pipeline  # noqa: F401

        return True
    except Exception:
        pass

    for root in _flydsl_root_candidates():
        if os.path.isdir(os.path.join(root, "kernels")) and root not in sys.path:
            sys.path.insert(0, root)
        try:
            import kernels.mfma_preshuffle_pipeline  # noqa: F401

            return True
        except Exception:
            continue
    return False


@functools.lru_cache(maxsize=1)
def _load_fwd_compile_fn():
    """Import the vendored forward / NT ``compile_blockscale_fwd_gemm`` (or None)."""
    if not _ensure_flydsl_kernels_on_path():
        return None
    try:
        from primus_turbo.flydsl.gemm.kernels.blockscale_fwd_gemm import (
            compile_blockscale_fwd_gemm,
        )

        return compile_blockscale_fwd_gemm
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _load_dgrad_compile_fn():
    """Import the vendored dgrad / NN ``compile_blockscale_dgrad_gemm`` (or None)."""
    if not _ensure_flydsl_kernels_on_path():
        return None
    try:
        from primus_turbo.flydsl.gemm.kernels.blockscale_dgrad_gemm import (
            compile_blockscale_dgrad_gemm,
        )

        return compile_blockscale_dgrad_gemm
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _load_wgrad_compile_fn():
    """Import the vendored ``compile_blockscale_wgrad_gemm`` (1Dx1D, or None)."""
    if not _ensure_flydsl_kernels_on_path():
        return None
    try:
        from primus_turbo.flydsl.gemm.kernels.blockscale_wgrad_gemm import (
            compile_blockscale_wgrad_gemm,
        )

        return compile_blockscale_wgrad_gemm
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _flyc():
    import flydsl.compiler as flyc  # type: ignore

    return flyc


@functools.lru_cache(maxsize=None)
def _device_arch(device_index: int = 0) -> str:
    try:
        name = torch.cuda.get_device_properties(device_index).gcnArchName
    except Exception:
        return ""
    return name.split(":")[0]


def is_flydsl_available() -> bool:
    """True iff FlyDSL + its ``kernels`` package can be imported on this host.

    Keyed on the vendored forward kernel (the NT dispatch path); the dgrad/wgrad
    copies share the same FlyDSL ``kernels`` helpers, so this is a valid proxy
    for all three.
    """
    return _load_fwd_compile_fn() is not None


def _select_tile(
    M: int, N: int, K: int, scale_block_k: int = _SCALE_BLOCK
) -> Optional[Tuple[int, int, int]]:
    """Pick a (tile_m, tile_n, tile_k) that the kernel can run for this shape.

    Returns ``None`` when no candidate satisfies the kernel's hard divisibility
    constraints, signalling the caller to fall back to another backend.
    """

    def _valid(tm: int, tn: int, tk: int) -> bool:
        return (
            N % tn == 0
            and K % tk == 0
            and tk % scale_block_k == 0
            and (tm * tk) % 256 == 0
            and (tm * tk) // 256 >= 16
        )

    valid = [t for t in _TILE_CANDIDATES if _valid(*t)]
    if not valid:
        return None

    def _score(t: Tuple[int, int, int]) -> int:
        tm, tn, tk = t
        s = 0
        total_blocks = ((M + tm - 1) // tm) * (N // tn)
        s += 15 if total_blocks >= 256 else (10 if total_blocks >= 128 else (5 if total_blocks >= 64 else 0))
        if M <= 48:
            s += 12 if tm == 16 else (8 if tm == 32 else 0)
        elif M <= 128:
            s += 10 if tm == 32 else (6 if tm == 16 else (4 if tm == 64 else 0))
        elif M <= 512:
            s += 12 if tm == 64 else (8 if tm == 32 else 0)
        else:
            s += 12 if tm == 64 else 0
        if M <= 128:
            s += 6 if tn == 64 else (4 if tn == 128 else (2 if tn == 256 else 0))
        else:
            # Round-2: for large shapes prefer the larger tile_n=256 variant
            # (num_acc_n=4). It doubles A-fragment / scale_a reuse per MFMA,
            # cutting per-MFMA LDS/address/scale VALU overhead on the
            # compute-bound backward (wgrad/dgrad) path. tile_n=256 now
            # outscores tile_n=128 by the same margin tile_n=128 led by before.
            s += 8 if tn == 256 else (6 if tn == 128 else (4 if tn == 64 else 0))
        s += 6 if tk == 128 else 3
        return s

    return max(valid, key=_score)


def flydsl_blockwise_gemm_supported(M: int, N: int, K: int) -> bool:
    """Cheap pre-flight check used by the backend's ``can_handle``."""
    if _device_arch() not in _SUPPORTED_ARCHS:
        return False
    # The kernel asserts K % scale_block_k == 0; N only needs to divide some
    # tile_n (weight scales use ceil(N / 128) rows, so N need not be a multiple
    # of 128). _select_tile enforces the N % tile_n / K % tile_k constraints.
    if K % _SCALE_BLOCK != 0:
        return False
    if N % 16 != 0:  # B pre-shuffle / MFMA layout needs N divisible by 16
        return False
    if _select_tile(M, N, K) is None:
        return False
    return is_flydsl_available()


def flydsl_blockwise_wgrad_supported(m: int, n: int, k: int) -> bool:
    """Pre-flight for the wgrad (TN, 1Dx1D) path.

    Logical dims from ``get_gemm_logical_shape(a[M,K], b[M,N], trans_a=True,
    trans_b=False)``: ``m = K`` (grad_b cols), ``n = N`` (grad_b rows),
    ``k = M`` (contraction). The kernel runs with (M_kernel=n, N_kernel=m,
    K_kernel=k); both operands carry 1D-block (1x128) scales along M.
    """
    if _device_arch() not in _SUPPORTED_ARCHS:
        return False
    if k % _SCALE_BLOCK != 0:  # contraction M must tile into 128-blocks (col-quant)
        return False
    if m % 16 != 0:  # grad_b cols (K) -> B pre-shuffle / MFMA layout needs %16
        return False
    if _select_tile(n, m, k) is None:  # kernel dims (M_k=n, N_k=m, K_k=k)
        return False
    return _load_wgrad_compile_fn() is not None


def shuffle_b(b: torch.Tensor, layout: Tuple[int, int] = (16, 16)) -> torch.Tensor:
    """Pre-shuffle a ``[N, K]`` FP8 weight into the kernel's MFMA-friendly layout.

    Equivalent to FlyDSL's / AITER's ``shuffle_weight(..., layout=(16, 16))``
    for a 2D tensor.
    """
    N, K = b.shape
    IN, IK = layout
    BK = IK * 2
    K_inner = 16 // b.element_size()
    BN = IN
    assert N % BN == 0 and K % BK == 0, f"shuffle_b: N={N} K={K} not divisible by ({BN}, {BK})"
    v = b.view(N // BN, BN, K // BK, BK // K_inner, K_inner)
    return v.permute(0, 2, 3, 1, 4).contiguous().view(N, K)


def gemm_fp8_blockwise_flydsl(
    a_fp8: torch.Tensor,  # [M, K] fp8 (row-major / row-quant)
    b_fp8: torch.Tensor,  # [N, K] fp8 (weight; NOT pre-shuffled)
    a_scale_inv: torch.Tensor,  # [M, K // 128] fp32
    b_scale_inv: torch.Tensor,  # [N // 128, K // 128] fp32
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Forward / NT blockwise FP8 GEMM via the FlyDSL blockscale kernel.

    Computes ``out[M, N] = (a_fp8 * a_scale) @ (b_fp8 * b_scale)^T`` with
    per-block scales (1xK-block for the activation, 128x128 for the weight).
    """
    assert a_fp8.ndim == 2 and b_fp8.ndim == 2, "a and b must be 2D"
    assert a_scale_inv.ndim == 2 and b_scale_inv.ndim == 2, "scales must be 2D"
    assert out_dtype in (torch.bfloat16, torch.float16), "out_dtype must be bf16 or fp16"

    M, K = a_fp8.shape
    N, Kb = b_fp8.shape
    assert K == Kb, f"K mismatch: a has K={K}, b has K={Kb}"

    compile_fn = _load_fwd_compile_fn()
    if compile_fn is None:
        raise RuntimeError(
            "FlyDSL forward kernel is unavailable. Install FlyDSL and make its source "
            "`kernels` package importable (set PRIMUS_TURBO_FLYDSL_ROOT or PYTHONPATH)."
        )

    tile = _select_tile(M, N, K)
    if tile is None:
        raise ValueError(
            f"No valid FlyDSL blockscale tile for (M={M}, N={N}, K={K}); "
            "N and K must be divisible by a supported (tile_n, tile_k)."
        )
    tile_m, tile_n, tile_k = tile

    out_dtype_str = "bf16" if out_dtype == torch.bfloat16 else "fp16"
    flyc = _flyc()

    # FlyDSL expects scale_a transposed to [K // 128, M] (flattened) and
    # scale_b row-major [N // 128, K // 128] (flattened).
    a_scale_t = a_scale_inv.transpose(0, 1).contiguous().view(-1)
    b_scale_flat = b_scale_inv.contiguous().view(-1)
    b_shuffled = shuffle_b(b_fp8)

    out = torch.empty((M, N), dtype=out_dtype, device=a_fp8.device)
    stream = torch.cuda.current_stream()

    key = ("fwd", M, N, K, tile_m, tile_n, tile_k, out_dtype_str)
    compiled = _compiled_cache.get(key)
    if compiled is None:
        exe = compile_fn(
            M=M,
            N=N,
            K=K,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            scale_block_k=_SCALE_BLOCK,
            out_dtype=out_dtype_str,
        )
        compiled = flyc.compile(exe, out, a_fp8, b_shuffled, a_scale_t, b_scale_flat, M, N, stream)
        _compiled_cache[key] = compiled

    compiled(out, a_fp8, b_shuffled, a_scale_t, b_scale_flat, M, N, stream)
    return out


def gemm_fp8_blockwise_flydsl_dgrad(
    grad_out_fp8: torch.Tensor,  # [M, N] fp8 (row-quant along N)
    b_fp8: torch.Tensor,  # [N, K] fp8 (forward weight, 2D-block)
    grad_out_scale_inv: torch.Tensor,  # [M, N // 128] fp32
    b_scale_inv: torch.Tensor,  # [N // 128, K // 128] fp32
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """dgrad (NN) blockwise FP8 GEMM: ``grad_a[M, K] = grad_out[M, N] @ b[N, K]``.

    Uses the dedicated dgrad kernel (``blockscale_dgrad_gemm.py``). The general
    kernel form is ``C[M, OUT] = A[M, CON] @ B[OUT, CON]^T``; dgrad maps to
    ``CON = N`` (contraction), ``OUT = K``:

        A = grad_out[M, N]                 (1D-block along N)
        B = b^T  i.e. [K, N]               (transpose of the [N, K] weight)
        scale_b = b_scale^T -> [K // 128, N // 128]

    BASELINE: this transposes + pre-shuffles the weight every call. Independent
    from the forward path, so the dgrad kernel can later be rewritten with a
    native NN global-load that consumes ``b[N, K]`` directly (dropping these
    transposes) without touching the forward kernel.
    """
    assert grad_out_fp8.ndim == 2 and b_fp8.ndim == 2, "inputs must be 2D"

    compile_fn = _load_dgrad_compile_fn()
    if compile_fn is None:
        raise RuntimeError(
            "FlyDSL dgrad kernel is unavailable. Install FlyDSL and make its source "
            "`kernels` package importable (set PRIMUS_TURBO_FLYDSL_ROOT or PYTHONPATH)."
        )

    M, N = grad_out_fp8.shape  # grad_out [M, N], contraction over N
    Nb, K = b_fp8.shape  # weight [N, K] -> output dim K
    assert N == Nb, f"N mismatch: grad_out has N={N}, b has N={Nb}"

    # B = b^T = [K, N]; its 2D-block scale transposes to [K // 128, N // 128].
    b_t = b_fp8.transpose(0, 1).contiguous()
    b_scale_t = b_scale_inv.transpose(0, 1).contiguous()

    # Kernel dims: rows M_kernel=M, output cols N_kernel=K, contraction K_kernel=N.
    tile = _select_tile(M, K, N)
    if tile is None:
        raise ValueError(
            f"No valid FlyDSL dgrad tile for grad_a[M={M}, K={K}] (contract N={N})."
        )
    tile_m, tile_n, tile_k = tile

    out_dtype_str = "bf16" if out_dtype == torch.bfloat16 else "fp16"
    flyc = _flyc()

    a_scale_t = grad_out_scale_inv.transpose(0, 1).contiguous().view(-1)  # [N//128, M]
    b_scale_flat = b_scale_t.contiguous().view(-1)
    b_shuffled = shuffle_b(b_t)

    out = torch.empty((M, K), dtype=out_dtype, device=grad_out_fp8.device)
    stream = torch.cuda.current_stream()

    key = ("dgrad", M, K, N, tile_m, tile_n, tile_k, out_dtype_str)
    compiled = _compiled_cache.get(key)
    if compiled is None:
        exe = compile_fn(
            M=M,
            N=K,
            K=N,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            scale_block_k=_SCALE_BLOCK,
            out_dtype=out_dtype_str,
        )
        compiled = flyc.compile(exe, out, grad_out_fp8, b_shuffled, a_scale_t, b_scale_flat, M, K, stream)
        _compiled_cache[key] = compiled

    compiled(out, grad_out_fp8, b_shuffled, a_scale_t, b_scale_flat, M, K, stream)
    return out


def gemm_fp8_blockwise_flydsl_wgrad(
    a_col_fp8: torch.Tensor,  # [M, K] fp8 (activation, col-quant: 1D-block along M)
    grad_out_col_fp8: torch.Tensor,  # [M, N] fp8 (grad_out, col-quant: 1D-block along M)
    a_col_scale_inv: torch.Tensor,  # [M // 128, K] fp32
    grad_out_col_scale_inv: torch.Tensor,  # [M // 128, N] fp32
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """wgrad (TN, 1Dx1D): ``grad_b[N, K] = grad_out[M, N]^T @ a[M, K]`` (contract M).

    Uses the native 1Dx1D blockscale kernel (per-output-column ``scale_b``). Maps
    to ``C[M_k, OUT] = A[M_k, CON] @ B[OUT, CON]^T`` with ``M_k=N, OUT=K, CON=M``:

        A = grad_out^T[N, M]            scale_a = grad_out_col_scale [M//128, N]
        B = a^T[K, M] (pre-shuffled)    scale_b = a_col_scale^T  [K, M//128]  (per column)

    Both operands are column-quantized (1D-block along the contraction dim M), so
    no extra re-quantization is needed in the backward pass.
    """
    assert a_col_fp8.ndim == 2 and grad_out_col_fp8.ndim == 2, "inputs must be 2D"
    M, K = a_col_fp8.shape
    M2, N = grad_out_col_fp8.shape
    assert M == M2, f"M mismatch: a_col has M={M}, grad_out_col has M={M2}"

    compile_fn = _load_wgrad_compile_fn()
    if compile_fn is None:
        raise RuntimeError(
            "FlyDSL wgrad kernel is unavailable. Ensure FlyDSL's source `kernels` "
            "package is importable (set PRIMUS_TURBO_FLYDSL_ROOT or PYTHONPATH)."
        )

    # Kernel dims: rows M_kernel=N, output cols N_kernel=K, contraction K_kernel=M.
    tile = _select_tile(N, K, M)
    if tile is None:
        raise ValueError(
            f"No valid FlyDSL wgrad tile for grad_b[N={N}, K={K}] (contract M={M}); "
            "K must divide a supported tile_n and M must tile into tile_k / 128-blocks."
        )
    tile_m, tile_n, tile_k = tile

    out_dtype_str = "bf16" if out_dtype == torch.bfloat16 else "fp16"
    flyc = _flyc()

    arg_a = grad_out_col_fp8.transpose(0, 1).contiguous()  # [N, M]
    arg_b = shuffle_b(a_col_fp8.transpose(0, 1).contiguous())  # pre-shuffle a^T [K, M]
    # scale_a already in [scale_con=M//128, rows=N]; scale_b per-output-column [K, M//128].
    a_scale_flat = grad_out_col_scale_inv.contiguous().view(-1)
    b_scale_flat = a_col_scale_inv.transpose(0, 1).contiguous().view(-1)

    out = torch.empty((N, K), dtype=out_dtype, device=a_col_fp8.device)
    stream = torch.cuda.current_stream()

    key = ("wgrad", N, K, M, tile_m, tile_n, tile_k, out_dtype_str)
    compiled = _compiled_cache.get(key)
    if compiled is None:
        exe = compile_fn(
            M=N,
            N=K,
            K=M,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            scale_block_k=_SCALE_BLOCK,
            out_dtype=out_dtype_str,
        )
        compiled = flyc.compile(exe, out, arg_a, arg_b, a_scale_flat, b_scale_flat, N, K, stream)
        _compiled_cache[key] = compiled

    compiled(out, arg_a, arg_b, a_scale_flat, b_scale_flat, N, K, stream)
    return out
