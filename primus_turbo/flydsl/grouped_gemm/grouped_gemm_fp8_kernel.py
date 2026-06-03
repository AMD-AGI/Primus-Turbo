###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL FP8 grouped GEMM wrapper.

Exposes:
    - grouped_gemm_fp8_tensorwise_flydsl_kernel              (fwd + dgrad via trans_b)
    - grouped_gemm_fp8_tensorwise_variable_k_flydsl_kernel   (wgrad; per-group dense fallback)

API is drop-in compatible with the Triton counterpart so it can be substituted
inside a backend dispatcher without callsite changes.

variable_k (wgrad) is implemented as a host-side per-group loop calling the
dense FlyDSL 8-wave kernel. Each per-group call has its own kernel launch and
JIT-compile (cached by K = M_g per group). This is slower than Triton's
single-launch variable-K kernel; replacement with a native FlyDSL grouped
variable-K kernel is a separate effort.

Source kernel lives in the FlyDSL repo (NOT vendored here). Discovery:
  1. Try importing `kernels.grouped_gemm_fp8_pertensor` directly (works when the
     FlyDSL repo root, e.g. `/workspace/code/FlyDSL`, is on PYTHONPATH).
  2. Fall back to inferring the repo root from `flydsl.__file__` and inserting
     it onto sys.path.

Only TENSORWISE scaling is supported. Constraints: K % 128 == 0, out_dtype must
be torch.bfloat16, A must be row-major contiguous (trans_a not supported). For
trans_b=False inputs (e.g. dgrad), the wrapper transposes B on host before
the launch -- correctness only, this overhead is paid each call.
"""

from __future__ import annotations

import functools
import os
import sys
from typing import Tuple

import torch

# ──────────────────────────────────────────────────────────────────────────────
# Lazy FlyDSL imports (so this module loads even if FlyDSL is not installed;
# can_handle() in the backend layer is the gate that prevents calling us).
# ──────────────────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def _flydsl_kernels():
    """Resolve FlyDSL imports. Returns (build_tile_arrays, compile_fn, flyc_module)."""
    try:
        from kernels.grouped_gemm_fp8_pertensor import (
            build_tile_arrays,
            compile_grouped_gemm_fp8_pertensor,
        )
    except ImportError:
        try:
            import flydsl  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "FlyDSL is not installed. Install via `pip install -e .` from the "
                "FlyDSL repo root, then ensure the repo root (containing `kernels/`) "
                "is on PYTHONPATH."
            ) from e
        flydsl_root = os.path.abspath(os.path.join(os.path.dirname(flydsl.__file__), "..", ".."))
        if flydsl_root not in sys.path:
            sys.path.insert(0, flydsl_root)
        from kernels.grouped_gemm_fp8_pertensor import (  # noqa: E402
            build_tile_arrays,
            compile_grouped_gemm_fp8_pertensor,
        )
    import flydsl.compiler as flyc

    return build_tile_arrays, compile_grouped_gemm_fp8_pertensor, flyc


@functools.lru_cache(maxsize=128)
def _compile_launch_for_k(K: int, BLOCK_M: int = 256, BLOCK_N: int = 256):
    """Compile the per-K specialization once and cache the launch function."""
    _, compile_fn, _ = _flydsl_kernels()
    return compile_fn(K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)


# Compiled-callable cache. Avoids paying the `flyc.compile` cost on every call
# (single-test benchmarks bind compiled = flyc.compile(...) once and reuse).
_COMPILED_GROUPED_CACHE: dict = {}


def _get_compiled_grouped(launch, args):
    _, _, flyc = _flydsl_kernels()
    key_parts = [id(launch)]
    for a in args:
        if isinstance(a, torch.Tensor):
            key_parts.append((tuple(a.shape), a.dtype))
        elif isinstance(a, int):
            key_parts.append(a)
        else:
            key_parts.append(type(a).__name__)
    key = tuple(key_parts)
    cached = _COMPILED_GROUPED_CACHE.get(key)
    if cached is None:
        cached = flyc.compile(launch, *args)
        _COMPILED_GROUPED_CACHE[key] = cached
    return cached


def flydsl_available() -> bool:
    """Cheap probe used by Layer-2 backend's can_handle."""
    try:
        _flydsl_kernels()
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────


def _as_i8_flat(t: torch.Tensor) -> torch.Tensor:
    """fp8 tensor → flat int8 1-D view for the FlyDSL launch arglist."""
    if "float8" in str(t.dtype):
        return t.contiguous().view(torch.int8).view(-1)
    return t.contiguous().view(-1)


def _broadcast_scale(scale: torch.Tensor, length: int, device: torch.device) -> torch.Tensor:
    """FlyDSL StoreC reads scale by base_row / base_col index. For a per-tensor
    scalar scale, broadcast to a length-`length` fp32 vector."""
    if scale.numel() == 1:
        scalar = float(scale.flatten()[0].item())
        return torch.full((length,), scalar, dtype=torch.float32, device=device)
    if scale.numel() == length:
        return scale.to(dtype=torch.float32, device=device).contiguous().view(-1)
    raise ValueError(f"per-tensor wrapper expected scale.numel() in {{1, {length}}}, got {scale.shape}")


def _build_routing(
    group_offs: torch.Tensor, BLOCK_M: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    build_tile_arrays, _, _ = _flydsl_kernels()
    m_starts = group_offs.tolist()
    tg, tbm, trl = build_tile_arrays(m_starts, BLOCK_M=BLOCK_M)
    device = group_offs.device
    tg_t = torch.tensor(tg, dtype=torch.int32, device=device)
    tbm_t = torch.tensor(tbm, dtype=torch.int32, device=device)
    trl_t = torch.tensor(trl, dtype=torch.int32, device=device)
    return tg_t, tbm_t, trl_t, tg_t.numel()


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point — API-aligned with the Triton counterpart
# ──────────────────────────────────────────────────────────────────────────────


def grouped_gemm_fp8_tensorwise_flydsl_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    group_offs: torch.Tensor,
    trans_b: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Persistent grouped FP8 GEMM (per-tensor scaling) via the FlyDSL kernel.

    Args:
        a: [M_total, K] fp8 e4m3 (or e5m2). trans_a is NOT supported (always row-major).
        b: [G, K, N] when trans_b=False; [G, N, K] when trans_b=True.
        a_scale: scalar fp32 (shape [] or [1]) -- broadcast to length-M internally.
        b_scale: scalar fp32 -- broadcast to length-N internally.
        group_offs: [G+1] int prefix sum of group lengths.
        trans_b: see b layout above.
        out_dtype: must be torch.bfloat16 (FlyDSL StoreC writes bf16 only).

    Returns:
        [M_total, N] bfloat16
    """
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 3, f"b must be 3D, got {b.shape}"
    if out_dtype != torch.bfloat16:
        raise NotImplementedError(
            f"FlyDSL wrapper currently only emits bf16 (StoreC fixed). Got {out_dtype}."
        )

    M_total, K_a = a.shape
    b.shape[0]
    if trans_b:
        N, K_b = b.shape[1], b.shape[2]
    else:
        K_b, N = b.shape[1], b.shape[2]
    assert K_a == K_b, f"K mismatch: a has K={K_a}, b has K={K_b}"
    K = K_a

    if K % 128 != 0:
        raise NotImplementedError(f"FlyDSL grouped GEMM requires K % 128 == 0 (BLOCK_K=128). Got K={K}.")

    # FlyDSL kernel internally expects B in [G, N, K] (K-contig per N-row).
    # If caller passed [G, K, N] (trans_b=False), pre-transpose on host.
    if trans_b:
        b_internal = b
    else:
        b_internal = b.transpose(-1, -2).contiguous()  # [G, K, N] -> [G, N, K]

    tg_t, tbm_t, trl_t, num_tiles_m = _build_routing(group_offs, BLOCK_M=256)

    a_scale_v = _broadcast_scale(a_scale, M_total, a.device)
    b_scale_v = _broadcast_scale(b_scale, N, a.device)

    out = torch.empty((M_total, N), dtype=out_dtype, device=a.device)

    launch = _compile_launch_for_k(K=K, BLOCK_M=256, BLOCK_N=256)

    args = (
        _as_i8_flat(a),
        _as_i8_flat(b_internal),
        out.contiguous().view(-1),
        a_scale_v,
        b_scale_v,
        tg_t.contiguous().view(-1),
        tbm_t.contiguous().view(-1),
        trl_t.contiguous().view(-1),
        num_tiles_m,
        M_total,
        N,
        torch.cuda.current_stream(),
    )
    _get_compiled_grouped(launch, args)(*args)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Variable-K (wgrad) — per-group dense fallback
# ──────────────────────────────────────────────────────────────────────────────


def grouped_gemm_fp8_tensorwise_variable_k_flydsl_kernel(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_scale: torch.Tensor,
    group_offs: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Variable-K grouped FP8 GEMM (wgrad). API-aligned with the Triton counterpart.

    Math (matching primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel):
        out[g] = lhs[m_s:m_e].T @ rhs[m_s:m_e]   for g = 0..G-1
        where m_s, m_e = group_offs[g], group_offs[g+1] and the contraction dim
        M_g = m_e - m_s varies per group.

    Output orientation control (trans_c in the gemm_fp8 autograd path) is the
    backend dispatcher's responsibility -- it swaps lhs/rhs before calling this
    wrapper so the natural [G, lhs.shape[1], rhs.shape[1]] output matches what
    the autograd Function expects for grad_b.

    P0 implementation: per-group dense fallback. The dense kernel requires
    K_inner (= M_g here) divisible by 128.

    Args:
        lhs: [M_total, K_lhs] fp8 row-major.
        rhs: [M_total, N_rhs] fp8 row-major.
        lhs_scale / rhs_scale: scalar fp32 per-tensor.
        group_offs: [G+1] int prefix sum.
        out_dtype: torch.bfloat16.

    Returns:
        out: [G, K_lhs, N_rhs]
    """
    assert lhs.dim() == 2 and rhs.dim() == 2
    assert lhs.shape[0] == rhs.shape[0], f"lhs and rhs must share M_total: lhs {lhs.shape}, rhs {rhs.shape}"
    if out_dtype != torch.bfloat16:
        raise NotImplementedError(f"FlyDSL variable_k wrapper only emits bf16. Got {out_dtype}.")

    M_total, K_lhs = lhs.shape
    _, N_rhs = rhs.shape
    m_starts = group_offs.tolist()
    G = len(m_starts) - 1

    out = torch.empty((G, K_lhs, N_rhs), dtype=out_dtype, device=lhs.device)

    # Lazy import to avoid circular dep at module load.
    from primus_turbo.flydsl.gemm.gemm_fp8_kernel import (
        gemm_fp8_tensorwise_flydsl_kernel,
    )

    for g in range(G):
        m_s, m_e = m_starts[g], m_starts[g + 1]
        M_g = m_e - m_s
        if M_g == 0:
            out[g].zero_()
            continue
        if M_g % 128 != 0:
            raise NotImplementedError(
                f"FlyDSL variable_k requires each group M_g % 128 == 0 (group {g} has M_g={M_g})."
            )
        # Per-group gemm:  out[g] [K_lhs, N_rhs] = lhs[m_s:m_e].T @ rhs[m_s:m_e]
        # Cast for the NT-native dense wrapper:
        #   pass lhs slice [M_g, K_lhs] with trans_a=True (kernel re-orients to [K_lhs, M_g])
        #   pass rhs slice [M_g, N_rhs] with trans_b=False (kernel transposes to [N_rhs, M_g])
        # gives gemm = lhs.T @ rhs = [K_lhs, N_rhs].
        out[g] = gemm_fp8_tensorwise_flydsl_kernel(
            lhs[m_s:m_e],
            lhs_scale,
            rhs[m_s:m_e],
            rhs_scale,
            trans_a=True,
            trans_b=False,
            out_dtype=out_dtype,
            trans_c=False,
        )
    return out
