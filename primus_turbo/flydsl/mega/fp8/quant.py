###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MXFP8 quantization helpers for the fused mega MoE forward path.

All activations/weights the mxfp8 mega GEMMs consume are per-1x32 E8M0 block-scaled
along the contraction (K) dim, matching what
``mega/fp8/grouped_gemm_mxfp8_kernel.py`` (and the dense mxfp8 kernel) expect:
raw E8M0 byte scales laid out ``[dim, K // 32]`` (the GEMM preshuffles them to the
broadcast int32 layout itself). These helpers just wrap the project mxfp8 quantizer
so the op/module layer has one place to produce (fp8 data, raw E8M0 scale) pairs.
"""

import torch

from primus_turbo.pytorch.core.low_precision import (
    ScalingGranularity,
    float8_e4m3,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8

MXFP8_BLOCK = 32

# Attribute name under which a weight tensor stashes its cached fp8 quant (keyed by _version).
_WQ_ATTR = "_mxfp8_grouped_q"


def quantize_rowwise_mxfp8(x: torch.Tensor, fmt=float8_e4m3):
    """Rowwise MXFP8 quant of a 2D ``[M, K]`` tensor along K (block=32).

    Returns ``(x_fp8 [M, K], x_scale [M, K // 32])`` where ``x_scale`` is raw E8M0
    (1 byte / 32-K block). Used to quantize dispatched tokens (K = hidden) and the
    SwiGLU output (K = intermediate) before the mxfp8 grouped GEMMs.
    """
    assert x.dim() == 2, f"expected 2D [M,K], got {tuple(x.shape)}"
    assert x.shape[1] % MXFP8_BLOCK == 0, f"K={x.shape[1]} must be a multiple of {MXFP8_BLOCK}"
    return quantize_fp8(x, fmt, ScalingGranularity.MX_BLOCKWISE, block_size=MXFP8_BLOCK, axis=1)


def quantize_grouped_weight_mxfp8(w: torch.Tensor, fmt=float8_e4m3):
    """Per-group MXFP8 quant of grouped weights ``[G, N, K]`` along K (block=32).

    Returns ``(w_fp8 [G, N, K], w_scale [G, N, K // 32])`` (raw E8M0), the B operand +
    scale for the grouped mxfp8 NT GEMM. Each group is quantized independently (its own
    per-1x32 block scales); N/K are the weight's out/in dims.

    Rowwise-along-K quant is per-row independent, so group boundaries don't matter: flatten
    ``[G, N, K] -> [G*N, K]`` and quantize in ONE kernel (bit-identical to per-group), then
    reshape back. Avoids the old ``G``-launch Python loop + two ``torch.stack`` copies (~2 ms at
    G=32 / DSv3 w1) -- a static-weight cost paid every step in real training.
    """
    assert w.dim() == 3, f"expected 3D [G,N,K], got {tuple(w.shape)}"
    G, N, K = w.shape
    q, s = quantize_rowwise_mxfp8(w.reshape(G * N, K), fmt)
    return q.view(G, N, K), s.view(G, N, K // MXFP8_BLOCK)


def quantize_grouped_weight_mxfp8_flydsl(w: torch.Tensor):
    """FlyDSL grouped MXFP8 weight quant -- a ~2.6-2.9x faster, BIT-IDENTICAL drop-in for
    ``quantize_grouped_weight_mxfp8`` (E4M3 only). Rowwise-along-K quant is per-row independent,
    so ``[G, N, K] -> [G*N, K]`` and run the hand-written FlyDSL rowwise kernel
    (``quantize_rowwise_mxfp8_flydsl``, ~5.9 TB/s, near HBM peak vs the generic ~2.3 TB/s), then
    reshape back. The scale is viewed as ``float8_e8m0fnu`` to match ``quantize_grouped_weight_mxfp8``'s
    return dtype (byte-identical raw E8M0). Returns ``(w_fp8 [G,N,K] e4m3, w_scale [G,N,K//32] e8m0)``."""
    from primus_turbo.flydsl.mega.fp8.quant_flydsl import quantize_rowwise_mxfp8_flydsl

    assert w.dim() == 3, f"expected 3D [G,N,K], got {tuple(w.shape)}"
    G, N, K = w.shape
    q, s = quantize_rowwise_mxfp8_flydsl(w.reshape(G * N, K))  # q e4m3 [G*N,K], s uint8 [G*N,K//32]
    return q.view(G, N, K), s.view(torch.float8_e8m0fnu).view(G, N, K // MXFP8_BLOCK)


def quantize_grouped_weight_mxfp8_cached(w: torch.Tensor, fmt=float8_e4m3):
    """Quantize a STATIC weight, caching the fp8 result ON the weight tensor, keyed by
    ``w._version``. Re-quantizes only when the weight changed in place (``optim.step()`` bumps
    ``_version``); otherwise returns the stashed ``(w_fp8, w_scale)``. First call always quantizes.

    Storing on the tensor (vs a global dict) makes it per-weight: auto-scales to many layers with
    no size cap / LRU thrash, and the fp8 copy is freed when the weight is. Correctness-safe
    (``_version`` guards against stale weights) and transfer-safe (keyed off the weight's own
    version, never an activation id -- Rule 11). Reuse pays off across a gradient-accumulation
    window (all micro-steps share one ``_version``) and, once the backward is mxfp8, across
    fwd+bwd; with grad-accum=1 and a forward-only consume it just quantizes every step."""
    v = getattr(w, "_version", 0)
    ent = getattr(w, _WQ_ATTR, None)
    if ent is not None and ent[0] == v and ent[1] is fmt:
        return ent[2], ent[3]
    # FlyDSL path (bit-identical, ~2.6x faster) for the E4M3 default; else the generic quant.
    if fmt is float8_e4m3:
        q, s = quantize_grouped_weight_mxfp8_flydsl(w)
    else:
        q, s = quantize_grouped_weight_mxfp8(w, fmt)
    try:
        setattr(w, _WQ_ATTR, (v, fmt, q, s))
    except (AttributeError, RuntimeError):
        pass  # can't stash on this tensor (rare) -> return freshly quantized, no caching
    return q, s
