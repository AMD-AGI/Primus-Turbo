###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Op-layer MXFP8 weight prep for the mega MoE combine GEMM.

The FlyDSL layer provides the basic primitives (grouped mxfp8 quant + scale preshuffle); this
module composes them into the two operands the combine GEMM consumes. Shared by the forward
(fc2 weight), the backward L1 dgrad (fc1^T combine weight), and the standalone benches, so the
"prepare a combine weight" concept lives once at the op layer rather than inside the kernel file.
"""

import torch

from primus_turbo.flydsl.mega.fp8 import (
    preshuffle_b_scale,
    quantize_grouped_weight_mxfp8_flydsl,
)

__all__ = ["prepare_dispatch_weight_fp8", "prepare_w1_fp8", "prepare_w2_fp8"]


def prepare_dispatch_weight_fp8(w: torch.Tensor):
    """Prepare a grouped weight ``[G, N, K]`` for the fp8 dispatch GEMM -> ``(wq, ws, flat, b_sp)``.

    Grouped mxfp8 quant + int8 flat + scale preshuffle (ScaleBComb, ``pack=1``): every weight
    derivative the dispatch GEMM contracts, so the kernel does no per-call weight work and needs no
    cache of its own. ``flat`` is a view of ``wq``, kept alongside it because the kernel still reads
    ``wq`` for shape and dtype. Static per weight version, so a version-keyed holder computes this
    once per ``optim.step``.
    """
    G, N, K = w.shape
    wq, ws = quantize_grouped_weight_mxfp8_flydsl(w)
    flat = wq.contiguous().reshape(G * N, K).view(torch.int8).reshape(-1)
    return wq, ws, flat, preshuffle_b_scale(ws, G, N, K, pack=1)


def prepare_w1_fp8(w1: torch.Tensor):
    """The L1 fc1 weight ``[G, 2I, H]`` prepped for the dispatch GEMM.

    Thin alias of :func:`prepare_dispatch_weight_fp8`; parallels :func:`prepare_w2_fp8` so both
    weights prep through one layer."""
    return prepare_dispatch_weight_fp8(w1)


def prepare_w2_fp8(l2_weights: torch.Tensor):
    """Prepare a grouped combine-GEMM weight ``[G, N, K]`` for the fp8 combine: grouped mxfp8 quant
    (FlyDSL) + scale preshuffle (ScaleBComb layout) + int8 flat -> ``(weight_flat int8 [G*N*K],
    b_sp int32)``, exactly the two operands the mxfp8 combine GEMM consumes. Static per weight
    version, so a stateful holder computes this ONCE per ``optim.step`` and passes it as ``w2_fp8``
    -- the combine then does NO per-call weight quant OR preshuffle. Used for the forward fc2 weight
    and, transposed, the backward L1 dgrad fc1^T combine weight."""
    G, N, K = l2_weights.shape
    w2q, w2s = quantize_grouped_weight_mxfp8_flydsl(l2_weights)
    b_sp = preshuffle_b_scale(w2s, G, N, K, pack=4)
    weight_flat = w2q.reshape(G * N, K).contiguous().view(torch.int8).reshape(-1)
    return weight_flat, b_sp
