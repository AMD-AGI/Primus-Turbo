###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Op-layer MXFP8 weight prep for the mega MoE combine GEMM.

The FlyDSL layer provides the basic primitives (grouped mxfp8 quant + scale preshuffle); this
module composes them into the two operands the combine GEMM consumes. Shared by the forward
(fc2 weight), the backward STEP3 (fc1^T combine weight), and the standalone benches, so the
"prepare a combine weight" concept lives once at the op layer rather than inside the kernel file.
"""

import torch

from primus_turbo.flydsl.mega.fp8 import (
    preshuffle_b_scale,
    quantize_grouped_weight_mxfp8_flydsl,
)

__all__ = ["prepare_w1_fp8", "prepare_w2_fp8"]


def prepare_w1_fp8(w1: torch.Tensor):
    """Prepare the L1 fc1 weight ``[G, 2I, H]`` for the fp8 dispatch GEMM: grouped mxfp8 quant
    (FlyDSL, E4M3) -> ``(w1q [G,2I,H] fp8, w1s [G,2I,H//32] raw E8M0)``. Unlike w2, the L1 dispatch
    GEMM preshuffles the weight scale internally, so this is just the raw grouped quant (no
    preshuffle / flatten). Parallels :func:`prepare_w2_fp8` so both weights prep through one layer."""
    return quantize_grouped_weight_mxfp8_flydsl(w1)


def prepare_w2_fp8(l2_weights: torch.Tensor):
    """Prepare a grouped combine-GEMM weight ``[G, N, K]`` for the fp8 combine: grouped mxfp8 quant
    (FlyDSL) + scale preshuffle (ScaleBComb layout) + int8 flat -> ``(weight_flat int8 [G*N*K],
    b_sp int32)``, exactly the two operands the mxfp8 combine GEMM consumes. Static per weight
    version, so a stateful holder computes this ONCE per ``optim.step`` and passes it as ``w2_fp8``
    -- the combine then does NO per-call weight quant OR preshuffle. Used for the forward fc2 weight
    and, transposed, the backward STEP3 fc1^T combine weight."""
    G, N, K = l2_weights.shape
    w2q, w2s = quantize_grouped_weight_mxfp8_flydsl(l2_weights)
    b_sp = preshuffle_b_scale(w2s, G, N, K)
    weight_flat = w2q.reshape(G * N, K).contiguous().view(torch.int8).reshape(-1)
    return weight_flat, b_sp
