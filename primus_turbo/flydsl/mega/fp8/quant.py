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
    """
    assert w.dim() == 3, f"expected 3D [G,N,K], got {tuple(w.shape)}"
    G = w.shape[0]
    qs, ss = zip(*(quantize_rowwise_mxfp8(w[g], fmt) for g in range(G)))
    return torch.stack(qs, 0), torch.stack(ss, 0)
