###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MXFP8 grouped-weight quantization for the fused mega MoE.

All activations/weights the mxfp8 mega GEMMs consume are per-1x32 E8M0 block-scaled
along the contraction (K) dim: raw E8M0 byte scales laid out ``[dim, K // 32]`` (the
GEMM preshuffles them to the broadcast int32 layout itself).
"""

import torch

MXFP8_BLOCK = 32


def quantize_grouped_weight_mxfp8_flydsl(w: torch.Tensor):
    """Per-group MXFP8 quant of grouped weights ``[G, N, K]`` along K (block=32), E4M3.

    Rowwise-along-K quant is per-row independent, so group boundaries don't matter:
    ``[G, N, K] -> [G*N, K]`` and run the hand-written FlyDSL rowwise kernel
    (``quantize_rowwise_mxfp8_flydsl``, ~5.9 TB/s, near HBM peak vs the generic ~2.3 TB/s),
    then reshape back -- one kernel instead of a ``G``-launch Python loop (~2 ms at G=32 /
    DSv3 w1, a static-weight cost otherwise paid every step). The scale is viewed as
    ``float8_e8m0fnu`` (byte-identical raw E8M0). Returns
    ``(w_fp8 [G,N,K] e4m3, w_scale [G,N,K//32] e8m0)``."""
    from primus_turbo.flydsl.mega.fp8.quant_flydsl import quantize_rowwise_mxfp8_flydsl

    assert w.dim() == 3, f"expected 3D [G,N,K], got {tuple(w.shape)}"
    G, N, K = w.shape
    q, s = quantize_rowwise_mxfp8_flydsl(w.reshape(G * N, K))  # q e4m3 [G*N,K], s uint8 [G*N,K//32]
    return q.view(G, N, K), s.view(torch.float8_e8m0fnu).view(G, N, K // MXFP8_BLOCK)
