###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Primus-Turbo FlyDSL blockscale GEMM kernel.

A single parameterized kernel (``blockscale_gemm.compile_blockscale_gemm``)
serves all three GEMM directions of the blockwise FP8 path; they differ only
along two compile-time axes (both fully resolved at compile time, so each
direction emits the same machine code as a hand-specialized kernel would):

    forward / NT  : out[M,N]    = a[M,K] @ b[N,K]^T     scale_b_mode="block2d", l2_group_m=16
    dgrad   / NN  : grad_a[M,K] = grad_out[M,N] @ b[N,K] scale_b_mode="block2d", l2_group_m=8
    wgrad   / TN  : grad_b[N,K] = grad_out[M,N]^T @ a[M,K] scale_b_mode="col1d", l2_group_m=1

  * ``scale_b_mode``: ``"block2d"`` reads one scale_b per 128-output-column block
    (2D-block weight); ``"col1d"`` reads one per output column (both operands
    1D-block / 1x128 column-quantized along the contraction dim).
  * ``l2_group_m``: L2-aware super-block rasterization height in M-tiles; ``<=1``
    disables the grouped ordering.

Thin per-direction wrappers (``compile_blockscale_{fwd,dgrad,wgrad}_gemm``) bind
the axis values; the launcher imports those.
"""
