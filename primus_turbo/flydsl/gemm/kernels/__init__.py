###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Primus-Turbo FlyDSL blockscale GEMM kernels — one independent source file per
GEMM layout, each a vendored baseline to be optimized separately:

    blockscale_fwd_gemm.py    forward / NT  : out[M,N]    = a[M,K] @ b[N,K]^T
    blockscale_dgrad_gemm.py  dgrad   / NN  : grad_a[M,K] = grad_out[M,N] @ b[N,K]
    blockscale_wgrad_gemm.py  wgrad   / TN  : grad_b[N,K] = grad_out[M,N]^T @ a[M,K]

The fwd and dgrad copies are byte-identical baselines today (dgrad is fed the
transposed weight by the launcher); the wgrad copy differs only in scale_b load
granularity (per-output-column 1Dx1D). Keeping three files lets each kernel be
tuned independently without cross-coupling.
"""
