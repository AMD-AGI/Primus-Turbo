###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pinned per-shape configs for the FlyDSL MXFP4 GEMM (``gemm_mxfp4_flydsl_kernel``) on prepacked scales.

Without them the first call of a shape runs a timed sweep (swizzle / wave layout; every candidate is bit-identical),
so which kernel runs depends on measurement noise and the first call pays the sweep. These are the configs the
256-wide, no-split-K autotune picked for the Flux training GEMM shapes (backward and forward MXFP4) and smaller-batch
variants of them, verified bit-exact over sustained calls; aiter's exact-shape A4W4 tilescale code objects are built
from the same configs.

(M, N, K) -> (cfg, mode): cfg = (group_m, group_n, num_xcds, wlv, elgk, taccw, coop), keyed in ``_MXFP4_CFG_CACHE`` by
(M, N, K, row_bytes=None, out_fp16=False); mode = (k-split mode, splits), keyed in ``_MXFP4_KSPLIT_CACHE`` with
scales_prepacked=True.
"""

PINNED = {
    (512, 1024, 1024): ((4, 0, 8, 10, 9, False, False), (0, 1)),
    (512, 1024, 3072): ((4, 0, 8, 10, 9, False, False), (0, 1)),
    (512, 1024, 4096): ((4, 0, 8, 10, 9, False, False), (0, 1)),
    (512, 4096, 1024): ((4, 0, 8, 10, 9, False, False), (0, 1)),
    (1024, 1024, 512): ((4, 0, 8, 10, 9, False, False), (0, 1)),
    (1024, 1024, 1024): ((4, 0, 8, 10, 9, False, False), (0, 1)),
    (1024, 1024, 3072): ((4, 0, 8, 10, 9, False, False), (0, 1)),
    (1024, 1024, 4096): ((4, 0, 8, 10, 9, False, False), (0, 1)),
    (1024, 4096, 512): ((4, 0, 8, 10, 9, False, False), (0, 1)),
    (1024, 4096, 1024): ((4, 0, 8, 10, 9, False, False), (0, 1)),
    (3072, 1024, 512): ((4, 0, 8, 10, 9, False, False), (0, 1)),
    (3072, 1024, 1024): ((4, 0, 8, 10, 9, False, False), (0, 1)),
    (3072, 3072, 8192): ((4, 4, 8, 16, 15, False, False), (0, 1)),
    (3072, 3072, 16384): ((4, 4, 8, 16, 15, False, False), (0, 1)),
    (3072, 12288, 8192): ((4, 0, 8, 10, 9, False, False), (0, 1)),
    (3072, 12288, 16384): ((4, 8, 8, 10, 9, False, False), (0, 1)),
    (4096, 1024, 512): ((4, 0, 8, 10, 9, False, False), (0, 1)),
    (4096, 1024, 1024): ((4, 0, 8, 10, 9, False, False), (0, 1)),
    (8192, 3072, 3072): ((4, 4, 8, 10, 9, False, False), (0, 1)),
    (8192, 3072, 9216): ((4, 4, 8, 10, 9, False, False), (0, 1)),
    (8192, 3072, 12288): ((4, 0, 8, 10, 9, False, False), (0, 1)),
    (8192, 12288, 3072): ((4, 4, 8, 10, 9, False, False), (0, 1)),
    (9216, 3072, 8192): ((4, 0, 8, 10, 9, False, False), (0, 1)),
    (9216, 3072, 16384): ((4, 8, 8, 10, 9, False, False), (0, 1)),
    (12288, 3072, 8192): ((4, 4, 8, 10, 9, False, False), (0, 1)),
    (12288, 3072, 16384): ((4, 4, 8, 10, 9, False, False), (0, 1)),
    (16384, 3072, 3072): ((4, 8, 8, 10, 9, False, False), (0, 1)),
    (16384, 3072, 9216): ((4, 4, 8, 10, 9, False, False), (0, 1)),
    (16384, 3072, 12288): ((4, 8, 8, 10, 9, False, False), (0, 1)),
    (16384, 12288, 3072): ((4, 8, 8, 10, 9, False, False), (0, 1)),
}
