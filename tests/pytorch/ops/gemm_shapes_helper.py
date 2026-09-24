###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shape lists shared by the dense and grouped GEMM unit tests."""

# ---------------------------------------------------------------------------
# Grouped GEMM: (B, M, N, K)
# ---------------------------------------------------------------------------
# Every B / M / (N, K) value of the former full sweeps appears at least once.
GROUPED_GEMM_SHAPES = [
    (1, 2048, 4096, 7168),  # single group -> dispatcher's non-grouped special case
    (2, 128, 2048, 1536),  # M below one tile
    (2, 1024, 1408, 2048),  # N not 256-aligned
    (3, 512, 2048, 1408),  # odd group count, K not 256-aligned
    (8, 256, 2816, 2048),
    (8, 2048, 3072, 5120),
    (16, 1024, 5120, 1536),
    (16, 128, 7168, 2048),  # many tiny groups, wide N
    (32, 512, 2048, 1536),  # max group count
]
# Pre-quantized inputs / graph capture reuse the kernels of the main sweep; only the
# input handling differs.
GROUPED_GEMM_SHAPES_SMALL = [(1, 2048, 4096, 7168), (3, 512, 2048, 1408), (16, 128, 7168, 2048)]

# ---------------------------------------------------------------------------
# Dense GEMM: (m, n, k)
# ---------------------------------------------------------------------------
# The precisions disagree on what a legal shape is (MX needs 16-aligned m / n / k,
# BLOCKWISE a 128-aligned k), so the lists stay per precision / granularity.

# bf16 / fp16 / fp32.
GEMM_SHAPES = [
    (1, 1, 1),
    (1, 4096, 2048),  # m=1 (GEMV-like)
    (2048, 1, 1024),  # n=1
    (512, 2048, 1),  # k=1
    (16, 16, 16),
    (128, 129, 127),  # odd n / k
    (256, 512, 255),
    (1024, 1024, 512),
    (2048, 4096, 2048),
]

# FP8 TENSORWISE / ROWWISE. m = 255 / 507 are not 32-aligned (CK declines them);
# k = 576 is not 128-aligned.
GEMM_FP8_SHAPES = [
    (255, 1024, 576),
    (256, 512, 256),
    (256, 4096, 576),
    (507, 4096, 1024),
    (512, 2048, 2048),
    (512, 1024, 512),
]
GEMM_FP8_SHAPES_SMALL = [(512, 1024, 512), (1024, 512, 1024)]

# FP8 BLOCKWISE (block_size=128).
GEMM_FP8_BLOCKWISE_SHAPES = [(512, 1024, 256), (1024, 4096, 1024), (512, 4096, 4096)]
GEMM_FP8_BLOCKWISE_SHAPES_SMALL = [(256, 4096, 1024), (1024, 256, 256), (512, 512, 1024)]

# MXFP8 and MXFP4 (NT, m / n / k multiples of 16). Shapes 128-aligned nowhere run on
# hipBLASLt / AITER / default dispatch only; TURBO needs m / n / k 128-aligned and
# >= 384, FlyDSL MXFP8 128-aligned and >= 256, FlyDSL MXFP4 64-aligned.
GEMM_MX_SHAPES = [
    (176, 352, 160),
    (256, 256, 128),
    (256, 352, 160),
    (256, 1024, 256),
    (512, 1024, 512),
    (512, 2048, 512),
    (1024, 1024, 1024),
    (1024, 2048, 1024),
]
GEMM_MX_SHAPES_SMALL = [
    (256, 352, 160),
    (256, 1024, 128),
    (512, 2048, 512),
    (1024, 256, 512),
    (1024, 1024, 1024),
]
