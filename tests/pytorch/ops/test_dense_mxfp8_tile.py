###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Correctness test for ``gemm_mxfp8_nt_tile`` + the RAW on-the-fly E8M0 scale loader
(``ScaleS2RRaw``) in ``primus_turbo/flydsl/mega/fp8/gemm_mxfp8_tile.py``.

This is the per-tile mxfp8 GEMM closure the FUSED dispatch kernel's gemm role uses;
unlike the grouped kernel it reads raw (un-preshuffled) E8M0 scales directly, so this
isolates that path. A/B are quantized once; the reference is an fp32 dequant of the
SAME quantized inputs matmul'd. High SNR (>= 25 dB E4M3) validates the raw scale MMA.
"""

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import (
    ScalingGranularity,
    check_mxfp8_support,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.ops.quantization import dequantize_fp8, quantize_fp8
from tests.pytorch.test_utils import compute_snr

torch.manual_seed(0)


def _mxq(x, fmt):
    return quantize_fp8(x, fmt, ScalingGranularity.MX_BLOCKWISE, block_size=32, axis=1)


def _mxdq(q, s):
    return dequantize_fp8(
        q, torch.float32, ScalingGranularity.MX_BLOCKWISE, block_size=32, axis=1, scale_inv=s
    )


def _run(M, N, K, fmt):
    from primus_turbo.flydsl.mega.fp8.gemm_mxfp8_tile import dense_mxfp8_tile_gemm

    dev = "cuda:0"
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    aq, a_s = _mxq(a, fmt)
    bq, b_s = _mxq(b, fmt)

    c_ref = _mxdq(aq, a_s) @ _mxdq(bq, b_s).T
    c = dense_mxfp8_tile_gemm(aq, a_s, bq, b_s)
    torch.cuda.synchronize()

    thr = 25.0 if fmt == float8_e4m3 else 20.0
    snr = compute_snr(c_ref, c)
    print(f"M={M} N={N} K={K} fmt={fmt}  SNR={snr:.2f} dB")
    assert c.shape == (M, N)
    assert snr > thr, f"dense mxfp8 tile SNR too low: {snr:.2f} dB (thr {thr})"


@pytest.mark.skipif(not check_mxfp8_support(), reason="MXFP8 requires gfx950")
@pytest.mark.parametrize(
    "M,N,K",
    [
        (256, 256, 256),
        (512, 512, 512),
        (256, 512, 1024),
        (768, 256, 2048),
        (512, 4096, 7168),  # mega L1-like
        (512, 7168, 2048),  # mega L2-like
    ],
)
def test_dense_mxfp8_tile_e4m3(M, N, K):
    _run(M, N, K, float8_e4m3)


@pytest.mark.skipif(not check_mxfp8_support(), reason="MXFP8 requires gfx950")
def test_dense_mxfp8_tile_e5m2():
    _run(512, 512, 512, float8_e5m2)
