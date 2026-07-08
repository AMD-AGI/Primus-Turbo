###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Correctness test for the FlyDSL grouped MXFP8 (per-1x32 E8M0) NT GEMM kernel
(``primus_turbo/flydsl/mega/fp8/grouped_gemm_mxfp8_kernel.py``).

Isolates GEMM correctness from quant error: A/B are quantized once with the project
mxfp8 quantizer; the reference is an fp32 dequant of those SAME quantized inputs,
matmul'd per group. The kernel is fed the identical fp8 data + raw E8M0 scales, so a
high SNR (>= 25 dB E4M3 / >= 20 dB E5M2) validates the block-scaled grouped matmul.
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

torch.manual_seed(42)


def _mxq(x, fmt):
    # rowwise mxfp8 quant along K (axis=1) -> (fp8 [.,K], e8m0 scale [.,K//32])
    return quantize_fp8(x, fmt, ScalingGranularity.MX_BLOCKWISE, block_size=32, axis=1)


def _mxdq(q, s):
    return dequantize_fp8(
        q, torch.float32, ScalingGranularity.MX_BLOCKWISE, block_size=32, axis=1, scale_inv=s
    )


def _run_grouped_mxfp8(group_sizes, N, K, fmt, group_m=0, num_xcd=1):
    from primus_turbo.flydsl.mega.fp8.grouped_gemm_mxfp8_kernel import (
        grouped_gemm_mxfp8_flydsl_kernel,
    )

    dev = "cuda:0"
    G = len(group_sizes)
    M = sum(group_sizes)
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(G, N, K, device=dev, dtype=torch.bfloat16)
    offs = torch.tensor(
        [0] + list(torch.tensor(group_sizes).cumsum(0)), device=dev, dtype=torch.int64
    )

    aq, a_s = _mxq(a, fmt)
    bqs, bss = zip(*(_mxq(b[g], fmt) for g in range(G)))
    bq, bs = torch.stack(bqs, 0), torch.stack(bss, 0)

    a_deq = _mxdq(aq, a_s)
    c_ref = torch.empty(M, N, device=dev, dtype=torch.float32)
    for g in range(G):
        s, e = int(offs[g]), int(offs[g + 1])
        if e > s:
            c_ref[s:e] = a_deq[s:e] @ _mxdq(bq[g], bs[g]).T

    c = grouped_gemm_mxfp8_flydsl_kernel(
        aq, a_s, bq, bs, offs, out_dtype=torch.bfloat16, group_m=group_m, num_xcd=num_xcd
    )
    torch.cuda.synchronize()

    thr = 25.0 if fmt == float8_e4m3 else 20.0
    snr = compute_snr(c_ref, c)
    print(f"G={G} M={M} N={N} K={K} fmt={fmt} gm={group_m} xcd={num_xcd}  SNR={snr:.2f} dB")
    assert c.shape == (M, N)
    assert snr > thr, f"grouped mxfp8 SNR too low: {snr:.2f} dB (thr {thr})"


@pytest.mark.skipif(not check_mxfp8_support(), reason="MXFP8 grouped GEMM requires gfx950")
@pytest.mark.parametrize(
    "group_sizes,N,K",
    [
        ([256, 256, 256, 256], 512, 512),  # balanced
        ([256, 512, 256, 768], 512, 512),  # unbalanced (each 256-aligned)
        ([256, 0, 512, 256], 512, 512),  # empty group in the middle
        ([256, 256], 4096, 7168),  # mega L1-like (K=H, N=2I)
        ([256, 256], 7168, 2048),  # mega L2-like (K=I, N=H)
    ],
)
def test_grouped_gemm_mxfp8_e4m3(group_sizes, N, K):
    _run_grouped_mxfp8(group_sizes, N, K, float8_e4m3)


@pytest.mark.skipif(not check_mxfp8_support(), reason="MXFP8 grouped GEMM requires gfx950")
def test_grouped_gemm_mxfp8_group_m_xcd():
    _run_grouped_mxfp8([512, 512], 1024, 1024, float8_e4m3, group_m=4, num_xcd=8)


@pytest.mark.skipif(not check_mxfp8_support(), reason="MXFP8 grouped GEMM requires gfx950")
def test_grouped_gemm_mxfp8_e5m2():
    _run_grouped_mxfp8([256, 256], 512, 512, float8_e5m2)
