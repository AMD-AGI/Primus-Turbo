###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import check_mxfp4_support, float4_e2m1fn_x2


def _hip_dual(x):
    # C++ reference dual (rowwise + colwise-transpose), 2d-block amax, no SR/shuffle.
    return torch.ops.primus_turbo_cpp_extension.quantize_mxfp4_dual(
        x, float4_e2m1fn_x2, 128, True, False, False, True, False, False
    )


def _eq(a, b):
    a8, b8 = a.view(torch.uint8), b.view(torch.uint8)
    return a8.shape == b8.shape and torch.equal(a8, b8)


def _assert_bitexact(hip_out, fly_out):
    for h, f in zip(hip_out, fly_out):
        assert _eq(h, f)


# (R, C): square, C>R (down-proj -> col_locality swap), R>C (gate/up).
@pytest.mark.skipif(not check_mxfp4_support(), reason="MXFP4 dual needs gfx950")
@pytest.mark.parametrize("shape", [(4096, 4096), (4096, 11008), (11008, 4096)])
def test_flydsl_dual_quant_2d_bitexact(shape):
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import flydsl_dual_quant

    R, C = shape
    torch.manual_seed(0)
    x = torch.randn((R, C), dtype=torch.bfloat16, device="cuda")
    fly = flydsl_dual_quant(x, float4_e2m1fn_x2, False, False, row_2d=True, col_2d=True)
    _assert_bitexact(_hip_dual(x), fly)


# (G, N, K): K<N (padded K tail), K>N (col_locality swap), aligned.
@pytest.mark.skipif(not check_mxfp4_support(), reason="MXFP4 dual needs gfx950")
@pytest.mark.parametrize("shape", [(8, 5760, 2880), (8, 4096, 7168), (8, 8192, 4096)])
def test_flydsl_dual_quant_batched_bitexact(shape):
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import flydsl_dual_quant_batched

    G, N, K = shape
    torch.manual_seed(0)
    x = torch.randn((G, N, K), dtype=torch.bfloat16, device="cuda")
    fly = flydsl_dual_quant_batched(x, float4_e2m1fn_x2, False, False, row_2d=True, col_2d=True)
    _assert_bitexact(_hip_dual(x), fly)
