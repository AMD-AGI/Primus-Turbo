###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import pytest
import torch

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.float8 import ScalingGranularity
from primus_turbo.pytorch.ops import quantize_fp8
from tests.pytorch.ref.quantization_ref import (
    quantize_fp8_rowwise_ref,
    quantize_fp8_tensorwise_ref,
)
from tests.pytorch.test_utils import get_tolerances


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("numel", [6 * 1 * 7168 * 8192])
@pytest.mark.parametrize("dynamic_quantize", [True, False])
@pytest.mark.parametrize("torch_compile", [True, False])
def test_quantize_fp8_tensorwise(orig_dtype, dest_dtype, numel, dynamic_quantize, torch_compile):
    torch._dynamo.reset()
    torch.manual_seed(42)

    x = torch.rand(numel, device="cuda", dtype=orig_dtype)
    x_ref = x.detach().clone()
    x_fp8_ref, x_scale_ref, x_scale_inv_ref = quantize_fp8_tensorwise_ref(x_ref, dest_dtype)

    scale = None
    if dynamic_quantize == False:
        scale = x_scale_ref.detach().clone()

    if torch_compile is True:
        compiled_func = torch.compile(
            lambda t: quantize_fp8(t, dest_dtype, granularity=ScalingGranularity.TENSORWISE, scale=scale),
            fullgraph=True,
            mode="max-autotune",
        )
        x_fp8, x_scale_inv = compiled_func(x)
    else:
        x_fp8, x_scale_inv = quantize_fp8(
            x, dest_dtype, granularity=ScalingGranularity.TENSORWISE, scale=scale
        )

    torch.testing.assert_close(x_scale_inv_ref, x_scale_inv, **get_tolerances(torch.float32))
    torch.testing.assert_close(
        x_fp8_ref.to(torch.float32) * x_scale_inv_ref,
        x_fp8.to(torch.float32) * x_scale_inv,
        **get_tolerances(dest_dtype)
    )


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("axis", [-1, -2, -3])
@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("M", [1, 7168])
@pytest.mark.parametrize("N", [4096])
def test_quantize_fp8_rowwise(orig_dtype, dest_dtype, axis, B, M, N):
    # print("\n", orig_dtype, dest_dtype, axis, B, M, N)

    torch.manual_seed(42)

    x = torch.rand((B, M, N), device="cuda", dtype=orig_dtype)
    x_ref = x.detach().clone()

    x_fp8, x_scale_inv = quantize_fp8(x, dest_dtype, granularity=ScalingGranularity.ROWWISE, axis=axis)
    x_fp8_ref, x_scale_inv_ref = quantize_fp8_rowwise_ref(x_ref, dest_dtype, axis)

    torch.testing.assert_close(x_scale_inv_ref, x_scale_inv, **get_tolerances(torch.float32))
    torch.testing.assert_close(
        x_fp8_ref.to(torch.float32) * x_scale_inv_ref,
        x_fp8.to(torch.float32) * x_scale_inv,
        **get_tolerances(dest_dtype)
    )
