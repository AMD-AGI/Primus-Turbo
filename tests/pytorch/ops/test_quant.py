###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

import primus_turbo.pytorch as turbo
from tests.pytorch.test_utils import get_tolerances


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("numel", [6 * 1 * 7168 * 8192])
def test_fp8_quant_dequant(orig_dtype, dest_dtype, numel):
    torch.manual_seed(42)
    x_ref = torch.rand(numel, device="cuda", dtype=orig_dtype)

    amax = x_ref.abs().max()
    scale = torch.full([1], fill_value=torch.finfo(dest_dtype).max / amax, device=x_ref.device)
    scale_inv = 1 / scale

    x_fp8 = torch.ops.primus_turbo_cpp_extension.fp8_quantize(x_ref, scale, dest_dtype)
    x_fp8_ref = (x_ref * scale).to(dest_dtype)

    x = torch.ops.primus_turbo_cpp_extension.fp8_dequantize(x_fp8, scale_inv, orig_dtype)

    torch.testing.assert_close(x_ref, x, **get_tolerances(dest_dtype))
    torch.testing.assert_close(x_fp8, x_fp8_ref, rtol=0, atol=0)
