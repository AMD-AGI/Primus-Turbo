###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Reject incompatible quantization requests before any device kernel can run."""

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float4_e2m1fn_x2, float8_e4m3
from primus_turbo.pytorch.ops import quantization as ops


@pytest.mark.parametrize(
    "name",
    [
        "quantize_fp8",
        "quantize_fp8_with_trans",
        "grouped_quantize_fp8",
        "grouped_quantize_fp8_with_trans",
        "dequantize_fp8",
        "grouped_dequantize_fp8",
        "quantize_fp4",
        "quantize_fp4_with_trans",
        "grouped_quantize_fp4",
        "grouped_quantize_fp4_with_trans",
        "dequantize_fp4",
        "grouped_dequantize_fp4",
    ],
)
def test_unknown_granularity_is_rejected(name):
    x = torch.zeros(2, 32)
    dtype = float4_e2m1fn_x2 if "fp4" in name else float8_e4m3
    kwargs = {}
    extra = []
    if name.startswith("grouped_quantize"):
        extra = [torch.tensor([2]), torch.tensor([0, 2])]
    if name == "grouped_dequantize_fp8":
        extra = [torch.tensor([0, 2]), torch.tensor([0, 128])]
    if "dequantize" in name:
        kwargs["scale_inv"] = torch.ones(1)
        dtype = torch.bfloat16
    with pytest.raises(NotImplementedError, match="Unknown granularity"):
        getattr(ops, name)(x, dtype, "unsupported", *extra, **kwargs)


def test_rowwise_dequantization_requires_axis():
    with pytest.raises(ValueError, match="axis must be specified"):
        ops.dequantize_fp8(
            torch.zeros(2, 32).to(float8_e4m3),
            torch.bfloat16,
            ScalingGranularity.ROWWISE,
            scale_inv=torch.ones(2, 1),
        )
