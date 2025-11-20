###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Union

import torch

from primus_turbo.pytorch.core.low_precision import (
    Float4QuantConfig,
    Format,
    ScalingGranularity,
)

__all__ = ["gemm_fp4"]


# TODO:
class FP4GemmMXFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float4QuantConfig,
    ):
        """Forward pass"""
        assert config.granularity == ScalingGranularity.MX_BLOCKWISE
        assert config.format == Format.E2M1_X2

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """Backward pass"""


def gemm_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
    config: Union[Float4QuantConfig, None] = None,
) -> torch.Tensor:
    """ """
    assert a.ndim == 2 and b.ndim == 2, "Only 2D tensors are supported"
    if out_dtype is None:
        out_dtype = torch.result_type(a, b)

    if config is None:
        config = Float4QuantConfig()

    # args = (a, b, trans_a, trans_b, out_dtype, config)
    if config.granularity == ScalingGranularity.MX_BLOCKWISE:
        raise NotImplementedError("MX_BLOCKWISE is not supported yet")
    else:
        raise ValueError(f"Unsupported FP4 ScalingGranularity: {config.granularity}")
