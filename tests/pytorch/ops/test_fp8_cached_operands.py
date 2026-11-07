###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Reuse quantized forward/gradient operands across dense and grouped FP8 GEMM."""

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    ScaleDtype,
    ScalingGranularity,
    ScalingRecipe,
    float8_e4m3,
)
from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor, QuantizedTensorPair
from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.ops import gemm_fp8, grouped_gemm_fp8
from tests.pytorch.ops.gradient_contract_helpers import caller_thread_backward  # noqa: F401
from tests.pytorch.test_utils import compute_snr

pytestmark = [
    pytest.mark.usefixtures("caller_thread_backward"),
    pytest.mark.skipif(not is_gfx950(), reason="This regression suite requires gfx950"),
]


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize(
    "granularity",
    [ScalingGranularity.TENSORWISE, ScalingGranularity.ROWWISE, ScalingGranularity.MX_BLOCKWISE],
)
@pytest.mark.parametrize("cache_a", [False, True])
def test_cached_operands_match_fp32(grouped, granularity, cache_a):
    torch.manual_seed(820)
    mx = granularity == ScalingGranularity.MX_BLOCKWISE
    config = Float8QuantConfig(
        granularity=granularity,
        block_size=32 if mx else None,
        scale_dtype=ScaleDtype.E8M0 if mx else ScaleDtype.FP32,
    )
    if granularity == ScalingGranularity.ROWWISE:
        GlobalBackendManager.set_gemm_backend(BackendType.TRITON)
        GlobalBackendManager.set_grouped_gemm_backend(BackendType.TRITON)
    a = torch.randn(384, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn((2, 128, 256) if grouped else (128, 256), device="cuda", dtype=torch.bfloat16)
    lens = torch.tensor([128, 256], device="cuda", dtype=torch.int64)
    args, leaves, refs = [], [], []
    for index, value in enumerate((a, b)):
        if index == 0 and not cache_a:
            leaf = value.detach().requires_grad_()
            arg = leaf
        else:
            extra = {"group_lens": lens} if grouped and index == 0 else {}
            quant_kwargs = dict(
                dest_dtype=float8_e4m3,
                granularity=granularity,
                block_size=config.block_size,
                scaling_recipe=ScalingRecipe(use_2d_block=mx and index == 1),
                **extra,
            )
            leaf = QuantizedTensor.quantize(value, axis=-1, **quant_kwargs).requires_grad_()
            col = (
                leaf.t().contiguous()
                if granularity == ScalingGranularity.TENSORWISE and not grouped
                else QuantizedTensor.quantize(leaf.dequantize(), axis=-2, **quant_kwargs)
            )
            arg = QuantizedTensorPair(leaf, col)
        args.append(arg)
        leaves.append(leaf)
        refs.append(
            (leaf.dequantize() if isinstance(leaf, QuantizedTensor) else leaf)
            .detach()
            .float()
            .requires_grad_()
        )
    if grouped:
        out = grouped_gemm_fp8(*args, lens, trans_b=True, config=config)
        ref = torch.cat([x @ w.T for x, w in zip(refs[0].split([128, 256]), refs[1])])
    else:
        out = gemm_fp8(*args, trans_b=True, config=config)
        ref = refs[0] @ refs[1].T
    grad = torch.randn_like(out)
    actual_grads = torch.autograd.grad(out, leaves, grad)
    expected_grads = torch.autograd.grad(ref, refs, grad.float())
    for actual, expected in zip((out, *actual_grads), (ref, *expected_grads)):
        assert actual.shape == expected.shape
        assert compute_snr(expected, actual.float()) > 24
