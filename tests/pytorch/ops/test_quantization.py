###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import pytest
import torch

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.low_precision import (
    MXScalingRecipe,
    ScalingGranularity,
    check_mxfp4_support,
)
from primus_turbo.pytorch.ops import dequantize_fp8, quantize_fp4, quantize_fp8
from primus_turbo.pytorch.ops.quantization import dequantize_fp4
from tests.pytorch.ref.quantization_ref import dequantize_fp8_ref, quantize_fp8_ref
from tests.pytorch.test_utils import get_tolerances


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("numel", [6 * 1 * 7168 * 8192])
@pytest.mark.parametrize("dynamic_quantize", [True, False])
@pytest.mark.parametrize("torch_compile", [True, False])
@pytest.mark.parametrize("granularity", [ScalingGranularity.TENSORWISE])
def test_quantize_fp8_tensorwise(orig_dtype, dest_dtype, numel, dynamic_quantize, torch_compile, granularity):
    torch.manual_seed(42)

    x = torch.rand(numel, device="cuda", dtype=orig_dtype)
    x_ref = x.detach().clone()
    x_fp8_ref, x_scale_ref, x_scale_inv_ref = quantize_fp8_ref(x_ref, dest_dtype, granularity)

    # Quantize
    scale = None
    if dynamic_quantize == False:
        scale = x_scale_ref.detach().clone()

    if torch_compile is True:
        torch._dynamo.reset()
        compiled_func = torch.compile(
            lambda t: quantize_fp8(t, dest_dtype, granularity=granularity, scale=scale),
            fullgraph=True,
            mode="max-autotune",
        )
        x_fp8, x_scale_inv = compiled_func(x)
    else:
        x_fp8, x_scale_inv = quantize_fp8(x, dest_dtype, granularity=granularity, scale=scale)

    torch.testing.assert_close(x_scale_inv_ref, x_scale_inv, **get_tolerances(torch.float32))
    torch.testing.assert_close(
        x_fp8_ref.to(torch.float32) * x_scale_inv_ref,
        x_fp8.to(torch.float32) * x_scale_inv,
        **get_tolerances(dest_dtype)
    )

    # DeQuantize
    x_dq = dequantize_fp8(x_fp8, orig_dtype, granularity, scale_inv=x_scale_inv)
    x_dq_ref = dequantize_fp8_ref(x_fp8_ref, orig_dtype, granularity, scale_inv=x_scale_inv_ref)
    torch.testing.assert_close(x_dq, x_dq_ref, **get_tolerances(dest_dtype))


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("axis", [-1, -2, -3, 0, 1, 2])
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("M", [1, 111, 7168])
@pytest.mark.parametrize("N", [1, 111, 4096])
@pytest.mark.parametrize("dynamic_quantize", [True, False])
@pytest.mark.parametrize("torch_compile", [True, False])
@pytest.mark.parametrize("granularity", [ScalingGranularity.ROWWISE])
def test_quantize_fp8_rowwise(
    orig_dtype, dest_dtype, axis, B, M, N, dynamic_quantize, torch_compile, granularity
):
    # print("\n", orig_dtype, dest_dtype, axis, B, M, N)
    torch.manual_seed(42)

    x = torch.rand((B, M, N), device="cuda", dtype=orig_dtype)
    x_ref = x.detach().clone()
    x_fp8_ref, x_scale_ref, x_scale_inv_ref = quantize_fp8_ref(x_ref, dest_dtype, granularity, axis)

    scale = None
    if dynamic_quantize == False:
        scale = x_scale_ref.detach().clone()

    if torch_compile is True:
        torch._dynamo.reset()
        compiled_func = torch.compile(
            lambda t: quantize_fp8(t, dest_dtype, granularity=granularity, axis=axis, scale=scale),
            fullgraph=True,
            mode="max-autotune",
        )
        x_fp8, x_scale_inv = compiled_func(x)
    else:
        x_fp8, x_scale_inv = quantize_fp8(x, dest_dtype, granularity=granularity, axis=axis, scale=scale)

    torch.testing.assert_close(x_scale_inv_ref, x_scale_inv, **get_tolerances(torch.float32))
    torch.testing.assert_close(
        x_fp8_ref.to(torch.float32) * x_scale_inv_ref,
        x_fp8.to(torch.float32) * x_scale_inv,
        **get_tolerances(dest_dtype)
    )


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("M", [32, 64, 256, 1024])
@pytest.mark.parametrize("N", [32, 64, 256, 1024])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("padding_align_size", [None, 128])
@pytest.mark.parametrize("granularity", [ScalingGranularity.MX_BLOCKWISE])
@pytest.mark.parametrize("use_2d_block", [True, False])
def test_quantize_mxfp8(orig_dtype, dest_dtype, B, M, N, axis, padding_align_size, granularity, use_2d_block):
    def padding_size(n: int, padding_align_size: int) -> int:
        return (n + padding_align_size - 1) // padding_align_size * padding_align_size - n

    MX_BLOCK_SIZE = 32
    torch.manual_seed(42)

    x = torch.randn((B, M, N), device="cuda", dtype=orig_dtype)
    x.detach().clone()

    row_length = x.size(-1)
    x_2d = x.view(-1, row_length)
    if padding_align_size is not None:
        if axis == 0:
            x_2d_ref = torch.cat(
                [
                    x_2d,
                    torch.zeros(
                        padding_size(x_2d.size(0), padding_align_size),
                        x_2d.size(1),
                        device=x.device,
                        dtype=orig_dtype,
                    ),
                ],
                dim=axis,
            )
        else:
            x_2d_ref = torch.cat(
                [
                    x_2d,
                    torch.zeros(
                        x_2d.size(0),
                        padding_size(x_2d.size(1), padding_align_size),
                        device=x.device,
                        dtype=orig_dtype,
                    ),
                ],
                dim=axis,
            )
    else:
        x_2d_ref = x_2d

    scaling_recipe = MXScalingRecipe(
        use_2d_block=use_2d_block,
    )

    x_fp8, x_scale_inv = quantize_fp8(
        x_2d,
        dest_dtype,
        granularity=granularity,
        axis=axis,
        block_size=MX_BLOCK_SIZE,
        padding_align_size=padding_align_size,
        scaling_recipe=scaling_recipe,
    )

    # check quantize and dequantize precision
    out = dequantize_fp8(
        x_fp8,
        orig_dtype,
        granularity=granularity,
        block_size=MX_BLOCK_SIZE,
        axis=axis,
        scale_inv=x_scale_inv,
        scaling_recipe=scaling_recipe,
    )

    torch.testing.assert_close(x_2d_ref, out, **get_tolerances(dest_dtype))


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "dest_dtype",
    [
        turbo.float4_e2m1fn_x2,
    ],
)
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("M", [32, 64, 256, 1024])
@pytest.mark.parametrize("N", [32, 64, 256, 1024])
@pytest.mark.parametrize("axis", [1])
@pytest.mark.parametrize("granularity", [ScalingGranularity.MX_BLOCKWISE])
@pytest.mark.parametrize("use_sr", [True, False])
@pytest.mark.parametrize("use_2d_block", [True, False])
def test_quantize_mxfp4(orig_dtype, dest_dtype, B, M, N, axis, granularity, use_sr, use_2d_block):
    # Skip unit test on gfx942.
    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    # NOTE: Fix seed and offset for reproducibility
    PHILOX_SEED = 42
    PHILOX_OFFSET = 0

    scaling_recipe = MXScalingRecipe(
        use_2d_block=use_2d_block,
        use_sr=use_sr,
        philox_seed=PHILOX_SEED,
        philox_offset=PHILOX_OFFSET,
    )

    MX_BLOCK_SIZE = 32
    torch.manual_seed(42)

    x = torch.randn((B, M, N), device="cuda", dtype=orig_dtype)
    x.detach().clone()

    row_length = x.size(-1)
    x_2d = x.view(-1, row_length)

    x_fp4, x_scale_inv = quantize_fp4(
        x_2d,
        dest_dtype,
        granularity=granularity,
        axis=axis,
        block_size=MX_BLOCK_SIZE,
        scaling_recipe=scaling_recipe,
    )

    # check quantize and dequantize precision
    out = dequantize_fp4(
        x_fp4,
        orig_dtype,
        granularity=granularity,
        block_size=MX_BLOCK_SIZE,
        axis=axis,
        scale_inv=x_scale_inv,
        scaling_recipe=scaling_recipe,
    )

    torch.testing.assert_close(x_2d, out, **get_tolerances(dest_dtype))
