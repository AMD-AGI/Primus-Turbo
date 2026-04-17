###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import pytest
import torch

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.low_precision import (
    MXFP4_BLOCK_SIZE,
    MXFP8_BLOCK_SIZE,
    ScalingGranularity,
    ScalingRecipe,
    check_mxfp4_support,
    check_mxfp8_support,
)
from primus_turbo.pytorch.ops import dequantize_fp8, quantize_fp4, quantize_fp8
from primus_turbo.pytorch.ops.quantization import (
    dequantize_fp4,
    quantize_fp4_with_trans,
    quantize_fp8_with_trans,
)
from tests.pytorch.ref.quantization_ref import dequantize_fp8_ref, quantize_fp8_ref
from tests.pytorch.test_utils import get_tolerances


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("numel", [6 * 1 * 7168 * 8192])
@pytest.mark.parametrize("torch_compile", [True, False])
@pytest.mark.parametrize("granularity", [ScalingGranularity.TENSORWISE])
def test_quantize_fp8_tensorwise(orig_dtype, dest_dtype, numel, torch_compile, granularity):
    torch.manual_seed(42)

    x = torch.rand(numel, device="cuda", dtype=orig_dtype)
    x_ref = x.detach().clone()
    x_fp8_ref, x_scale_ref, x_scale_inv_ref = quantize_fp8_ref(x_ref, dest_dtype, granularity)

    if torch_compile is True:
        torch._dynamo.reset()
        compiled_func = torch.compile(
            lambda t: quantize_fp8(t, dest_dtype, granularity=granularity),
            fullgraph=True,
            mode="max-autotune",
        )
        x_fp8, x_scale_inv = compiled_func(x)
    else:
        x_fp8, x_scale_inv = quantize_fp8(x, dest_dtype, granularity=granularity)

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
@pytest.mark.parametrize("torch_compile", [True, False])
@pytest.mark.parametrize("granularity", [ScalingGranularity.ROWWISE])
def test_quantize_fp8_rowwise(orig_dtype, dest_dtype, axis, B, M, N, torch_compile, granularity):
    # print("\n", orig_dtype, dest_dtype, axis, B, M, N)
    torch.manual_seed(42)

    x = torch.rand((B, M, N), device="cuda", dtype=orig_dtype)
    x_ref = x.detach().clone()
    x_fp8_ref, x_scale_ref, x_scale_inv_ref = quantize_fp8_ref(x_ref, dest_dtype, granularity, axis)

    if torch_compile is True:
        torch._dynamo.reset()
        compiled_func = torch.compile(
            lambda t: quantize_fp8(t, dest_dtype, granularity=granularity, axis=axis),
            fullgraph=True,
            mode="max-autotune",
        )
        x_fp8, x_scale_inv = compiled_func(x)
    else:
        x_fp8, x_scale_inv = quantize_fp8(x, dest_dtype, granularity=granularity, axis=axis)

    torch.testing.assert_close(x_scale_inv_ref, x_scale_inv, **get_tolerances(torch.float32))
    torch.testing.assert_close(
        x_fp8_ref.to(torch.float32) * x_scale_inv_ref,
        x_fp8.to(torch.float32) * x_scale_inv,
        **get_tolerances(dest_dtype)
    )


def padding_size(n: int, padding_align_size: int) -> int:
    return (n + padding_align_size - 1) // padding_align_size * padding_align_size - n


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("M", [32, 64, 256, 1024])
@pytest.mark.parametrize("N", [32, 64, 256, 1024])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("granularity", [ScalingGranularity.MX_BLOCKWISE])
@pytest.mark.parametrize("use_2d_block", [True, False])
def test_quantize_mxfp8(orig_dtype, dest_dtype, B, M, N, axis, granularity, use_2d_block):
    padding_align_size = 128

    # Skip unit test on gfx942.
    mxfp8_supported, reason = check_mxfp8_support()
    if not mxfp8_supported:
        pytest.skip(reason)

    MX_BLOCK_SIZE = 32
    torch.manual_seed(42)

    x = torch.randn((B, M, N), device="cuda", dtype=orig_dtype)

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

    scaling_recipe = ScalingRecipe(
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


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("M", [32, 64, 256, 1024])
@pytest.mark.parametrize("N", [32, 64, 256, 1024])
@pytest.mark.parametrize("granularity", [ScalingGranularity.MX_BLOCKWISE])
@pytest.mark.parametrize("use_2d_block", [True, False])
def test_quantize_mxfp8_with_trans(orig_dtype, dest_dtype, B, M, N, granularity, use_2d_block):
    padding_align_size = 128

    mxfp8_supported, reason = check_mxfp8_support()
    if not mxfp8_supported:
        pytest.skip(reason)

    scaling_recipe = ScalingRecipe(
        use_2d_block=use_2d_block,
    )

    MX_BLOCK_SIZE = 32
    torch.manual_seed(42)

    x = torch.ones((B, M, N), device="cuda", dtype=orig_dtype) * 6

    row_length = x.size(-1)
    x_2d = x.view(-1, row_length)
    M_actual = x_2d.size(0)
    N_actual = x_2d.size(1)

    x_fp8_rowwise, x_scale_inv_rowwise, x_fp8_colwise, x_scale_inv_colwise = quantize_fp8_with_trans(
        x_2d,
        dest_dtype,
        granularity=granularity,
        block_size=MX_BLOCK_SIZE,
        scaling_recipe=scaling_recipe,
        scaling_recipe_for_trans=scaling_recipe,
    )

    # Test 2: Dequantize and compare with zero-padded reference.
    # Rowwise dequantize: output shape [M, N_pad]
    x_2d_ref_rowwise = torch.cat(
        [
            x_2d,
            torch.zeros(
                M_actual,
                padding_size(N_actual, padding_align_size),
                device=x_2d.device,
                dtype=orig_dtype,
            ),
        ],
        dim=1,
    )

    out_rowwise = dequantize_fp8(
        x_fp8_rowwise,
        orig_dtype,
        granularity=granularity,
        block_size=MX_BLOCK_SIZE,
        axis=1,
        scale_inv=x_scale_inv_rowwise,
        scaling_recipe=scaling_recipe,
    )
    torch.testing.assert_close(x_2d_ref_rowwise, out_rowwise, **get_tolerances(dest_dtype))

    # Colwise dequantize: output shape [M_pad, N]
    x_2d_ref_colwise = torch.cat(
        [
            x_2d,
            torch.zeros(
                padding_size(M_actual, padding_align_size),
                N_actual,
                device=x_2d.device,
                dtype=orig_dtype,
            ),
        ],
        dim=0,
    )

    out_colwise = dequantize_fp8(
        x_fp8_colwise,
        orig_dtype,
        granularity=granularity,
        block_size=MX_BLOCK_SIZE,
        axis=0,
        scale_inv=x_scale_inv_colwise,
        scaling_recipe=scaling_recipe,
    )
    torch.testing.assert_close(x_2d_ref_colwise, out_colwise, **get_tolerances(dest_dtype))


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "dest_dtype",
    [
        turbo.float8_e4m3,
        turbo.float8_e5m2,
    ],
)
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("M", [32, 64, 256, 1024])
@pytest.mark.parametrize("N", [32, 64, 256, 1024])
@pytest.mark.parametrize("granularity", [ScalingGranularity.MX_BLOCKWISE])
@pytest.mark.parametrize("use_2d_block", [True, False])
def test_quantize_mxfp8_shuffle(orig_dtype, dest_dtype, B, M, N, granularity, use_2d_block):
    # Skip unit test on gfx942.
    mxfp8_supported, reason = check_mxfp8_support()
    if not mxfp8_supported:
        pytest.skip(reason)

    torch.manual_seed(42)

    x = torch.randn((B, M, N), device="cuda", dtype=orig_dtype)

    row_length = x.size(-1)
    x_2d = x.view(-1, row_length)

    scaling_recipe = ScalingRecipe(
        use_2d_block=use_2d_block,
    )
    _, rowwise_scale, _, colwise_scale = quantize_fp8_with_trans(
        x_2d,
        dest_dtype,
        granularity=granularity,
        block_size=MXFP8_BLOCK_SIZE,
        scaling_recipe=scaling_recipe,
        scaling_recipe_for_trans=scaling_recipe,
    )

    rowwise_scale_shuffle = torch.ops.primus_turbo_cpp_extension.shuffle_scale(rowwise_scale, [16, 16])
    colwise_scale_shuffle = torch.ops.primus_turbo_cpp_extension.shuffle_scale(colwise_scale, [16, 16])

    scaling_recipe_with_shuffle = ScalingRecipe(
        use_2d_block=use_2d_block,
        shuffle_scale=True,
        shuffle_out=False,
    )
    _, rowwise_scale_shuffle_ref, _, colwise_scale_shuffle_ref = quantize_fp8_with_trans(
        x_2d,
        dest_dtype,
        block_size=MXFP8_BLOCK_SIZE,
        granularity=granularity,
        scaling_recipe=scaling_recipe_with_shuffle,
        scaling_recipe_for_trans=scaling_recipe_with_shuffle,
    )

    # TODO(ruibin): Add shuffle weight for MXFP8.
    torch.testing.assert_close(
        rowwise_scale_shuffle.view(torch.uint8), rowwise_scale_shuffle_ref.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(
        colwise_scale_shuffle.view(torch.uint8), colwise_scale_shuffle_ref.view(torch.uint8), atol=0, rtol=0
    )


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "dest_dtype",
    [
        turbo.float4_e2m1fn_x2,
    ],
)
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("M", [32, 64, 256, 1024])
@pytest.mark.parametrize("N", [32, 64, 256, 1024])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("granularity", [ScalingGranularity.MX_BLOCKWISE])
@pytest.mark.parametrize("use_2d_block", [True, False])
def test_quantize_mxfp4(orig_dtype, dest_dtype, B, M, N, axis, granularity, use_2d_block):
    # Hardcode padding align size to 128.
    padding_align_size = 128

    # Skip unit test on gfx942.
    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    scaling_recipe = ScalingRecipe(
        use_2d_block=use_2d_block,
    )

    MX_BLOCK_SIZE = 32
    torch.manual_seed(42)

    # x = torch.randn((B, M, N), device="cuda", dtype=orig_dtype)
    x = torch.ones((B, M, N), device="cuda", dtype=orig_dtype) * 6

    row_length = x.size(-1)
    x_2d = x.view(-1, row_length)
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

    torch.testing.assert_close(x_2d_ref, out, **get_tolerances(dest_dtype))


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "dest_dtype",
    [
        turbo.float4_e2m1fn_x2,
    ],
)
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("M", [32, 64, 256, 1024])
@pytest.mark.parametrize("N", [32, 64, 256, 1024])
@pytest.mark.parametrize("granularity", [ScalingGranularity.MX_BLOCKWISE])
@pytest.mark.parametrize("use_2d_block", [True, False])
def test_quantize_mxfp4_with_trans(orig_dtype, dest_dtype, B, M, N, granularity, use_2d_block):
    padding_align_size = 128

    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    scaling_recipe = ScalingRecipe(
        use_2d_block=use_2d_block,
    )

    MX_BLOCK_SIZE = 32
    torch.manual_seed(42)

    x = torch.ones((B, M, N), device="cuda", dtype=orig_dtype) * 6

    row_length = x.size(-1)
    x_2d = x.view(-1, row_length)
    M_actual = x_2d.size(0)
    N_actual = x_2d.size(1)

    x_fp4, x_scale_inv, x_t_fp4, x_t_scale_inv = quantize_fp4_with_trans(
        x_2d,
        dest_dtype,
        granularity=granularity,
        block_size=MX_BLOCK_SIZE,
        scaling_recipe=scaling_recipe,
        scaling_recipe_for_trans=scaling_recipe,
    )

    # Test 2: Dequantize and compare with zero-padded reference.
    # Rowwise dequantize: output shape [M, N_pad]
    x_2d_ref_rowwise = torch.cat(
        [
            x_2d,
            torch.zeros(
                M_actual,
                padding_size(N_actual, padding_align_size),
                device=x_2d.device,
                dtype=orig_dtype,
            ),
        ],
        dim=1,
    )

    out_rowwise = dequantize_fp4(
        x_fp4,
        orig_dtype,
        granularity=granularity,
        block_size=MX_BLOCK_SIZE,
        axis=1,
        scale_inv=x_scale_inv,
        scaling_recipe=scaling_recipe,
    )
    torch.testing.assert_close(x_2d_ref_rowwise, out_rowwise, **get_tolerances(dest_dtype))

    # Colwise dequantize: output shape [M_pad, N]
    x_2d_ref_colwise = torch.cat(
        [
            x_2d,
            torch.zeros(
                padding_size(M_actual, padding_align_size),
                N_actual,
                device=x_2d.device,
                dtype=orig_dtype,
            ),
        ],
        dim=0,
    )

    out_colwise = dequantize_fp4(
        x_t_fp4,
        orig_dtype,
        granularity=granularity,
        block_size=MX_BLOCK_SIZE,
        axis=0,
        scale_inv=x_t_scale_inv,
        scaling_recipe=scaling_recipe,
    )
    torch.testing.assert_close(x_2d_ref_colwise, out_colwise, **get_tolerances(dest_dtype))


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "dest_dtype",
    [
        turbo.float4_e2m1fn_x2,
    ],
)
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("M", [32, 64, 256, 1024])
@pytest.mark.parametrize("N", [32, 64, 256, 1024])
@pytest.mark.parametrize("granularity", [ScalingGranularity.MX_BLOCKWISE])
@pytest.mark.parametrize("use_2d_block", [True, False])
def test_quantize_mxfp4_shuffle(orig_dtype, dest_dtype, B, M, N, granularity, use_2d_block):
    # Skip unit test on gfx942.
    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    torch.manual_seed(42)

    x = torch.randn((B, M, N), device="cuda", dtype=orig_dtype)

    row_length = x.size(-1)
    x_2d = x.view(-1, row_length)

    scaling_recipe = ScalingRecipe(
        use_2d_block=use_2d_block,
    )
    rowwise_out, rowwise_scale, colwise_out, colwise_scale = quantize_fp4_with_trans(
        x_2d,
        dest_dtype,
        granularity=granularity,
        block_size=MXFP4_BLOCK_SIZE,
        scaling_recipe=scaling_recipe,
        scaling_recipe_for_trans=scaling_recipe,
    )

    rowwise_out_shuffle = torch.ops.primus_turbo_cpp_extension.shuffle_weight(rowwise_out, [16, 16])
    rowwise_scale_shuffle = torch.ops.primus_turbo_cpp_extension.shuffle_scale(rowwise_scale, [16, 16])
    colwise_out_shuffle = torch.ops.primus_turbo_cpp_extension.shuffle_weight(colwise_out, [16, 16])
    colwise_scale_shuffle = torch.ops.primus_turbo_cpp_extension.shuffle_scale(colwise_scale, [16, 16])

    scaling_recipe_with_shuffle = ScalingRecipe(
        use_2d_block=use_2d_block,
        shuffle_scale=True,
        shuffle_out=True,
    )
    rowwise_out_shuffle_ref, rowwise_scale_shuffle_ref, colwise_out_shuffle_ref, colwise_scale_shuffle_ref = (
        quantize_fp4_with_trans(
            x_2d,
            dest_dtype,
            granularity=granularity,
            block_size=MXFP4_BLOCK_SIZE,
            scaling_recipe=scaling_recipe_with_shuffle,
            scaling_recipe_for_trans=scaling_recipe_with_shuffle,
        )
    )

    torch.testing.assert_close(
        rowwise_out_shuffle.view(torch.uint8), rowwise_out_shuffle_ref.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(
        rowwise_scale_shuffle.view(torch.uint8), rowwise_scale_shuffle_ref.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(
        colwise_out_shuffle.view(torch.uint8), colwise_out_shuffle_ref.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(
        colwise_scale_shuffle.view(torch.uint8), colwise_scale_shuffle_ref.view(torch.uint8), atol=0, rtol=0
    )
