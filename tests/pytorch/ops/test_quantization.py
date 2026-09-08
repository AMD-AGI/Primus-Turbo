###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from unittest.mock import Mock

import pytest
import torch

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.low_precision import (
    DEFAULT_BLOCK_SIZE,
    MXFP4_BLOCK_SIZE,
    MXFP8_BLOCK_SIZE,
    Float4QuantConfig,
    ScalingGranularity,
    ScalingRecipe,
    check_mxfp4_support,
    check_mxfp8_support,
)
from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor, create_quantized_weight
from primus_turbo.pytorch.ops import dequantize_fp8, quantize_fp4, quantize_fp8
from primus_turbo.pytorch.ops.quantization import (
    dequantize_fp4,
    quantize_fp4_with_trans,
    quantize_fp8_with_trans,
)
from tests.pytorch.ref.quantization_ref import dequantize_fp8_ref, quantize_fp8_ref
from tests.pytorch.test_utils import get_tolerances


def _load_mxfp4_flydsl_kernel(require_gfx950=False):
    """Import FlyDSL lazily, optionally requiring the supported gfx950 path."""
    if require_gfx950:
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        mxfp4_supported, reason = check_mxfp4_support()
        if not mxfp4_supported:
            pytest.skip(reason)

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if (props.major, props.minor) != (9, 5):
            pytest.skip("FlyDSL MXFP4 quantization requires gfx950")
    else:
        pytest.importorskip("flydsl")

    # Once gfx950 is supported, an internal import failure must fail the test.
    from primus_turbo.flydsl.quantization import mxfp4_quant_kernel

    return mxfp4_quant_kernel


def _hip_quantize_mxfp4_dual(x, row_recipe, col_recipe, scale_rounding_mode=0):
    """Call the HIP oracle directly, bypassing Python backend dispatch."""
    return torch.ops.primus_turbo_cpp_extension.quantize_mxfp4_dual(
        x,
        turbo.float4_e2m1fn_x2,
        128,
        row_recipe.use_2d_block,
        row_recipe.use_sr,
        row_recipe.use_rht,
        col_recipe.use_2d_block,
        col_recipe.use_sr,
        col_recipe.use_rht,
        False,
        False,
        False,
        False,
        scale_rounding_mode,
    )


def _assert_byte_exact(actual, expected):
    assert len(actual) == len(expected)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor.view(torch.uint8),
            expected_tensor.view(torch.uint8),
            rtol=0,
            atol=0,
        )


def test_mxfp4_scale_rounding_mode_contract():
    """Validate the config contract and its mapping to the three supported biases."""
    kernel = _load_mxfp4_flydsl_kernel()
    assert Float4QuantConfig().scale_rounding_mode == 0
    for mode, expected in ((0, 1 << 21), (1, 1 << 22), (2, 3 << 19)):
        config = Float4QuantConfig(scale_rounding_mode=mode)
        assert kernel._mxfp4_scale_rounding_bias(config.scale_rounding_mode) == expected

    for mode in (-1, 3, "1"):
        with pytest.raises(AssertionError, match="must be 0, 1, or 2"):
            Float4QuantConfig(scale_rounding_mode=mode)
        with pytest.raises(ValueError, match="must be 0, 1, or 2"):
            kernel._mxfp4_scale_rounding_bias(mode)


@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize(
    "orig_dtype,numel,torch_compile,check_unaligned_fallback",
    [
        (orig_dtype, 6 * 1 * 7168 * 8192, torch_compile, False)
        for orig_dtype in [torch.bfloat16, torch.float16, torch.float32]
        for torch_compile in [True, False]
    ]
    + [
        (orig_dtype, numel, False, True)
        for orig_dtype in [torch.bfloat16, torch.float16]
        for numel in [15, 16, 17, 2880, 2881]
    ],
)
@pytest.mark.parametrize("granularity", [ScalingGranularity.TENSORWISE])
def test_quantize_fp8_tensorwise(
    orig_dtype,
    dest_dtype,
    numel,
    torch_compile,
    check_unaligned_fallback,
    granularity,
):
    torch.manual_seed(42)

    x = torch.rand(numel, device="cuda", dtype=orig_dtype)
    if check_unaligned_fallback:
        storage = torch.empty(numel + 1, device="cuda", dtype=orig_dtype)
        x_unaligned = storage[1:]
        x_unaligned.copy_(x)

        assert x.data_ptr() % 32 == 0
        assert x_unaligned.is_contiguous()
        assert x_unaligned.data_ptr() % 32 != 0

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
        **get_tolerances(dest_dtype),
    )

    # DeQuantize
    x_dq = dequantize_fp8(x_fp8, orig_dtype, granularity, scale_inv=x_scale_inv)
    x_dq_ref = dequantize_fp8_ref(x_fp8_ref, orig_dtype, granularity, scale_inv=x_scale_inv_ref)
    torch.testing.assert_close(x_dq, x_dq_ref, **get_tolerances(dest_dtype))

    if check_unaligned_fallback:
        x_fp8_fallback, x_scale_inv_fallback = quantize_fp8(
            x_unaligned,
            dest_dtype,
            granularity=granularity,
        )

        assert x_fp8.data_ptr() % 16 == 0
        assert torch.equal(x_fp8.view(torch.uint8), x_fp8_fallback.view(torch.uint8))
        torch.testing.assert_close(x_scale_inv, x_scale_inv_fallback, rtol=0, atol=0)


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("granularity", [ScalingGranularity.TENSORWISE])
@pytest.mark.parametrize(
    "shape,spike_pos",
    [
        ((1, 100), -1),
        ((1, 8193), -1),
        ((512, 3072), 8300),
        ((512, 3072), -1),
        ((1024, 4096), -1),
    ],
)
def test_quantize_fp8_tensorwise_amax_correctness(orig_dtype, dest_dtype, granularity, shape, spike_pos):
    """Regression test for partial-tile amax reduction bug in reduce_row_kernel."""
    x = torch.ones(shape, device="cuda", dtype=orig_dtype) * 0.5
    x.view(-1)[spike_pos] = 100.0
    x_ref = x.detach().clone()

    x_fp8_ref, x_scale_ref, x_scale_inv_ref = quantize_fp8_ref(x_ref, dest_dtype, granularity)
    x_fp8, x_scale_inv = quantize_fp8(x, dest_dtype, granularity=granularity)

    torch.testing.assert_close(x_scale_inv_ref, x_scale_inv, **get_tolerances(torch.float32))
    torch.testing.assert_close(
        x_fp8_ref.to(torch.float32) * x_scale_inv_ref,
        x_fp8.to(torch.float32) * x_scale_inv,
        **get_tolerances(dest_dtype),
    )

    # Opt-in K/N pad computes amax over the real data, so it keeps the tight scale, leaves the
    # real block byte-identical, and zero-fills the widened tail (aligned dims make it a no-op).
    N_dim, K_dim = shape
    x_fp8_pad, x_scale_inv_pad = quantize_fp8(
        x, dest_dtype, granularity=granularity, pad_align_last=128, pad_align_penultimate=128
    )
    Np = ((N_dim + 127) // 128) * 128
    Kp = ((K_dim + 127) // 128) * 128
    assert tuple(x_fp8_pad.shape) == (Np, Kp)
    torch.testing.assert_close(x_scale_inv_pad, x_scale_inv, rtol=0, atol=0)
    torch.testing.assert_close(
        x_fp8_pad[:N_dim, :K_dim].to(torch.float32), x_fp8.to(torch.float32), rtol=0, atol=0
    )
    assert (x_fp8_pad[N_dim:, :].to(torch.float32) != 0).sum() == 0
    assert (x_fp8_pad[:, K_dim:].to(torch.float32) != 0).sum() == 0


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
        **get_tolerances(dest_dtype),
    )

    x_dq = dequantize_fp8(x_fp8, orig_dtype, granularity, axis=axis, scale_inv=x_scale_inv)
    x_dq_ref = dequantize_fp8_ref(x_fp8_ref, orig_dtype, granularity, axis=axis, scale_inv=x_scale_inv_ref)
    torch.testing.assert_close(x_dq, x_dq_ref, **get_tolerances(dest_dtype))


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("M", [128, 256, 320])
@pytest.mark.parametrize("N", [128, 256, 320])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("granularity", [ScalingGranularity.BLOCKWISE])
def test_quantize_fp8_blockwise(orig_dtype, dest_dtype, M, N, axis, granularity):
    """1D-block (activation) blockwise FP8 quant/dequant round-trip."""
    torch.manual_seed(42)

    x = torch.randn((M, N), device="cuda", dtype=orig_dtype)

    x_fp8, x_scale_inv = quantize_fp8(
        x,
        dest_dtype,
        granularity=granularity,
        axis=axis,
        block_size=DEFAULT_BLOCK_SIZE,
    )

    out = dequantize_fp8(
        x_fp8,
        orig_dtype,
        granularity=granularity,
        block_size=DEFAULT_BLOCK_SIZE,
        axis=axis,
        scale_inv=x_scale_inv,
    )

    torch.testing.assert_close(x, out, **get_tolerances(dest_dtype))


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("M", [128, 256, 320])
@pytest.mark.parametrize("N", [128, 256, 320])
@pytest.mark.parametrize("granularity", [ScalingGranularity.BLOCKWISE])
def test_quantize_fp8_blockwise_with_trans(orig_dtype, dest_dtype, M, N, granularity):
    """Fused row + col blockwise FP8 quant, dequantized in both directions."""
    torch.manual_seed(42)

    x = torch.randn((M, N), device="cuda", dtype=orig_dtype)

    x_fp8_row, x_scale_inv_row, x_fp8_col, x_scale_inv_col = quantize_fp8_with_trans(
        x,
        dest_dtype,
        granularity=granularity,
        block_size=DEFAULT_BLOCK_SIZE,
    )

    # Rowwise (axis == 1) dequantize.
    out_row = dequantize_fp8(
        x_fp8_row,
        orig_dtype,
        granularity=granularity,
        block_size=DEFAULT_BLOCK_SIZE,
        axis=1,
        scale_inv=x_scale_inv_row,
    )
    torch.testing.assert_close(x, out_row, **get_tolerances(dest_dtype))

    # Colwise (axis == 0) dequantize.
    out_col = dequantize_fp8(
        x_fp8_col,
        orig_dtype,
        granularity=granularity,
        block_size=DEFAULT_BLOCK_SIZE,
        axis=0,
        scale_inv=x_scale_inv_col,
    )
    torch.testing.assert_close(x, out_col, **get_tolerances(dest_dtype))


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("M", [128, 256, 320])
@pytest.mark.parametrize("N", [128, 256, 320])
@pytest.mark.parametrize("granularity", [ScalingGranularity.BLOCKWISE])
def test_quantize_fp8_blockwise_for_weight(orig_dtype, dest_dtype, batched, B, M, N, granularity):
    """2D-block (weight) blockwise FP8 quant/dequant round-trip for 2D/3D weights."""
    torch.manual_seed(42)

    if batched:
        x = torch.randn((B, M, N), device="cuda", dtype=orig_dtype)
    else:
        x = torch.randn((M, N), device="cuda", dtype=orig_dtype)

    # use_2d_block ignores axis; scales along both dims.
    scaling_recipe = ScalingRecipe(use_2d_block=True)

    x_fp8, x_scale_inv = quantize_fp8(
        x,
        dest_dtype,
        granularity=granularity,
        block_size=DEFAULT_BLOCK_SIZE,
        scaling_recipe=scaling_recipe,
    )

    out = dequantize_fp8(
        x_fp8,
        orig_dtype,
        granularity=granularity,
        block_size=DEFAULT_BLOCK_SIZE,
        scale_inv=x_scale_inv,
        scaling_recipe=scaling_recipe,
    )

    torch.testing.assert_close(x, out, **get_tolerances(dest_dtype))


def padding_size(n: int, padding_align_size: int) -> int:
    return (n + padding_align_size - 1) // padding_align_size * padding_align_size - n


def mxfp8_padded_ref(x: torch.Tensor, axis: int, padding_align_size: int, dtype: torch.dtype) -> torch.Tensor:
    """Build zero-padded reference matching MXFP8 quantize/dequantize padding."""
    if x.dim() == 2:
        if axis == 0:
            # Colwise: dequant output [M_pad, N].
            pad_amt = padding_size(x.size(0), padding_align_size)
            zeros = torch.zeros(pad_amt, x.size(1), device=x.device, dtype=dtype)
            return torch.cat([x, zeros], dim=0)
        # Rowwise: dequant output [M, N_pad].
        pad_amt = padding_size(x.size(1), padding_align_size)
        zeros = torch.zeros(x.size(0), pad_amt, device=x.device, dtype=dtype)
        return torch.cat([x, zeros], dim=1)

    # 3D batched [B, M, N]
    if axis == 1:
        # Colwise: quant/dequant layout [B, N, M_pad].
        x_bn = x.transpose(1, 2).contiguous()
        pad_amt = padding_size(x_bn.size(2), padding_align_size)
        zeros = torch.zeros(x_bn.size(0), x_bn.size(1), pad_amt, device=x.device, dtype=dtype)
        return torch.cat([x_bn, zeros], dim=2)
    # Rowwise (axis == 2): dequant output [B, M, N_pad].
    pad_amt = padding_size(x.size(2), padding_align_size)
    zeros = torch.zeros(x.size(0), x.size(1), pad_amt, device=x.device, dtype=dtype)
    return torch.cat([x, zeros], dim=2)


def mxfp4_padded_ref(x: torch.Tensor, axis: int, padding_align_size: int, dtype: torch.dtype) -> torch.Tensor:
    """Build zero-padded reference matching MXFP4 quantize/dequantize padding."""
    if x.dim() == 2:
        if axis == 0:
            # Colwise: dequant output [M_pad, N].
            pad_amt = padding_size(x.size(0), padding_align_size)
            zeros = torch.zeros(pad_amt, x.size(1), device=x.device, dtype=dtype)
            return torch.cat([x, zeros], dim=0)
        # Rowwise: dequant output [M, N_pad].
        pad_amt = padding_size(x.size(1), padding_align_size)
        zeros = torch.zeros(x.size(0), pad_amt, device=x.device, dtype=dtype)
        return torch.cat([x, zeros], dim=1)

    # 3D batched [B, M, N]
    if axis == 1:
        # Colwise: quant/dequant layout [B, N, M_pad].
        x_bn = x.transpose(1, 2).contiguous()
        pad_amt = padding_size(x_bn.size(2), padding_align_size)
        zeros = torch.zeros(x_bn.size(0), x_bn.size(1), pad_amt, device=x.device, dtype=dtype)
        return torch.cat([x_bn, zeros], dim=2)
    # Rowwise (axis == 2): dequant output [B, M, N_pad].
    pad_amt = padding_size(x.size(2), padding_align_size)
    zeros = torch.zeros(x.size(0), x.size(1), pad_amt, device=x.device, dtype=dtype)
    return torch.cat([x, zeros], dim=2)


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("M", [32, 64, 256, 1024])
@pytest.mark.parametrize("N", [32, 64, 256, 1024])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("granularity", [ScalingGranularity.MX_BLOCKWISE])
@pytest.mark.parametrize("use_2d_block", [True, False])
def test_quantize_mxfp8(orig_dtype, dest_dtype, batched, B, M, N, axis, granularity, use_2d_block):
    padding_align_size = 128

    # Skip unit test on gfx942.
    mxfp8_supported, reason = check_mxfp8_support()
    if not mxfp8_supported:
        pytest.skip(reason)

    MX_BLOCK_SIZE = 32
    torch.manual_seed(42)

    if batched:
        x = torch.randn((B, M, N), device="cuda", dtype=orig_dtype)
        # 3D MXFP8: axis 1 = colwise, axis 2 = rowwise (inner-K).
        quantize_axis = axis + 1
    else:
        x = torch.randn((M, N), device="cuda", dtype=orig_dtype)
        # 2D MXFP8: axis 0 = colwise, axis 1 = rowwise.
        quantize_axis = axis

    x_ref = mxfp8_padded_ref(x, quantize_axis, padding_align_size, orig_dtype)

    scaling_recipe = ScalingRecipe(
        use_2d_block=use_2d_block,
    )

    x_fp8, x_scale_inv = quantize_fp8(
        x,
        dest_dtype,
        granularity=granularity,
        axis=quantize_axis,
        block_size=MX_BLOCK_SIZE,
        scaling_recipe=scaling_recipe,
    )

    # check quantize and dequantize precision
    out = dequantize_fp8(
        x_fp8,
        orig_dtype,
        granularity=granularity,
        block_size=MX_BLOCK_SIZE,
        axis=quantize_axis,
        scale_inv=x_scale_inv,
        scaling_recipe=scaling_recipe,
    )

    torch.testing.assert_close(x_ref, out, **get_tolerances(dest_dtype))


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

    x = torch.randn((B, M, N), device="cuda", dtype=orig_dtype)

    row_length = x.size(-1)
    x_2d = x.view(-1, row_length)
    M_actual = x_2d.size(0)
    N_actual = x_2d.size(1)

    x_fp8_rowwise, x_scale_inv_rowwise, x_fp8_t, x_scale_inv_colwise = quantize_fp8_with_trans(
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
        x_fp8_t,
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
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("M", [32, 64, 256, 1024])
@pytest.mark.parametrize("N", [32, 64, 256, 1024])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("granularity", [ScalingGranularity.MX_BLOCKWISE])
@pytest.mark.parametrize("use_2d_block", [True, False])
def test_quantize_mxfp4(orig_dtype, dest_dtype, batched, B, M, N, axis, granularity, use_2d_block):
    # Hardcode padding align size to 128.
    padding_align_size = 128

    # Skip unit test on gfx942.
    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    MX_BLOCK_SIZE = 32
    torch.manual_seed(42)

    if batched:
        x = torch.randn((B, M, N), device="cuda", dtype=orig_dtype)
        # 3D MXFP4: axis 1 = colwise, axis 2 = rowwise (inner-K).
        quantize_axis = axis + 1
    else:
        x = torch.randn((M, N), device="cuda", dtype=orig_dtype)
        # 2D MXFP4: axis 0 = colwise, axis 1 = rowwise.
        quantize_axis = axis

    x_ref = mxfp4_padded_ref(x, quantize_axis, padding_align_size, orig_dtype)

    scaling_recipe = ScalingRecipe(
        use_2d_block=use_2d_block,
    )

    x_fp4, x_scale_inv = quantize_fp4(
        x,
        dest_dtype,
        granularity=granularity,
        axis=quantize_axis,
        block_size=MX_BLOCK_SIZE,
        scaling_recipe=scaling_recipe,
    )

    # check quantize and dequantize precision
    out = dequantize_fp4(
        x_fp4,
        orig_dtype,
        granularity=granularity,
        block_size=MX_BLOCK_SIZE,
        axis=quantize_axis,
        scale_inv=x_scale_inv,
        scaling_recipe=scaling_recipe,
    )

    torch.testing.assert_close(x_ref, out, **get_tolerances(dest_dtype))


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "dest_dtype",
    [
        turbo.float4_e2m1fn_x2,
    ],
)
@pytest.mark.parametrize("B", [None, 4], ids=["2d", "3d"])
@pytest.mark.parametrize("M", [32, 64, 192, 256])
@pytest.mark.parametrize("N", [32, 64, 192, 256])
@pytest.mark.parametrize("granularity", [ScalingGranularity.MX_BLOCKWISE])
@pytest.mark.parametrize("use_2d_block", [True, False])
def test_quantize_mxfp4_with_trans(orig_dtype, dest_dtype, B, M, N, granularity, use_2d_block):
    """Validate the 2D and real 3D batched dual-quant paths.

    The 64/192 shapes guard selective tail-zeroing in both output directions
    when the kernel allocates its output buffers with torch.empty.
    """
    padding_align_size = 128

    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    scaling_recipe = ScalingRecipe(
        use_2d_block=use_2d_block,
    )

    MX_BLOCK_SIZE = 32
    torch.manual_seed(42)

    shape = (M, N) if B is None else (B, M, N)
    x = torch.randn(shape, device="cuda", dtype=orig_dtype)

    x_fp4_rowwise, x_scale_inv_rowwise, x_fp4_t, x_scale_inv_colwise = quantize_fp4_with_trans(
        x,
        dest_dtype,
        granularity=granularity,
        block_size=MX_BLOCK_SIZE,
        scaling_recipe=scaling_recipe,
        scaling_recipe_for_trans=scaling_recipe,
    )

    if B is not None:
        rowwise_ref = torch.nn.functional.pad(x, (0, padding_size(N, padding_align_size)))
        colwise_ref = torch.nn.functional.pad(x.transpose(1, 2), (0, padding_size(M, padding_align_size)))
        rowwise_axis = colwise_axis = 2
    else:
        rowwise_ref = torch.nn.functional.pad(x, (0, padding_size(N, padding_align_size)))
        colwise_ref = torch.nn.functional.pad(x, (0, 0, 0, padding_size(M, padding_align_size)))
        rowwise_axis, colwise_axis = 1, 0

    out_rowwise = dequantize_fp4(
        x_fp4_rowwise,
        orig_dtype,
        granularity=granularity,
        block_size=MX_BLOCK_SIZE,
        axis=rowwise_axis,
        scale_inv=x_scale_inv_rowwise,
        scaling_recipe=scaling_recipe,
    )
    torch.testing.assert_close(rowwise_ref, out_rowwise, **get_tolerances(dest_dtype))

    out_colwise = dequantize_fp4(
        x_fp4_t,
        orig_dtype,
        granularity=granularity,
        block_size=MX_BLOCK_SIZE,
        axis=colwise_axis,
        scale_inv=x_scale_inv_colwise,
        scaling_recipe=scaling_recipe,
    )
    torch.testing.assert_close(colwise_ref, out_colwise, **get_tolerances(dest_dtype))


def test_mxfp4_scale_rounding_dense_runtime_switch(monkeypatch):
    """The public dense path uses one FlyDSL kernel for all byte-exact modes."""
    kernel = _load_mxfp4_flydsl_kernel(require_gfx950=True)
    recipe = ScalingRecipe()
    assert kernel.dual_eligible(384, 256, recipe, recipe)

    # Exact bf16 thresholds make mode 1 differ at 1.5 and mode 2 differ at 1.75.
    block_values = torch.tensor([1.5, 1.75] * 4, device="cuda", dtype=torch.bfloat16)
    x = block_values.repeat_interleave(MXFP4_BLOCK_SIZE).expand(384, -1).contiguous()

    traced_flydsl_dual_quant = Mock(wraps=kernel.flydsl_dual_quant)
    monkeypatch.setattr(kernel, "flydsl_dual_quant", traced_flydsl_dual_quant)

    results = {}
    for mode in (0, 1, 2):
        config = Float4QuantConfig(scale_rounding_mode=mode)
        fly = quantize_fp4_with_trans(
            x,
            turbo.float4_e2m1fn_x2,
            granularity=ScalingGranularity.MX_BLOCKWISE,
            block_size=MXFP4_BLOCK_SIZE,
            scaling_recipe=recipe,
            scaling_recipe_for_trans=recipe,
            scale_rounding_mode=config.scale_rounding_mode,
        )
        hip = _hip_quantize_mxfp4_dual(x, recipe, recipe, config.scale_rounding_mode)
        _assert_byte_exact(fly, hip)
        results[mode] = tuple(tensor.clone() for tensor in fly)

    assert traced_flydsl_dual_quant.call_count == 3
    assert not torch.equal(results[0][1].view(torch.uint8), results[1][1].view(torch.uint8))
    assert not torch.equal(results[0][1].view(torch.uint8), results[2][1].view(torch.uint8))

    # Exercise config propagation through QuantizedTensor metadata, which prevents
    # a cached weight from being reused under a different rounding mode.
    config = Float4QuantConfig(scale_rounding_mode=2)
    quantized_weight, _ = create_quantized_weight(x, turbo.float4_e2m1fn_x2, config)
    expected_weight = quantize_fp4(
        x,
        turbo.float4_e2m1fn_x2,
        granularity=config.granularity,
        block_size=config.block_size,
        axis=1,
        scaling_recipe=ScalingRecipe(use_2d_block=True),
        scale_rounding_mode=config.scale_rounding_mode,
    )
    _assert_byte_exact((quantized_weight.qdata, quantized_weight.scale_inv), expected_weight)
    assert quantized_weight.scale_rounding_mode == config.scale_rounding_mode

    keys, metadata = quantized_weight.__tensor_flatten__()
    restored = QuantizedTensor.__tensor_unflatten__(
        {key: getattr(quantized_weight, key) for key in keys},
        metadata,
        quantized_weight.shape,
        quantized_weight.stride(),
    )
    assert restored.scale_rounding_mode == config.scale_rounding_mode


def test_mxfp4_scale_rounding_dense_special_recipes_match_hip():
    """Representative RHT and 2D-scale variants propagate the selected mode."""
    kernel = _load_mxfp4_flydsl_kernel(require_gfx950=True)
    torch.manual_seed(42)
    x = torch.randn((384, 256), device="cuda", dtype=torch.bfloat16)

    for mode, row_2d, col_2d, row_rht, col_rht in (
        (1, False, False, False, True),
        (2, True, True, False, False),
    ):
        row_recipe = ScalingRecipe(use_2d_block=row_2d, use_rht=row_rht)
        col_recipe = ScalingRecipe(use_2d_block=col_2d, use_rht=col_rht)
        assert kernel.dual_eligible(x.shape[0], x.shape[1], row_recipe, col_recipe)

        fly = kernel.flydsl_dual_quant(
            x,
            turbo.float4_e2m1fn_x2,
            row_rht,
            col_rht,
            row_2d=row_2d,
            col_2d=col_2d,
            scale_rounding_mode=mode,
        )
        _assert_byte_exact(fly, _hip_quantize_mxfp4_dual(x, row_recipe, col_recipe, mode))


def test_mxfp4_scale_rounding_batched_3d_padding_and_cache():
    """Padded 3D quant caches distinct compiled variants for modes 0 and 2."""
    kernel = _load_mxfp4_flydsl_kernel(require_gfx950=True)
    recipe = ScalingRecipe(use_2d_block=True)
    assert kernel.dual3_eligible(192, 64, recipe, recipe)

    block_values = torch.tensor([1.5, 1.75], device="cuda", dtype=torch.bfloat16)
    x = block_values.repeat_interleave(MXFP4_BLOCK_SIZE).expand(2, 192, -1).contiguous()

    results = {}
    for mode in (0, 2):
        fly = kernel.flydsl_dual_quant_batched(
            x,
            turbo.float4_e2m1fn_x2,
            False,
            False,
            row_2d=True,
            col_2d=True,
            scale_rounding_mode=mode,
        )
        _assert_byte_exact(fly, _hip_quantize_mxfp4_dual(x, recipe, recipe, mode))
        results[mode] = tuple(tensor.clone() for tensor in fly)

    assert not torch.equal(results[0][1].view(torch.uint8), results[2][1].view(torch.uint8))


def test_mxfp4_scale_rounding_hip_rejects_invalid_mode():
    """The HIP entry point independently rejects invalid low-level mode values."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    x = torch.zeros((32, 32), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="must be 0, 1, or 2"):
        quantize_fp4(
            x,
            turbo.float4_e2m1fn_x2,
            granularity=ScalingGranularity.MX_BLOCKWISE,
            axis=1,
            block_size=MXFP4_BLOCK_SIZE,
            scaling_recipe=ScalingRecipe(),
            scale_rounding_mode=3,
        )


def test_grouped_mxfp4_scale_rounding_flydsl_matches_hip():
    """Irregular and empty K=256 groups preserve HIP bytes with and without RHT."""
    _load_mxfp4_flydsl_kernel(require_gfx950=True)
    from primus_turbo.flydsl.quantization import mxfp4_grouped_quant

    scale_rounding_mode = 2
    group_lens_values = (0, 1, 255, 257, 511, 513)
    group_lens = torch.tensor(group_lens_values, device="cuda", dtype=torch.int64)
    group_offs = torch.cat([torch.zeros(1, device="cuda", dtype=torch.int64), group_lens.cumsum(0)])

    torch.manual_seed(42)
    x = torch.randn((sum(group_lens_values), 256), device="cuda", dtype=torch.bfloat16)

    fly_lens = [((length + 255) // 256) * 256 for length in group_lens_values]
    hip_lens = [((length + 127) // 128) * 128 for length in group_lens_values]
    fly_offs, hip_offs = [0], [0]
    for fly_length, hip_length in zip(fly_lens, hip_lens):
        fly_offs.append(fly_offs[-1] + fly_length)
        hip_offs.append(hip_offs[-1] + hip_length)

    expected_fly_lens = torch.tensor(fly_lens, device="cuda", dtype=torch.int64)
    expected_fly_offs = torch.tensor(fly_offs, device="cuda", dtype=torch.int64)
    expected_hip_lens = torch.tensor(hip_lens, device="cuda", dtype=torch.int64)
    expected_hip_offs = torch.tensor(hip_offs, device="cuda", dtype=torch.int64)

    for use_rht in (False, True):
        # The FlyDSL colwise layout is 256-aligned per group; HIP uses 128 alignment.
        fly = mxfp4_grouped_quant.grouped_quant_mxfp4_raw(
            x,
            group_lens,
            group_offs,
            turbo.float4_e2m1fn_x2,
            use_rht,
            use_rht,
            scale_rounding_mode=scale_rounding_mode,
        )
        hip = torch.ops.primus_turbo_cpp_extension.grouped_quantize_mxfp4_dual(
            x,
            group_lens,
            group_offs,
            turbo.float4_e2m1fn_x2,
            False,
            False,
            use_rht,
            False,
            False,
            use_rht,
            scale_rounding_mode,
        )

        _assert_byte_exact(fly[:2], hip[:2])
        torch.testing.assert_close(hip[4], group_lens, rtol=0, atol=0)
        torch.testing.assert_close(hip[5], group_offs, rtol=0, atol=0)
        torch.testing.assert_close(fly[4], expected_fly_lens, rtol=0, atol=0)
        torch.testing.assert_close(fly[5], expected_fly_offs, rtol=0, atol=0)
        torch.testing.assert_close(hip[6], expected_hip_lens, rtol=0, atol=0)
        torch.testing.assert_close(hip[7], expected_hip_offs, rtol=0, atol=0)

        fly_col, fly_scale = fly[2].view(torch.uint8), fly[3].view(torch.uint8)
        hip_col, hip_scale = hip[2].view(torch.uint8), hip[3].view(torch.uint8)
        for fly_start, fly_length, hip_start, hip_length in zip(fly_offs, fly_lens, hip_offs, hip_lens):
            torch.testing.assert_close(
                fly_col[:, fly_start // 2 : (fly_start + hip_length) // 2],
                hip_col[:, hip_start // 2 : (hip_start + hip_length) // 2],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                fly_scale[
                    :,
                    fly_start // MXFP4_BLOCK_SIZE : (fly_start + hip_length) // MXFP4_BLOCK_SIZE,
                ],
                hip_scale[
                    :,
                    hip_start // MXFP4_BLOCK_SIZE : (hip_start + hip_length) // MXFP4_BLOCK_SIZE,
                ],
                rtol=0,
                atol=0,
            )
            assert (
                torch.count_nonzero(
                    fly_col[:, (fly_start + hip_length) // 2 : (fly_start + fly_length) // 2]
                ).item()
                == 0
            )
            assert (
                torch.count_nonzero(
                    fly_scale[
                        :,
                        (fly_start + hip_length) // MXFP4_BLOCK_SIZE : (fly_start + fly_length)
                        // MXFP4_BLOCK_SIZE,
                    ]
                ).item()
                == 0
            )


def test_mxfp4_scale_rounding_sr_scale_parity():
    """SR changes samples but leaves FlyDSL and HIP scale selection byte-exact."""
    kernel = _load_mxfp4_flydsl_kernel(require_gfx950=True)
    scale_rounding_mode = 2

    torch.manual_seed(123)
    x = torch.randn((384, 256), device="cuda", dtype=torch.bfloat16)
    row_recipe = ScalingRecipe(use_sr=True)
    col_recipe = ScalingRecipe(use_sr=True, use_rht=True)
    assert kernel.dual_eligible(x.shape[0], x.shape[1], row_recipe, col_recipe)

    fly1 = kernel.flydsl_dual_quant(
        x,
        turbo.float4_e2m1fn_x2,
        row_recipe.use_rht,
        col_recipe.use_rht,
        row_sr=True,
        col_sr=True,
        scale_rounding_mode=scale_rounding_mode,
    )
    fly2 = kernel.flydsl_dual_quant(
        x,
        turbo.float4_e2m1fn_x2,
        row_recipe.use_rht,
        col_recipe.use_rht,
        row_sr=True,
        col_sr=True,
        scale_rounding_mode=scale_rounding_mode,
    )
    hip = _hip_quantize_mxfp4_dual(x, row_recipe, col_recipe, scale_rounding_mode)

    assert not torch.equal(fly1[0].view(torch.uint8), fly2[0].view(torch.uint8))
    assert not torch.equal(fly1[2].view(torch.uint8), fly2[2].view(torch.uint8))
    for scale_index in (1, 3):
        torch.testing.assert_close(
            fly1[scale_index].view(torch.uint8),
            hip[scale_index].view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            fly2[scale_index].view(torch.uint8),
            hip[scale_index].view(torch.uint8),
            rtol=0,
            atol=0,
        )


def test_mxfp4_dense_dual_covers_every_eligible_row():
    """A dense launch must not leave tail rows unwritten for an eligible shape."""
    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if (props.major, props.minor) != (9, 5):
        pytest.skip("FlyDSL MXFP4 quantization requires gfx950")

    from primus_turbo.flydsl.quantization import mxfp4_quant_kernel as kernel

    torch.manual_seed(42)
    x = torch.randn((128, 256), device="cuda", dtype=torch.bfloat16)
    row_recipe = ScalingRecipe()
    col_recipe = ScalingRecipe(use_rht=True)

    assert kernel.dual_eligible(x.shape[0], x.shape[1], row_recipe, col_recipe)
    assert x.shape[0] % kernel._TR == 0

    fly = kernel.flydsl_dual_quant(
        x,
        turbo.float4_e2m1fn_x2,
        row_recipe.use_rht,
        col_recipe.use_rht,
    )
    hip = torch.ops.primus_turbo_cpp_extension.quantize_mxfp4_dual(
        x,
        turbo.float4_e2m1fn_x2,
        128,
        row_recipe.use_2d_block,
        row_recipe.use_sr,
        row_recipe.use_rht,
        col_recipe.use_2d_block,
        col_recipe.use_sr,
        col_recipe.use_rht,
        False,
        False,
        False,
        False,
    )
    for actual, expected in zip(fly, hip):
        torch.testing.assert_close(
            actual.view(torch.uint8),
            expected.view(torch.uint8),
            rtol=0,
            atol=0,
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
        colwise_scale_shuffle.view(torch.uint8),
        colwise_scale_shuffle_ref.view(torch.uint8),
        atol=0,
        rtol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mxfp4_sr_consecutive_calls_differ():
    """Two quantize_fp4 calls with use_sr=True on the same input should produce different outputs."""
    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    torch.manual_seed(42)
    x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)

    sr_recipe = ScalingRecipe(use_sr=True)

    out1, scale1 = quantize_fp4(
        x,
        turbo.float4_e2m1fn_x2,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        axis=1,
        block_size=MXFP4_BLOCK_SIZE,
        scaling_recipe=sr_recipe,
    )

    out2, scale2 = quantize_fp4(
        x,
        turbo.float4_e2m1fn_x2,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        axis=1,
        block_size=MXFP4_BLOCK_SIZE,
        scaling_recipe=sr_recipe,
    )

    assert not torch.equal(out1.view(torch.uint8), out2.view(torch.uint8)), (
        "SR-quantized outputs should differ across consecutive calls"
    )
