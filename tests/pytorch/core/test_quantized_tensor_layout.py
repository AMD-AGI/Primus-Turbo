###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import (
    ScalingGranularity,
    check_mxfp4_support,
    check_mxfp8_support,
    float4_e2m1fn_x2,
    float8_e4m3,
)
from primus_turbo.pytorch.core.quantized_tensor import (
    QuantizedTensor,
    _compute_scale_shape,
    _normalize_axis,
    _pad_inner_dim,
)


@pytest.mark.parametrize(
    "shape, align, expected",
    [((), 32, ()), ((2, 3), 0, (2, 3)), ((2, 3), 1, (2, 3)), ((2, 32), 32, (2, 32)), ((2, 33), 32, (2, 64))],
)
def test_padding_shape(shape, align, expected):
    assert _pad_inner_dim(shape, align) == expected


@pytest.mark.parametrize("axis, ndim, expected", [(None, 2, None), (-1, 2, 1), (-2, 3, 1), (0, 3, 0)])
def test_axis_normalization(axis, ndim, expected):
    assert _normalize_axis(axis, ndim) == expected


@pytest.mark.parametrize(
    "shape, block, packing, expected", [((2, 17), 32, 2, (2, 2)), ((3, 2, 33), 32, 1, (3, 2, 2))]
)
def test_scale_shape_uses_logical_packed_width(shape, block, packing, expected):
    assert _compute_scale_shape(shape, block, packing) == expected


def make_tensor(*, fp4=False, mx=False):
    dtype = float4_e2m1fn_x2 if fp4 else float8_e4m3
    packing = 2 if fp4 else 1
    # Both target layouts have aligned inner dimensions and identical storage size.
    data = (
        torch.arange(4 * 256 // packing, dtype=torch.int64)
        .to(torch.uint8)
        .reshape(4, 256 // packing)
        .view(dtype)
    )
    scale = torch.ones((4, 8) if mx else (), dtype=torch.float32)
    return QuantizedTensor(
        data,
        scale,
        shape=torch.Size((4, 256)),
        orig_dtype=torch.bfloat16,
        dest_dtype=dtype,
        granularity=ScalingGranularity.MX_BLOCKWISE if mx else ScalingGranularity.TENSORWISE,
        block_size=32 if mx else None,
    )


@pytest.mark.parametrize(
    "fp4, mx",
    [
        (False, False),
        (False, True),
        pytest.param(
            True,
            True,
            marks=pytest.mark.skipif(
                not check_mxfp4_support()[0], reason="MXFP4 requires supported hardware"
            ),
        ),
    ],
)
@pytest.mark.parametrize("operation", ["view", "reshape"])
def test_reshape_preserves_payload_and_scale_mapping(fp4, mx, operation):
    tensor = make_tensor(fp4=fp4, mx=mx)
    out = getattr(tensor, operation)(2, 512)
    assert out.shape == (2, 512)
    assert out.qdata.shape == (2, 256 if fp4 else 512)
    assert out.qdata.data_ptr() == tensor.qdata.data_ptr()
    torch.testing.assert_close(
        out.qdata.view(torch.uint8).flatten(), tensor.qdata.view(torch.uint8).flatten()
    )
    assert out.scale_inv.shape == ((2, 16) if mx else ())
    assert out.scale_inv.data_ptr() == tensor.scale_inv.data_ptr()
    same = getattr(tensor, operation)(4, 256)
    assert same.qdata.data_ptr() == tensor.qdata.data_ptr()


def test_detach_select_and_unsupported_operation():
    tensor = make_tensor()
    detached = tensor.detach()
    assert isinstance(detached, QuantizedTensor)
    assert detached.qdata.data_ptr() == tensor.qdata.data_ptr()
    selected = tensor[1]
    torch.testing.assert_close(selected.view(torch.uint8), tensor.qdata[1].view(torch.uint8))
    with pytest.raises(NotImplementedError, match="Unsupported dispatch"):
        torch.sin(tensor)


def test_identity_transpose_preserves_metadata():
    tensor = make_tensor(mx=True)
    out = tensor.transpose(-1, 1)
    assert out.shape == tensor.shape
    assert out.qdata.data_ptr() == tensor.qdata.data_ptr()
    assert out.granularity == tensor.granularity


@pytest.mark.parametrize("operation", ["view", "reshape"])
def test_layout_autograd_restores_original_gradient_shape(operation):
    tensor = make_tensor()
    tensor.requires_grad_(True)
    output = getattr(tensor, operation)(2, 512)
    grad = torch.arange(1024, dtype=torch.bfloat16).reshape(2, 512)
    with torch.autograd.set_multithreading_enabled(False):
        output.backward(grad)
    torch.testing.assert_close(tensor.grad, grad.reshape(4, 256))


@pytest.mark.parametrize("operation", [torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default])
def test_aten_view_preserves_quantized_storage(operation):
    tensor = make_tensor(mx=True)
    output = operation(tensor, [2, 512])
    assert output.shape == (2, 512)
    assert output.qdata.data_ptr() == tensor.qdata.data_ptr()
    assert output.scale_inv.shape == (2, 16)


@pytest.mark.parametrize(
    "granularity",
    [
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
        ScalingGranularity.BLOCKWISE,
        pytest.param(
            ScalingGranularity.MX_BLOCKWISE,
            marks=pytest.mark.skipif(
                not check_mxfp8_support()[0], reason="MXFP8 requires supported hardware"
            ),
        ),
    ],
)
def test_quantized_weight_cache_preserves_logical_values(granularity):
    from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, ScaleDtype
    from primus_turbo.pytorch.core.quantized_tensor import create_quantized_weight

    torch.manual_seed(13)
    weight = torch.randn(256, 384, device="cuda", dtype=torch.bfloat16)
    config = Float8QuantConfig(
        granularity=granularity,
        block_size={ScalingGranularity.MX_BLOCKWISE: 32, ScalingGranularity.BLOCKWISE: 128}.get(granularity),
        scale_dtype=ScaleDtype.E8M0 if granularity == ScalingGranularity.MX_BLOCKWISE else ScaleDtype.FP32,
    )
    quantized, transposed = create_quantized_weight(
        weight, float8_e4m3, config, need_weight_transpose_cache=True
    )
    assert transposed is not None
    assert quantized.granularity == transposed.granularity == granularity
    torch.testing.assert_close(quantized.dequantize(), weight, rtol=0.15, atol=0.03)
    # Block-scaled caches change the quantization axis, while dequantize()
    # restores the original logical matrix. Tensor/rowwise caches explicitly
    # transpose the input; a non-square shape distinguishes the two contracts.
    expected = (
        weight
        if granularity in (ScalingGranularity.BLOCKWISE, ScalingGranularity.MX_BLOCKWISE)
        else weight.T.contiguous()
    )
    torch.testing.assert_close(transposed.dequantize(), expected, rtol=0.15, atol=0.03)
