###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    ScaleDtype,
    ScalingGranularity,
    ScalingRecipe,
    float8_e4m3,
)
from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.kernels.grouped_gemm import grouped_gemm_fp8_impl as impl


def tensor(shape, dtype=float8_e4m3):
    return torch.empty(shape, device="meta", dtype=dtype)


def operands(*, variable_k=False, trans_b=True):
    return dict(
        a=tensor((8, 32)),
        b=tensor((8, 48) if variable_k else ((3, 48, 32) if trans_b else (3, 32, 48))),
        a_scales=tensor((), torch.float32),
        b_scales=tensor((), torch.float32),
        group_lens=tensor((3,), torch.int64),
        group_offs=tensor((4,), torch.int64),
        trans_a=variable_k,
        trans_b=trans_b,
        out_dtype=torch.bfloat16,
        granularity=ScalingGranularity.TENSORWISE.value,
        num_cu=None,
        default_backend=0,
    )


@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("n_real, width", [(None, 48), (43, 43)])
def test_registered_fake_grouped_output(trans_b, n_real, width):
    # Call the registered custom op, so this also verifies its Fake/Meta dispatch.
    out = impl.grouped_gemm_fp8_impl(**operands(trans_b=trans_b), n_real=n_real)
    assert out.shape == (8, width)
    assert out.dtype == torch.bfloat16 and out.device.type == "meta"


@pytest.mark.parametrize(
    "trans_c, overrides, shape",
    [(False, {}, (3, 32, 48)), (True, {}, (3, 48, 32)), (False, {"m_real": 29, "n_real": 43}, (3, 29, 43))],
)
def test_variable_k_logical_shape(trans_c, overrides, shape):
    args = operands(variable_k=True, trans_b=False)
    out = impl.grouped_gemm_fp8_variable_k_impl(**args, trans_c=trans_c, **overrides)
    assert out.shape == shape
    assert out.dtype == args["out_dtype"]


@pytest.mark.parametrize("trans_c, shape", [(False, (3, 32, 48)), (True, (3, 48, 32))])
def test_mx_variable_k_swaps_feature_axes(trans_c, shape):
    args = operands(variable_k=True, trans_b=False)
    args.update(a=tensor((32, 128)), b=tensor((48, 128)), granularity=ScalingGranularity.MX_BLOCKWISE.value)
    out = impl.grouped_gemm_fp8_variable_k_impl(**args, trans_c=trans_c)
    assert out.shape == shape


@pytest.mark.parametrize(
    "update, message",
    [
        ({"a": tensor((1, 8, 32))}, "a must be 2D"),
        ({"b": tensor((32, 48))}, "b must be 3D"),
        ({"a": tensor((8, 32), torch.float32)}, "a must be fp8"),
        ({"b": tensor((3, 48, 32), torch.float32)}, "b must be fp8"),
        ({"out_dtype": torch.float32}, "out_dtype"),
        ({"trans_a": True}, "trans_a=False"),
    ],
)
def test_fake_grouped_rejects_invalid_contract(update, message):
    args = operands()
    args.update(update)
    with pytest.raises(AssertionError, match=message):
        impl.grouped_gemm_fp8_impl(**args)


def glu_operands(trans_b=True):
    args = operands(trans_b=trans_b)
    del args["granularity"], args["default_backend"]
    args.update(
        probs=tensor((8,), torch.float32),
        config=Float8QuantConfig(),
        out_quant_dtype=float8_e4m3,
        out_row_scaling_recipe=ScalingRecipe(),
        out_col_scaling_recipe=ScalingRecipe(),
    )
    return args


@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("alignment, padded", [(1, 24), (32, 32)])
def test_glu_meta_separates_intermediate_and_padded_activation(trans_b, alignment, padded):
    intermediate, act, scales = impl.grouped_gemm_fp8_glu_impl(**glu_operands(trans_b), k_align=alignment)
    assert intermediate.shape == (8, 48) and intermediate.dtype == torch.bfloat16
    assert act.shape == (8, padded) and act.dtype == float8_e4m3
    assert scales.shape == () and scales.dtype == torch.float32


@pytest.mark.parametrize("i_real, width", [(None, 48), (43, 43)])
def test_dglu_meta_restores_unpadded_intermediate(i_real, width):
    grads, intermediate, scales = impl.grouped_gemm_fp8_dglu_impl(
        **glu_operands(), intermediate=tensor((8, 96), torch.bfloat16), i_real=i_real
    )
    assert grads.shape == (8,) and grads.dtype == torch.float32
    assert intermediate.shape == (8, 2 * width) and intermediate.dtype == float8_e4m3
    assert scales.shape == () and scales.dtype == torch.float32


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_gfx950(), reason="MXFP8 fused GLU requires gfx950"
)
def test_mx_glu_meta_padded_row_and_column_operands(backward):
    args = glu_operands()
    del args["out_quant_dtype"]
    args.update(
        a=tensor((64, 32)),
        b=tensor((3, 48, 32)),
        probs=tensor((17,), torch.float32),
        config=Float8QuantConfig(
            granularity=ScalingGranularity.MX_BLOCKWISE, block_size=32, scale_dtype=ScaleDtype.E8M0
        ),
    )
    if backward:
        outputs = impl.grouped_gemm_mxfp8_dglu_impl(**args, intermediate=tensor((17, 96), torch.bfloat16))
        expected_first, feature = (17,), 96
    else:
        outputs = impl.grouped_gemm_mxfp8_glu_impl(**args)
        expected_first, feature = (17, 48), 24
    first, row, row_scale, col, col_scale = outputs
    assert first.shape == expected_first
    assert row.shape == (64, 128) and row_scale.shape == (64, 4)
    assert col.shape == (feature, 512) and col_scale.shape == (feature, 16)
    assert row.dtype == col.dtype == float8_e4m3
    assert row_scale.element_size() == col_scale.element_size() == 1
