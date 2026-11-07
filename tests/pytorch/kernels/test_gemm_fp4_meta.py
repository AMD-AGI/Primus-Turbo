###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float4_e2m1fn_x2
from primus_turbo.pytorch.kernels.gemm import gemm_fp4_impl as impl


def tensor(shape, dtype=float4_e2m1fn_x2):
    return torch.empty(shape, device="meta", dtype=dtype)


@pytest.mark.parametrize("trans_a, trans_b", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("trans_c", [False, True])
def test_meta_output_obeys_logical_layout(trans_a, trans_b, trans_c):
    a = tensor((32, 8) if trans_a else (8, 32))
    b = tensor((48, 32) if trans_b else (32, 48))
    out = impl.gemm_fp4_impl(
        a,
        tensor((1,)),
        trans_a,
        b,
        tensor((1,)),
        trans_b,
        torch.bfloat16,
        trans_c,
        ScalingGranularity.MX_BLOCKWISE.value,
        0,
    )
    assert out.shape == ((48, 8) if trans_c else (8, 48))
    assert out.dtype == torch.bfloat16 and out.device.type == "meta"


def test_meta_rejects_contraction_mismatch():
    with pytest.raises(AssertionError, match="GEMM K mismatch"):
        impl.gemm_fp4_impl(
            tensor((8, 32)),
            tensor((1,)),
            False,
            tensor((48, 64)),
            tensor((1,)),
            True,
            torch.bfloat16,
            False,
            ScalingGranularity.MX_BLOCKWISE.value,
            0,
        )


@pytest.mark.parametrize("trans_c", [False, True])
def test_meta_accumulation_requires_exact_output_shape(trans_c):
    args = dict(
        a=tensor((8, 32)),
        a_scale_inv=tensor((1,)),
        trans_a=False,
        b=tensor((48, 32)),
        b_scale_inv=tensor((1,)),
        trans_b=True,
        out_dtype=torch.bfloat16,
        trans_c=trans_c,
        granularity=ScalingGranularity.MX_BLOCKWISE.value,
        default_backend=0,
    )
    valid = tensor((48, 8) if trans_c else (8, 48), torch.bfloat16)
    assert impl.gemm_fp4_accum_impl(**args, out=valid) is None
    with pytest.raises(AssertionError, match="out shape"):
        impl.gemm_fp4_accum_impl(**args, out=tensor((8, 8), torch.bfloat16))


@pytest.mark.parametrize("width", [64, 128])
def test_glu_and_dglu_meta_pack_and_pad_both_orientations(width):
    args = dict(
        a=tensor((17, 32)),
        a_scale_inv=tensor((17, 2)),
        b=tensor((2 * width, 32)),
        b_scale_inv=tensor((2 * width, 2)),
        probs=tensor((17,), torch.float32),
        out_dtype=torch.bfloat16,
        row_use_sr=False,
        col_use_sr=False,
        scale_rounding_mode=0,
        activation="silu",
        clamp_limit=None,
    )
    intermediate, row, row_scale, col, col_scale = impl.gemm_fp4_glu_quant_impl(**args)
    assert intermediate.shape == (17, 2 * width)
    assert row.shape == (17, 64) and row_scale.shape == (17, 4)
    assert col.shape == (width, 128) and col_scale.shape == (width, 8)
    assert row.dtype == col.dtype == float4_e2m1fn_x2
    args.update(b=tensor((width, 32)), b_scale_inv=tensor((width, 2)))
    row, row_scale, col, col_scale = impl.gemm_fp4_dglu_quant_impl(**args, l1=intermediate)
    assert row.shape == (17, width) and row_scale.shape == (17, 2 * width // 32)
    assert col.shape == (2 * width, 128) and col_scale.shape == (2 * width, 8)
