###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Compiler-visible grouped FP4 shapes, packing and accumulation side effects."""

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import (
    Float4QuantConfig,
    ScalingGranularity,
    ScalingRecipe,
    float4_e2m1fn_x2,
)
from primus_turbo.pytorch.kernels.grouped_gemm import grouped_gemm_fp4_impl as impl


def tensor(shape, dtype=torch.bfloat16):
    return torch.empty(shape, device="meta", dtype=dtype)


@pytest.mark.parametrize("groups,rows,n,k", [(2, 65, 128, 128), (3, 129, 256, 256)])
def test_grouped_fp4_forward_fake_matches_logical_shape(groups, rows, n, k):
    result = impl.grouped_gemm_fp4_impl(
        tensor((rows, k // 2), float4_e2m1fn_x2),
        tensor((groups, n, k // 2), float4_e2m1fn_x2),
        tensor((rows, k // 32), torch.float8_e8m0fnu),
        tensor((groups, n, k // 32), torch.float8_e8m0fnu),
        tensor((groups,), torch.int64),
        tensor((groups + 1,), torch.int64),
        False,
        True,
        torch.bfloat16,
        ScalingGranularity.MX_BLOCKWISE.value,
        None,
        0,
    )
    assert result.shape == (rows, n) and result.dtype == torch.bfloat16


@pytest.mark.parametrize("backward", [False, True])
def test_glu_fake_preserves_packed_row_and_column_contract(backward):
    rows, k, i, groups = 65, 128, 128, 2
    a = tensor((rows, k // 2), float4_e2m1fn_x2)
    b = tensor((groups, 2 * i if not backward else i, k // 2), float4_e2m1fn_x2)
    args = [
        a,
        b,
        tensor((rows, 4), torch.float8_e8m0fnu),
        tensor((groups, b.shape[1], 4), torch.float8_e8m0fnu),
        tensor((groups,), torch.int64),
        tensor((groups + 1,), torch.int64),
        False,
        True,
        torch.bfloat16,
        None,
        tensor((rows,), torch.float32),
    ]
    if backward:
        args.append(tensor((rows, 2 * i)))
    args.extend([Float4QuantConfig(), ScalingRecipe(), ScalingRecipe(use_rht=True)])
    op = impl.grouped_gemm_fp4_dglu_impl if backward else impl.grouped_gemm_fp4_glu_impl
    head, row, sr, col, sc = op(*args)
    width = 2 * i if backward else i
    assert head.shape == ((rows,) if backward else (rows, 2 * i))
    assert row.shape[0] == rows and row.shape[1] * 2 >= width
    assert col.shape[0] == width and col.shape[1] * 2 >= rows
    assert sr.shape[-1] * 32 == row.shape[-1] * 2
    assert sc.shape[-1] * 32 == col.shape[-1] * 2
    assert row.dtype == col.dtype == float4_e2m1fn_x2


@pytest.mark.parametrize("bad_contraction", [False, True])
def test_accumulation_fake_returns_no_tensor_and_checks_contraction(bad_contraction):
    args = dict(
        a=tensor((128, 128), float4_e2m1fn_x2),
        b=tensor((256, 64 if bad_contraction else 128), float4_e2m1fn_x2),
        a_scales=tensor((128, 8)),
        b_scales=tensor((256, 8)),
        group_lens=tensor((2,), torch.int64),
        group_offs=tensor((3,), torch.int64),
        trans_a=False,
        trans_b=True,
        trans_c=False,
        out_dtype=torch.bfloat16,
        granularity=ScalingGranularity.MX_BLOCKWISE.value,
        num_cu=None,
        default_backend=0,
        out=tensor((2, 128, 256)),
    )
    if bad_contraction:
        with pytest.raises(AssertionError, match="contraction dimensions"):
            impl.grouped_gemm_fp4_variable_k_accum_impl(**args)
    else:
        assert impl.grouped_gemm_fp4_variable_k_accum_impl(**args) is None
