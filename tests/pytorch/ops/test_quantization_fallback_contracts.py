###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Numerical HIP fallback and gather/zero-padding contracts for MX quantization."""

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import ScalingRecipe, float4_e2m1fn_x2, float8_e4m3
from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.kernels.quantization import quantization_impl as impl

pytestmark = pytest.mark.skipif(not is_gfx950(), reason="MX quantization requires gfx950")


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("rank", [1, 4])
def test_mx_rejects_unsupported_rank(bits, rank):
    fn = getattr(impl, f"quantize_mxfp{bits}_impl")
    dtype = float4_e2m1fn_x2 if bits == 4 else float8_e4m3
    with pytest.raises(ValueError, match="must be 2D or 3D"):
        fn(torch.zeros([2] * rank, device="cuda"), dtype, 0, 32)


@pytest.mark.parametrize("probs", [None, [1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.5, 0.0]])
def test_materialized_gather_preserves_source_and_zeros_padding(probs):
    x = torch.arange(24, device="cuda").reshape(6, 4).float()
    original = x.clone()
    index = torch.tensor([5, 0, 5, 2], device="cuda", dtype=torch.int32)
    p = None if probs is None else torch.tensor(probs, device="cuda")
    actual = impl._materialize_gathered_activation(x, index, p)
    expected = torch.stack([original[5], original[0], original[5], original[2]])
    if p is not None:
        expected[p == 0] = 0
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x, original)


@pytest.mark.parametrize("bits", [4, 8])
def test_grouped_dual_2d_recipe_hip_matches_single_direction(bits):
    torch.manual_seed(910)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    lens = torch.tensor([32, 32], device="cuda", dtype=torch.int64)
    offs = torch.tensor([0, 32, 64], device="cuda", dtype=torch.int64)
    recipe = ScalingRecipe(use_2d_block=True)
    dtype = float4_e2m1fn_x2 if bits == 4 else float8_e4m3
    fn = getattr(impl, f"grouped_quantize_mxfp{bits}_impl")
    dual = fn(x, dtype, None, 32, lens, offs, True, recipe, recipe)
    single = fn(x, dtype, 1, 32, lens, offs, False, recipe)
    # Compare exact packed bytes and E8M0 scale bytes, not float4 cast semantics.
    for group in range(2):
        dual_start, single_start = int(dual[5][group]), int(single[3][group])
        for actual, expected in zip(dual[:2], single[:2]):
            torch.testing.assert_close(
                actual[dual_start : dual_start + 32].view(torch.uint8),
                expected[single_start : single_start + 32].view(torch.uint8),
            )
    assert len(dual) == 8
    assert all(t.is_cuda for t in dual)


@pytest.mark.parametrize("zero_padding", [False, True])
def test_fp4_gather_fallback_quantizes_destination_rows(zero_padding):
    torch.manual_seed(911)
    x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
    index = torch.arange(64, device="cuda", dtype=torch.int32).remainder(32).flip(0)
    probs = torch.ones(64, device="cuda")
    if zero_padding:
        probs[::7] = 0
    lens = torch.tensor([32, 32], device="cuda", dtype=torch.int64)
    offs = torch.tensor([0, 32, 64], device="cuda", dtype=torch.int64)
    recipe = ScalingRecipe(use_2d_block=True)
    args = (float4_e2m1fn_x2, None, 32, lens, offs, True, recipe, recipe)
    actual = impl.grouped_quantize_mxfp4_impl(x, *args, dest2src=index, permuted_probs=probs, total_m=64)
    expected_input = torch.stack([x[int(i)] for i in index.cpu()])
    expected_input[probs == 0] = 0
    expected = impl.grouped_quantize_mxfp4_impl(expected_input, *args)
    for i, (a, b) in enumerate(zip(actual, expected)):
        if i in (2, 3):
            used = int(expected[7][-1]) // (2 if i == 2 else 32)
            a, b = a[:, :used], b[:, :used]
        torch.testing.assert_close(a.view(torch.uint8), b.view(torch.uint8))
