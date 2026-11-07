###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Check real quantization outputs against their fake kernels and independent scales."""

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import float8_e4m3
from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.kernels.quantization import quantization_impl as impl

pytestmark = pytest.mark.skipif(not is_gfx950(), reason="This regression suite requires gfx950")


@pytest.mark.parametrize("shape", [(129, 65), (2, 129, 65)])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_block_weight_scales_and_fake(shape, block_size, noncontiguous):
    torch.manual_seed(801)
    w = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    if noncontiguous:
        w = w.transpose(-1, -2)
    quant, scales = impl.quant_fp8_blockwise_for_weight_impl(w, float8_e4m3, block_size)
    expected_scales = torch.empty_like(scales)
    expected = torch.empty_like(w)
    for m in range(0, w.shape[-2], block_size):
        for n in range(0, w.shape[-1], block_size):
            tile = w[..., m : m + block_size, n : n + block_size].float()
            multiplier = torch.finfo(float8_e4m3).max / tile.abs().amax(dim=(-1, -2)).clamp_min(1e-4)
            scale = multiplier.reciprocal()
            expected_scales[..., m // block_size, n // block_size] = scale
            rounded = (tile * multiplier[..., None, None]).to(float8_e4m3).float()
            expected[..., m : m + block_size, n : n + block_size] = rounded * scale[..., None, None]
    torch.testing.assert_close(scales, expected_scales)
    dequant = impl.dequant_fp8_blockwise_for_weight_impl(quant, scales, w.dtype, block_size)
    torch.testing.assert_close(dequant, expected)
    # Also exercise a strided FP8 input to dequantization.
    transposed = impl.dequant_fp8_blockwise_for_weight_impl(
        quant.transpose(-1, -2), scales.transpose(-1, -2).contiguous(), w.dtype, block_size
    )
    torch.testing.assert_close(transposed, expected.transpose(-1, -2))
    for op, args in [
        (impl.quant_fp8_blockwise_for_weight_impl, (w, float8_e4m3, block_size)),
        (impl.dequant_fp8_blockwise_for_weight_impl, (quant, scales, w.dtype, block_size)),
    ]:
        torch.library.opcheck(op, args, test_utils=("test_faketensor",))


@pytest.mark.parametrize("axis", [-2, -1, 0, 1])
def test_row_column_fake_matches_real(axis):
    x = torch.randn(65, 129, device="cuda", dtype=torch.bfloat16)
    q, scales = impl.quant_fp8_blockwise_impl(x, float8_e4m3, axis, 64)
    restored = impl.dequant_fp8_blockwise_impl(q, scales, torch.float32, axis, 64)
    expanded = scales.repeat_interleave(64, axis % 2)[:65, :129]
    torch.testing.assert_close(restored, q.float() * expanded)
    for op, args in [
        (impl.quant_fp8_blockwise_impl, (x, float8_e4m3, axis, 64)),
        (impl.dequant_fp8_blockwise_impl, (q, scales, torch.float32, axis, 64)),
        (impl.quant_fp8_blockwise_dual_impl, (x, float8_e4m3, 64)),
    ]:
        torch.library.opcheck(op, args, test_utils=("test_faketensor",))
    row, sr, col, sc = impl.quant_fp8_blockwise_dual_impl(x, float8_e4m3, 64)
    selected, scale = (row, sr) if axis % 2 else (col, sc)
    torch.testing.assert_close(selected.float(), q.float(), rtol=0, atol=0)
    torch.testing.assert_close(scale, scales, rtol=0, atol=0)


def test_segment_padding_fake_matches_real():
    x = torch.randn(192, 128, device="cuda", dtype=torch.bfloat16)
    lens = torch.tensor([65, 127], device="cuda", dtype=torch.int64)
    offs = torch.tensor([0, 65, 192], device="cuda", dtype=torch.int64)
    op = impl.quant_fp8_blockwise_segment_m_row_col_impl
    torch.library.opcheck(op, (x, float8_e4m3, 128, lens, offs), test_utils=("test_faketensor",))
    _, _, _, _, padded_lens, padded_offs = op(x, float8_e4m3, 128, lens, offs)
    torch.testing.assert_close(padded_lens, torch.tensor([128, 128], device="cuda"))
    torch.testing.assert_close(padded_offs, torch.tensor([0, 128, 256], device="cuda"))


@pytest.mark.parametrize("lazy_seam", [False, True])
@pytest.mark.parametrize("zero_probs", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gather_fallback_matches_materialized(dtype, zero_probs, lazy_seam):
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity, ScalingRecipe, float4_e2m1fn_x2
    from primus_turbo.pytorch.ops.quantization import grouped_quantize_fp4_with_trans

    torch.manual_seed(806)
    source = torch.randn(97, 256, device="cuda", dtype=dtype)
    mapping = torch.randint(0, 97, (192,), device="cuda", dtype=torch.int32)
    probs = torch.ones(192, device="cuda") if zero_probs else None
    if probs is not None:
        probs[::3] = 0
    explicit = source[mapping.long()].clone()
    if probs is not None:
        explicit[probs == 0] = 0
    lens = torch.tensor([65, 0, 127], device="cuda", dtype=torch.int64)
    offs = torch.tensor([0, 65, 65, 192], device="cuda", dtype=torch.int64)
    options = dict(
        block_size=32, scaling_recipe=ScalingRecipe(), scaling_recipe_for_trans=ScalingRecipe(use_rht=True)
    )
    expected = grouped_quantize_fp4_with_trans(
        explicit, float4_e2m1fn_x2, ScalingGranularity.MX_BLOCKWISE, lens, offs, **options
    )
    if lazy_seam:
        from primus_turbo.pytorch.ops.moe_gather import register_permuted_activation_seam

        placeholder = torch.empty(1, 256, device="cuda", dtype=dtype).expand(192, 256)
        register_permuted_activation_seam(placeholder, source, mapping, probs)
        got = grouped_quantize_fp4_with_trans(
            placeholder, float4_e2m1fn_x2, ScalingGranularity.MX_BLOCKWISE, lens, offs, **options
        )
    else:
        got = grouped_quantize_fp4_with_trans(
            source,
            float4_e2m1fn_x2,
            ScalingGranularity.MX_BLOCKWISE,
            lens,
            offs,
            **options,
            dest2src=mapping,
            permuted_probs=probs,
            total_m=192,
        )
    for index, (a, b) in enumerate(zip(got, expected)):
        if index in (0, 1, 2, 3):
            a, b = a.view(torch.uint8), b.view(torch.uint8)
        if index in (2, 3):
            used = int(expected[7][-1])
            divisor = 2 if index == 2 else 32
            a, b = a[:, : used // divisor], b[:, : used // divisor]
        torch.testing.assert_close(a, b, atol=0, rtol=0)
