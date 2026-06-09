###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the fused FP8 cast+transpose+amax kernel (C++ and Triton)."""

import pytest
import torch

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.kernels.quantization.cast_transpose_fp8 import (
    cast_transpose_fp8_triton,
)
from tests.pytorch.test_utils import get_tolerances

DEVICE = "cuda:0"

SHAPES = [
    (4096, 3072),
    (100, 200),
    (33, 65),
    (64, 64),
    (1, 128),
    (128, 1),
]


def _reference_cast_transpose(x, dest_dtype, scale):
    """Reference: C++ quantize with pre-computed scale, then transpose."""
    fp8_out, scale_inv = torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(
        x.reshape(-1), dest_dtype, scale
    )
    cast_ref = fp8_out.reshape(x.shape)
    trans_ref = cast_ref.t().contiguous()
    return cast_ref, trans_ref, scale_inv


def _make_scale(x, dest_dtype):
    """Compute a tensorwise scale from input, matching delayed-scaling usage."""
    fp8_max = torch.finfo(dest_dtype).max
    amax = x.float().abs().amax()
    return torch.tensor(fp8_max / amax.item(), dtype=torch.float32, device=x.device)


# ---------------------------------------------------------------------------
# C++ kernel tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("shape", SHAPES)
def test_cast_transpose_fp8_fused_correctness(orig_dtype, dest_dtype, shape):
    """C++ cast_transpose_fp8_fused produces correct cast and transpose outputs."""
    torch.manual_seed(42)
    x = torch.randn(shape, device=DEVICE, dtype=orig_dtype)
    scale = _make_scale(x, dest_dtype)

    cast_ref, trans_ref, scale_inv_ref = _reference_cast_transpose(x, dest_dtype, scale)

    result = torch.ops.primus_turbo_cpp_extension.cast_transpose_fp8_fused(x, dest_dtype, scale)
    cast_out, trans_out, scale_inv = result[0], result[1], result[2]

    assert cast_out.shape == x.shape
    assert trans_out.shape == (shape[1], shape[0])
    torch.testing.assert_close(cast_out, cast_ref, atol=0, rtol=0)
    torch.testing.assert_close(trans_out, trans_ref, atol=0, rtol=0)
    torch.testing.assert_close(scale_inv, scale_inv_ref, **get_tolerances(torch.float32))


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("shape", SHAPES)
def test_cast_transpose_fp8_fused_amax(orig_dtype, dest_dtype, shape):
    """C++ cast_transpose_fp8_fused correctly captures amax_out."""
    torch.manual_seed(42)
    x = torch.randn(shape, device=DEVICE, dtype=orig_dtype)
    scale = _make_scale(x, dest_dtype)

    amax_out = torch.zeros((), dtype=torch.float32, device=DEVICE)
    torch.ops.primus_turbo_cpp_extension.cast_transpose_fp8_fused(x, dest_dtype, scale, amax_out)

    expected_amax = x.float().abs().amax()
    torch.testing.assert_close(amax_out, expected_amax, **get_tolerances(torch.float32))


# ---------------------------------------------------------------------------
# Triton kernel tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("shape", SHAPES)
def test_cast_transpose_fp8_triton_correctness(orig_dtype, dest_dtype, shape):
    """Triton cast_transpose_fp8_triton produces correct cast and transpose outputs."""
    torch.manual_seed(42)
    x = torch.randn(shape, device=DEVICE, dtype=orig_dtype)
    scale = _make_scale(x, dest_dtype)

    cast_ref, trans_ref, scale_inv_ref = _reference_cast_transpose(x, dest_dtype, scale)

    amax_out = torch.zeros((), dtype=torch.float32, device=DEVICE)
    cast_out, trans_out, scale_inv = cast_transpose_fp8_triton(x, dest_dtype, scale, amax_out)

    assert cast_out.shape == x.shape
    assert trans_out.shape == (shape[1], shape[0])
    torch.testing.assert_close(cast_out, cast_ref, atol=0, rtol=0)
    torch.testing.assert_close(trans_out, trans_ref, atol=0, rtol=0)
    torch.testing.assert_close(scale_inv, scale_inv_ref, **get_tolerances(torch.float32))


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3])
@pytest.mark.parametrize("shape", [(64, 64), (100, 200)])
def test_cast_transpose_fp8_triton_compile(orig_dtype, dest_dtype, shape):
    """Triton cast_transpose_fp8_triton works under torch.compile(fullgraph=True)."""
    torch.manual_seed(42)
    x = torch.randn(shape, device=DEVICE, dtype=orig_dtype)
    scale = _make_scale(x, dest_dtype)
    amax_buf = torch.zeros((), dtype=torch.float32, device=DEVICE)

    cast_eager, trans_eager, si_eager = cast_transpose_fp8_triton(x, dest_dtype, scale, amax_buf)

    torch._dynamo.reset()

    @torch.compile(fullgraph=True)
    def fn(x, scale, amax_buf):
        return cast_transpose_fp8_triton(x, dest_dtype, scale, amax_buf)

    amax_compiled = torch.zeros((), dtype=torch.float32, device=DEVICE)
    cast_compiled, trans_compiled, si_compiled = fn(x, scale, amax_compiled)

    torch.testing.assert_close(cast_compiled, cast_eager, atol=0, rtol=0)
    torch.testing.assert_close(trans_compiled, trans_eager, atol=0, rtol=0)
    torch.testing.assert_close(si_compiled, si_eager, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Cross-backend comparison
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("shape", [(4096, 3072), (100, 200), (33, 65)])
def test_cast_transpose_fp8_cpp_vs_triton(orig_dtype, dest_dtype, shape):
    """C++ and Triton backends produce identical results."""
    torch.manual_seed(42)
    x = torch.randn(shape, device=DEVICE, dtype=orig_dtype)
    scale = _make_scale(x, dest_dtype)

    amax_cpp = torch.zeros((), dtype=torch.float32, device=DEVICE)
    result_cpp = torch.ops.primus_turbo_cpp_extension.cast_transpose_fp8_fused(x, dest_dtype, scale, amax_cpp)
    cast_cpp, trans_cpp, si_cpp = result_cpp[0], result_cpp[1], result_cpp[2]

    amax_triton = torch.zeros((), dtype=torch.float32, device=DEVICE)
    cast_triton, trans_triton, si_triton = cast_transpose_fp8_triton(x, dest_dtype, scale, amax_triton)

    torch.testing.assert_close(cast_cpp, cast_triton, atol=0, rtol=0)
    torch.testing.assert_close(trans_cpp, trans_triton, atol=0, rtol=0)
    torch.testing.assert_close(si_cpp, si_triton, **get_tolerances(torch.float32))
    torch.testing.assert_close(amax_cpp, amax_triton, **get_tolerances(torch.float32))
