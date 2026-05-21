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
    bias_gelu_cast_transpose_fp8,
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
    """Reference: C++ fused quantize with pre-computed scale, then transpose."""
    fp8_out, scale_inv = torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise_fused(
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


# ---------------------------------------------------------------------------
# Fused bias+GELU+cast_transpose tests
# ---------------------------------------------------------------------------


def _openai_gelu_no_jit(x):
    """GELU tanh approximation matching Flux's activation function."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


FUSED_SHAPES = [
    (4096, 3072),
    (8192, 3072),
    (2048, 3072),
    (8192, 12288),
    (2048, 12288),
    (64, 64),
    (33, 65),
]


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("shape", FUSED_SHAPES)
def test_bias_gelu_cast_transpose_correctness(orig_dtype, dest_dtype, shape):
    """Fused bias+GELU+cast_transpose matches fp32 reference within FP8 tolerance.

    The fused kernel computes bias+GELU in fp32 before quantizing to FP8,
    which is more precise than the unfused path (which truncates to bf16/f16
    between GELU and cast_transpose).  We compare against an fp32 reference
    and allow +-1 ULP of the FP8 type.
    """
    torch.manual_seed(42)
    M, N = shape
    x = torch.randn(shape, device=DEVICE, dtype=orig_dtype)
    bias = torch.randn(N, device=DEVICE, dtype=orig_dtype)

    # fp32 reference: matches what the fused kernel computes internally
    # (load bf16 -> fp32, bias fp32, GELU fp32, scale -> FP8)
    val_f32 = _openai_gelu_no_jit((x.float() + bias.float()))
    scale = _make_scale(val_f32.to(orig_dtype), dest_dtype)
    fp8_max = torch.finfo(dest_dtype).max
    ref_scaled = (val_f32 * scale.item()).clamp(-fp8_max, fp8_max)
    cast_ref = ref_scaled.to(dest_dtype)
    trans_ref = cast_ref.t().contiguous()
    si_ref = torch.tensor(1.0 / scale.item(), dtype=torch.float32, device=DEVICE)

    # Fused kernel
    amax_fused = torch.zeros((), dtype=torch.float32, device=DEVICE)
    cast_fused, trans_fused, si_fused = bias_gelu_cast_transpose_fp8(
        x,
        bias,
        dest_dtype,
        scale,
        amax_fused,
    )

    assert cast_fused.shape == (M, N)
    assert trans_fused.shape == (N, M)
    # libdevice.tanh and torch.tanh can differ at fp32 rounding boundaries,
    # which occasionally flips a single FP8 value.  Allow <=0.001% mismatch.
    diff = (cast_fused.float() - cast_ref.float()).abs()
    mismatch_frac = (diff > 0).float().mean().item()
    assert mismatch_frac < 1e-5, f"Cast mismatch fraction {mismatch_frac:.6f} exceeds 0.001% threshold"
    diff_t = (trans_fused.float() - trans_ref.float()).abs()
    mismatch_frac_t = (diff_t > 0).float().mean().item()
    assert mismatch_frac_t < 1e-5, f"Trans mismatch fraction {mismatch_frac_t:.6f} exceeds 0.001% threshold"
    torch.testing.assert_close(si_fused, si_ref, atol=0, rtol=0)


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3])
@pytest.mark.parametrize("shape", FUSED_SHAPES)
def test_bias_gelu_cast_transpose_amax(orig_dtype, dest_dtype, shape):
    """Fused kernel captures correct amax (of the post-GELU, pre-scale values)."""
    torch.manual_seed(42)
    M, N = shape
    x = torch.randn(shape, device=DEVICE, dtype=orig_dtype)
    bias = torch.randn(N, device=DEVICE, dtype=orig_dtype)

    val_f32 = _openai_gelu_no_jit(x.float() + bias.float())
    expected_amax = val_f32.abs().amax()

    scale = _make_scale(val_f32.to(orig_dtype), dest_dtype)
    amax_out = torch.zeros((), dtype=torch.float32, device=DEVICE)
    bias_gelu_cast_transpose_fp8(x, bias, dest_dtype, scale, amax_out)

    # libdevice.tanh vs torch.tanh can produce slightly different fp32
    # values, which affects amax; use standard fp32 tolerance.
    torch.testing.assert_close(amax_out, expected_amax, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dest_dtype", [turbo.float8_e4m3])
@pytest.mark.parametrize("shape", [(64, 64), (100, 200)])
def test_bias_gelu_cast_transpose_compile(orig_dtype, dest_dtype, shape):
    """Fused bias_gelu_cast_transpose_fp8 works under torch.compile(fullgraph=True)."""
    torch.manual_seed(42)
    M, N = shape
    x = torch.randn(shape, device=DEVICE, dtype=orig_dtype)
    bias = torch.randn(N, device=DEVICE, dtype=orig_dtype)
    scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    amax_buf = torch.zeros((), dtype=torch.float32, device=DEVICE)

    cast_eager, trans_eager, si_eager = bias_gelu_cast_transpose_fp8(
        x,
        bias,
        dest_dtype,
        scale,
        amax_buf,
    )

    torch._dynamo.reset()

    @torch.compile(fullgraph=True)
    def fn(x, bias, scale, amax_buf):
        return bias_gelu_cast_transpose_fp8(x, bias, dest_dtype, scale, amax_buf)

    amax_compiled = torch.zeros((), dtype=torch.float32, device=DEVICE)
    cast_compiled, trans_compiled, si_compiled = fn(x, bias, scale, amax_compiled)

    torch.testing.assert_close(cast_compiled, cast_eager, atol=0, rtol=0)
    torch.testing.assert_close(trans_compiled, trans_eager, atol=0, rtol=0)
    torch.testing.assert_close(si_compiled, si_eager, atol=0, rtol=0)
