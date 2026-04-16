###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    MXFP8_BLOCK_SIZE,
    Float8QuantConfig,
    Format,
    MXScalingRecipe,
    ScaleDtype,
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
    is_fp8_dtype,
)
from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl
from primus_turbo.pytorch.ops.quantization import quantize_fp8_with_trans

DEVICE = "cuda"
M, N = 256, 512


def _make_config(granularity, block_size=None):
    if granularity == ScalingGranularity.MX_BLOCKWISE:
        return Float8QuantConfig(
            granularity=granularity,
            block_size=block_size or MXFP8_BLOCK_SIZE,
            scale_dtype=ScaleDtype.E8M0,
        )
    return Float8QuantConfig(granularity=granularity, block_size=block_size)


# =====================================================================
# QuantizedGemm — autograd Function that accepts QuantizedTensor
# =====================================================================
def _get_fp8_dtype(fmt: Format, is_fwd: bool):
    if fmt == Format.E4M3:
        return float8_e4m3
    elif fmt == Format.E5M2:
        return float8_e5m2
    elif fmt == Format.HYBRID:
        return float8_e4m3 if is_fwd else float8_e5m2
    raise ValueError(f"Unsupported format: {fmt}")


class QuantizedGemmFunction(torch.autograd.Function):
    """GEMM that accepts QuantizedTensor inputs (with_trans=True, NT layout).

    Forward:  C = A_fp8 @ B_fp8^T
    Backward: dA = dC_fp8 @ B_t_fp8^T  (bf16),  dB = dC_t_fp8 @ A_t_fp8^T  (bf16)
    """

    @staticmethod
    def forward(ctx, a: QuantizedTensor, b: QuantizedTensor, out_dtype: torch.dtype):
        out = gemm_fp8_impl(
            a._data,
            a._scale,
            False,
            b._data,
            b._scale,
            True,
            out_dtype,
            False,
            granularity=a._config.granularity.value,
            default_backend=BackendType.TURBO.value,
        )

        ctx.save_for_backward(a._data_t, a._scale_t, b._data_t, b._scale_t)
        ctx.out_dtype = out_dtype
        ctx.config = a._config
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a_t_fp8, a_t_scale, b_t_fp8, b_t_scale = ctx.saved_tensors
        cfg = ctx.config
        grad_dtype = _get_fp8_dtype(cfg.format, False)

        grad_out = grad_out.contiguous()
        go_fp8, go_scale, go_t_fp8, go_t_scale = quantize_fp8_with_trans(
            grad_out,
            grad_dtype,
            cfg.granularity,
            block_size=cfg.block_size,
        )

        grad_a = gemm_fp8_impl(
            go_fp8,
            go_scale,
            False,
            b_t_fp8,
            b_t_scale,
            True,
            ctx.out_dtype,
            False,
            granularity=cfg.granularity.value,
            default_backend=BackendType.TURBO.value,
        )

        grad_b = gemm_fp8_impl(
            go_t_fp8,
            go_t_scale,
            False,
            a_t_fp8,
            a_t_scale,
            True,
            ctx.out_dtype,
            False,
            granularity=cfg.granularity.value,
            default_backend=BackendType.TURBO.value,
        )

        return grad_a, grad_b, None


def quantized_gemm(a: QuantizedTensor, b: QuantizedTensor, out_dtype=None):
    """GEMM with pre-quantized QuantizedTensor inputs (NT layout).

    Args:
        a: QuantizedTensor [M, K] with with_trans=True
        b: QuantizedTensor [N, K] with with_trans=True
        out_dtype: Output dtype, defaults to a's original dtype
    """
    assert isinstance(a, QuantizedTensor), "a must be a QuantizedTensor"
    assert isinstance(b, QuantizedTensor), "b must be a QuantizedTensor"
    assert a._data_t is not None, "a must be created with with_trans=True"
    assert b._data_t is not None, "b must be created with with_trans=True"

    if out_dtype is None:
        out_dtype = a._orig_dtype

    return QuantizedGemmFunction.apply(a, b, out_dtype)


# =====================================================================
# Basic construction & properties
# =====================================================================
class TestQuantizedTensorBasic:

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_create_and_properties(self, dtype):
        x = torch.randn(M, N, dtype=dtype, device=DEVICE)
        cfg = _make_config(ScalingGranularity.MX_BLOCKWISE)
        qt = QuantizedTensor(x, float8_e4m3, cfg)

        assert qt.dtype == dtype
        assert is_fp8_dtype(qt.real_dtype)
        assert qt.data is not None
        assert qt.scale is not None
        assert qt.shape == torch.Size([M, N])
        assert qt.device.type == "cuda"

    def test_data_is_fp8(self):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        cfg = _make_config(ScalingGranularity.MX_BLOCKWISE)
        qt = QuantizedTensor(x, float8_e4m3, cfg)

        assert is_fp8_dtype(qt.data.dtype)
        assert qt.data.shape == torch.Size([M, N])

    def test_scale_shape_mx_blockwise(self):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        cfg = _make_config(ScalingGranularity.MX_BLOCKWISE)
        qt = QuantizedTensor(x, float8_e4m3, cfg)

        assert qt.scale.shape[0] == M
        assert qt.scale.shape[1] == N // MXFP8_BLOCK_SIZE


# =====================================================================
# Dequantization accuracy
# =====================================================================
class TestDequantize:

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_dequantize_roundtrip(self, dtype):
        x = torch.randn(M, N, dtype=dtype, device=DEVICE)
        cfg = _make_config(ScalingGranularity.MX_BLOCKWISE)
        qt = QuantizedTensor(x, float8_e4m3, cfg)

        x_recon = qt.dequantize()
        assert x_recon.dtype == dtype
        assert x_recon.shape == x.shape

        mean_abs_err = (x - x_recon).abs().mean().item()
        assert mean_abs_err < 0.1, f"Mean abs error too high: {mean_abs_err}"

    def test_dequantize_snr(self):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        cfg = _make_config(ScalingGranularity.MX_BLOCKWISE)
        qt = QuantizedTensor(x, float8_e4m3, cfg)

        x_recon = qt.dequantize()
        noise = (x - x_recon).float()
        signal_power = x.float().pow(2).mean()
        noise_power = noise.pow(2).mean()
        snr_db = 10 * torch.log10(signal_power / noise_power).item()
        assert snr_db > 20, f"SNR too low: {snr_db:.2f} dB"


# =====================================================================
# with_trans (dual quantization)
# =====================================================================
class TestWithTrans:

    def test_with_trans_mx_blockwise(self):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        cfg = _make_config(ScalingGranularity.MX_BLOCKWISE)
        qt = QuantizedTensor(x, float8_e4m3, cfg, with_trans=True)

        data_t, scale_t = qt.t
        assert data_t is not None
        assert scale_t is not None
        assert is_fp8_dtype(data_t.dtype)

    def test_without_trans_raises(self):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        cfg = _make_config(ScalingGranularity.MX_BLOCKWISE)
        qt = QuantizedTensor(x, float8_e4m3, cfg, with_trans=False)

        with pytest.raises(AssertionError):
            _ = qt.t


# =====================================================================
# Repr
# =====================================================================
class TestRepr:

    def test_repr_no_trans(self):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        cfg = _make_config(ScalingGranularity.MX_BLOCKWISE)
        qt = QuantizedTensor(x, float8_e4m3, cfg)

        r = repr(qt)
        assert "QuantizedTensor" in r
        assert "MX_BLOCKWISE" in r
        assert "has_transpose" not in r

    def test_repr_with_trans(self):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        cfg = _make_config(ScalingGranularity.MX_BLOCKWISE)
        qt = QuantizedTensor(x, float8_e4m3, cfg, with_trans=True)

        r = repr(qt)
        assert "has_transpose" in r


# =====================================================================
# Serialisation (flatten / unflatten)
# =====================================================================
class TestSerialization:

    def test_flatten_unflatten_roundtrip(self):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        cfg = _make_config(ScalingGranularity.MX_BLOCKWISE)
        qt = QuantizedTensor(x, float8_e4m3, cfg)

        keys, metadata = qt.__tensor_flatten__()
        assert "_data" in keys
        assert "_scale" in keys

        inner = {"_data": qt._data, "_scale": qt._scale}
        qt2 = QuantizedTensor.__tensor_unflatten__(inner, metadata, qt.shape, qt.stride())
        assert qt2.dtype == qt.dtype
        assert qt2.shape == qt.shape
        assert torch.equal(qt2._data, qt._data)
        assert torch.equal(qt2._scale, qt._scale)

    def test_flatten_unflatten_with_trans(self):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        cfg = _make_config(ScalingGranularity.MX_BLOCKWISE)
        qt = QuantizedTensor(x, float8_e4m3, cfg, with_trans=True)

        keys, metadata = qt.__tensor_flatten__()
        assert "_data_t" in keys
        assert "_scale_t" in keys

        inner = {
            "_data": qt._data,
            "_scale": qt._scale,
            "_data_t": qt._data_t,
            "_scale_t": qt._scale_t,
        }
        qt2 = QuantizedTensor.__tensor_unflatten__(inner, metadata, qt.shape, qt.stride())
        assert torch.equal(qt2._data_t, qt._data_t)
        assert torch.equal(qt2._scale_t, qt._scale_t)


# =====================================================================
# QuantizedGemm autograd tests
# =====================================================================
class TestQuantizedGemm:

    def _quantize(self, x, with_2d_block=False):
        cfg = _make_config(ScalingGranularity.MX_BLOCKWISE)
        recipe = MXScalingRecipe(use_2d_block=with_2d_block) if with_2d_block else None
        return QuantizedTensor(
            x,
            float8_e4m3,
            cfg,
            with_trans=True,
            scaling_recipe=recipe,
            scaling_recipe_for_trans=recipe,
        )

    @pytest.mark.parametrize("M_,K,N_", [(256, 512, 256), (128, 384, 640)])
    def test_forward_shape(self, M_, K, N_):
        a = torch.randn(M_, K, dtype=torch.bfloat16, device=DEVICE)
        b = torch.randn(N_, K, dtype=torch.bfloat16, device=DEVICE)
        qt_a = self._quantize(a)
        qt_b = self._quantize(b, with_2d_block=True)

        out = quantized_gemm(qt_a, qt_b)
        assert out.shape == (M_, N_)
        assert out.dtype == torch.bfloat16

    def test_forward_accuracy(self):
        K = 512
        a = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)
        b = torch.randn(N, K, dtype=torch.bfloat16, device=DEVICE)
        qt_a = self._quantize(a)
        qt_b = self._quantize(b, with_2d_block=True)

        out = quantized_gemm(qt_a, qt_b)
        ref = torch.mm(a, b.t())

        noise = out.float() - ref.float()
        signal_power = ref.float().pow(2).mean()
        noise_power = noise.pow(2).mean()
        snr = 10 * torch.log10(signal_power / noise_power).item()
        assert snr > 15, f"Forward SNR too low: {snr:.2f} dB"

    def test_backward_runs(self):
        K = 512
        a = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        b = torch.randn(N, K, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        qt_a = self._quantize(a)
        qt_b = self._quantize(b, with_2d_block=True)

        out = quantized_gemm(qt_a, qt_b)
        assert out.grad_fn is not None
        loss = out.sum()
        loss.backward()

    def test_backward_grad_shape(self):
        K = 512
        a = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        b = torch.randn(N, K, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        qt_a = self._quantize(a)
        qt_b = self._quantize(b, with_2d_block=True)
        qt_a.retain_grad()
        qt_b.retain_grad()

        out = quantized_gemm(qt_a, qt_b)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)

        assert qt_a.grad is not None, "qt_a.grad is None"
        assert qt_b.grad is not None, "qt_b.grad is None"
        assert qt_a.grad.shape == (M, K)
        assert qt_b.grad.shape == (N, K)
        assert qt_a.grad.dtype == torch.bfloat16
        assert qt_b.grad.dtype == torch.bfloat16
