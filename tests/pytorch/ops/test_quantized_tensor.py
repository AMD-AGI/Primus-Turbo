###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import (
    MXFP8_BLOCK_SIZE,
    ScalingGranularity,
    ScalingRecipe,
    float8_e4m3,
    is_fp8_dtype,
)
from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor

DEVICE = "cuda"
M, N = 256, 512
BLOCK_SIZE_1D = 128
BLOCK_SIZE_2D = 128


def _make_quantized_tensor(
    x: torch.Tensor,
    granularity: ScalingGranularity = ScalingGranularity.MX_BLOCKWISE,
    block_size=None,
    keep_trans_cache: bool = False,
    use_2d_block: bool = False,
) -> QuantizedTensor:
    """Unified helper: construct a QuantizedTensor with sensible defaults for each granularity.

    - TENSORWISE: no block_size, no recipe
    - ROWWISE:    no block_size, no recipe
    - BLOCKWISE:  block_size required; `use_2d_block=True` for weight-style 2D blocks
    - MX_BLOCKWISE: block_size + ScalingRecipe required
    """
    kwargs = dict(keep_trans_cache=keep_trans_cache)

    if granularity == ScalingGranularity.MX_BLOCKWISE:
        bs = block_size or MXFP8_BLOCK_SIZE
        recipe = ScalingRecipe(use_2d_block=use_2d_block)
        kwargs["block_size"] = bs
        kwargs["scaling_recipe"] = recipe
        kwargs["scaling_recipe_for_trans"] = recipe
    elif granularity == ScalingGranularity.BLOCKWISE:
        bs = block_size or BLOCK_SIZE_1D
        kwargs["block_size"] = bs
        if use_2d_block:
            kwargs["scaling_recipe"] = ScalingRecipe(use_2d_block=True)
            kwargs["scaling_recipe_for_trans"] = ScalingRecipe(use_2d_block=True)
    # TENSORWISE / ROWWISE: nothing else needed.

    return QuantizedTensor(x, float8_e4m3, granularity, **kwargs)


def _expected_scale_shape(granularity: ScalingGranularity, M_: int, N_: int, block_size, use_2d_block: bool):
    """Return the expected scale_inv shape for a [M, N] tensor at axis=1."""
    if granularity == ScalingGranularity.TENSORWISE:
        return ()
    if granularity == ScalingGranularity.ROWWISE:
        return (M_, 1)
    if granularity == ScalingGranularity.BLOCKWISE:
        if use_2d_block:
            return (M_ // block_size, N_ // block_size)
        return (M_, N_ // block_size)
    if granularity == ScalingGranularity.MX_BLOCKWISE:
        return (M_, N_ // block_size)


# Granularity × block_size combos for parametrization (excluding BLOCKWISE-2D,
# which is weight-only and tested separately).
_GRAN_1D_CASES = [
    (ScalingGranularity.TENSORWISE, None),
    (ScalingGranularity.ROWWISE, None),
    (ScalingGranularity.BLOCKWISE, BLOCK_SIZE_1D),
    (ScalingGranularity.MX_BLOCKWISE, MXFP8_BLOCK_SIZE),
]


# =====================================================================
# Basic construction & properties (all granularities)
# =====================================================================
class TestQuantizedTensorBasic:

    @pytest.mark.parametrize("granularity,block_size", _GRAN_1D_CASES)
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_create_and_properties(self, granularity, block_size, dtype):
        x = torch.randn(M, N, dtype=dtype, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, block_size=block_size)

        assert qt.dtype == dtype
        assert is_fp8_dtype(qt.real_dtype)
        assert qt.data is not None
        assert qt.scale_inv is not None
        assert qt.shape == torch.Size([M, N])
        assert qt.device.type == "cuda"
        assert qt.granularity == granularity
        assert qt.block_size == block_size

    @pytest.mark.parametrize("granularity,block_size", _GRAN_1D_CASES)
    def test_data_is_fp8(self, granularity, block_size):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, block_size=block_size)

        assert is_fp8_dtype(qt.data.dtype)
        assert qt.data.shape == torch.Size([M, N])

    @pytest.mark.parametrize("granularity,block_size", _GRAN_1D_CASES)
    def test_scale_shape(self, granularity, block_size):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, block_size=block_size)

        expected = _expected_scale_shape(granularity, M, N, block_size, use_2d_block=False)
        assert (
            tuple(qt.scale_inv.shape) == expected
        ), f"{granularity.name}: expected scale shape {expected}, got {tuple(qt.scale_inv.shape)}"

    def test_blockwise_2d_weight(self):
        """BLOCKWISE with use_2d_block=True produces a (M/bs, N/bs) weight-style scale."""
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(
            x, granularity=ScalingGranularity.BLOCKWISE, block_size=BLOCK_SIZE_2D, use_2d_block=True
        )

        assert is_fp8_dtype(qt.data.dtype)
        assert qt.data.shape == torch.Size([M, N])
        assert tuple(qt.scale_inv.shape) == (M // BLOCK_SIZE_2D, N // BLOCK_SIZE_2D)


# =====================================================================
# Dequantization accuracy
# NOTE: dequantize_fp8 currently only supports TENSORWISE and MX_BLOCKWISE.
# ROWWISE / BLOCKWISE dequantize kernels are not implemented yet.
# =====================================================================
_DEQUANT_CASES = [
    (ScalingGranularity.TENSORWISE, None, 20),
    (ScalingGranularity.MX_BLOCKWISE, MXFP8_BLOCK_SIZE, 20),
]


class TestDequantize:

    @pytest.mark.parametrize("granularity,block_size,_snr", _DEQUANT_CASES)
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_dequantize_roundtrip(self, granularity, block_size, _snr, dtype):
        x = torch.randn(M, N, dtype=dtype, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, block_size=block_size)

        x_recon = qt.dequantize()
        assert x_recon.dtype == dtype
        assert x_recon.shape == x.shape

        mean_abs_err = (x - x_recon).abs().mean().item()
        assert mean_abs_err < 0.1, f"{granularity.name}: mean abs error too high: {mean_abs_err}"

    @pytest.mark.parametrize("granularity,block_size,snr_threshold", _DEQUANT_CASES)
    def test_dequantize_snr(self, granularity, block_size, snr_threshold):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, block_size=block_size)

        x_recon = qt.dequantize()
        noise = (x - x_recon).float()
        signal_power = x.float().pow(2).mean()
        noise_power = noise.pow(2).mean()
        snr_db = 10 * torch.log10(signal_power / noise_power).item()
        assert snr_db > snr_threshold, f"{granularity.name}: SNR={snr_db:.2f} dB < threshold {snr_threshold}"

    @pytest.mark.parametrize(
        "granularity,block_size",
        [
            (ScalingGranularity.ROWWISE, None),
            (ScalingGranularity.BLOCKWISE, BLOCK_SIZE_1D),
        ],
    )
    def test_dequantize_raises_for_unimplemented(self, granularity, block_size):
        """ROWWISE / BLOCKWISE dequantize kernels aren't implemented yet; verify
        the NotImplementedError is surfaced so users get a clear signal."""
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, block_size=block_size)

        with pytest.raises(NotImplementedError):
            qt.dequantize()


# =====================================================================
# keep_trans_cache (dual quantization)
# =====================================================================
class TestKeepTransCache:

    @pytest.mark.parametrize("granularity,block_size", _GRAN_1D_CASES)
    def test_keep_trans_cache_populates_fields(self, granularity, block_size):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, block_size=block_size, keep_trans_cache=True)

        assert qt._data_t is not None
        assert qt._scale_inv_t is not None
        assert is_fp8_dtype(qt._data_t.dtype)

    @pytest.mark.parametrize("granularity,block_size", _GRAN_1D_CASES)
    def test_no_trans_cache_leaves_fields_none(self, granularity, block_size):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, block_size=block_size, keep_trans_cache=False)

        assert qt._data_t is None
        assert qt._scale_inv_t is None

    @pytest.mark.parametrize(
        "granularity,block_size",
        [
            (ScalingGranularity.ROWWISE, None),
            (ScalingGranularity.BLOCKWISE, BLOCK_SIZE_1D),
            (ScalingGranularity.MX_BLOCKWISE, MXFP8_BLOCK_SIZE),
        ],
    )
    def test_trans_scale_shape_matches_axis0(self, granularity, block_size):
        """For non-TENSORWISE, the cached transpose corresponds to axis=0 quantization,
        so the scale shape should reflect the column direction."""
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, block_size=block_size, keep_trans_cache=True)

        s = qt._scale_inv
        s_t = qt._scale_inv_t
        # axis=0 (col-wise) has a different shape than axis=1 (row-wise)
        # except for corner cases where dimensions coincidentally match.
        if granularity == ScalingGranularity.ROWWISE:
            # axis=1 -> (M, 1); axis=0 -> (1, N)
            assert tuple(s.shape) == (M, 1)
            assert tuple(s_t.shape) == (1, N)
        elif granularity == ScalingGranularity.BLOCKWISE:
            # axis=1 -> (M, N/bs); axis=0 -> (M/bs, N)
            assert tuple(s.shape) == (M, N // block_size)
            assert tuple(s_t.shape) == (M // block_size, N)
        elif granularity == ScalingGranularity.MX_BLOCKWISE:
            # axis=1 -> (M, N/bs); the MX with_trans kernel transposes axis 0.
            assert tuple(s.shape) == (M, N // block_size)
            assert s_t is not None and s_t.numel() > 0


# =====================================================================
# Repr
# =====================================================================
class TestRepr:

    @pytest.mark.parametrize("granularity,block_size", _GRAN_1D_CASES)
    def test_repr_no_trans(self, granularity, block_size):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, block_size=block_size)

        r = repr(qt)
        assert "QuantizedTensor" in r
        assert granularity.name in r
        assert "has_trans_cache" not in r

    @pytest.mark.parametrize("granularity,block_size", _GRAN_1D_CASES)
    def test_repr_with_trans(self, granularity, block_size):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, block_size=block_size, keep_trans_cache=True)

        r = repr(qt)
        assert "has_trans_cache" in r


# =====================================================================
# Serialisation (flatten / unflatten)
# =====================================================================
class TestSerialization:

    @pytest.mark.parametrize("granularity,block_size", _GRAN_1D_CASES)
    def test_flatten_unflatten_roundtrip(self, granularity, block_size):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, block_size=block_size)

        keys, metadata = qt.__tensor_flatten__()
        assert "_data" in keys
        assert "_scale_inv" in keys
        assert metadata["_granularity"] == granularity
        assert metadata["_block_size"] == block_size

        inner = {"_data": qt._data, "_scale_inv": qt._scale_inv}
        qt2 = QuantizedTensor.__tensor_unflatten__(inner, metadata, qt.shape, qt.stride())
        assert qt2.dtype == qt.dtype
        assert qt2.shape == qt.shape
        assert qt2.granularity == qt.granularity
        assert qt2.block_size == qt.block_size
        assert torch.equal(qt2._data, qt._data)
        assert torch.equal(qt2._scale_inv, qt._scale_inv)

    @pytest.mark.parametrize("granularity,block_size", _GRAN_1D_CASES)
    def test_flatten_unflatten_with_trans(self, granularity, block_size):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, block_size=block_size, keep_trans_cache=True)

        keys, metadata = qt.__tensor_flatten__()
        assert "_data_t" in keys
        assert "_scale_inv_t" in keys

        inner = {
            "_data": qt._data,
            "_scale_inv": qt._scale_inv,
            "_data_t": qt._data_t,
            "_scale_inv_t": qt._scale_inv_t,
        }
        qt2 = QuantizedTensor.__tensor_unflatten__(inner, metadata, qt.shape, qt.stride())
        assert torch.equal(qt2._data_t, qt._data_t)
        assert torch.equal(qt2._scale_inv_t, qt._scale_inv_t)
