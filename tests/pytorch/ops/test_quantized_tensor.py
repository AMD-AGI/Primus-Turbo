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
    float8_e5m2,
    is_fp8_dtype,
)
from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor
from tests.pytorch.test_utils import get_tolerances

DEVICE = "cuda"
M, N = 256, 512
BLOCK_SIZE_1D = 128
BLOCK_SIZE_2D = 128

# FP8 dtypes to cover across tests.
_FP8_DTYPES = [float8_e4m3, float8_e5m2]


def _make_quantized_tensor(
    x: torch.Tensor,
    granularity: ScalingGranularity = ScalingGranularity.MX_BLOCKWISE,
    block_size: int = MXFP8_BLOCK_SIZE,
    dest_dtype: torch.dtype = float8_e4m3,
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
        recipe = ScalingRecipe(use_2d_block=use_2d_block)
        kwargs["block_size"] = block_size
        kwargs["scaling_recipe"] = recipe
        kwargs["scaling_recipe_for_trans"] = recipe
    elif granularity == ScalingGranularity.BLOCKWISE:
        kwargs["block_size"] = block_size
        if use_2d_block:
            kwargs["scaling_recipe"] = ScalingRecipe(use_2d_block=True)
            kwargs["scaling_recipe_for_trans"] = ScalingRecipe(use_2d_block=True)
    # TENSORWISE / ROWWISE: nothing else needed.

    return QuantizedTensor(x, dest_dtype, granularity, **kwargs)


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
    @pytest.mark.parametrize("dest_dtype", _FP8_DTYPES)
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_create_and_properties(self, granularity, block_size, dest_dtype, dtype):
        x = torch.randn(M, N, dtype=dtype, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, dest_dtype=dest_dtype, block_size=block_size)

        assert qt.dtype == dtype
        assert qt.real_dtype == dest_dtype
        assert is_fp8_dtype(qt.real_dtype)
        assert qt._data is not None
        assert qt._scale_inv is not None
        assert qt.shape == torch.Size([M, N])
        assert qt.device.type == "cuda"
        assert qt.granularity == granularity
        assert qt.block_size == block_size

    @pytest.mark.parametrize("granularity,block_size", _GRAN_1D_CASES)
    @pytest.mark.parametrize("dest_dtype", _FP8_DTYPES)
    def test_data_is_fp8(self, granularity, block_size, dest_dtype):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, dest_dtype=dest_dtype, block_size=block_size)

        assert qt._data.dtype == dest_dtype
        assert qt._data.shape == torch.Size([M, N])

    @pytest.mark.parametrize("granularity,block_size", _GRAN_1D_CASES)
    @pytest.mark.parametrize("dest_dtype", _FP8_DTYPES)
    def test_scale_shape(self, granularity, block_size, dest_dtype):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, dest_dtype=dest_dtype, block_size=block_size)

        expected = _expected_scale_shape(granularity, M, N, block_size, use_2d_block=False)
        assert (
            tuple(qt._scale_inv.shape) == expected
        ), f"{granularity.name}/{dest_dtype}: expected scale shape {expected}, got {tuple(qt._scale_inv.shape)}"

    @pytest.mark.parametrize("dest_dtype", _FP8_DTYPES)
    def test_blockwise_2d_weight(self, dest_dtype):
        """BLOCKWISE with use_2d_block=True produces a (M/bs, N/bs) weight-style scale."""
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(
            x,
            granularity=ScalingGranularity.BLOCKWISE,
            dest_dtype=dest_dtype,
            block_size=BLOCK_SIZE_2D,
            use_2d_block=True,
        )

        assert qt._data.dtype == dest_dtype
        assert qt._data.shape == torch.Size([M, N])
        assert tuple(qt._scale_inv.shape) == (M // BLOCK_SIZE_2D, N // BLOCK_SIZE_2D)


# =====================================================================
# Dequantization accuracy
# Uses the same tolerance policy as tests/pytorch/ops/test_quantization.py:
#   torch.testing.assert_close(..., **get_tolerances(dest_dtype))
# NOTE: dequantize_fp8 currently only supports TENSORWISE and MX_BLOCKWISE.
# ROWWISE / BLOCKWISE dequantize kernels are not implemented yet.
# =====================================================================
_DEQUANT_CASES = [
    (ScalingGranularity.TENSORWISE, None),
    (ScalingGranularity.MX_BLOCKWISE, MXFP8_BLOCK_SIZE),
]


class TestDequantize:

    @pytest.mark.parametrize("granularity,block_size", _DEQUANT_CASES)
    @pytest.mark.parametrize("dest_dtype", _FP8_DTYPES)
    def test_dequantize_roundtrip_close(self, granularity, block_size, dest_dtype):
        """Quant -> dequant roundtrip should be close to the original within
        FP8 tolerances (rtol=atol=1e-1), matching test_quantization.py policy.
        """
        torch.manual_seed(42)
        x = torch.rand(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, dest_dtype=dest_dtype, block_size=block_size)

        x_recon = qt.dequantize()
        torch.testing.assert_close(x_recon, x, **get_tolerances(dest_dtype))


# =====================================================================
# keep_trans_cache (dual quantization)
# =====================================================================
class TestKeepTransCache:

    @pytest.mark.parametrize("granularity,block_size", _GRAN_1D_CASES)
    @pytest.mark.parametrize("dest_dtype", _FP8_DTYPES)
    def test_keep_trans_cache_populates_fields(self, granularity, block_size, dest_dtype):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(
            x,
            granularity=granularity,
            dest_dtype=dest_dtype,
            block_size=block_size,
            keep_trans_cache=True,
        )

        assert qt._data_t is not None
        assert qt._scale_inv_t is not None
        assert qt._data_t.dtype == dest_dtype

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
    @pytest.mark.parametrize("dest_dtype", _FP8_DTYPES)
    def test_flatten_unflatten_roundtrip(self, granularity, block_size, dest_dtype):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, dest_dtype=dest_dtype, block_size=block_size)

        keys, metadata = qt.__tensor_flatten__()
        assert "_data" in keys
        assert "_scale_inv" in keys
        assert metadata["_granularity"] == granularity
        assert metadata["_block_size"] == block_size

        inner = {"_data": qt._data, "_scale_inv": qt._scale_inv}
        qt2 = QuantizedTensor.__tensor_unflatten__(inner, metadata, qt.shape, qt.stride())
        assert qt2.dtype == qt.dtype
        assert qt2.real_dtype == dest_dtype
        assert qt2.shape == qt.shape
        assert qt2.granularity == qt.granularity
        assert qt2.block_size == qt.block_size
        assert torch.equal(qt2._data, qt._data)
        assert torch.equal(qt2._scale_inv, qt._scale_inv)

    @pytest.mark.parametrize("granularity,block_size", _GRAN_1D_CASES)
    @pytest.mark.parametrize("dest_dtype", _FP8_DTYPES)
    def test_flatten_unflatten_with_trans(self, granularity, block_size, dest_dtype):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(
            x,
            granularity=granularity,
            dest_dtype=dest_dtype,
            block_size=block_size,
            keep_trans_cache=True,
        )

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


# =====================================================================
# from_tensor factory
# =====================================================================
class TestFromTensor:
    """Tests for ``QuantizedTensor.from_tensor`` which wraps pre-quantized
    data + scale_inv without re-running the quantization kernels."""

    @pytest.mark.parametrize("granularity,block_size", _GRAN_1D_CASES)
    @pytest.mark.parametrize("dest_dtype", _FP8_DTYPES)
    @pytest.mark.parametrize("orig_dtype", [torch.bfloat16, torch.float16])
    def test_basic_properties(self, granularity, block_size, dest_dtype, orig_dtype):
        """from_tensor should produce correct dtype / real_dtype / shape /
        device / granularity / block_size."""
        x = torch.randn(M, N, dtype=orig_dtype, device=DEVICE)
        qt_ref = _make_quantized_tensor(
            x,
            granularity=granularity,
            dest_dtype=dest_dtype,
            block_size=block_size,
        )

        scaling_recipe = qt_ref._scaling_recipe
        qt = QuantizedTensor.from_tensor(
            data=qt_ref._data,
            scale_inv=qt_ref._scale_inv,
            orig_size=torch.Size([M, N]),
            orig_dtype=orig_dtype,
            granularity=granularity,
            block_size=block_size,
            scaling_recipe=scaling_recipe,
        )

        assert qt.dtype == orig_dtype
        assert qt.real_dtype == dest_dtype
        assert qt.shape == torch.Size([M, N])
        assert qt.device.type == "cuda"
        assert qt.granularity == granularity
        assert qt.block_size == block_size
        assert qt.scaling_recipe == scaling_recipe

    @pytest.mark.parametrize("granularity,block_size", _GRAN_1D_CASES)
    @pytest.mark.parametrize("dest_dtype", _FP8_DTYPES)
    def test_data_and_scale_preserved(self, granularity, block_size, dest_dtype):
        """The internal _data / _scale_inv should be exactly the tensors we
        passed in (no copy, no re-quantization)."""
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt_ref = _make_quantized_tensor(
            x,
            granularity=granularity,
            dest_dtype=dest_dtype,
            block_size=block_size,
        )

        qt = QuantizedTensor.from_tensor(
            data=qt_ref._data,
            scale_inv=qt_ref._scale_inv,
            orig_size=torch.Size([M, N]),
            orig_dtype=torch.bfloat16,
            granularity=granularity,
            block_size=block_size,
            scaling_recipe=qt_ref._scaling_recipe,
        )

        assert qt._data.data_ptr() == qt_ref._data.data_ptr()
        assert qt._scale_inv.data_ptr() == qt_ref._scale_inv.data_ptr()

    def test_orig_size_differs_from_data_size(self):
        """When the physical data has been padded (e.g. MX_BLOCKWISE), the
        wrapper shape should reflect orig_size, not data.size()."""
        orig_m, orig_k = 100, 200
        padded_k = 224  # ceil(200/32)*32 = 224
        data = torch.zeros(orig_m, padded_k, dtype=float8_e4m3, device=DEVICE)
        scale_inv = torch.ones(orig_m, padded_k // MXFP8_BLOCK_SIZE, dtype=torch.uint8, device=DEVICE)

        qt = QuantizedTensor.from_tensor(
            data=data,
            scale_inv=scale_inv,
            orig_size=torch.Size([orig_m, orig_k]),
            orig_dtype=torch.bfloat16,
            granularity=ScalingGranularity.MX_BLOCKWISE,
            block_size=MXFP8_BLOCK_SIZE,
            scaling_recipe=ScalingRecipe(),
        )

        assert qt.shape == torch.Size([orig_m, orig_k])
        assert qt._data.shape == torch.Size([orig_m, padded_k])

    @pytest.mark.parametrize("granularity,block_size", _DEQUANT_CASES)
    @pytest.mark.parametrize("dest_dtype", _FP8_DTYPES)
    def test_dequantize_matches_normal_construction(self, granularity, block_size, dest_dtype):
        """A from_tensor QuantizedTensor should dequantize identically to
        one created via the normal __new__ path."""
        torch.manual_seed(42)
        x = torch.rand(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt_ref = _make_quantized_tensor(
            x,
            granularity=granularity,
            dest_dtype=dest_dtype,
            block_size=block_size,
        )

        qt = QuantizedTensor.from_tensor(
            data=qt_ref._data,
            scale_inv=qt_ref._scale_inv,
            orig_size=qt_ref.shape,
            orig_dtype=torch.bfloat16,
            granularity=granularity,
            block_size=block_size,
            scaling_recipe=qt_ref._scaling_recipe,
        )

        x_ref = qt_ref.dequantize()
        x_ft = qt.dequantize()
        torch.testing.assert_close(x_ft, x_ref, rtol=0, atol=0)

    @pytest.mark.parametrize("granularity,block_size", _GRAN_1D_CASES)
    @pytest.mark.parametrize("dest_dtype", _FP8_DTYPES)
    def test_flatten_unflatten_roundtrip(self, granularity, block_size, dest_dtype):
        """from_tensor objects should survive flatten → unflatten."""
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt_ref = _make_quantized_tensor(
            x,
            granularity=granularity,
            dest_dtype=dest_dtype,
            block_size=block_size,
        )

        qt = QuantizedTensor.from_tensor(
            data=qt_ref._data,
            scale_inv=qt_ref._scale_inv,
            orig_size=qt_ref.shape,
            orig_dtype=torch.bfloat16,
            granularity=granularity,
            block_size=block_size,
            scaling_recipe=qt_ref._scaling_recipe,
        )

        keys, metadata = qt.__tensor_flatten__()
        inner = {"_data": qt._data, "_scale_inv": qt._scale_inv}
        qt2 = QuantizedTensor.__tensor_unflatten__(inner, metadata, qt.shape, qt.stride())

        assert qt2.shape == qt.shape
        assert qt2.dtype == qt.dtype
        assert qt2.real_dtype == qt.real_dtype
        assert qt2.granularity == qt.granularity
        assert qt2.block_size == qt.block_size
        assert torch.equal(qt2._data, qt._data)
        assert torch.equal(qt2._scale_inv, qt._scale_inv)
