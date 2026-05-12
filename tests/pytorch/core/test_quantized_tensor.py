###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import (
    MXFP4_BLOCK_SIZE,
    MXFP8_BLOCK_SIZE,
    ScalingGranularity,
    ScalingRecipe,
    check_mxfp4_support,
    check_mxfp8_support,
    float4_e2m1fn_x2,
    float8_e4m3,
)
from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor
from tests.pytorch.test_utils import get_tolerances

MXFP8_SUPPORT, _ = check_mxfp8_support()
MXFP4_SUPPORT, _ = check_mxfp4_support()

DEVICE = "cuda"
M, N = 256, 512
BLOCK_SIZE = 128


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
_GRAN_1D_CASES = set(
    [
        (ScalingGranularity.TENSORWISE, None, float8_e4m3),
        (ScalingGranularity.ROWWISE, None, float8_e4m3),
        (ScalingGranularity.BLOCKWISE, BLOCK_SIZE, float8_e4m3),
    ]
)

if MXFP8_SUPPORT:
    _GRAN_1D_CASES.add((ScalingGranularity.MX_BLOCKWISE, MXFP8_BLOCK_SIZE, float8_e4m3))

if MXFP4_SUPPORT:
    _GRAN_1D_CASES.add((ScalingGranularity.MX_BLOCKWISE, MXFP4_BLOCK_SIZE, float4_e2m1fn_x2))


# =====================================================================
# Basic construction & properties (all granularities)
# =====================================================================
class TestQuantizedTensorBasic:

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_1D_CASES)
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_create_and_properties(self, granularity, block_size, dest_dtype, dtype):
        x = torch.randn(M, N, dtype=dtype, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, dest_dtype=dest_dtype, block_size=block_size)

        assert qt.dtype == dtype
        assert qt.real_dtype == dest_dtype
        assert qt.data is not None
        assert qt.scale_inv is not None
        assert qt.shape == torch.Size([M, N])
        assert qt.device.type == "cuda"
        assert qt.granularity == granularity
        assert qt.block_size == block_size

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_1D_CASES)
    def test_scale_shape(self, granularity, block_size, dest_dtype):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, dest_dtype=dest_dtype, block_size=block_size)

        expected = _expected_scale_shape(granularity, M, N, block_size, use_2d_block=False)
        assert (
            tuple(qt.scale_inv.shape) == expected
        ), f"{granularity.name}/{dest_dtype}: expected scale shape {expected}, got {tuple(qt.scale_inv.shape)}"

    @pytest.mark.parametrize(
        "granularity,block_size,dest_dtype", [(ScalingGranularity.BLOCKWISE, BLOCK_SIZE, float8_e4m3)]
    )
    def test_blockwise_2d_weight(self, granularity, block_size, dest_dtype):
        """BLOCKWISE with use_2d_block=True produces a (M/bs, N/bs) weight-style scale."""
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(
            x,
            granularity=granularity,
            dest_dtype=dest_dtype,
            block_size=block_size,
            use_2d_block=True,
        )

        assert qt.data.dtype == dest_dtype
        assert qt.data.shape == torch.Size([M, N])
        assert tuple(qt.scale_inv.shape) == (M // BLOCK_SIZE, N // BLOCK_SIZE)


# =====================================================================
# Dequantization accuracy
# Uses the same tolerance policy as tests/pytorch/ops/test_quantization.py:
#   torch.testing.assert_close(..., **get_tolerances(dest_dtype))
# NOTE: dequantize_fp8 currently only supports TENSORWISE and MX_BLOCKWISE.
# ROWWISE / BLOCKWISE dequantize kernels are not implemented yet.
# =====================================================================
_DEQUANT_CASES = set(
    [
        (ScalingGranularity.TENSORWISE, None, float8_e4m3),
        # TODO(ruibin): rowwise and blockwise dequantization is not implemented yet.
    ]
)

if MXFP8_SUPPORT:
    _DEQUANT_CASES.add((ScalingGranularity.MX_BLOCKWISE, MXFP8_BLOCK_SIZE, float8_e4m3))

if MXFP4_SUPPORT:
    _DEQUANT_CASES.add((ScalingGranularity.MX_BLOCKWISE, MXFP4_BLOCK_SIZE, float4_e2m1fn_x2))


class TestDequantize:

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _DEQUANT_CASES)
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

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_1D_CASES)
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

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_1D_CASES)
    def test_no_trans_cache_leaves_fields_none(self, granularity, block_size, dest_dtype):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(
            x, granularity=granularity, block_size=block_size, dest_dtype=dest_dtype, keep_trans_cache=False
        )

        assert qt._data_t is None
        assert qt._scale_inv_t is None

    @pytest.mark.parametrize(
        "granularity,block_size,dest_dtype",
        _GRAN_1D_CASES,
    )
    def test_trans_scale_shape_matches_axis0(self, granularity, block_size, dest_dtype):
        """For non-TENSORWISE, the cached transpose corresponds to axis=0 quantization,
        so the scale shape should reflect the column direction."""
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(
            x, granularity=granularity, block_size=block_size, dest_dtype=dest_dtype, keep_trans_cache=True
        )

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

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_1D_CASES)
    def test_repr_no_trans(self, granularity, block_size, dest_dtype):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, block_size=block_size, dest_dtype=dest_dtype)

        r = repr(qt)
        assert "QuantizedTensor" in r
        assert granularity.name in r
        assert "has_trans_cache" not in r

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_1D_CASES)
    def test_repr_with_trans(self, granularity, block_size, dest_dtype):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(
            x, granularity=granularity, block_size=block_size, dest_dtype=dest_dtype, keep_trans_cache=True
        )

        r = repr(qt)
        assert "has_trans_cache" in r


# =====================================================================
# Serialisation (flatten / unflatten)
# =====================================================================
class TestSerialization:

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_1D_CASES)
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

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_1D_CASES)
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
# View / reshape  (rank-preserving only)
# =====================================================================

_VIEW_GRAN_2D_CASES = set(
    [
        (ScalingGranularity.TENSORWISE, None, float8_e4m3),
        (ScalingGranularity.ROWWISE, None, float8_e4m3),
        (ScalingGranularity.BLOCKWISE, BLOCK_SIZE, float8_e4m3),
    ]
)

_VIEW_GRAN_3D_CASES = set(
    [
        (ScalingGranularity.TENSORWISE, None, float8_e4m3),
        (ScalingGranularity.ROWWISE, None, float8_e4m3),
        (ScalingGranularity.BLOCKWISE, BLOCK_SIZE, float8_e4m3),
    ]
)

if MXFP8_SUPPORT:
    _VIEW_GRAN_2D_CASES.add((ScalingGranularity.MX_BLOCKWISE, MXFP8_BLOCK_SIZE, float8_e4m3))

if MXFP4_SUPPORT:
    _VIEW_GRAN_2D_CASES.add((ScalingGranularity.MX_BLOCKWISE, MXFP4_BLOCK_SIZE, float4_e2m1fn_x2))


class TestViewFunc:
    """View / reshape on a 2D wrapper (``[M, N]``).

    ``M, N = 256, 512`` from the module-level constants — both multiples of
    every block size used here, so no inner-dim padding kicks in.
    """

    # ---- Happy path: same-shape view ----------------------------------
    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _VIEW_GRAN_2D_CASES)
    def test_view_2d_same_shape_identity(self, granularity, block_size, dest_dtype):
        """Viewing to the wrapper's own shape returns the same data /
        scale buffers via the early-return path in
        ``_view_data_and_transpose``."""
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, block_size=block_size, dest_dtype=dest_dtype)

        viewed = qt.view(M, N)

        assert viewed.shape == qt.shape
        assert viewed.orig_ndim == qt.orig_ndim == 2
        # Early-return path: same underlying buffers.
        assert viewed._data.data_ptr() == qt._data.data_ptr()
        assert viewed._scale_inv.data_ptr() == qt._scale_inv.data_ptr()

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _VIEW_GRAN_2D_CASES)
    def test_view_2d_metadata_propagated(self, granularity, block_size, dest_dtype):
        """All wrapper metadata (granularity, dtypes, block_size,
        orig_ndim, keep_trans_cache flag) survives a view."""
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(
            x, granularity=granularity, block_size=block_size, dest_dtype=dest_dtype, keep_trans_cache=True
        )

        viewed = qt.view(M, N)

        assert viewed.orig_ndim == qt.orig_ndim == 2
        assert viewed._granularity == qt._granularity
        assert viewed._block_size == qt._block_size
        assert viewed._orig_dtype == qt._orig_dtype
        assert viewed._dest_dtype == qt._dest_dtype
        assert viewed._keep_trans_cache == qt._keep_trans_cache

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _VIEW_GRAN_2D_CASES)
    def test_view_2d_trans_cache_propagated(self, granularity, block_size, dest_dtype):
        """When the source wrapper has a transpose cache populated, the
        viewed wrapper inherits the same buffers (no re-quantization)."""
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(
            x, granularity=granularity, block_size=block_size, dest_dtype=dest_dtype, keep_trans_cache=True
        )

        viewed = qt.view(M, N)

        if granularity == ScalingGranularity.TENSORWISE:
            # TENSORWISE doesn't populate _data_t / _scale_inv_t (no
            # separate column-quantization is needed for a global scale).
            return
        assert viewed._data_t is not None and qt._data_t is not None
        assert viewed._data_t.data_ptr() == qt._data_t.data_ptr()
        assert viewed._scale_inv_t.data_ptr() == qt._scale_inv_t.data_ptr()


class TestViewFunc3D:
    """View / reshape on a 3D wrapper (``[G, K, N]`` — stacked weights).

    Only TENSORWISE / ROWWISE are exercised here; MX_BLOCKWISE is 2D-only at
    the kernel layer.  BLOCKWISE on 3D requires ``use_2d_block=True`` (weight
    kernel) and is covered separately in :class:`TestQuantizedTensor3DBlockwise`.
    """

    G_, K_, N_ = 4, 128, 256

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _VIEW_GRAN_3D_CASES)
    def test_view_3d_same_shape_identity(self, granularity, block_size, dest_dtype):
        x = torch.randn(self.G_, self.K_, self.N_, dtype=torch.bfloat16, device=DEVICE)
        if granularity == ScalingGranularity.BLOCKWISE:
            use_2d_block = True
        else:
            use_2d_block = False
        qt = _make_quantized_tensor(
            x,
            granularity=granularity,
            block_size=block_size,
            dest_dtype=dest_dtype,
            use_2d_block=use_2d_block,
        )

        viewed = qt.view(self.G_, self.K_, self.N_)

        assert viewed.shape == qt.shape
        assert viewed.orig_ndim == qt.orig_ndim == 3
        assert viewed._data.data_ptr() == qt._data.data_ptr()
        assert viewed._scale_inv.data_ptr() == qt._scale_inv.data_ptr()

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _VIEW_GRAN_3D_CASES)
    def test_view_3d_metadata_propagated(self, granularity, block_size, dest_dtype):
        x = torch.randn(self.G_, self.K_, self.N_, dtype=torch.bfloat16, device=DEVICE)
        if granularity == ScalingGranularity.BLOCKWISE:
            use_2d_block = True
        else:
            use_2d_block = False
        qt = _make_quantized_tensor(
            x,
            granularity=granularity,
            block_size=block_size,
            dest_dtype=dest_dtype,
            use_2d_block=use_2d_block,
        )

        viewed = qt.view(self.G_, self.K_, self.N_)

        assert viewed.orig_ndim == 3
        assert viewed._granularity == qt._granularity
        assert viewed._block_size == qt._block_size
        assert viewed._orig_dtype == qt._orig_dtype
        assert viewed._dest_dtype == qt._dest_dtype
