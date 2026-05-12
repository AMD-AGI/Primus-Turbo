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
    check_mxfp4_support,
    check_mxfp8_support,
    float8_e4m3,
)
from primus_turbo.pytorch.core.quantized_tensor import GroupedQuantizedTensor
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
    group_offs_from_lens,
)
from tests.pytorch.test_utils import get_tolerances

MXFP8_SUPPORT, _ = check_mxfp8_support()
MXFP4_SUPPORT, _ = check_mxfp4_support()

DEVICE = "cuda"

# Packed-M activation:  ``TOTAL_M`` tokens routed across ``NUM_GROUPS`` experts.
# ``GROUP_LENS`` sums to ``TOTAL_M`` so the wrapper sees a contiguous 2D buffer
# of shape ``[TOTAL_M, K]``.
TOTAL_M, K = 256, 512
NUM_GROUPS = 4
BLOCK_SIZE = 128

# Equal-sized groups by default — keeps tests deterministic and avoids
# pathological partitions interfering with the wrapper-level checks.
DEFAULT_GROUP_LENS = [TOTAL_M // NUM_GROUPS] * NUM_GROUPS  # [64, 64, 64, 64]


def _make_group_lens(lens=DEFAULT_GROUP_LENS, device: str = DEVICE) -> torch.Tensor:
    return torch.tensor(lens, dtype=torch.int64, device=device)


def _make_grouped_quantized_tensor(
    x: torch.Tensor,
    group_lens: torch.Tensor,
    granularity: ScalingGranularity = ScalingGranularity.MX_BLOCKWISE,
    block_size: int = MXFP8_BLOCK_SIZE,
    dest_dtype: torch.dtype = float8_e4m3,
    keep_trans_cache: bool = False,
) -> GroupedQuantizedTensor:
    """Unified helper: construct a GroupedQuantizedTensor with sensible
    defaults for each granularity.

    - TENSORWISE: no block_size, no recipe (a single global scale)
    - ROWWISE:    no block_size, no recipe
    - BLOCKWISE:  block_size required (1D blockwise; 2D-block / weight-style
                  blockwise does not apply to packed-M activations)
    - MX_BLOCKWISE: block_size + ScalingRecipe required
    """
    kwargs = dict(keep_trans_cache=keep_trans_cache)

    if granularity == ScalingGranularity.MX_BLOCKWISE:
        recipe = ScalingRecipe()
        kwargs["block_size"] = block_size
        kwargs["scaling_recipe"] = recipe
        kwargs["scaling_recipe_for_trans"] = recipe
    elif granularity == ScalingGranularity.BLOCKWISE:
        kwargs["block_size"] = block_size
    # TENSORWISE / ROWWISE: nothing else needed.

    return GroupedQuantizedTensor(x, dest_dtype, granularity, group_lens, **kwargs)


def _expected_scale_shape(granularity: ScalingGranularity, M_: int, N_: int):
    """Expected ``scale_inv`` shape for a packed-M [M, K] tensor at axis=1.

    Packed-M quantization is currently group-agnostic — scales are computed
    over the whole [M, K] buffer just like a plain 2D tensor.
    """
    if granularity == ScalingGranularity.TENSORWISE:
        return ()
    if granularity == ScalingGranularity.ROWWISE:
        return (M_, 1)
    if granularity == ScalingGranularity.BLOCKWISE:
        return (M_, N_ // BLOCK_SIZE)
    if granularity == ScalingGranularity.MX_BLOCKWISE:
        return (M_, N_ // MXFP8_BLOCK_SIZE)


# Granularity × block_size combos for parametrization.
_GRAN_1D_CASES = set(
    [
        (ScalingGranularity.TENSORWISE, None, float8_e4m3),
        (ScalingGranularity.ROWWISE, None, float8_e4m3),
    ]
)


# =====================================================================
# Basic construction & properties (all granularities)
# =====================================================================
class TestGroupedQuantizedTensorBasic:

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_1D_CASES)
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_create_and_properties(self, granularity, block_size, dest_dtype, dtype):
        x = torch.randn(TOTAL_M, K, dtype=dtype, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_grouped_quantized_tensor(
            x, group_lens, granularity=granularity, dest_dtype=dest_dtype, block_size=block_size
        )

        assert qt.dtype == dtype
        assert qt.real_dtype == dest_dtype
        assert qt.data is not None
        assert qt.scale_inv is not None
        assert qt.shape == torch.Size([TOTAL_M, K])
        assert qt.device.type == "cuda"
        assert qt.granularity == granularity
        assert qt.block_size == block_size
        # GroupedQuantizedTensor is always 2D packed-M.
        assert qt.orig_ndim == 2

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_1D_CASES)
    def test_scale_shape(self, granularity, block_size, dest_dtype):
        x = torch.randn(TOTAL_M, K, dtype=torch.bfloat16, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_grouped_quantized_tensor(
            x, group_lens, granularity=granularity, dest_dtype=dest_dtype, block_size=block_size
        )

        expected = _expected_scale_shape(granularity, TOTAL_M, K)
        assert (
            tuple(qt.scale_inv.shape) == expected
        ), f"{granularity.name}/{dest_dtype}: expected scale shape {expected}, got {tuple(qt.scale_inv.shape)}"


# =====================================================================
# Group-specific behaviour
# =====================================================================
class TestGroupSpecific:
    """Behaviours unique to GroupedQuantizedTensor — group_lens / group_offs
    bookkeeping plus the constructor's hard constraints.
    """

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_1D_CASES)
    def test_group_lens_and_offs(self, granularity, block_size, dest_dtype):
        """``group_lens`` is preserved; ``group_offs`` is derived as a
        cumulative-sum (with a leading 0)."""
        x = torch.randn(TOTAL_M, K, dtype=torch.bfloat16, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_grouped_quantized_tensor(
            x, group_lens, granularity=granularity, block_size=block_size, dest_dtype=dest_dtype
        )

        assert torch.equal(qt.group_lens, group_lens)
        # ``group_offs`` is computed by ``group_offs_from_lens``; check the
        # wrapper's value matches a fresh computation.
        assert torch.equal(qt.group_offs, group_offs_from_lens(group_lens))

    def test_uneven_group_lens(self):
        """Sums to TOTAL_M but with non-uniform group sizes — the wrapper
        must accept this without padding or rebalancing."""
        lens = torch.tensor([20, 80, 60, 96], dtype=torch.int64, device=DEVICE)
        assert int(lens.sum().item()) == TOTAL_M
        x = torch.randn(TOTAL_M, K, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_grouped_quantized_tensor(x, lens, granularity=ScalingGranularity.ROWWISE)
        assert torch.equal(qt.group_lens, lens)
        assert torch.equal(qt.group_offs, group_offs_from_lens(lens))


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
        # TODO(ruibin): rowwise dequantization is not implemented yet.
        # (ScalingGranularity.ROWWISE, None),
    ]
)


class TestDequantize:

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _DEQUANT_CASES)
    def test_dequantize_roundtrip_close(self, granularity, block_size, dest_dtype):
        """Quant -> dequant roundtrip should be close to the original within
        FP8 tolerances (rtol=atol=1e-1), matching test_quantization.py policy.
        """
        torch.manual_seed(42)
        x = torch.rand(TOTAL_M, K, dtype=torch.bfloat16, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_grouped_quantized_tensor(
            x, group_lens, granularity=granularity, dest_dtype=dest_dtype, block_size=block_size
        )

        x_recon = qt.dequantize()
        torch.testing.assert_close(x_recon, x, **get_tolerances(dest_dtype))


# =====================================================================
# keep_trans_cache (dual quantization)
# =====================================================================
class TestKeepTransCache:

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_1D_CASES)
    def test_keep_trans_cache_populates_fields(self, granularity, block_size, dest_dtype):
        x = torch.randn(TOTAL_M, K, dtype=torch.bfloat16, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_grouped_quantized_tensor(
            x,
            group_lens,
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
        x = torch.randn(TOTAL_M, K, dtype=torch.bfloat16, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_grouped_quantized_tensor(
            x,
            group_lens,
            granularity=granularity,
            block_size=block_size,
            dest_dtype=dest_dtype,
            keep_trans_cache=False,
        )

        assert qt._data_t is None
        assert qt._scale_inv_t is None

    @pytest.mark.parametrize(
        "granularity,block_size,dest_dtype",
        _GRAN_1D_CASES,
    )
    def test_trans_scale_shape_matches_axis0(self, granularity, block_size, dest_dtype):
        """For non-TENSORWISE, the cached transpose corresponds to axis=0
        quantization, so the scale shape should reflect the column direction."""
        x = torch.randn(TOTAL_M, K, dtype=torch.bfloat16, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_grouped_quantized_tensor(
            x,
            group_lens,
            granularity=granularity,
            block_size=block_size,
            dest_dtype=dest_dtype,
            keep_trans_cache=True,
        )

        s = qt._scale_inv
        s_t = qt._scale_inv_t
        if granularity == ScalingGranularity.ROWWISE:
            # axis=1 -> (M, 1); axis=0 -> (1, N)
            assert tuple(s.shape) == (TOTAL_M, 1)
            assert tuple(s_t.shape) == (1, K)


# =====================================================================
# Repr
# =====================================================================
class TestRepr:

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_1D_CASES)
    def test_repr_no_trans(self, granularity, block_size, dest_dtype):
        x = torch.randn(TOTAL_M, K, dtype=torch.bfloat16, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_grouped_quantized_tensor(
            x, group_lens, granularity=granularity, block_size=block_size, dest_dtype=dest_dtype
        )

        r = repr(qt)
        assert "GroupedQuantizedTensor" in r
        assert "group_lens" in r
        assert granularity.name in r
        assert "has_trans_cache" not in r

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_1D_CASES)
    def test_repr_with_trans(self, granularity, block_size, dest_dtype):
        x = torch.randn(TOTAL_M, K, dtype=torch.bfloat16, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_grouped_quantized_tensor(
            x,
            group_lens,
            granularity=granularity,
            block_size=block_size,
            dest_dtype=dest_dtype,
            keep_trans_cache=True,
        )

        r = repr(qt)
        assert "GroupedQuantizedTensor" in r
        assert "has_trans_cache" in r


# =====================================================================
# Serialisation (flatten / unflatten)
# =====================================================================
class TestSerialization:

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_1D_CASES)
    def test_flatten_unflatten_roundtrip(self, granularity, block_size, dest_dtype):
        x = torch.randn(TOTAL_M, K, dtype=torch.bfloat16, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_grouped_quantized_tensor(
            x, group_lens, granularity=granularity, dest_dtype=dest_dtype, block_size=block_size
        )

        keys, metadata = qt.__tensor_flatten__()
        assert "_data" in keys
        assert "_scale_inv" in keys
        assert metadata["_granularity"] == granularity
        assert metadata["_block_size"] == block_size
        # GroupedQuantizedTensor must serialise its packed-M metadata.
        assert "_group_lens" in metadata
        assert metadata["_orig_ndim"] == 2

        inner = {"_data": qt._data, "_scale_inv": qt._scale_inv}
        qt2 = GroupedQuantizedTensor.__tensor_unflatten__(inner, metadata, qt.shape, qt.stride())
        assert isinstance(qt2, GroupedQuantizedTensor)
        assert qt2.dtype == qt.dtype
        assert qt2.real_dtype == dest_dtype
        assert qt2.shape == qt.shape
        assert qt2.granularity == qt.granularity
        assert qt2.block_size == qt.block_size
        assert torch.equal(qt2._data, qt._data)
        assert torch.equal(qt2._scale_inv, qt._scale_inv)
        # group metadata round-trips: lens preserved and offs recomputed.
        assert torch.equal(qt2.group_lens, qt.group_lens)
        assert torch.equal(qt2.group_offs, qt.group_offs)

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_1D_CASES)
    def test_flatten_unflatten_with_trans(self, granularity, block_size, dest_dtype):
        x = torch.randn(TOTAL_M, K, dtype=torch.bfloat16, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_grouped_quantized_tensor(
            x,
            group_lens,
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
        qt2 = GroupedQuantizedTensor.__tensor_unflatten__(inner, metadata, qt.shape, qt.stride())
        assert torch.equal(qt2._data_t, qt._data_t)
        assert torch.equal(qt2._scale_inv_t, qt._scale_inv_t)
        assert torch.equal(qt2.group_lens, qt.group_lens)


# =====================================================================
# View / reshape  (rank-preserving only — packed-M is strictly 2D)
# =====================================================================
class TestViewFunc:
    """View / reshape on a GroupedQuantizedTensor (always 2D packed-M).

    The wrapper enforces ``len(target_shape) == orig_ndim == 2``; cross-rank
    views are rejected.  ``group_lens`` / ``group_offs`` must propagate to
    the viewed wrapper through ``_make_like``.
    """

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_1D_CASES)
    def test_view_same_shape_identity(self, granularity, block_size, dest_dtype):
        """Viewing to the wrapper's own shape returns the same data /
        scale buffers via the early-return path in
        ``_view_data_and_transpose``."""
        x = torch.randn(TOTAL_M, K, dtype=torch.bfloat16, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_grouped_quantized_tensor(x, group_lens, granularity=granularity, block_size=block_size)

        viewed = qt.view(TOTAL_M, K)

        assert isinstance(viewed, GroupedQuantizedTensor)
        assert viewed.shape == qt.shape
        assert viewed.orig_ndim == qt.orig_ndim == 2
        # Early-return path: same underlying buffers.
        assert viewed._data.data_ptr() == qt._data.data_ptr()
        assert viewed._scale_inv.data_ptr() == qt._scale_inv.data_ptr()

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_1D_CASES)
    def test_view_metadata_propagated(self, granularity, block_size, dest_dtype):
        """All wrapper metadata (granularity, dtypes, block_size,
        keep_trans_cache flag) survives a view."""
        x = torch.randn(TOTAL_M, K, dtype=torch.bfloat16, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_grouped_quantized_tensor(
            x,
            group_lens,
            granularity=granularity,
            block_size=block_size,
            dest_dtype=dest_dtype,
            keep_trans_cache=True,
        )

        viewed = qt.view(TOTAL_M, K)

        assert viewed.orig_ndim == qt.orig_ndim == 2
        assert viewed._granularity == qt._granularity
        assert viewed._block_size == qt._block_size
        assert viewed._orig_dtype == qt._orig_dtype
        assert viewed._dest_dtype == qt._dest_dtype
        assert viewed._keep_trans_cache == qt._keep_trans_cache

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_1D_CASES)
    def test_view_group_metadata_propagated(self, granularity, block_size, dest_dtype):
        """``group_lens`` / ``group_offs`` flow through ``_make_like`` —
        the viewed wrapper shares the same tensors as the source."""
        x = torch.randn(TOTAL_M, K, dtype=torch.bfloat16, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_grouped_quantized_tensor(
            x, group_lens, granularity=granularity, dest_dtype=dest_dtype, block_size=block_size
        )

        viewed = qt.view(TOTAL_M, K)

        assert viewed.group_lens.data_ptr() == qt.group_lens.data_ptr()
        assert viewed.group_offs.data_ptr() == qt.group_offs.data_ptr()

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_1D_CASES)
    def test_view_trans_cache_propagated(self, granularity, block_size, dest_dtype):
        """When the source wrapper has a transpose cache populated, the
        viewed wrapper inherits the same buffers (no re-quantization)."""
        x = torch.randn(TOTAL_M, K, dtype=torch.bfloat16, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_grouped_quantized_tensor(
            x,
            group_lens,
            granularity=granularity,
            block_size=block_size,
            dest_dtype=dest_dtype,
            keep_trans_cache=True,
        )

        viewed = qt.view(TOTAL_M, K)

        assert viewed._data_t is not None and qt._data_t is not None
        assert viewed._data_t.data_ptr() == qt._data_t.data_ptr()
        assert viewed._scale_inv_t.data_ptr() == qt._scale_inv_t.data_ptr()
