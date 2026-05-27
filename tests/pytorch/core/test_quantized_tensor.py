###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

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
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
    group_offs_from_lens,
)
from tests.pytorch.test_utils import get_tolerances

MXFP8_SUPPORT, _ = check_mxfp8_support()
MXFP4_SUPPORT, _ = check_mxfp4_support()

SKIP_MXFP8 = pytest.mark.skipif(not MXFP8_SUPPORT, reason="MXFP8 not supported on this device")
SKIP_MXFP4 = pytest.mark.skipif(not MXFP4_SUPPORT, reason="MXFP4 not supported on this device")

DEVICE = "cuda"
M, N = 256, 512


def _make_quantized_tensor(
    x: torch.Tensor,
    granularity: ScalingGranularity = ScalingGranularity.MX_BLOCKWISE,
    block_size: int = MXFP8_BLOCK_SIZE,
    dest_dtype: torch.dtype = float8_e4m3,
    axis: int = -1,
    use_2d_block: bool = False,
    group_lens: Optional[torch.Tensor] = None,
) -> QuantizedTensor:
    """Unified helper for the ``QuantizedTensor.quantize`` API.

    Defaults are filled in per granularity:

    - TENSORWISE:   no ``block_size`` / ``scaling_recipe`` needed.
    - ROWWISE:      no ``block_size`` / ``scaling_recipe`` needed; ``axis``
                    selects the direction of the per-row scale.
    - MX_BLOCKWISE: ``block_size`` + ``ScalingRecipe`` required.

    BLOCKWISE is **not** supported by ``QuantizedTensor`` — callers should
    use the low-level ``quant_fp8_blockwise_impl`` /
    ``quant_fp8_blockwise_for_weight_impl`` kernels directly, so it is not
    exercised by these tests.

    ``group_lens`` makes the wrapper grouped (``_is_grouped_tensor=True``);
    grouped wrappers are restricted to 2D + ROWWISE / TENSORWISE.
    """
    kwargs: dict = {"axis": axis}
    if group_lens is not None:
        kwargs["group_lens"] = group_lens

    if granularity == ScalingGranularity.MX_BLOCKWISE:
        kwargs["block_size"] = block_size
        kwargs["scaling_recipe"] = ScalingRecipe(use_2d_block=use_2d_block)
    # TENSORWISE / ROWWISE: nothing else needed.

    return QuantizedTensor.quantize(x, dest_dtype, granularity, **kwargs)


def _expected_scale_shape(granularity: ScalingGranularity, M_: int, N_: int, block_size, use_2d_block: bool):
    """Return the expected scale_inv shape for a [M, N] tensor at axis=-1."""
    if granularity == ScalingGranularity.TENSORWISE:
        return ()
    if granularity == ScalingGranularity.ROWWISE:
        return (M_, 1)
    if granularity == ScalingGranularity.MX_BLOCKWISE:
        return (M_, N_ // block_size)


# Granularity × block_size combos for parametrization.
#
# BLOCKWISE is intentionally absent — the wrapper no longer supports it (see
# the docstring on ``_make_quantized_tensor``).  MX_BLOCKWISE cases use
# ``pytest.param`` with skipif marks so that devices lacking MXFP8 / MXFP4
# support produce a deterministic "skipped" test instead of crashing inside
# the kernel layer.
_GRAN_CASES = [
    (ScalingGranularity.TENSORWISE, None, float8_e4m3),
    (ScalingGranularity.ROWWISE, None, float8_e4m3),
    pytest.param(
        ScalingGranularity.MX_BLOCKWISE,
        MXFP8_BLOCK_SIZE,
        float8_e4m3,
        marks=SKIP_MXFP8,
        id="MX_BLOCKWISE-fp8",
    ),
    pytest.param(
        ScalingGranularity.MX_BLOCKWISE,
        MXFP4_BLOCK_SIZE,
        float4_e2m1fn_x2,
        marks=SKIP_MXFP4,
        id="MX_BLOCKWISE-fp4",
    ),
]


# =====================================================================
# Basic construction & properties (all granularities)
# =====================================================================
class TestQuantizedTensorBasic:

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_CASES)
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_create_and_properties(self, granularity, block_size, dest_dtype, dtype):
        x = torch.randn(M, N, dtype=dtype, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, dest_dtype=dest_dtype, block_size=block_size)

        assert qt.dtype == dtype
        assert qt.real_dtype == dest_dtype
        assert qt.qdata is not None
        assert qt.scale_inv is not None
        assert qt.shape == torch.Size([M, N])
        assert qt.device.type == "cuda"
        assert qt.granularity == granularity
        assert qt.block_size == block_size
        # Plain (non-grouped) construction.
        assert qt._is_grouped_tensor is False

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_CASES)
    def test_scale_shape(self, granularity, block_size, dest_dtype):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, dest_dtype=dest_dtype, block_size=block_size)

        expected = _expected_scale_shape(granularity, M, N, block_size, use_2d_block=False)
        assert (
            tuple(qt.scale_inv.shape) == expected
        ), f"{granularity.name}/{dest_dtype}: expected scale shape {expected}, got {tuple(qt.scale_inv.shape)}"


# =====================================================================
# Quantize-API constraint checks (the new API tightened these)
# =====================================================================
class TestQuantizeConstraints:
    """Negative tests covering the assertions in ``QuantizedTensor.quantize``."""

    @SKIP_MXFP8
    def test_mx_blockwise_requires_scaling_recipe(self):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        with pytest.raises(AssertionError):
            QuantizedTensor.quantize(
                x,
                float8_e4m3,
                ScalingGranularity.MX_BLOCKWISE,
                axis=1,
                block_size=MXFP8_BLOCK_SIZE,
            )

    def test_rank_must_be_2d_or_3d(self):
        x = torch.randn(2, M, N, 4, dtype=torch.bfloat16, device=DEVICE)
        with pytest.raises(AssertionError):
            QuantizedTensor.quantize(
                x,
                float8_e4m3,
                ScalingGranularity.TENSORWISE,
                axis=-1,
            )


# =====================================================================
# Dequantization accuracy
# Uses the same tolerance policy as tests/pytorch/ops/test_quantization.py:
#   torch.testing.assert_close(..., **get_tolerances(dest_dtype))
# NOTE: dequantize currently only supports TENSORWISE and MX_BLOCKWISE.
# ROWWISE dequantize kernels are not implemented yet.
# =====================================================================
_DEQUANT_CASES = [
    (ScalingGranularity.TENSORWISE, None, float8_e4m3),
    # TODO(ruibin): rowwise dequantization is not implemented yet.
    pytest.param(
        ScalingGranularity.MX_BLOCKWISE,
        MXFP8_BLOCK_SIZE,
        float8_e4m3,
        marks=SKIP_MXFP8,
        id="MX_BLOCKWISE-fp8",
    ),
    pytest.param(
        ScalingGranularity.MX_BLOCKWISE,
        MXFP4_BLOCK_SIZE,
        float4_e2m1fn_x2,
        marks=SKIP_MXFP4,
        id="MX_BLOCKWISE-fp4",
    ),
]


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
# Serialisation (flatten / unflatten)
# =====================================================================
class TestSerialization:

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GRAN_CASES)
    def test_flatten_unflatten_roundtrip(self, granularity, block_size, dest_dtype):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, dest_dtype=dest_dtype, block_size=block_size)

        keys, metadata = qt.__tensor_flatten__()

        # Inner-tensor keys ------------------------------------------------
        assert "_data" in keys
        assert "_scale_inv" in keys
        # Non-grouped wrapper must NOT serialise grouped buffers.
        assert "_group_lens" not in keys
        assert "_group_offs" not in keys

        # Metadata ---------------------------------------------------------
        assert metadata["_orig_dtype"] == qt._orig_dtype
        assert metadata["_dest_dtype"] == dest_dtype
        assert metadata["_granularity"] == granularity
        assert metadata["_block_size"] == block_size
        assert metadata["_scaling_recipe"] == qt._scaling_recipe
        assert metadata["_is_grouped_tensor"] is False

        # Round-trip -------------------------------------------------------
        inner = {k: getattr(qt, k) for k in keys}
        qt2 = QuantizedTensor.__tensor_unflatten__(inner, metadata, qt.shape, qt.stride())
        assert isinstance(qt2, QuantizedTensor)
        assert qt2.dtype == qt.dtype
        assert qt2.real_dtype == dest_dtype
        assert qt2.shape == qt.shape
        assert qt2.granularity == qt.granularity
        assert qt2.block_size == qt.block_size
        assert qt2.scaling_recipe == qt.scaling_recipe
        assert qt2._is_grouped_tensor is False
        assert torch.equal(qt2._data, qt._data)
        assert torch.equal(qt2._scale_inv, qt._scale_inv)


# =====================================================================
# View / reshape  (rank-preserving only)
# =====================================================================

_VIEW_GRAN_2D_CASES = [
    (ScalingGranularity.TENSORWISE, None, float8_e4m3),
    (ScalingGranularity.ROWWISE, None, float8_e4m3),
    pytest.param(
        ScalingGranularity.MX_BLOCKWISE,
        MXFP8_BLOCK_SIZE,
        float8_e4m3,
        marks=SKIP_MXFP8,
        id="MX_BLOCKWISE-fp8",
    ),
    pytest.param(
        ScalingGranularity.MX_BLOCKWISE,
        MXFP4_BLOCK_SIZE,
        float4_e2m1fn_x2,
        marks=SKIP_MXFP4,
        id="MX_BLOCKWISE-fp4",
    ),
]

# MX_BLOCKWISE is 2D-only at the kernel layer, so 3D view tests don't include
# the MX cases here.
_VIEW_GRAN_3D_CASES = [
    (ScalingGranularity.TENSORWISE, None, float8_e4m3),
    (ScalingGranularity.ROWWISE, None, float8_e4m3),
]


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
        ``_view_data_and_scale_inv``."""
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, block_size=block_size, dest_dtype=dest_dtype)

        viewed = qt.view(M, N)

        assert viewed.shape == qt.shape
        # Early-return path: same underlying buffers.
        assert viewed._data.data_ptr() == qt._data.data_ptr()
        assert viewed._scale_inv.data_ptr() == qt._scale_inv.data_ptr()

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _VIEW_GRAN_2D_CASES)
    def test_view_2d_metadata_propagated(self, granularity, block_size, dest_dtype):
        """All wrapper metadata (granularity, dtypes, block_size, scaling
        recipe, grouped flag) survives a view."""
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=granularity, block_size=block_size, dest_dtype=dest_dtype)

        viewed = qt.view(M, N)

        assert viewed._granularity == qt._granularity
        assert viewed._block_size == qt._block_size
        assert viewed._orig_dtype == qt._orig_dtype
        assert viewed._dest_dtype == qt._dest_dtype
        assert viewed._scaling_recipe == qt._scaling_recipe
        assert viewed._is_grouped_tensor == qt._is_grouped_tensor


class TestViewFunc3D:
    """View / reshape on a 3D wrapper (``[G, K, N]`` — stacked weights).

    Only TENSORWISE / ROWWISE are exercised here; MX_BLOCKWISE is 2D-only at
    the kernel layer and BLOCKWISE is not supported by ``QuantizedTensor`` at
    all (it goes through the low-level kernels directly).
    """

    G_, K_, N_ = 4, 128, 256

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _VIEW_GRAN_3D_CASES)
    def test_view_3d_same_shape_identity(self, granularity, block_size, dest_dtype):
        x = torch.randn(self.G_, self.K_, self.N_, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(
            x,
            granularity=granularity,
            block_size=block_size,
            dest_dtype=dest_dtype,
        )

        viewed = qt.view(self.G_, self.K_, self.N_)

        assert viewed.shape == qt.shape
        assert viewed._data.data_ptr() == qt._data.data_ptr()
        assert viewed._scale_inv.data_ptr() == qt._scale_inv.data_ptr()

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _VIEW_GRAN_3D_CASES)
    def test_view_3d_metadata_propagated(self, granularity, block_size, dest_dtype):
        x = torch.randn(self.G_, self.K_, self.N_, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(
            x,
            granularity=granularity,
            block_size=block_size,
            dest_dtype=dest_dtype,
        )

        viewed = qt.view(self.G_, self.K_, self.N_)

        assert viewed._granularity == qt._granularity
        assert viewed._block_size == qt._block_size
        assert viewed._orig_dtype == qt._orig_dtype
        assert viewed._dest_dtype == qt._dest_dtype
        assert viewed._scaling_recipe == qt._scaling_recipe


# =====================================================================
# Grouped QuantizedTensor (packed-M activations carrying ``group_lens``)
# =====================================================================

NUM_GROUPS = 4
DEFAULT_GROUP_LENS = [M // NUM_GROUPS] * NUM_GROUPS  # [64, 64, 64, 64]


def _make_group_lens(lens=DEFAULT_GROUP_LENS, device: str = DEVICE) -> torch.Tensor:
    return torch.tensor(lens, dtype=torch.int64, device=device)


# Grouped tensors only support ROWWISE and TENSORWISE (see
# QuantizedTensor.quantize assertions).
_GROUPED_GRAN_CASES = [
    (ScalingGranularity.TENSORWISE, None, float8_e4m3),
    (ScalingGranularity.ROWWISE, None, float8_e4m3),
]

_GROUPED_DEQUANT_CASES = [
    (ScalingGranularity.TENSORWISE, None, float8_e4m3),
    # TODO(ruibin): rowwise dequantization is not implemented yet.
]


# ---------------------------------------------------------------------
# Basic construction & properties
# ---------------------------------------------------------------------
class TestGroupedQuantizedTensorBasic:

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GROUPED_GRAN_CASES)
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_create_and_properties(self, granularity, block_size, dest_dtype, dtype):
        x = torch.randn(M, N, dtype=dtype, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_quantized_tensor(
            x,
            granularity=granularity,
            dest_dtype=dest_dtype,
            block_size=block_size,
            group_lens=group_lens,
        )

        assert qt.dtype == dtype
        assert qt.real_dtype == dest_dtype
        assert qt.qdata is not None
        assert qt.scale_inv is not None
        assert qt.shape == torch.Size([M, N])
        assert qt.device.type == "cuda"
        assert qt.granularity == granularity
        assert qt.block_size == block_size
        # Wrapper is grouped — exposes the grouped predicate and group_lens.
        assert qt._is_grouped_tensor is True

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GROUPED_GRAN_CASES)
    def test_scale_shape(self, granularity, block_size, dest_dtype):
        """Packed-M quantization is group-agnostic, so scale shape matches
        the equivalent non-grouped 2D wrapper."""
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_quantized_tensor(
            x,
            granularity=granularity,
            dest_dtype=dest_dtype,
            block_size=block_size,
            group_lens=group_lens,
        )

        expected = _expected_scale_shape(granularity, M, N, block_size, use_2d_block=False)
        assert (
            tuple(qt.scale_inv.shape) == expected
        ), f"{granularity.name}/{dest_dtype}: expected scale shape {expected}, got {tuple(qt.scale_inv.shape)}"


# ---------------------------------------------------------------------
# Group-specific behaviour
# ---------------------------------------------------------------------
class TestGroupSpecific:
    """Behaviours unique to grouped ``QuantizedTensor`` — ``group_lens`` /
    ``group_offs`` bookkeeping plus the constructor's hard constraints.
    """

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GROUPED_GRAN_CASES)
    def test_group_lens_and_offs(self, granularity, block_size, dest_dtype):
        """``group_lens`` is preserved; ``group_offs`` is derived as a
        cumulative-sum (with a leading 0).  In the new implementation the
        offsets are eagerly materialised inside ``__new__``.
        """
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_quantized_tensor(
            x,
            granularity=granularity,
            block_size=block_size,
            dest_dtype=dest_dtype,
            group_lens=group_lens,
        )

        assert torch.equal(qt.group_lens, group_lens)
        # ``group_offs`` matches a fresh prefix-sum.
        assert torch.equal(qt.group_offs, group_offs_from_lens(group_lens))
        # And it is already materialised (no lazy fallback needed).
        assert qt._group_offs is not None

    def test_uneven_group_lens(self):
        """Sums to TOTAL_M (== M) but with non-uniform group sizes — the
        wrapper must accept this without padding or rebalancing."""
        lens = torch.tensor([20, 80, 60, 96], dtype=torch.int64, device=DEVICE)
        assert int(lens.sum().item()) == M
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        qt = _make_quantized_tensor(x, granularity=ScalingGranularity.ROWWISE, group_lens=lens)
        assert torch.equal(qt.group_lens, lens)
        assert torch.equal(qt.group_offs, group_offs_from_lens(lens))


# ---------------------------------------------------------------------
# Dequantization accuracy
# ---------------------------------------------------------------------
class TestGroupedDequantize:

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GROUPED_DEQUANT_CASES)
    def test_dequantize_roundtrip_close(self, granularity, block_size, dest_dtype):
        """Quant -> dequant roundtrip should be close to the original within
        FP8 tolerances (rtol=atol=1e-1), matching test_quantization.py policy.
        """
        torch.manual_seed(42)
        x = torch.rand(M, N, dtype=torch.bfloat16, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_quantized_tensor(
            x,
            granularity=granularity,
            dest_dtype=dest_dtype,
            block_size=block_size,
            group_lens=group_lens,
        )

        x_recon = qt.dequantize()
        torch.testing.assert_close(x_recon, x, **get_tolerances(dest_dtype))


# ---------------------------------------------------------------------
# Serialisation (flatten / unflatten)
# ---------------------------------------------------------------------
class TestGroupedSerialization:

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GROUPED_GRAN_CASES)
    def test_flatten_unflatten_roundtrip(self, granularity, block_size, dest_dtype):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_quantized_tensor(
            x,
            granularity=granularity,
            dest_dtype=dest_dtype,
            block_size=block_size,
            group_lens=group_lens,
        )

        keys, metadata = qt.__tensor_flatten__()

        # Inner-tensor keys: data/scale + grouped buffers.
        assert "_data" in keys
        assert "_scale_inv" in keys
        assert "_group_lens" in keys
        # ``_group_offs`` is eagerly materialised in ``__new__`` and must
        # therefore be part of the flatten payload.
        assert "_group_offs" in keys

        # Metadata: includes scaling_recipe and the grouped predicate.
        assert metadata["_granularity"] == granularity
        assert metadata["_block_size"] == block_size
        assert metadata["_orig_dtype"] == qt._orig_dtype
        assert metadata["_dest_dtype"] == dest_dtype
        assert metadata["_scaling_recipe"] == qt._scaling_recipe
        assert metadata["_is_grouped_tensor"] is True

        inner = {k: getattr(qt, k) for k in keys}
        qt2 = QuantizedTensor.__tensor_unflatten__(inner, metadata, qt.shape, qt.stride())
        assert isinstance(qt2, QuantizedTensor)
        assert qt2._is_grouped_tensor is True
        assert qt2.dtype == qt.dtype
        assert qt2.real_dtype == dest_dtype
        assert qt2.shape == qt.shape
        assert qt2.granularity == qt.granularity
        assert qt2.block_size == qt.block_size
        assert qt2.scaling_recipe == qt.scaling_recipe
        assert torch.equal(qt2._data, qt._data)
        assert torch.equal(qt2._scale_inv, qt._scale_inv)
        # group metadata round-trips: lens preserved and offs preserved.
        assert torch.equal(qt2.group_lens, qt.group_lens)
        assert torch.equal(qt2.group_offs, qt.group_offs)


# ---------------------------------------------------------------------
# View / reshape (rank-preserving — packed-M is strictly 2D)
# ---------------------------------------------------------------------
class TestGroupedViewFunc:
    """View / reshape on a grouped ``QuantizedTensor`` (always 2D packed-M)."""

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GROUPED_GRAN_CASES)
    def test_view_same_shape_identity(self, granularity, block_size, dest_dtype):
        """Viewing to the wrapper's own shape returns the same data /
        scale buffers via the early-return path in
        ``_view_data_and_scale_inv``."""
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_quantized_tensor(x, granularity=granularity, block_size=block_size, group_lens=group_lens)

        viewed = qt.view(M, N)

        assert isinstance(viewed, QuantizedTensor)
        assert viewed._is_grouped_tensor is True
        assert viewed.shape == qt.shape
        # Early-return path: same underlying buffers.
        assert viewed._data.data_ptr() == qt._data.data_ptr()
        assert viewed._scale_inv.data_ptr() == qt._scale_inv.data_ptr()

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GROUPED_GRAN_CASES)
    def test_view_metadata_propagated(self, granularity, block_size, dest_dtype):
        """All wrapper metadata (granularity, dtypes, block_size,
        scaling_recipe) survives a view."""
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_quantized_tensor(
            x,
            granularity=granularity,
            block_size=block_size,
            dest_dtype=dest_dtype,
            group_lens=group_lens,
        )

        viewed = qt.view(M, N)

        assert viewed._granularity == qt._granularity
        assert viewed._block_size == qt._block_size
        assert viewed._orig_dtype == qt._orig_dtype
        assert viewed._dest_dtype == qt._dest_dtype
        assert viewed._scaling_recipe == qt._scaling_recipe

    @pytest.mark.parametrize("granularity,block_size,dest_dtype", _GROUPED_GRAN_CASES)
    def test_view_group_metadata_propagated(self, granularity, block_size, dest_dtype):
        """``group_lens`` / ``group_offs`` flow through ``_make_like`` —
        the viewed wrapper shares the same tensors as the source.

        Since ``__new__`` now eagerly materialises ``_group_offs``, both the
        source and the viewed wrapper already hold the same offsets tensor
        and pointer identity is preserved through ``_make_like``.
        """
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        group_lens = _make_group_lens()
        qt = _make_quantized_tensor(
            x,
            granularity=granularity,
            dest_dtype=dest_dtype,
            block_size=block_size,
            group_lens=group_lens,
        )

        viewed = qt.view(M, N)

        assert viewed.group_lens.data_ptr() == qt.group_lens.data_ptr()
        assert viewed.group_offs.data_ptr() == qt.group_offs.data_ptr()
