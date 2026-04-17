###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Tests for primus_turbo.pytorch.core.low_precision – covers dataclass validation
logic and enum semantics added/changed by recent PRs.

These tests are CPU-only and do not require a GPU.
"""

import pytest

from primus_turbo.pytorch.core.low_precision import (
    Float4QuantConfig,
    Float8QuantConfig,
    Format,
    MXScalingRecipe,
    ScaleDtype,
    ScalingGranularity,
    ScalingStrategy,
)


# ---------------------------------------------------------------------------
# Float8QuantConfig validation
# ---------------------------------------------------------------------------


class TestFloat8QuantConfig:
    def test_default_fields(self):
        cfg = Float8QuantConfig()
        assert cfg.format == Format.E4M3
        assert cfg.granularity == ScalingGranularity.TENSORWISE
        assert cfg.strategy == ScalingStrategy.DYNAMIC
        assert cfg.scale_dtype == ScaleDtype.FP32
        assert cfg.block_size is None

    def test_tensorwise_no_block_size_required(self):
        cfg = Float8QuantConfig(granularity=ScalingGranularity.TENSORWISE)
        assert cfg.block_size is None

    def test_rowwise_no_block_size_required(self):
        cfg = Float8QuantConfig(granularity=ScalingGranularity.ROWWISE)
        assert cfg.block_size is None

    def test_blockwise_requires_block_size(self):
        with pytest.raises(AssertionError):
            Float8QuantConfig(granularity=ScalingGranularity.BLOCKWISE)

    def test_blockwise_with_valid_block_size(self):
        cfg = Float8QuantConfig(granularity=ScalingGranularity.BLOCKWISE, block_size=128)
        assert cfg.block_size == 128

    def test_mx_blockwise_requires_block_size_32(self):
        with pytest.raises(AssertionError):
            Float8QuantConfig(
                granularity=ScalingGranularity.MX_BLOCKWISE,
                block_size=64,
                scale_dtype=ScaleDtype.E8M0,
            )

    def test_mx_blockwise_requires_e8m0_scale_dtype(self):
        with pytest.raises(AssertionError):
            Float8QuantConfig(
                granularity=ScalingGranularity.MX_BLOCKWISE,
                block_size=32,
                scale_dtype=ScaleDtype.FP32,
            )

    def test_mx_blockwise_valid(self):
        cfg = Float8QuantConfig(
            granularity=ScalingGranularity.MX_BLOCKWISE,
            block_size=32,
            scale_dtype=ScaleDtype.E8M0,
        )
        assert cfg.block_size == 32
        assert cfg.scale_dtype == ScaleDtype.E8M0

    def test_hybrid_format_tensorwise(self):
        cfg = Float8QuantConfig(format=Format.HYBRID, granularity=ScalingGranularity.TENSORWISE)
        assert cfg.format == Format.HYBRID

    def test_all_formats_accepted_for_tensorwise(self):
        for fmt in (Format.E4M3, Format.E5M2, Format.HYBRID):
            cfg = Float8QuantConfig(format=fmt, granularity=ScalingGranularity.TENSORWISE)
            assert cfg.format == fmt


# ---------------------------------------------------------------------------
# Float4QuantConfig validation
# ---------------------------------------------------------------------------


class TestFloat4QuantConfig:
    def test_default_fields(self):
        cfg = Float4QuantConfig()
        assert cfg.format == Format.E2M1_X2
        assert cfg.granularity == ScalingGranularity.MX_BLOCKWISE
        assert cfg.block_size == 32
        assert cfg.scale_dtype == ScaleDtype.E8M0

    def test_only_mx_blockwise_granularity_allowed(self):
        with pytest.raises(AssertionError):
            Float4QuantConfig(granularity=ScalingGranularity.TENSORWISE)

    def test_block_size_must_be_32(self):
        with pytest.raises(AssertionError):
            Float4QuantConfig(block_size=64)

    def test_format_must_be_e2m1_x2(self):
        with pytest.raises(AssertionError):
            Float4QuantConfig(format=Format.E4M3)

    def test_scale_dtype_must_be_e8m0(self):
        with pytest.raises(AssertionError):
            Float4QuantConfig(scale_dtype=ScaleDtype.FP32)


# ---------------------------------------------------------------------------
# MXScalingRecipe defaults
# ---------------------------------------------------------------------------


class TestMXScalingRecipe:
    def test_defaults_all_false(self):
        r = MXScalingRecipe()
        assert r.use_2d_block is False
        assert r.use_sr is False
        assert r.use_rht is False
        assert r.shuffle_scale is False
        assert r.shuffle_out is False

    def test_set_fields(self):
        r = MXScalingRecipe(use_2d_block=True, use_sr=True)
        assert r.use_2d_block is True
        assert r.use_sr is True
        assert r.use_rht is False
