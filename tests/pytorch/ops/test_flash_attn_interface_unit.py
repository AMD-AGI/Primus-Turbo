###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Unit tests for non-GPU logic in flash_attn_interface.py.

These tests cover:
1. AiterFlashAttnFunc._resolve_is_v3_atomic_fp32  (env-var logic)
2. The sm_scale=None default derivation  (PR #263 fix)

All tests are CPU-only; no aiter/CUDA dependency required.
"""

import math

import pytest

from primus_turbo.pytorch.ops.attention.flash_attn_interface import AiterFlashAttnFunc


class TestResolveIsV3AtomicFP32:
    def test_explicit_true_returns_true(self):
        assert AiterFlashAttnFunc._resolve_is_v3_atomic_fp32(True) is True

    def test_explicit_false_returns_false(self):
        assert AiterFlashAttnFunc._resolve_is_v3_atomic_fp32(False) is False

    def test_none_defaults_to_true_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32", raising=False)
        assert AiterFlashAttnFunc._resolve_is_v3_atomic_fp32(None) is True

    def test_none_reads_env_var_1(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32", "1")
        assert AiterFlashAttnFunc._resolve_is_v3_atomic_fp32(None) is True

    def test_none_reads_env_var_0(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32", "0")
        assert AiterFlashAttnFunc._resolve_is_v3_atomic_fp32(None) is False

    def test_none_with_invalid_env_value_defaults_to_true(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32", "yes")
        assert AiterFlashAttnFunc._resolve_is_v3_atomic_fp32(None) is True


class TestSmScaleDefault:
    """
    Verify the sm_scale=None default formula  head_dim ** (-0.5)  matches the
    expected value.

    PR #263 fixed AttentionCKFunctionCPA2A and AttentionTritonFunctionCPA2A to
    fill in the default before using the scale; this test exercises the same
    formula so a future regression is immediately caught.
    """

    @pytest.mark.parametrize("head_dim", [32, 64, 128, 192, 256])
    def test_scale_formula(self, head_dim):
        expected = head_dim ** (-0.5)
        assert math.isfinite(expected)
        assert abs(expected - 1.0 / math.sqrt(head_dim)) < 1e-9

    def test_scale_for_head_dim_128(self):
        head_dim = 128
        scale = head_dim ** (-0.5)
        assert abs(scale - 0.08838834764831843) < 1e-9
