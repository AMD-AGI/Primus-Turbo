###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Tests for helper functions in attention_aiter_impl.py.

PR #262 added _normalize_sink_window() to translate GPT-OSS-style causal
sliding-window sizes (left, 0) into the aiter Triton sentinel (left, -1).

These tests are CPU-only and do not require a GPU or aiter to be installed.
"""

import pytest

from primus_turbo.pytorch.kernels.attention.attention_aiter_impl import (
    _is_power_of_2,
    _normalize_sink_window,
)


class TestIsPowerOf2:
    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    def test_powers_of_two(self, n):
        assert _is_power_of_2(n) is True

    @pytest.mark.parametrize("n", [0, 3, 5, 6, 7, 9, 10, 12, 100, 192])
    def test_non_powers_of_two(self, n):
        assert _is_power_of_2(n) is False

    def test_negative_is_not_power_of_two(self):
        assert _is_power_of_2(-4) is False


class TestNormalizeSinkWindow:
    # Causal + right==0  =>  (left, -1)  — the GPT-OSS → aiter translation
    def test_causal_right_zero_returns_sentinel(self):
        left, right = _normalize_sink_window(causal=True, window_size_left=64, window_size_right=0)
        assert left == 64
        assert right == -1

    def test_causal_right_zero_with_unlimited_left(self):
        left, right = _normalize_sink_window(causal=True, window_size_left=-1, window_size_right=0)
        assert left == -1
        assert right == -1

    # Non-causal or right != 0  =>  pass-through, no sentinel substitution
    def test_causal_right_nonzero_passthrough(self):
        left, right = _normalize_sink_window(causal=True, window_size_left=32, window_size_right=16)
        assert left == 32
        assert right == 16

    def test_non_causal_right_zero_passthrough(self):
        left, right = _normalize_sink_window(causal=False, window_size_left=64, window_size_right=0)
        assert left == 64
        assert right == 0

    def test_non_causal_right_nonzero_passthrough(self):
        left, right = _normalize_sink_window(causal=False, window_size_left=-1, window_size_right=-1)
        assert left == -1
        assert right == -1

    def test_causal_left_zero_right_zero_returns_sentinel(self):
        left, right = _normalize_sink_window(causal=True, window_size_left=0, window_size_right=0)
        assert left == 0
        assert right == -1
