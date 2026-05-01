###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Unit tests for ``primus_turbo.pytorch.ops.attention.attention_utils``.

These tests cover pure-Python helpers introduced/touched by the recent
attention layout work (PR #275 / #304) and the env-var unification (PR #305):

* ``_infer_qkv_format`` / ``_infer_format`` — strict stride-based detection
  of ``bshd`` / ``sbhd`` / ``bhsd`` layouts and the layout-mismatch guard.
* ``_resolve_is_v3_atomic_fp32_from_env`` — ``PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32``
  parsing, including the "unrecognized → default true" branch.
* ``block_scaling_node`` — ``use_fp8=False`` shortcut returns the input
  unchanged with a unit scale tensor.

All tests run on CPU so that they remain deterministic and fast on any
worker, independent of the ROCm runtime.
"""

import pytest
import torch

from primus_turbo.pytorch.ops.attention.attention_utils import (
    _infer_qkv_format,
    _resolve_is_v3_atomic_fp32_from_env,
    block_scaling_node,
)


# ---------------------------------------------------------------------------
# _infer_qkv_format / _infer_format
# ---------------------------------------------------------------------------


def _make_bshd(b=2, s=4, h=3, d=8):
    """Allocate a contiguous tensor with logical [B, S, H, D] layout.

    Strides will be ``(S*H*D, H*D, D, 1)`` → bshd path.
    """
    return torch.empty(b, s, h, d)


def _make_sbhd_view(b=2, s=4, h=3, d=8):
    """Return a view whose logical shape is [B, S, H, D] but memory is sbhd.

    Done by allocating ``[S, B, H, D]`` contiguously and transposing
    dims 0 and 1.  The resulting tensor has stride ``(H*D, B*H*D, D, 1)``
    → ``s1 > s0 > s2`` invariant which selects ``sbhd``.
    """
    return torch.empty(s, b, h, d).transpose(0, 1)


def _make_bhsd_view(b=2, s=4, h=3, d=8):
    """Return a view whose logical shape is [B, S, H, D] but memory is bhsd.

    Done by allocating ``[B, H, S, D]`` contiguously and transposing
    dims 1 and 2.  Stride becomes ``(H*S*D, D, S*D, 1)`` →
    ``s0 > s2 > s1`` invariant which selects ``bhsd``.
    """
    return torch.empty(b, h, s, d).transpose(1, 2)


class TestInferQkvFormat:

    def test_bshd_contiguous(self):
        q, k, v = _make_bshd(), _make_bshd(), _make_bshd()
        assert _infer_qkv_format(q, k, v) == "bshd"

    def test_sbhd_via_transpose(self):
        q, k, v = _make_sbhd_view(), _make_sbhd_view(), _make_sbhd_view()
        assert _infer_qkv_format(q, k, v) == "sbhd"

    def test_bhsd_via_transpose(self):
        """Regression for PR #304: ``s0 > s2 > s1`` must resolve to bhsd."""
        q, k, v = _make_bhsd_view(), _make_bhsd_view(), _make_bhsd_view()
        assert _infer_qkv_format(q, k, v) == "bhsd"

    def test_layout_mismatch_raises(self):
        """q is bshd while k is sbhd → must raise an explicit AssertionError."""
        q = _make_bshd()
        k = _make_sbhd_view()
        v = _make_bshd()
        with pytest.raises(AssertionError, match="Layout mismatch"):
            _infer_qkv_format(q, k, v)

    def test_layout_mismatch_bhsd_vs_bshd_raises(self):
        q = _make_bshd()
        k = _make_bhsd_view()
        v = _make_bshd()
        with pytest.raises(AssertionError, match="Layout mismatch"):
            _infer_qkv_format(q, k, v)

    def test_non_contiguous_innermost_dim_raises(self):
        """If ``stride[-1] != 1`` (innermost dim non-contiguous) we must
        refuse to guess a layout; this is a critical safety check because
        the downstream FlashAttention kernel assumes a unit head-dim stride.
        """
        # Allocate [B, S, H, D] then transpose innermost dims to break
        # the unit stride invariant on the last axis.
        bad = torch.empty(2, 4, 3, 8).transpose(2, 3)
        # bad.stride()[-1] is no longer 1.
        assert bad.stride()[-1] != 1
        with pytest.raises(AssertionError, match="contiguous innermost dim"):
            _infer_qkv_format(bad, bad, bad)

    def test_q_must_be_4d(self):
        q = torch.empty(2, 4, 3)  # 3-D
        k = _make_bshd()
        v = _make_bshd()
        with pytest.raises(AssertionError, match="Expected 4-D tensor for q"):
            _infer_qkv_format(q, k, v)

    def test_kv_must_be_4d(self):
        q = _make_bshd()
        bad = torch.empty(2, 4, 3)  # 3-D
        with pytest.raises(AssertionError, match="Expected 4-D tensor for k"):
            _infer_qkv_format(q, bad, _make_bshd())
        with pytest.raises(AssertionError, match="Expected 4-D tensor for v"):
            _infer_qkv_format(q, _make_bshd(), bad)


# ---------------------------------------------------------------------------
# _resolve_is_v3_atomic_fp32_from_env
# ---------------------------------------------------------------------------


class TestResolveAttnV3AtomicFp32Env:

    ENV = "PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32"

    def test_default_true_when_unset(self, monkeypatch):
        monkeypatch.delenv(self.ENV, raising=False)
        assert _resolve_is_v3_atomic_fp32_from_env() is True

    def test_explicit_disable(self, monkeypatch):
        monkeypatch.setenv(self.ENV, "0")
        assert _resolve_is_v3_atomic_fp32_from_env() is False

    def test_explicit_enable(self, monkeypatch):
        monkeypatch.setenv(self.ENV, "1")
        assert _resolve_is_v3_atomic_fp32_from_env() is True

    @pytest.mark.parametrize("bad_value", ["", "true", "yes", "2", "-1"])
    def test_unrecognized_value_falls_back_to_true(self, monkeypatch, bad_value):
        """The helper deliberately treats anything outside ``{"0","1"}`` as
        "default to True"; this prevents typos in user shells from silently
        flipping the FP32-atomic accumulation off (which would degrade
        attention numerics on gfx942 for 16-bit dtypes).
        """
        monkeypatch.setenv(self.ENV, bad_value)
        assert _resolve_is_v3_atomic_fp32_from_env() is True


# ---------------------------------------------------------------------------
# block_scaling_node — non-FP8 short-circuit path
# ---------------------------------------------------------------------------


class TestBlockScalingNode:

    def test_use_fp8_false_returns_input_unchanged(self):
        t = torch.randn(2, 4, 3, 8)
        out, scale = block_scaling_node(t, use_fp8=False)
        assert out is t, "use_fp8=False must short-circuit and return the same tensor"
        # Scale must be a 1-element tensor of value 1.0 on the same device.
        assert scale.shape == (1,)
        assert scale.device == t.device
        assert torch.equal(scale, torch.tensor([1.0], device=t.device))
