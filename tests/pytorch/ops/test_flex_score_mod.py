###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for ``flex/_score_mod.py``: ALiBi / soft-cap detection and the sign-convention self-check."""

import math

import pytest
import torch

from primus_turbo.pytorch.ops.attention.flex import _config as flex_config
from primus_turbo.pytorch.ops.attention.flex import _score_mod as flex_score_mod
from primus_turbo.pytorch.ops.attention.flex._score_mod import (
    _detect_alibi_slopes,
    _detect_softcap,
    _is_identity_score_mod,
)

from .flex_test_utils import _alibi_score_mod, _softcap_score_mod


def test_detect_alibi_recovers_slopes():
    H = 8
    slopes = _detect_alibi_slopes(_alibi_score_mod(H), B=1, Hq=H, q_len=64, kv_len=64)
    assert slopes is not None
    expected = [2.0 ** (-8.0 * h / H) for h in range(H)]
    for got, exp in zip(slopes.tolist(), expected):
        assert abs(got - exp) < 1e-3


def test_detect_alibi_none_for_scale_score():
    # Multiplicative modification is not an additive (kv-q) bias.
    def score_mod(score, b, h, q_idx, kv_idx):
        return score + 0.1 * score

    assert _detect_alibi_slopes(score_mod, B=1, Hq=8, q_len=64, kv_len=64) is None


def test_detect_alibi_none_for_constant_bias():
    # Constant additive bias (non-zero at delta 0) is not ALiBi.
    def score_mod(score, b, h, q_idx, kv_idx):
        return score + 1.0

    assert _detect_alibi_slopes(score_mod, B=1, Hq=8, q_len=64, kv_len=64) is None


def test_detect_alibi_none_for_position_dependent():
    # Depends on absolute position (q_idx*kv_idx), not on (kv - q).
    def score_mod(score, b, h, q_idx, kv_idx):
        return score + 0.001 * (q_idx * kv_idx)

    assert _detect_alibi_slopes(score_mod, B=1, Hq=8, q_len=64, kv_len=64) is None


def test_detect_alibi_zero_for_identity():
    # A no-op score_mod is representable with zero slopes.
    def score_mod(score, b, h, q_idx, kv_idx):
        return score

    slopes = _detect_alibi_slopes(score_mod, B=1, Hq=4, q_len=64, kv_len=64)
    assert slopes is not None
    assert max(abs(x) for x in slopes.tolist()) < 1e-6


def test_detect_alibi_none_for_batch_dependent():
    H = 8

    def score_mod(score, b, h, q_idx, kv_idx):
        slope = 2.0 ** (-8.0 * float(h) / H)
        return score + (1.0 + float(b)) * slope * (kv_idx - q_idx)

    assert _detect_alibi_slopes(score_mod, B=2, Hq=H, q_len=64, kv_len=64) is None


@pytest.mark.parametrize("cap", [20.0, 30.0, 50.0])
def test_detect_softcap_recovers_cap(cap):
    got = _detect_softcap(_softcap_score_mod(cap), B=1, Hq=8, q_len=64, kv_len=64)
    assert got is not None
    assert abs(got - cap) < 1e-2 * cap


def test_detect_softcap_none_for_identity():
    def score_mod(score, b, h, q_idx, kv_idx):
        return score

    assert _detect_softcap(score_mod, B=1, Hq=8, q_len=64, kv_len=64) is None


def test_detect_softcap_none_for_linear_scale():
    def score_mod(score, b, h, q_idx, kv_idx):
        return 1.1 * score

    assert _detect_softcap(score_mod, B=1, Hq=8, q_len=64, kv_len=64) is None


def test_detect_softcap_none_for_constant_bias():
    def score_mod(score, b, h, q_idx, kv_idx):
        return score + 1.0

    assert _detect_softcap(score_mod, B=1, Hq=8, q_len=64, kv_len=64) is None


def test_detect_softcap_none_for_alibi():
    # ALiBi depends on (kv - q) -> not a pure (position-independent) soft-cap.
    def score_mod(score, b, h, q_idx, kv_idx):
        slope = 2.0 ** (-8.0 * float(h) / 8)
        return score + slope * (kv_idx - q_idx)

    assert _detect_softcap(score_mod, B=1, Hq=8, q_len=64, kv_len=64) is None


def test_detect_softcap_none_for_hard_clamp():
    # A hard clamp saturates like a soft-cap but is not tanh-shaped -> rejected.
    cap = 30.0

    def score_mod(score, b, h, q_idx, kv_idx):
        return max(-cap, min(cap, score))

    assert _detect_softcap(score_mod, B=1, Hq=8, q_len=64, kv_len=64) is None


def test_detect_softcap_none_for_alibi_softcap_combo():
    # cap*tanh((score + alibi)/cap) is position-dependent -> not a pure soft-cap.
    cap = 30.0

    def score_mod(score, b, h, q_idx, kv_idx):
        slope = 2.0 ** (-8.0 * float(h) / 8)
        return cap * math.tanh((score + slope * (kv_idx - q_idx)) / cap)

    assert _detect_softcap(score_mod, B=1, Hq=8, q_len=64, kv_len=64) is None


def test_softcap_not_misdetected_as_zero_alibi():
    # Regression guard: the ALiBi detector only probes score=0 and would read a
    # soft-cap as zero slopes; the soft-cap detector must catch it first so the
    # dispatcher never silently drops the cap.
    cap = 30.0
    sm = _softcap_score_mod(cap)
    slopes = _detect_alibi_slopes(sm, B=1, Hq=8, q_len=64, kv_len=64)
    # ALiBi misreads it as (near-)zero slopes ...
    assert slopes is not None and max(abs(x) for x in slopes.tolist()) < 1e-2
    # ... but the soft-cap detector recognises it, so callers route it away from Turbo.
    assert _detect_softcap(sm, B=1, Hq=8, q_len=64, kv_len=64) is not None


def test_is_identity_score_mod_true_for_identity():
    assert _is_identity_score_mod(lambda s, b, h, q, kv: s, B=1, Hq=4, q_len=16, kv_len=16)


def test_is_identity_score_mod_true_for_add_zero():
    assert _is_identity_score_mod(lambda s, b, h, q, kv: s + 0.0, B=2, Hq=4, q_len=16, kv_len=16)


def test_is_identity_score_mod_false_for_alibi():
    def sm(s, b, h, q, kv):
        return s + 0.5 * (kv - q)

    assert not _is_identity_score_mod(sm, B=1, Hq=4, q_len=16, kv_len=16)


def test_is_identity_score_mod_false_for_constant_bias():
    assert not _is_identity_score_mod(lambda s, b, h, q, kv: s + 1.0, B=1, Hq=4, q_len=16, kv_len=16)


def _fake_alibi_backend(sign):
    """A CPU stand-in for flash_attn_func that applies ALiBi with the given sign."""

    def fake(q, k, v, causal=False, alibi_slopes=None, **kwargs):
        q_b, k_b, v_b = (t.transpose(1, 2) for t in (q, k, v))
        out = flex_score_mod._reference_attention_with_alibi(q_b, k_b, v_b, alibi_slopes, sign, causal=causal)
        return out.transpose(1, 2).to(q.dtype)

    return fake


@pytest.fixture
def patch_alibi_backend(monkeypatch):
    def _apply(sign):
        monkeypatch.setattr(
            "primus_turbo.pytorch.ops.attention.flash_attn_interface.flash_attn_func",
            _fake_alibi_backend(sign),
        )

    return _apply


def test_check_alibi_sign_detects_plus(patch_alibi_backend):
    patch_alibi_backend(1.0)
    rep = flex_score_mod.check_alibi_sign_convention(
        device="cpu", dtype=torch.float32, seqlen=64, head_dim=32
    )
    assert rep["sign"] == 1.0
    assert rep["matches_assumption"] is True
    assert rep["plus_err"] < rep["minus_err"]


def test_check_alibi_sign_detects_minus(patch_alibi_backend):
    patch_alibi_backend(-1.0)
    rep = flex_score_mod.check_alibi_sign_convention(
        device="cpu", dtype=torch.float32, seqlen=64, head_dim=32
    )
    assert rep["sign"] == -1.0
    assert rep["matches_assumption"] is False


def test_assert_alibi_sign_passes_on_matching_build(patch_alibi_backend):
    patch_alibi_backend(flex_config._ASSUMED_ALIBI_SIGN)
    rep = flex_score_mod.assert_alibi_sign_convention(
        device="cpu", dtype=torch.float32, seqlen=64, head_dim=32
    )
    assert rep["matches_assumption"] is True


def test_assert_alibi_sign_raises_on_flipped_build(patch_alibi_backend):
    patch_alibi_backend(-flex_config._ASSUMED_ALIBI_SIGN)
    with pytest.raises(RuntimeError, match="ALiBi sign convention"):
        flex_score_mod.assert_alibi_sign_convention(device="cpu", dtype=torch.float32, seqlen=64, head_dim=32)


def test_check_alibi_sign_reports_unknown_for_unrelated_backend(monkeypatch):
    # A backend that ignores ALiBi entirely matches neither hypothesis: report None
    # rather than picking the "less wrong" sign.
    def no_alibi(q, k, v, causal=False, alibi_slopes=None, **kwargs):
        zeros = torch.zeros_like(alibi_slopes)
        q_b, k_b, v_b = (t.transpose(1, 2) for t in (q, k, v))
        out = flex_score_mod._reference_attention_with_alibi(q_b, k_b, v_b, zeros, 1.0, causal=causal)
        return out.transpose(1, 2).to(q.dtype)

    monkeypatch.setattr("primus_turbo.pytorch.ops.attention.flash_attn_interface.flash_attn_func", no_alibi)
    rep = flex_score_mod.check_alibi_sign_convention(
        device="cpu", dtype=torch.float32, seqlen=64, head_dim=32
    )
    assert rep["sign"] is None
    assert rep["matches_assumption"] is False
