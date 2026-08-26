###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared fixtures and builders for the flex compat-layer unit tests.

Not a test module (no ``test_`` prefix) so pytest imports it rather than collecting it."""

import math

import pytest
import torch

from primus_turbo.pytorch.ops.attention.flex._cache import clear_classification_cache
from primus_turbo.pytorch.ops.attention.flex._routing import clear_backend_overrides


class _DummyBlockMask:
    def __init__(self, mask_mod):
        self.mask_mod = mask_mod


@pytest.fixture(autouse=True)
def _reset_backend_overrides():
    """The override registry and classification caches are module-global; keep
    tests independent (cold cache each test)."""
    clear_backend_overrides()
    clear_classification_cache()
    yield
    clear_backend_overrides()
    clear_classification_cache()


_CAUSAL_CFG = {"kind": "causal", "causal": True, "window_size": (-1, -1)}


def _alibi_score_mod(num_heads):
    def score_mod(score, b, h, q_idx, kv_idx):
        slope = 2.0 ** (-8.0 * float(h) / num_heads)
        return score + slope * (kv_idx - q_idx)

    return score_mod


def _softcap_score_mod(cap):
    def score_mod(score, b, h, q_idx, kv_idx):
        return cap * math.tanh(score / cap)

    return score_mod


@pytest.fixture
def capture_backend(monkeypatch):
    captured = {}

    def fake_flash_attn_func(q, k, v, **kwargs):
        captured["called"] = True
        captured["kwargs"] = kwargs
        captured["q_shape"] = tuple(q.shape)
        if kwargs.get("return_lse"):
            b, s, h, d = q.shape
            return q.clone(), torch.zeros((b, h, s), dtype=torch.float32)
        return q.clone()

    monkeypatch.setattr(
        "primus_turbo.pytorch.ops.attention.flash_attn_interface.flash_attn_func",
        fake_flash_attn_func,
        raising=True,
    )
    return captured


def _make_qkv(B=1, Hq=4, S=16, D=16, dtype=torch.float16):
    q = torch.randn(B, Hq, S, D, dtype=dtype)
    k = torch.randn(B, Hq, S, D, dtype=dtype)
    v = torch.randn(B, Hq, S, D, dtype=dtype)
    return q, k, v


def _cu_from_seqlens(seqlens, device="cpu"):
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(0)
    return cu, max(seqlens), int(cu[-1].item())


def _make_thd(total, H, D, dtype=torch.float16):
    return torch.randn(total, H, D, dtype=dtype)


# ---- end-to-end dispatch (backend mocked on CPU) --------------------------


@pytest.fixture
def capture_varlen_backend(monkeypatch):
    captured = {}

    def fake_flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs
    ):
        captured["called"] = True
        captured["kwargs"] = kwargs
        captured["cu_q"] = cu_seqlens_q
        captured["cu_k"] = cu_seqlens_k
        captured["max_q"] = max_seqlen_q
        captured["max_k"] = max_seqlen_k
        captured["q_shape"] = tuple(q.shape)
        captured["k_shape"] = tuple(k.shape)
        captured["v_shape"] = tuple(v.shape)
        if kwargs.get("return_lse"):
            return q.clone(), torch.zeros((q.shape[1], q.shape[0]), dtype=torch.float32)
        return q.clone()

    monkeypatch.setattr(
        "primus_turbo.pytorch.ops.attention.flash_attn_interface.flash_attn_varlen_func",
        fake_flash_attn_varlen_func,
        raising=True,
    )
    return captured


def _doc_causal_dense_mask(seg_lens):
    total = sum(seg_lens)
    document_id = torch.cat([torch.full((s,), i, dtype=torch.int64) for i, s in enumerate(seg_lens)])
    qi = torch.arange(total).view(total, 1)
    ki = torch.arange(total).view(1, total)
    return (document_id.view(total, 1) == document_id.view(1, total)) & (qi >= ki)


def _doc_causal_block_mask(seg_lens):
    document_id = torch.cat([torch.full((s,), i, dtype=torch.int64) for i, s in enumerate(seg_lens)])

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_id[q_idx] == document_id[kv_idx]
        return same_doc & (q_idx >= kv_idx)

    return _DummyBlockMask(mask_mod)


# ---- end-to-end: dense entry routes a document mask to the varlen backend --


def _make_bhsd(B, H, S, D, dtype=torch.float16):
    return torch.randn(B, H, S, D, dtype=dtype)


def _make_bshd(B=2, S=32, H=4, D=16, dtype=torch.float16):
    return torch.randn(B, S, H, D, dtype=dtype)
