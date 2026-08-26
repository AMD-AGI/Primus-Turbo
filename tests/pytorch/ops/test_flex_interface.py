###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for ``flex_attention_interface.py``: the public entry points, end-to-end with the backend mocked on CPU."""

import pytest
import torch

from primus_turbo.pytorch.ops.attention.flex_attention_interface import (
    flex_attention,
    flex_attention_bshd,
    flex_attention_varlen,
)

from .flex_test_utils import (
    _cu_from_seqlens,
    _doc_causal_block_mask,
    _DummyBlockMask,
    _make_bhsd,
    _make_bshd,
    _make_qkv,
    _make_thd,
)


def test_explicit_alibi_slopes_passed_through_and_bypasses_detection(capture_backend):
    H = 4
    q, k, v = _make_qkv(Hq=H)
    slopes = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float32)
    out = flex_attention(q, k, v, alibi_slopes=slopes.clone())
    assert out.shape == (1, H, 16, 16)
    passed = capture_backend["kwargs"]["alibi_slopes"]
    assert passed is not None
    assert torch.allclose(passed.cpu(), slopes)


def test_explicit_alibi_equivalent_to_autodetected_slopes(capture_backend):
    # The explicit path and the score_mod auto-detect path must thread the *same*
    # per-head slopes to the backend for an equivalent ALiBi definition.
    H = 8
    q, k, v = _make_qkv(Hq=H, S=32)
    slopes = torch.tensor([2.0 ** (-8.0 * h / H) for h in range(H)], dtype=torch.float32)

    flex_attention(q, k, v, alibi_slopes=slopes.clone())
    explicit_passed = capture_backend["kwargs"]["alibi_slopes"].clone()

    def alibi_score_mod(score, b, h, qi, ki):
        slope = 2.0 ** (-8.0 * float(h) / H)
        return score + slope * (ki - qi)

    capture_backend.clear()
    flex_attention(q, k, v, score_mod=alibi_score_mod)
    detected_passed = capture_backend["kwargs"]["alibi_slopes"].clone()

    assert torch.allclose(explicit_passed.cpu(), detected_passed.cpu(), atol=1e-5)


def test_explicit_alibi_with_causal_mask(capture_backend):
    # ALiBi is commonly paired with causal; the explicit slopes must still flow.
    H = 4
    q, k, v = _make_qkv(Hq=H)
    slopes = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float32)
    bm = _DummyBlockMask(lambda b, h, qi, ki: qi >= ki)
    flex_attention(q, k, v, alibi_slopes=slopes.clone(), block_mask=bm)
    assert capture_backend["kwargs"]["causal"] is True
    assert torch.allclose(capture_backend["kwargs"]["alibi_slopes"].cpu(), slopes)


def test_explicit_alibi_with_nontrivial_score_mod_raises(capture_backend):
    H = 4
    q, k, v = _make_qkv(Hq=H)
    slopes = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float32)

    def score_mod(score, b, h, qi, ki):
        return score + 0.1 * (ki - qi)

    with pytest.raises(ValueError):
        flex_attention(q, k, v, alibi_slopes=slopes, score_mod=score_mod)


def test_explicit_alibi_with_identity_score_mod_allowed(capture_backend):
    H = 4
    q, k, v = _make_qkv(Hq=H)
    slopes = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float32)

    def identity(score, b, h, qi, ki):
        return score

    flex_attention(q, k, v, alibi_slopes=slopes.clone(), score_mod=identity)
    assert torch.allclose(capture_backend["kwargs"]["alibi_slopes"].cpu(), slopes)


def test_explicit_alibi_invalid_length_raises_via_entry(capture_backend):
    q, k, v = _make_qkv(Hq=8)
    slopes = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32)  # len 3 != Hq=8
    with pytest.raises(ValueError):
        flex_attention(q, k, v, alibi_slopes=slopes)


def test_explicit_softcap_positive_raises_not_implemented(capture_backend):
    q, k, v = _make_qkv()
    with pytest.raises(NotImplementedError):
        flex_attention(q, k, v, softcap=30.0)
    # The gate fires before any backend dispatch: the cap is never silently dropped.
    assert "called" not in capture_backend


def test_explicit_softcap_zero_and_none_are_noops(capture_backend):
    q, k, v = _make_qkv()
    out = flex_attention(q, k, v, softcap=0.0)
    assert out.shape == (1, 4, 16, 16)
    assert capture_backend["kwargs"]["alibi_slopes"] is None

    capture_backend.clear()
    flex_attention(q, k, v, softcap=None)
    assert capture_backend["called"] is True


def test_explicit_softcap_positive_raises_even_with_explicit_alibi(capture_backend):
    # softcap>0 is the blocker; it must still hard-error alongside explicit alibi.
    H = 4
    q, k, v = _make_qkv(Hq=H)
    slopes = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float32)
    with pytest.raises(NotImplementedError):
        flex_attention(q, k, v, alibi_slopes=slopes, softcap=15.0)


def test_no_explicit_args_matches_plain_dispatch(capture_backend):
    # Zero-regression guard: without the extension args the turbo path is taken
    # with alibi_slopes=None, exactly as before.
    q, k, v = _make_qkv()
    out = flex_attention(q, k, v)
    assert out.shape == (1, 4, 16, 16)
    assert capture_backend["kwargs"]["alibi_slopes"] is None
    assert capture_backend["kwargs"]["causal"] is False


def test_dropout_p_default_zero_and_sink_none_passthrough(capture_backend):
    # Zero-regression: defaults thread dropout_p=0.0 and sink=None to the backend.
    q, k, v = _make_qkv()
    out = flex_attention(q, k, v)
    assert out.shape == (1, 4, 16, 16)
    assert capture_backend["kwargs"]["dropout_p"] == 0.0
    assert capture_backend["kwargs"]["sink"] is None


def test_dropout_p_positive_passed_through(capture_backend):
    q, k, v = _make_qkv()
    flex_attention(q, k, v, dropout_p=0.1)
    assert abs(capture_backend["kwargs"]["dropout_p"] - 0.1) < 1e-9


def test_dropout_p_out_of_range_raises_via_entry(capture_backend):
    q, k, v = _make_qkv()
    with pytest.raises(ValueError):
        flex_attention(q, k, v, dropout_p=1.0)
    # The validation fires before any backend dispatch.
    assert "called" not in capture_backend


def test_sink_passed_through(capture_backend):
    # _make_qkv default D=16 (power of two), Hq=4 -> a valid len-4 fp32 sink.
    q, k, v = _make_qkv(Hq=4, D=16)
    sink = torch.arange(4, dtype=torch.float32)
    flex_attention(q, k, v, sink=sink.clone())
    passed = capture_backend["kwargs"]["sink"]
    assert passed is not None
    assert passed.shape == (4,)
    assert passed.dtype == torch.float32
    assert torch.allclose(passed.cpu(), sink)


def test_sink_invalid_length_raises_via_entry(capture_backend):
    q, k, v = _make_qkv(Hq=4, D=16)
    sink = torch.zeros(3, dtype=torch.float32)  # len 3 != Hq=4
    with pytest.raises(ValueError):
        flex_attention(q, k, v, sink=sink)
    assert "called" not in capture_backend


def test_sink_non_pow2_head_dim_raises_via_entry(capture_backend):
    # D=48 is not a power of two -> the sink kernel-path constraint rejects it.
    q, k, v = _make_qkv(Hq=4, D=48)
    sink = torch.zeros(4, dtype=torch.float32)
    with pytest.raises(ValueError):
        flex_attention(q, k, v, sink=sink)
    assert "called" not in capture_backend


def test_dropout_and_sink_default_off_matches_plain_dispatch(capture_backend):
    # Full zero-regression guard for the two new args together.
    q, k, v = _make_qkv()
    flex_attention(q, k, v)
    assert capture_backend["kwargs"]["dropout_p"] == 0.0
    assert capture_backend["kwargs"]["sink"] is None
    assert capture_backend["kwargs"]["alibi_slopes"] is None
    assert capture_backend["kwargs"]["bias"] is None


def test_bias_passed_through_adapted(capture_backend):
    # A 2D [Sq,Skv] fp32 bias is adapted to q's dtype and threaded to the backend.
    q, k, v = _make_qkv(Hq=4, S=16, D=16, dtype=torch.bfloat16)
    bias = torch.randn(16, 16, dtype=torch.float32)
    flex_attention(q, k, v, bias=bias)
    passed = capture_backend["kwargs"]["bias"]
    assert passed is not None
    assert passed.shape == (16, 16)
    assert passed.dtype == torch.bfloat16  # adapted to q's dtype


def test_bias_default_none_passthrough(capture_backend):
    q, k, v = _make_qkv()
    flex_attention(q, k, v)
    assert capture_backend["kwargs"]["bias"] is None


def test_bias_per_head_raises_via_entry(capture_backend):
    q, k, v = _make_qkv(Hq=4, S=16, D=16)
    bias = torch.randn(1, 4, 16, 16, dtype=torch.float16)  # per-head (H=4) not supported
    with pytest.raises(ValueError):
        flex_attention(q, k, v, bias=bias)
    assert "called" not in capture_backend


def test_bias_leading_singleton_4d_passed_through(capture_backend):
    # [1,1,Sq,Skv] is accepted (shared across batch/head) and squeezed to [Sq,Skv].
    q, k, v = _make_qkv(Hq=4, S=16, D=16, dtype=torch.float16)
    bias = torch.randn(1, 1, 16, 16, dtype=torch.float16)
    flex_attention(q, k, v, bias=bias)
    passed = capture_backend["kwargs"]["bias"]
    assert passed is not None
    assert passed.shape == (16, 16)
    assert passed.dtype == torch.float16


def test_varlen_causal_thd_passthrough(capture_varlen_backend):
    # THD is threaded to the backend verbatim (no transpose), with causal=True and
    # the defaults for every optional arg.
    H, D = 8, 128
    q = _make_thd(512, H, D)
    cu, max_s, total = _cu_from_seqlens([128, 128, 256])
    out = flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True)
    assert capture_varlen_backend["q_shape"] == (512, H, D)  # unchanged: no transpose
    assert out.shape == (512, H, D)
    kw = capture_varlen_backend["kwargs"]
    assert kw["causal"] is True
    assert kw["window_size"] == (-1, -1)
    assert kw["dropout_p"] == 0.0
    assert kw["alibi_slopes"] is None
    # sink is threaded only when supplied (newer-backend feature) -> absent here.
    assert kw.get("sink") is None
    # bias is not exposed by the varlen entry -> left to the backend default (absent).
    assert kw.get("bias") is None
    assert kw["deterministic"] is False
    assert capture_varlen_backend["cu_q"] is cu  # passed through unmodified
    assert capture_varlen_backend["max_q"] == max_s


def test_varlen_window_passed_through(capture_varlen_backend):
    q = _make_thd(512, 8, 128)
    cu, max_s, total = _cu_from_seqlens([256, 256])
    flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True, window_size=(128, 0))
    assert capture_varlen_backend["kwargs"]["window_size"] == (128, 0)


def test_varlen_scale_passed_through(capture_varlen_backend):
    q = _make_thd(256, 8, 128)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True, scale=0.5)
    assert capture_varlen_backend["kwargs"]["softmax_scale"] == 0.5


def test_varlen_dropout_passed_through(capture_varlen_backend):
    q = _make_thd(256, 8, 128)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True, dropout_p=0.1)
    assert abs(capture_varlen_backend["kwargs"]["dropout_p"] - 0.1) < 1e-9


def test_varlen_dropout_out_of_range_raises(capture_varlen_backend):
    q = _make_thd(256, 8, 128)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    with pytest.raises(ValueError):
        flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, dropout_p=1.0)
    assert "called" not in capture_varlen_backend


def test_varlen_alibi_passed_through(capture_varlen_backend):
    H, D = 4, 128
    q = _make_thd(256, H, D)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    slopes = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float32)
    flex_attention_varlen(
        q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True, alibi_slopes=slopes.clone()
    )
    passed = capture_varlen_backend["kwargs"]["alibi_slopes"]
    assert passed is not None
    assert passed.dtype == torch.float32
    assert torch.allclose(passed.cpu(), slopes)


def test_varlen_alibi_invalid_length_raises(capture_varlen_backend):
    q = _make_thd(256, 8, 128)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    slopes = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32)  # len 3 != Hq=8
    with pytest.raises(ValueError):
        flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, alibi_slopes=slopes)
    assert "called" not in capture_varlen_backend


def test_varlen_sink_passed_through(capture_varlen_backend):
    H, D = 4, 128  # D power of two, Hq=4
    q = _make_thd(256, H, D)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    sink = torch.arange(H, dtype=torch.float32)
    flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True, sink=sink.clone())
    passed = capture_varlen_backend["kwargs"]["sink"]
    assert passed is not None
    assert passed.shape == (H,)
    assert torch.allclose(passed.cpu(), sink)


def test_varlen_sink_invalid_length_raises(capture_varlen_backend):
    q = _make_thd(256, 4, 128)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    sink = torch.zeros(3, dtype=torch.float32)  # len 3 != Hq=4
    with pytest.raises(ValueError):
        flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, sink=sink)
    assert "called" not in capture_varlen_backend


def test_varlen_softcap_positive_raises_not_implemented(capture_varlen_backend):
    q = _make_thd(256, 8, 128)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    with pytest.raises(NotImplementedError):
        flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True, softcap=30.0)
    # The gate fires before any backend dispatch: the cap is never silently dropped.
    assert "called" not in capture_varlen_backend


def test_varlen_softcap_zero_and_none_are_noops(capture_varlen_backend):
    q = _make_thd(256, 8, 128)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True, softcap=0.0)
    assert capture_varlen_backend["called"] is True
    capture_varlen_backend.clear()
    flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True, softcap=None)
    assert capture_varlen_backend["called"] is True


def test_varlen_invalid_cu_raises_before_dispatch(capture_varlen_backend):
    q = _make_thd(512, 8, 128)
    cu, max_s, total = _cu_from_seqlens([128, 128, 256])
    bad = cu.to(torch.int64)  # wrong dtype
    with pytest.raises(ValueError):
        flex_attention_varlen(q, q.clone(), q.clone(), bad, bad, max_s, max_s, causal=True)
    assert "called" not in capture_varlen_backend


def test_varlen_cu_total_mismatch_raises(capture_varlen_backend):
    # q has 512 tokens but cu_seqlens says 384 -> rejected before dispatch.
    q = _make_thd(512, 8, 128)
    cu, max_s, _ = _cu_from_seqlens([128, 256])  # total 384 != 512
    with pytest.raises(ValueError):
        flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True)
    assert "called" not in capture_varlen_backend


def test_varlen_return_lse_returns_tuple(capture_varlen_backend):
    q = _make_thd(256, 8, 128)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    out = flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True, return_lse=True)
    assert isinstance(out, tuple) and len(out) == 2
    assert capture_varlen_backend["kwargs"]["return_lse"] is True


def test_varlen_gqa_passthrough(capture_varlen_backend):
    Hq, Hkv, D = 8, 2, 128
    q = _make_thd(256, Hq, D)
    k = _make_thd(256, Hkv, D)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    flex_attention_varlen(q, k, k.clone(), cu, cu, max_s, max_s, causal=True)
    assert capture_varlen_backend["q_shape"] == (256, Hq, D)
    assert capture_varlen_backend["k_shape"] == (256, Hkv, D)


def test_varlen_full_cross_attention_dispatches(capture_varlen_backend):
    # causal=False with different q/k per-segment lengths (cross attention) dispatches.
    q = _make_thd(384, 8, 128)
    kv = _make_thd(512, 8, 128)
    cu_q, max_q, total_q = _cu_from_seqlens([128, 256])
    cu_k, max_k, total_k = _cu_from_seqlens([256, 256])
    flex_attention_varlen(q, kv, kv.clone(), cu_q, cu_k, max_q, max_k, causal=False)
    kw = capture_varlen_backend["kwargs"]
    assert kw["causal"] is False
    assert capture_varlen_backend["q_shape"] == (384, 8, 128)
    assert capture_varlen_backend["k_shape"] == (512, 8, 128)


def test_flex_document_routes_to_varlen(capture_varlen_backend):
    seg = [128, 128, 256]
    total, H, D = 512, 8, 128
    q = _make_bhsd(1, H, total, D)
    bm = _doc_causal_block_mask(seg)
    out = flex_attention(q, q.clone(), q.clone(), block_mask=bm)
    assert capture_varlen_backend["called"] is True
    # bhsd [1,H,S,D] packed to THD [B*S, H, D].
    assert capture_varlen_backend["q_shape"] == (total, H, D)
    kw = capture_varlen_backend["kwargs"]
    assert kw["causal"] is True
    assert capture_varlen_backend["cu_q"].tolist() == [0, 128, 256, 512]
    assert capture_varlen_backend["max_q"] == 256
    # output unpacked back to bhsd
    assert out.shape == (1, H, total, D)


def test_flex_document_b2_cu_replicated(capture_varlen_backend):
    # Batch > 1 with identical (batch-independent) doc structure replicates cu_seqlens.
    seg = [128, 128]
    total, B, H, D = 256, 2, 8, 128
    q = _make_bhsd(B, H, total, D)
    bm = _doc_causal_block_mask(seg)
    out = flex_attention(q, q.clone(), q.clone(), block_mask=bm)
    assert capture_varlen_backend["cu_q"].tolist() == [0, 128, 256, 384, 512]
    assert capture_varlen_backend["q_shape"] == (B * total, H, D)
    assert out.shape == (B, H, total, D)


def test_flex_document_with_explicit_alibi(capture_varlen_backend):
    seg = [128, 128]
    total, H, D = 256, 4, 128
    q = _make_bhsd(1, H, total, D)
    bm = _doc_causal_block_mask(seg)
    slopes = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float32)
    flex_attention(q, q.clone(), q.clone(), block_mask=bm, alibi_slopes=slopes.clone())
    passed = capture_varlen_backend["kwargs"]["alibi_slopes"]
    assert passed is not None
    assert torch.allclose(passed.cpu(), slopes)
    assert capture_varlen_backend["kwargs"]["causal"] is True


def test_flex_document_bias_rejected(capture_varlen_backend):
    seg = [128, 128]
    total, H, D = 256, 8, 128
    q = _make_bhsd(1, H, total, D)
    bm = _doc_causal_block_mask(seg)
    bias = torch.randn(total, total, dtype=torch.float16)
    with pytest.raises(NotImplementedError):
        flex_attention(q, q.clone(), q.clone(), block_mask=bm, bias=bias)
    assert "called" not in capture_varlen_backend


def test_flex_document_return_lse_rejected(capture_varlen_backend):
    seg = [128, 128]
    total, H, D = 256, 8, 128
    q = _make_bhsd(1, H, total, D)
    bm = _doc_causal_block_mask(seg)
    with pytest.raises(NotImplementedError):
        flex_attention(q, q.clone(), q.clone(), block_mask=bm, return_lse=True)
    assert "called" not in capture_varlen_backend


def test_bshd_entry_returns_bshd_and_dispatches(capture_backend):
    B, S, H, D = 2, 32, 4, 16
    q, k, v = (_make_bshd(B, S, H, D) for _ in range(3))
    out = flex_attention_bshd(q, k, v)
    assert capture_backend["called"] is True
    # Backend sees bshd, and so does the caller -- no layout round-trip.
    assert capture_backend["q_shape"] == (B, S, H, D)
    assert out.shape == (B, S, H, D)


def test_bshd_entry_passes_qkv_without_copying(monkeypatch):
    seen = {}

    def fake(q, k, v, **kwargs):
        seen["q_ptr"] = q.data_ptr()
        seen["k_ptr"] = k.data_ptr()
        seen["v_ptr"] = v.data_ptr()
        return q.clone()

    monkeypatch.setattr("primus_turbo.pytorch.ops.attention.flash_attn_interface.flash_attn_func", fake)
    q, k, v = (_make_bshd() for _ in range(3))
    flex_attention_bshd(q, k, v)
    # A bshd-contiguous input reaches the kernel as the very same buffer.
    assert seen["q_ptr"] == q.data_ptr()
    assert seen["k_ptr"] == k.data_ptr()
    assert seen["v_ptr"] == v.data_ptr()


def test_bshd_entry_returns_backend_output_without_copying(monkeypatch):
    made = {}

    def fake(q, k, v, **kwargs):
        out = q.clone()
        made["ptr"] = out.data_ptr()
        return out

    monkeypatch.setattr("primus_turbo.pytorch.ops.attention.flash_attn_interface.flash_attn_func", fake)
    q, k, v = (_make_bshd() for _ in range(3))
    out = flex_attention_bshd(q, k, v)
    assert out.data_ptr() == made["ptr"]


def test_bhsd_entry_still_returns_bhsd(capture_backend):
    # The torch-compatible entry is untouched: bhsd in, bhsd out.
    B, H, S, D = 2, 4, 32, 16
    q, k, v = _make_qkv(B=B, Hq=H, S=S, D=D)
    out = flex_attention(q, k, v)
    assert capture_backend["q_shape"] == (B, S, H, D)
    assert out.shape == (B, H, S, D)


def test_bshd_and_bhsd_entries_agree(monkeypatch):
    # Same numbers either way: the bshd entry is a layout change, not a semantic one.
    def fake(q, k, v, **kwargs):
        return (q.float() + k.float() + v.float()).to(q.dtype)

    monkeypatch.setattr("primus_turbo.pytorch.ops.attention.flash_attn_interface.flash_attn_func", fake)
    B, S, H, D = 2, 32, 4, 16
    q, k, v = (_make_bshd(B, S, H, D) for _ in range(3))
    out_bshd = flex_attention_bshd(q, k, v)
    out_bhsd = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    assert torch.equal(out_bshd, out_bhsd.transpose(1, 2))


def test_bshd_entry_return_lse(capture_backend):
    B, S, H, D = 1, 32, 4, 16
    q, k, v = (_make_bshd(B, S, H, D) for _ in range(3))
    out, lse = flex_attention_bshd(q, k, v, return_lse=True)
    assert out.shape == (B, S, H, D)
    assert lse.shape == (B, H, S)


def test_bshd_entry_forwards_turbo_extension_args(capture_backend):
    B, S, H, D = 1, 32, 4, 16
    q, k, v = (_make_bshd(B, S, H, D) for _ in range(3))
    slopes = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float32)
    flex_attention_bshd(q, k, v, alibi_slopes=slopes.clone(), dropout_p=0.1, scale=0.5)
    kw = capture_backend["kwargs"]
    assert torch.allclose(kw["alibi_slopes"].cpu(), slopes)
    assert kw["dropout_p"] == pytest.approx(0.1)
    assert kw["softmax_scale"] == pytest.approx(0.5)


def test_bshd_entry_validation_errors_match_bhsd():
    # Same rejections as the bhsd entry (softcap is gated, GQA needs the flag).
    q, k, v = (_make_bshd() for _ in range(3))
    with pytest.raises(NotImplementedError):
        flex_attention_bshd(q, k, v, softcap=30.0)
    q8 = _make_bshd(H=8)
    with pytest.raises(ValueError):
        flex_attention_bshd(q8, k, v)


def test_bshd_entry_document_mask_returns_bshd(capture_varlen_backend):
    seg = [128, 128]
    S, H, D = 256, 4, 64
    q = _make_bshd(1, S, H, D)
    bm = _doc_causal_block_mask(seg)
    out = flex_attention_bshd(q, q.clone(), q.clone(), block_mask=bm)
    assert capture_varlen_backend["called"] is True
    assert out.shape == (1, S, H, D)


# ---------------------------------------------------------------------------
# Layout plumbing: the bhsd entry must not copy q/k/v for nothing.
#
# flash_attn_func takes a [B,S,H,D]-shaped tensor and reads the real memory order
# out of the strides; bhsd is one of the orders it addresses natively, so the
# transpose can stay a view. See primus_turbo/pytorch/ops/attention/flex/_layout.py.
# ---------------------------------------------------------------------------


def _ptr_capture(monkeypatch, out_layout="bhsd"):
    """Patch flash_attn_func to record the buffers it was handed.

    ``out_layout`` mimics how the aiter forward allocates its output: bhsd-ordered
    memory when it was given bhsd, plain bshd otherwise.
    """
    seen = {}

    def fake(q, k, v, **kwargs):
        seen["q_ptr"] = q.data_ptr()
        seen["k_ptr"] = k.data_ptr()
        seen["v_ptr"] = v.data_ptr()
        seen["q_stride"] = tuple(q.stride())
        seen["q_shape"] = tuple(q.shape)
        b, s, h, d = q.shape
        if out_layout == "bhsd":
            out = torch.empty((b, h, s, d), dtype=q.dtype).permute(0, 2, 1, 3)
        else:
            out = torch.empty((b, s, h, d), dtype=q.dtype)
        seen["out_ptr"] = out.data_ptr()
        return out

    monkeypatch.setattr(
        "primus_turbo.pytorch.ops.attention.flash_attn_interface.flash_attn_func",
        fake,
        raising=True,
    )
    return seen


def test_bhsd_entry_passes_qkv_without_copying(monkeypatch):
    seen = _ptr_capture(monkeypatch)
    B, H, S, D = 2, 4, 32, 16
    q, k, v = (_make_bhsd(B, H, S, D) for _ in range(3))
    flex_attention(q, k, v)
    # Same buffers, only re-strided: this is the whole point of the change.
    assert seen["q_ptr"] == q.data_ptr()
    assert seen["k_ptr"] == k.data_ptr()
    assert seen["v_ptr"] == v.data_ptr()
    # Backend still sees the [B,S,H,D] logical shape it documents...
    assert seen["q_shape"] == (B, S, H, D)
    # ...with bhsd strides (s0 >= s2 >= s1), which _infer_qkv_format reads back as bhsd.
    s0, s1, s2, s3 = seen["q_stride"]
    assert s3 == 1 and s0 >= s2 >= s1


def test_bhsd_entry_returns_backend_output_without_copying(monkeypatch):
    seen = _ptr_capture(monkeypatch, out_layout="bhsd")
    q, k, v = (_make_bhsd(2, 4, 32, 16) for _ in range(3))
    out = flex_attention(q, k, v)
    assert out.shape == (2, 4, 32, 16)
    assert out.is_contiguous()
    # The backend allocated in bhsd order, so transposing back is free.
    assert out.data_ptr() == seen["out_ptr"]


def test_bhsd_entry_still_returns_contiguous_when_backend_is_bshd(monkeypatch):
    # A backend that allocates plain bshd (older/other kernels) must still yield a
    # contiguous [B,H,S,D] to the caller -- there the copy is genuinely required.
    seen = _ptr_capture(monkeypatch, out_layout="bshd")
    q, k, v = (_make_bhsd(2, 4, 32, 16) for _ in range(3))
    out = flex_attention(q, k, v)
    assert out.shape == (2, 4, 32, 16)
    assert out.is_contiguous()
    assert out.data_ptr() != seen["out_ptr"]


def test_bhsd_entry_materialises_at_batch_one(monkeypatch):
    # B == 1 keeps the bshd-contiguous copy so the sbhd-only FlyDSL / HipKittens
    # backends stay eligible (attention_impl._sbhd_layout).
    seen = _ptr_capture(monkeypatch)
    q, k, v = (_make_bhsd(1, 4, 32, 16) for _ in range(3))
    flex_attention(q, k, v)
    assert seen["q_ptr"] != q.data_ptr()
    s0, s1, s2, s3 = seen["q_stride"]
    assert s3 == 1 and s0 >= s1 >= s2  # plain bshd


def test_bshd_entry_unaffected_by_passthrough_at_batch_one(monkeypatch):
    # flex_attention_bshd hands in a transposed view of bshd memory; collapsing it
    # lands back on the caller's own buffer, so nothing is copied at any batch size.
    seen = _ptr_capture(monkeypatch, out_layout="bshd")
    for b in (1, 2):
        q, k, v = (_make_bshd(b, 32, 4, 16) for _ in range(3))
        flex_attention_bshd(q, k, v)
        assert seen["q_ptr"] == q.data_ptr()


def test_bhsd_and_bshd_entries_agree_on_backend_arguments(monkeypatch):
    # The layout branch must change buffers only -- never what the kernel is asked for.
    seen_bhsd = {}
    seen_bshd = {}

    def make_fake(sink_dict):
        def fake(q, k, v, **kwargs):
            sink_dict.update(kwargs)
            b, s, h, d = q.shape
            return torch.empty((b, s, h, d), dtype=q.dtype)

        return fake

    monkeypatch.setattr(
        "primus_turbo.pytorch.ops.attention.flash_attn_interface.flash_attn_func",
        make_fake(seen_bhsd),
        raising=True,
    )
    q = _make_bhsd(2, 4, 32, 16)
    flex_attention(q, q.clone(), q.clone(), scale=0.25, dropout_p=0.1)

    monkeypatch.setattr(
        "primus_turbo.pytorch.ops.attention.flash_attn_interface.flash_attn_func",
        make_fake(seen_bshd),
        raising=True,
    )
    qb = _make_bshd(2, 32, 4, 16)
    flex_attention_bshd(qb, qb.clone(), qb.clone(), scale=0.25, dropout_p=0.1)

    assert seen_bhsd.keys() == seen_bshd.keys()
    for key in seen_bhsd:
        if isinstance(seen_bhsd[key], torch.Tensor):
            continue
        assert seen_bhsd[key] == seen_bshd[key], key
