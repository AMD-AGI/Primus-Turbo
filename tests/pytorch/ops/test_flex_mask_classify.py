###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for ``flex/_mask_classify.py``: block_mask -> kernel mask parameters."""

import pytest
import torch

from primus_turbo.pytorch.ops.attention.flex import _config as flex_config
from primus_turbo.pytorch.ops.attention.flex import _mask_classify as flex_mask_classify
from primus_turbo.pytorch.ops.attention.flex._mask_classify import (
    _classify_block_mask,
    _detect_document_causal_segments,
    _locate_left_window,
)
from primus_turbo.pytorch.ops.attention.flex_attention_interface import flex_attention

from .flex_test_utils import (
    _doc_causal_block_mask,
    _doc_causal_dense_mask,
    _DummyBlockMask,
    _make_bhsd,
)


def test_classify_full_mask():
    block_mask = _DummyBlockMask(lambda b, h, q, kv: True)
    cfg = _classify_block_mask(block_mask, B=1, H=1, q_len=16, kv_len=16)
    assert cfg["kind"] == "full"
    assert cfg["causal"] is False
    assert cfg["window_size"] == (-1, -1)


def test_classify_causal_mask():
    block_mask = _DummyBlockMask(lambda b, h, q, kv: q >= kv)
    cfg = _classify_block_mask(block_mask, B=1, H=1, q_len=16, kv_len=16)
    assert cfg["kind"] == "causal"
    assert cfg["causal"] is True
    assert cfg["window_size"] == (-1, -1)


def test_classify_none_is_full():
    cfg = _classify_block_mask(None, B=2, H=4, q_len=128, kv_len=128)
    assert cfg["kind"] == "full"
    assert cfg["causal"] is False


@pytest.mark.parametrize("window", [1, 5, 63, 128, 256])
def test_classify_sliding_window_causal_mask(window):
    block_mask = _DummyBlockMask(lambda b, h, q, kv: (q >= kv) & ((q - kv) <= window))
    cfg = _classify_block_mask(block_mask, B=1, H=1, q_len=512, kv_len=512)
    assert cfg["kind"] == "sliding_window_causal"
    assert cfg["causal"] is True
    assert cfg["window_size"] == (window, 0)


def test_classify_sliding_window_python_and_scalar_fallback():
    # ``and`` only works on python scalars -> exercises the element-wise fallback.
    window = 5
    block_mask = _DummyBlockMask(lambda b, h, q, kv: (q >= kv) and ((q - kv) <= window))
    cfg = _classify_block_mask(block_mask, B=1, H=1, q_len=32, kv_len=32)
    assert cfg["kind"] == "sliding_window_causal"
    assert cfg["window_size"] == (window, 0)


def test_classify_random_mask_unsupported():
    block_mask = _DummyBlockMask(lambda b, h, q, kv: ((q + kv) % 2) == 0)
    with pytest.raises(NotImplementedError):
        _classify_block_mask(block_mask, B=1, H=1, q_len=16, kv_len=16)


# ---- bidirectional band windows -------------------------------------------------
# A non-causal band ``-R <= q - kv <= L`` is exactly what flash_attn spells
# ``window_size=(L, R)``, and the csrc/CK entry forwards both edges to the forward and
# the backward alike, so the mapping is exact rather than an approximation.


def test_classify_symmetric_band():
    width = 2
    block_mask = _DummyBlockMask(lambda b, h, q, kv: (q - kv).abs() <= width)
    cfg = _classify_block_mask(block_mask, B=1, H=1, q_len=32, kv_len=32)
    assert cfg["kind"] == "sliding_window"
    assert cfg["causal"] is False
    assert cfg["window_size"] == (width, width)


def test_classify_asymmetric_band():
    left, right = 4, 1
    block_mask = _DummyBlockMask(lambda b, h, q, kv: ((q - kv) <= left) & ((q - kv) >= -right))
    cfg = _classify_block_mask(block_mask, B=1, H=1, q_len=48, kv_len=48)
    assert cfg["kind"] == "sliding_window"
    assert cfg["causal"] is False
    assert cfg["window_size"] == (left, right)


def test_classify_band_python_and_scalar_fallback():
    # ``and`` forces the element-wise probe path, as for the causal window above.
    block_mask = _DummyBlockMask(lambda b, h, q, kv: ((q - kv) <= 3) and ((kv - q) <= 2))
    cfg = _classify_block_mask(block_mask, B=1, H=1, q_len=32, kv_len=32)
    assert cfg["kind"] == "sliding_window"
    assert cfg["window_size"] == (3, 2)


def test_classify_holey_band_unsupported():
    # Band-shaped extent but with a hole in it: the reconstructed band would not match,
    # so this must be refused rather than widened into a real band.
    def mask_mod(b, h, q, kv):
        return ((q - kv).abs() <= 3) & ((q - kv) != 1)

    with pytest.raises(NotImplementedError):
        _classify_block_mask(_DummyBlockMask(mask_mod), B=1, H=1, q_len=32, kv_len=32)


def test_classify_band_wider_than_probe_unsupported():
    # Left edge parked on the probe boundary: on the truncated grid this is
    # indistinguishable from any wider left edge, so refuse instead of guessing. The
    # narrow right edge keeps the probe from being all-visible, which would classify as
    # a full mask before the band logic is ever reached.
    long_len = 4096

    def mask_mod(b, h, q, kv):
        return ((q - kv) <= long_len) & ((kv - q) <= 2)

    with pytest.raises(NotImplementedError):
        _classify_block_mask(_DummyBlockMask(mask_mod), B=1, H=1, q_len=long_len, kv_len=long_len)


def test_classify_position_dependent_band_unsupported():
    # Band width that grows with q is not translation invariant, so a single
    # window_size cannot describe it. The far-q edge probes catch this.
    long_len = 4096

    def mask_mod(b, h, q, kv):
        return (q - kv).abs() <= (q // 64 + 1)

    with pytest.raises(NotImplementedError):
        _classify_block_mask(_DummyBlockMask(mask_mod), B=1, H=1, q_len=long_len, kv_len=long_len)


def test_classify_head_dependent_unsupported():
    # Head 0 causal, other heads full -> depends on h, cannot map to one kernel call.
    def mask_mod(b, h, q, kv):
        return (q >= kv) | (h > 0)

    with pytest.raises(NotImplementedError):
        _classify_block_mask(_DummyBlockMask(mask_mod), B=1, H=4, q_len=32, kv_len=32)


def test_classify_batch_dependent_unsupported():
    def mask_mod(b, h, q, kv):
        return (q >= kv) | (b > 0)

    with pytest.raises(NotImplementedError):
        _classify_block_mask(_DummyBlockMask(mask_mod), B=4, H=1, q_len=32, kv_len=32)


# ---- _detect_document_causal_segments -------------------------------------


def test_detect_document_segments_basic():
    seg = [128, 128, 256]
    mask = _doc_causal_dense_mask(seg)
    got = _detect_document_causal_segments(mask, q_len=512, kv_len=512, q_probe=512, kv_probe=512)
    assert got == seg


def test_detect_document_segments_two_docs():
    seg = [200, 312]
    mask = _doc_causal_dense_mask(seg)
    got = _detect_document_causal_segments(mask, q_len=512, kv_len=512, q_probe=512, kv_probe=512)
    assert got == seg


def test_detect_document_segments_none_for_plain_causal():
    n = 64
    qi = torch.arange(n).view(n, 1)
    ki = torch.arange(n).view(1, n)
    mask = qi >= ki
    assert _detect_document_causal_segments(mask, q_len=n, kv_len=n, q_probe=n, kv_probe=n) is None


def test_detect_document_segments_none_for_swa():
    n, W = 64, 8
    qi = torch.arange(n).view(n, 1)
    ki = torch.arange(n).view(1, n)
    mask = (qi >= ki) & ((qi - ki) <= W)
    assert _detect_document_causal_segments(mask, q_len=n, kv_len=n, q_probe=n, kv_probe=n) is None


def test_detect_document_segments_none_when_truncated():
    # Probe smaller than the sequence -> boundaries beyond the probe are unknowable.
    seg = [128, 128]
    mask = _doc_causal_dense_mask(seg)[:128, :128]
    assert _detect_document_causal_segments(mask, q_len=256, kv_len=256, q_probe=128, kv_probe=128) is None


def test_detect_document_segments_none_for_non_square():
    seg = [128, 128]
    mask = _doc_causal_dense_mask(seg)
    assert _detect_document_causal_segments(mask, q_len=256, kv_len=200, q_probe=256, kv_probe=200) is None


def test_detect_document_segments_none_for_doc_with_hole():
    # Block diagonal + causal but with one interior position masked out -> not exact.
    seg = [128, 128]
    mask = _doc_causal_dense_mask(seg).clone()
    mask[10, 5] = False  # a hole inside doc 0's causal region
    assert _detect_document_causal_segments(mask, q_len=256, kv_len=256, q_probe=256, kv_probe=256) is None


# ---- classify returns document_causal -------------------------------------


def test_classify_document_causal():
    bm = _doc_causal_block_mask([128, 128, 256])
    cfg = _classify_block_mask(bm, B=1, H=1, q_len=512, kv_len=512)
    assert cfg["kind"] == "document_causal"
    assert cfg["causal"] is True
    assert cfg["window_size"] == (-1, -1)
    assert cfg["doc_seglens"] == [128, 128, 256]


def test_classify_document_single_doc_is_causal():
    # A single document is just plain causal, not the document-packing kind.
    bm = _doc_causal_block_mask([256])
    cfg = _classify_block_mask(bm, B=1, H=1, q_len=256, kv_len=256)
    assert cfg["kind"] == "causal"


def test_classify_document_beyond_probe_short_docs():
    # Docs shorter than the probe grid, sequence longer than it: the probed corner is
    # block-diagonal (neither causal nor a window), so recovery goes through the
    # full-sequence locator.
    seg = [128] * 8  # total 1024 > 512
    bm = _doc_causal_block_mask(seg)
    cfg = _classify_block_mask(bm, B=1, H=1, q_len=1024, kv_len=1024)
    assert cfg["kind"] == "document_causal"
    assert cfg["doc_seglens"] == seg


def test_classify_document_beyond_probe_first_doc_longer_than_probe():
    # The first document covers the whole probe grid, so the corner looks exactly
    # causal; only the far-position check plus the locator can tell it apart from a
    # plain causal / large sliding window mask.
    seg = [1024, 512, 512]
    bm = _doc_causal_block_mask(seg)
    cfg = _classify_block_mask(bm, B=1, H=1, q_len=2048, kv_len=2048)
    assert cfg["kind"] == "document_causal"
    assert cfg["doc_seglens"] == seg


def test_classify_document_beyond_probe_uneven_segments():
    seg = [37, 611, 1, 200, 175]
    total = sum(seg)
    bm = _doc_causal_block_mask(seg)
    cfg = _classify_block_mask(bm, B=1, H=1, q_len=total, kv_len=total)
    assert cfg["kind"] == "document_causal"
    assert cfg["doc_seglens"] == seg


def test_locate_document_segments_direct():
    seg = [300, 300, 424]
    bm = _doc_causal_block_mask(seg)
    assert flex_mask_classify._locate_document_segments(bm.mask_mod, q_len=1024, kv_len=1024) == seg


def test_locate_document_segments_single_doc_returns_none():
    # No boundary at all == plain causal; the locator must not claim a document mask.
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    assert flex_mask_classify._locate_document_segments(causal, q_len=1024, kv_len=1024) is None


def test_locate_document_segments_rejects_holed_mask():
    # Block-diagonal boundaries look right on the sub-diagonal, but an extra hole makes
    # the exact reconstruction fail -> None (caller raises, never silently wrong).
    seg = [256, 256, 512]
    base = _doc_causal_block_mask(seg).mask_mod

    def holed(b, h, q_idx, kv_idx):
        return base(b, h, q_idx, kv_idx) & ~((q_idx == 700) & (kv_idx == 600))

    assert flex_mask_classify._locate_document_segments(holed, q_len=1024, kv_len=1024) is None


def test_locate_document_segments_rejects_window_mask():
    # A large sliding window also has an invisible far corner; it must not be mistaken
    # for document packing.
    def swa(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= 600)

    assert flex_mask_classify._locate_document_segments(swa, q_len=1024, kv_len=1024) is None


def test_locate_document_segments_rejects_non_square():
    seg = [128, 128]
    bm = _doc_causal_block_mask(seg)
    assert flex_mask_classify._locate_document_segments(bm.mask_mod, q_len=256, kv_len=512) is None


def test_locate_document_segments_beyond_exact_verify_limit():
    # Past _DOC_EXACT_VERIFY_LIMIT the O(S^2) verification is refused rather than
    # downgraded to sampling: we decline to classify instead of risking a wrong route.
    seg = [512, 512]
    bm = _doc_causal_block_mask(seg)
    limit = flex_config._DOC_EXACT_VERIFY_LIMIT
    assert (
        flex_mask_classify._locate_document_segments(bm.mask_mod, q_len=limit + 1, kv_len=limit + 1) is None
    )


def test_flex_document_beyond_probe_routes_to_varlen(capture_varlen_backend):
    # Used to raise NotImplementedError (S > 512); now recovered and routed to varlen.
    seg = [128] * 8  # total 1024 > 512
    total, H, D = 1024, 8, 128
    q = _make_bhsd(1, H, total, D)
    bm = _doc_causal_block_mask(seg)
    out = flex_attention(q, q.clone(), q.clone(), block_mask=bm)
    assert capture_varlen_backend["called"] is True
    assert capture_varlen_backend["cu_q"].tolist() == [0, 128, 256, 384, 512, 640, 768, 896, 1024]
    assert capture_varlen_backend["max_q"] == 128
    assert out.shape == (1, H, total, D)


def test_flex_document_too_long_rejected(capture_varlen_backend):
    # Beyond the exact-verification limit the boundaries cannot be verified in full ->
    # not routed, falls through to NotImplementedError (never silently wrong).
    total = flex_config._DOC_EXACT_VERIFY_LIMIT + 512
    seg = [512] * (total // 512)
    H, D = 2, 64
    q = _make_bhsd(1, H, total, D)
    bm = _doc_causal_block_mask(seg)
    with pytest.raises(NotImplementedError):
        flex_attention(q, q.clone(), q.clone(), block_mask=bm)
    assert "called" not in capture_varlen_backend


def _swa_block_mask(window):
    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= window)

    return _DummyBlockMask(mask_mod)


def _swa_mask_mod(window):
    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= window)

    return mask_mod


# ---- _locate_left_window direct ------------------------------------------


@pytest.mark.parametrize("window", [256, 512, 1024, 2048, 4096])
def test_locate_left_window_recovers_w(window):
    S = 8192
    got = _locate_left_window(_swa_mask_mod(window), q_len=S, kv_len=S)
    assert got == window


def test_locate_left_window_none_for_full_causal():
    # No window (far end still visible) -> not a window; caller keeps it causal.
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    assert _locate_left_window(causal, q_len=8192, kv_len=8192) is None


def test_locate_left_window_none_for_non_translation_invariant():
    # Window shrinks at the sampled midpoint row -> exact per-row check fails.
    W = 1024
    S = 8192

    def mask_mod(b, h, q_idx, kv_idx):
        # At the exact middle row the window is smaller; everywhere else it is W.
        w = torch.where(q_idx == (S // 2), torch.as_tensor(256), torch.as_tensor(W))
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= w)

    assert _locate_left_window(mask_mod, q_len=S, kv_len=S) is None


def test_locate_left_window_none_for_hole_on_last_row():
    W = 1024
    S = 8192

    def mask_mod(b, h, q_idx, kv_idx):
        base = (q_idx >= kv_idx) and ((q_idx - kv_idx) <= W)
        hole = (q_idx == (S - 1)) and (kv_idx == (S - 1 - 500))  # inside the window
        return base and not hole

    assert _locate_left_window(mask_mod, q_len=S, kv_len=S) is None


def test_locate_left_window_none_for_non_square():
    assert _locate_left_window(_swa_mask_mod(1024), q_len=8192, kv_len=4096) is None


# ---- classification integration (the real fix) ---------------------------


@pytest.mark.parametrize("window", [256, 512, 1024, 2048, 4096])
def test_classify_large_window_on_long_sequence(window):
    # The exact case the baseline benchmark hit: SWA(W>512) on S=8192 was raising
    # NotImplementedError; it must now classify as sliding_window_causal(W, 0).
    cfg = _classify_block_mask(_swa_block_mask(window), B=2, H=32, q_len=8192, kv_len=8192)
    assert cfg["kind"] == "sliding_window_causal"
    assert cfg["causal"] is True
    assert cfg["window_size"] == (window, 0)


def test_classify_large_window_w1024_matches_task_case():
    cfg = _classify_block_mask(_swa_block_mask(1024), B=2, H=32, q_len=2048, kv_len=2048)
    assert cfg["window_size"] == (1024, 0)


def test_classify_full_causal_still_causal_on_long_sequence():
    # No window: far end visible -> stays causal (not misdetected as a window).
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    cfg = _classify_block_mask(_DummyBlockMask(causal), B=1, H=1, q_len=8192, kv_len=8192)
    assert cfg["kind"] == "causal"
    assert cfg["window_size"] == (-1, -1)


def test_classify_window_larger_than_seq_is_causal():
    # A window >= sequence length is effectively full causal (far end visible).
    cfg = _classify_block_mask(_swa_block_mask(100000), B=1, H=1, q_len=8192, kv_len=8192)
    assert cfg["kind"] == "causal"


def test_classify_nonstandard_long_mask_still_raises():
    # Causal in the corner + invisible far corner, but NOT a clean window (window
    # depends on position at a sampled row) -> still NotImplementedError.
    S = 4096

    def mask_mod(b, h, q_idx, kv_idx):
        w = torch.where(q_idx < (S // 2), torch.as_tensor(1024), torch.as_tensor(300))
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= w)

    with pytest.raises(NotImplementedError):
        _classify_block_mask(_DummyBlockMask(mask_mod), B=1, H=1, q_len=S, kv_len=S)


# ---- document packing: the general (non-autoregressive / windowed) shapes --------
# Packing is same_doc(q, kv) intersected with an optional causal term and an optional
# within-document window. flash_attn_varlen_func applies causal and window_size *per
# segment* on top of cu_seqlens, so all four combinations lower exactly. The
# bidirectional ones are the ones image / video diffusion models actually use: those
# models are not autoregressive, and multi-resolution training packs several samples
# into one sequence.


def _doc_id(seg_lens):
    return torch.cat([torch.full((s,), i, dtype=torch.int64) for i, s in enumerate(seg_lens)])


def _bidirectional_doc_mask(seg_lens):
    document_id = _doc_id(seg_lens)

    def mask_mod(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    return _DummyBlockMask(mask_mod)


def test_classify_bidirectional_document_packing():
    seg = [6, 10, 4]
    cfg = _classify_block_mask(_bidirectional_doc_mask(seg), B=1, H=1, q_len=sum(seg), kv_len=sum(seg))
    assert cfg["kind"] == "document"
    assert cfg["causal"] is False
    assert cfg["window_size"] == (-1, -1)
    assert cfg["doc_seglens"] == seg


def test_classify_bidirectional_document_packing_beyond_the_probe():
    # Longer than the 512 probe grid, so the boundaries have to come from mask_mod over
    # the full sequence and be verified chunk by chunk.
    seg = [700, 500, 824]
    cfg = _classify_block_mask(_bidirectional_doc_mask(seg), B=1, H=1, q_len=sum(seg), kv_len=sum(seg))
    assert cfg["kind"] == "document"
    assert cfg["causal"] is False
    assert cfg["doc_seglens"] == seg


def test_classify_document_packing_with_within_document_window():
    seg = [12, 20]
    document_id = _doc_id(seg)

    def mask_mod(b, h, q_idx, kv_idx):
        same = document_id[q_idx] == document_id[kv_idx]
        return same & (q_idx >= kv_idx) & ((q_idx - kv_idx) <= 3)

    cfg = _classify_block_mask(_DummyBlockMask(mask_mod), B=1, H=1, q_len=sum(seg), kv_len=sum(seg))
    assert cfg["kind"] == "document"
    assert cfg["causal"] is True
    assert cfg["window_size"] == (3, 0)
    assert cfg["doc_seglens"] == seg


def test_classify_bidirectional_document_packing_with_window():
    seg = [14, 18]
    document_id = _doc_id(seg)

    def mask_mod(b, h, q_idx, kv_idx):
        same = document_id[q_idx] == document_id[kv_idx]
        return same & ((q_idx - kv_idx) <= 2) & ((kv_idx - q_idx) <= 2)

    cfg = _classify_block_mask(_DummyBlockMask(mask_mod), B=1, H=1, q_len=sum(seg), kv_len=sum(seg))
    assert cfg["kind"] == "document"
    assert cfg["causal"] is False
    assert cfg["window_size"] == (2, 2)
    assert cfg["doc_seglens"] == seg


def test_plain_document_causal_keeps_its_historical_kind():
    # Regression: the autoregressive shape must still take the original path, so
    # anything keying on kind == "document_causal" keeps working.
    seg = [8, 8, 16]
    cfg = _classify_block_mask(_doc_causal_block_mask(seg), B=1, H=1, q_len=sum(seg), kv_len=sum(seg))
    assert cfg["kind"] == "document_causal"
    assert cfg["causal"] is True
    assert cfg["doc_seglens"] == seg


def test_classify_cross_document_leak_unsupported():
    # Block diagonal except one stray visible position across a boundary: the exact
    # reconstruction must reject it rather than round it to clean packing.
    seg = [8, 8]
    document_id = _doc_id(seg)

    def mask_mod(b, h, q_idx, kv_idx):
        same = document_id[q_idx] == document_id[kv_idx]
        return same | ((q_idx == 9) & (kv_idx == 0))

    with pytest.raises(NotImplementedError):
        _classify_block_mask(_DummyBlockMask(mask_mod), B=1, H=1, q_len=sum(seg), kv_len=sum(seg))


def test_classify_identity_mask_is_a_degenerate_band_not_packing():
    # Every "document" would be one token long, so this is not packing -- but it *is* a
    # width-zero band, which window_size=(0, 0) expresses exactly. Take the band route
    # rather than routing S single-token segments through varlen.
    block_mask = _DummyBlockMask(lambda b, h, q, kv: q == kv)
    cfg = _classify_block_mask(block_mask, B=1, H=1, q_len=32, kv_len=32)
    assert cfg["window_size"] == (0, 0)
    assert "doc_seglens" not in cfg


def test_all_visible_probe_corner_does_not_silently_become_full():
    # The probed corner is entirely visible, but the mask closes past the probe limit.
    # Answering "full attention" on the strength of the corner would silently train a
    # different model, so this must raise instead.
    n = 4096
    block_mask = _DummyBlockMask(lambda b, h, q, kv: (q - kv <= 2000) & (kv - q <= 2000))
    with pytest.raises(NotImplementedError, match="fully visible"):
        _classify_block_mask(block_mask, B=1, H=1, q_len=n, kv_len=n)


def test_genuinely_full_mask_still_classifies_as_full_past_the_probe():
    block_mask = _DummyBlockMask(lambda b, h, q, kv: q >= 0)
    cfg = _classify_block_mask(block_mask, B=1, H=1, q_len=2048, kv_len=2048)
    assert cfg["kind"] == "full"
