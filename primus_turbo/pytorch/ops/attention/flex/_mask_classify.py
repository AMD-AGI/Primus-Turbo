###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Recover ``flash_attn`` kernel parameters from an opaque ``block_mask``.

The flex interface expresses a mask as an arbitrary Python predicate; the aiter
kernels take a fixed form (``is_causal`` + ``window_size_left`` + ``cu_seqlens``).
This module bridges the two by *runtime classification*: probe ``mask_mod`` on an
index grid, match the result against the full / causal / sliding-window-causal /
document-causal templates, and verify the match exactly before accepting it.
Anything that does not match exactly is rejected rather than approximated.
"""

from typing import Any, Callable, Dict, Optional

import torch

from ._cache import _CACHE_MISS, _CLASSIFY_CACHE, _cache_get, _cache_put
from ._config import _DOC_EXACT_VERIFY_LIMIT, _DOC_VERIFY_CHUNK, _MASK_PROBE_LIMIT
from ._probe import (
    _call_mask_mod,
    _mask_is_bh_dependent,
    _probe_mask_grid,
    _probe_mask_pairs,
    _probe_mask_row,
)


def _detect_document_causal_segments(
    mask: torch.Tensor,
    *,
    q_len: int,
    kv_len: int,
    q_probe: int,
    kv_probe: int,
) -> Optional[list]:
    """Recover per-document segment lengths from a block-diagonal causal mask.

    Recognises the *document packing* pattern ``same_doc(q,kv) & (q >= kv)`` (block
    diagonal along the sequence, causal within each document) and returns the list of
    document lengths, or ``None`` if the mask is not exactly that pattern.

    Correctness-first, never silently wrong:

    * Only a **square, fully-probed** self-attention mask is considered (``q_len ==
      kv_len`` and the probe covers the whole sequence, i.e. ``S <= _MASK_PROBE_LIMIT``);
      a truncated probe cannot see document boundaries past the limit, so it bails.
    * Document boundaries are read off the sub-diagonal (token ``i`` starts a new
      document iff it may not attend token ``i-1``), then the full block-diagonal +
      causal mask is *reconstructed and compared for exact equality*. Any deviation
      (a window, an off-diagonal hole, a non-causal block, ...) fails the check and
      returns ``None`` -- so the caller falls through to its normal handling.
    * A single document (``len < 2``) is plain causal, handled elsewhere; only genuine
      multi-document packing is returned here.
    """
    if q_len != kv_len or q_probe != q_len or kv_probe != kv_len:
        return None
    n = q_len
    if n <= 1:
        return None

    # Diagonal must be fully visible (each token attends itself); a hole here means it
    # is not a document-causal mask.
    if not bool(mask.diagonal().all().item()):
        return None

    sub_diag = mask.diagonal(offset=-1)  # mask[i, i-1] for i in 1..n-1
    seg_lens = []
    start = 0
    for i in range(1, n):
        if not bool(sub_diag[i - 1].item()):
            seg_lens.append(i - start)
            start = i
    seg_lens.append(n - start)
    if len(seg_lens) < 2:
        return None

    doc_id = torch.empty(n, dtype=torch.int64)
    pos = 0
    for d, s in enumerate(seg_lens):
        doc_id[pos : pos + s] = d
        pos += s
    qi = torch.arange(n).view(n, 1)
    ki = torch.arange(n).view(1, n)
    expected = (doc_id.view(n, 1) == doc_id.view(1, n)) & (qi >= ki)
    if not torch.equal(mask, expected):
        return None
    return seg_lens


def _locate_document_segments(
    mask_mod: Callable, *, q_len: int, kv_len: int, causal: bool = True
) -> Optional[list]:
    """Recover document lengths for a packed mask whose sequence exceeds the probe grid.

    :func:`_detect_document_causal_segments` only looks at the already-probed
    ``<= _MASK_PROBE_LIMIT`` corner, so packed sequences longer than that used to raise
    ``NotImplementedError`` even though the pattern is perfectly expressible. This
    variant works directly on ``mask_mod`` over the *full* sequence:

    ``causal`` selects which packed pattern to verify against: ``same_doc(q,kv) & (q>=kv)``
    (autoregressive packing) or plain ``same_doc(q,kv)`` (bidirectional packing, i.e. what
    diffusion / encoder models pack). Both lower onto the same block-diagonal
    ``cu_seqlens``; only the ``causal`` flag handed to the varlen kernel differs.

    1. Read the diagonal and the sub-diagonal with two vectorised calls (O(S) elements,
       O(1) python-level mask_mod invocations). Token ``i`` starts a new document iff it
       may not attend token ``i-1``; the diagonal must be fully visible.
    2. **Verify exactly, never by sampling**: reconstruct ``same_doc(q,kv) & (q>=kv)``
       and compare it row-block by row-block against the real mask (one vectorised call
       per ``_DOC_VERIFY_CHUNK`` rows, so peak memory stays at a few MB instead of
       ``S^2``). Any deviation returns ``None`` and the caller raises, exactly as before.

    Guarded by ``_DOC_EXACT_VERIFY_LIMIT``: beyond it the full comparison would cost
    ``O(S^2)``, so we decline rather than downgrade to sampled verification -- an
    unverifiable mask must never be routed.
    """
    if q_len != kv_len:
        return None
    n = q_len
    if n <= 1 or n > _DOC_EXACT_VERIFY_LIMIT:
        return None

    idx = torch.arange(n)
    if not bool(_probe_mask_pairs(mask_mod, idx, idx).all().item()):
        return None  # a hole on the diagonal: not document packing

    sub_diag = _probe_mask_pairs(mask_mod, idx[1:], idx[:-1])  # mask[i, i-1]
    boundaries = (~sub_diag).nonzero().flatten().add(1).tolist()
    if not boundaries:
        return None  # single document == plain causal, handled elsewhere

    seg_lens = []
    start = 0
    for b in boundaries:
        seg_lens.append(b - start)
        start = b
    seg_lens.append(n - start)
    if len(seg_lens) < 2:
        return None

    doc_id = torch.empty(n, dtype=torch.int64)
    pos = 0
    for d, s in enumerate(seg_lens):
        doc_id[pos : pos + s] = d
        pos += s

    kv_idx = torch.arange(n).view(1, n)
    for lo in range(0, n, _DOC_VERIFY_CHUNK):
        hi = min(n, lo + _DOC_VERIFY_CHUNK)
        q_rows = torch.arange(lo, hi)
        q_idx = q_rows.view(-1, 1)
        try:
            raw = mask_mod(0, 0, q_idx, kv_idx)
            block = torch.as_tensor(raw, dtype=torch.bool)
            if block.shape != (hi - lo, n):
                block = block.broadcast_to((hi - lo, n))
        except Exception:
            block = torch.stack([_probe_mask_row(mask_mod, int(q), n) for q in q_rows])
        expected = doc_id[q_rows].view(-1, 1) == doc_id.view(1, n)
        if causal:
            expected = expected & (q_idx >= kv_idx)
        if not torch.equal(block, expected):
            return None
    return seg_lens


def _detect_document_blocks(
    mask: torch.Tensor,
    *,
    q_len: int,
    kv_len: int,
    q_probe: int,
    kv_probe: int,
) -> Optional[Dict[str, Any]]:
    """Recover packing *and* the within-document pattern from a fully-probed mask.

    :func:`_detect_document_causal_segments` only recognises the autoregressive shape
    ``same_doc(q,kv) & (q>=kv)``. This generalisation additionally covers the three
    combinations that the varlen kernel can express just as exactly, because
    ``flash_attn_varlen_func`` takes ``causal`` and ``window_size`` alongside
    ``cu_seqlens`` and applies them *within* each packed segment:

    * ``same_doc``                                  -> causal=False, window (-1,-1)
    * ``same_doc & (q>=kv)``                        -> causal=True,  window (-1,-1)
    * ``same_doc & (q>=kv) & (q-kv<=L)``            -> causal=True,  window (L, 0)
    * ``same_doc & (-R<=q-kv<=L)``                  -> causal=False, window (L, R)

    The bidirectional entries are the ones that matter in practice: image and video
    diffusion models are not autoregressive, and multi-resolution / multi-duration
    training packs several samples into one sequence, which is exactly
    ``document_id[q] == document_id[kv]`` with no causal term.

    Returns ``{"seglens", "causal", "window_size"}`` or ``None``. Recognition is by
    *exact reconstruction*: boundaries are read off the sub-diagonal, the candidate
    pattern is rebuilt and compared for equality, and the first exact match wins
    (unbounded before windowed, so a plain packed mask is never described as a window).
    Anything that does not reproduce the probed mask bit for bit returns ``None``.
    """
    if q_len != kv_len or q_probe != q_len or kv_probe != kv_len:
        return None
    n = q_len
    if n <= 1:
        return None

    # Every token must at least attend itself; a hole here is not document packing.
    if not bool(mask.diagonal().all().item()):
        return None

    # Token ``i`` starts a new document iff it may not attend token ``i-1``. This read
    # is only a hypothesis -- the reconstruction below is what makes it safe.
    sub_diag = mask.diagonal(offset=-1)
    boundaries = (~sub_diag).nonzero().flatten().add(1).tolist()
    if not boundaries:
        return None  # single document: plain causal / band, handled by the other paths

    seg_lens = []
    start = 0
    for b in boundaries:
        seg_lens.append(b - start)
        start = b
    seg_lens.append(n - start)
    if len(seg_lens) < 2 or max(seg_lens) < 2:
        # All-length-1 "documents" means the mask is the identity. Nothing is gained by
        # routing S single-token segments through varlen, and the shape is far more
        # likely to be an arbitrary mask that happens to hide the sub-diagonal.
        return None

    doc_id = torch.empty(n, dtype=torch.int64)
    pos = 0
    for d, s in enumerate(seg_lens):
        doc_id[pos : pos + s] = d
        pos += s
    same_doc = doc_id.view(n, 1) == doc_id.view(1, n)

    qi = torch.arange(n).view(n, 1)
    ki = torch.arange(n).view(1, n)
    delta = qi - ki
    visible_delta = delta[mask]
    if visible_delta.numel() == 0:
        return None
    left = int(visible_delta.max().item())
    right = int((-visible_delta).max().item())

    candidates = (
        (True, (-1, -1), same_doc & (delta >= 0)),
        (False, (-1, -1), same_doc),
        (True, (left, 0), same_doc & (delta >= 0) & (delta <= left)),
        (False, (left, right), same_doc & (delta <= left) & (delta >= -right)),
    )
    for causal, window, expected in candidates:
        if torch.equal(mask, expected):
            return {"seglens": seg_lens, "causal": causal, "window_size": window}
    return None


def _locate_left_window(mask_mod: Callable, *, q_len: int, kv_len: int) -> Optional[int]:
    """Locate a large left-window size ``W`` whose boundary sits beyond the probe grid.

    Called only when the probed corner is *exactly causal* yet the far corner
    (``mask_mod(q_len-1, 0)``) is invisible -- i.e. there is a sliding window whose
    left edge ``W`` is bigger than ``_MASK_PROBE_LIMIT`` (e.g. W=1024/2048/4096 on a
    long sequence). We binary-search the last query row for the True->False flip
    (``W = max d s.t. mask_mod(S-1, S-1-d)`` is visible), then *verify exactly* that
    the mask is a standard left-window causal ``(q>=kv) & (q-kv<=W)`` by comparing
    full rows at several sampled query positions. Returns ``W`` (``>= 0``) or ``None``
    if it is not a clean translation-invariant left window (caller then falls back to
    ``NotImplementedError``, i.e. behaviour is unchanged for anything non-standard).
    """
    if q_len != kv_len:
        return None  # a windowed causal mask is square self-attention
    n = q_len
    last = n - 1
    # Endpoints of the search: diagonal (d=0) must be visible; the far end (d=last)
    # must be invisible for a window to exist (the caller established this, re-checked
    # here to be self-contained / robust to caller changes).
    if not _call_mask_mod(mask_mod, 0, 0, last, last):
        return None
    if _call_mask_mod(mask_mod, 0, 0, last, 0):
        return None

    lo, hi = 0, last  # visible at lo, invisible at hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _call_mask_mod(mask_mod, 0, 0, last, last - mid):
            lo = mid
        else:
            hi = mid
    window = lo

    # Exact verification: sample diverse rows (the boundary rows, the probe limit, and
    # ~16 evenly-spaced rows spanning the sequence) and require each to match the window
    # pattern bit-for-bit. Full-row equality catches holes; the spread of rows catches
    # non-translation-invariant windows -- so we never silently accept a non-window
    # mask (anything unclean returns None -> caller raises, unchanged behaviour).
    kv_idx = torch.arange(kv_len)
    sample_rows = {last, window, window + 1, min(n - 1, _MASK_PROBE_LIMIT), n // 2, (3 * n) // 4, n // 4}
    stride = max(1, n // 16)
    sample_rows.update(range(0, n, stride))
    for q_pos in sorted(sample_rows):
        if q_pos < 0 or q_pos >= n:
            continue
        row = _probe_mask_row(mask_mod, q_pos, kv_len)
        expected = (kv_idx <= q_pos) & ((q_pos - kv_idx) <= window)
        if not torch.equal(row, expected):
            return None
    return window


def _classify_band_mask(
    mask: torch.Tensor,
    *,
    delta: torch.Tensor,
    mask_mod: Optional[Callable],
    q_len: int,
    kv_len: int,
    q_probe: int,
    kv_probe: int,
) -> Optional[Dict[str, Any]]:
    """Recognise a non-causal band mask ``-R <= q - kv <= L`` (bidirectional local
    attention), or return ``None`` if the probed mask is not exactly that.

    ``flash_attn``'s ``window_size=(left, right)`` *is* this band, and the csrc/CK
    entry forwards both edges verbatim, so the mapping is exact rather than an
    approximation. Only reached once the caller has established that the mask is not
    causal (something is visible above the diagonal) and not all-ones.

    Never silently wrong:

    * ``L`` and ``R`` are read off the probed grid and the band is then *reconstructed
      and compared for exact equality*; any hole, block structure, or asymmetry fails.
    * When the probe did not cover the whole sequence, an edge is only accepted if it
      sits strictly inside the probed grid (so it was actually observed, not clipped)
      **and** ``mask_mod`` confirms the same edge at a far query position -- a band that
      is not translation invariant cannot be expressed as one ``window_size``.
    """
    visible_delta = delta[mask]
    if visible_delta.numel() == 0:
        return None
    left = int(visible_delta.max().item())
    right = int((-visible_delta).max().item())
    if right <= 0:
        # Causal or left-window causal; the caller's existing paths handle those and
        # this function must not change their classification.
        return None
    if left < 0:
        # Strictly above the diagonal: the query cannot even see itself. flash_attn's
        # window is anchored on the diagonal and cannot express that.
        return None

    band = (delta <= left) & (delta >= -right)
    if not torch.equal(mask, band):
        return None

    truncated = q_len > q_probe or kv_len > kv_probe
    if truncated:
        if mask_mod is None:
            return None
        # An edge that lands on the last probed offset may really be larger and merely
        # clipped by the grid, so refuse rather than guess.
        if left >= q_probe - 1 or right >= kv_probe - 1:
            raise NotImplementedError(
                f"Turbo flex compat layer detected a band mask whose edge may exceed the probe "
                f"limit {_MASK_PROBE_LIMIT}; the window size could not be verified. Please express "
                "the window explicitly via create_block_mask on a shorter probe, or use the "
                "codegen path."
            )
        far_q = q_len - 1
        checks = []
        if far_q - left >= 0:
            checks.append((_call_mask_mod(mask_mod, 0, 0, far_q, far_q - left), True))
        if far_q - left - 1 >= 0:
            checks.append((_call_mask_mod(mask_mod, 0, 0, far_q, far_q - left - 1), False))
        if far_q + right < kv_len:
            checks.append((_call_mask_mod(mask_mod, 0, 0, far_q, far_q + right), True))
        if far_q + right + 1 < kv_len:
            checks.append((_call_mask_mod(mask_mod, 0, 0, far_q, far_q + right + 1), False))
        if any(bool(got) is not want for got, want in checks):
            raise NotImplementedError(
                "Turbo flex compat layer does not support this block_mask: the band around the "
                "diagonal is not translation-invariant over long sequences, so a single "
                "window_size cannot be determined."
            )

    return {
        "kind": "sliding_window",
        "causal": False,
        "window_size": (left, right),
    }


def _classify_document_mask(
    mask: torch.Tensor,
    *,
    mask_mod: Optional[Callable],
    q_len: int,
    kv_len: int,
    q_probe: int,
    kv_probe: int,
    causal_hint: bool,
) -> Optional[Dict[str, Any]]:
    """Try to classify ``mask`` as document packing, on a full or a truncated probe.

    Returns a routing config with ``kind="document"`` (dispatched through the varlen
    backend exactly like ``document_causal``, but carrying the recovered ``causal`` flag
    and within-document ``window_size``) or ``None``.

    On a truncated probe only the *unwindowed* pattern is recoverable, because the
    boundaries then come from ``mask_mod`` over the full sequence and are verified
    against ``same_doc [& causal]``; ``causal_hint`` says which of the two to verify.
    A window whose edge lies past the probe would be unverifiable there, so it is not
    guessed at.
    """
    blocks = _detect_document_blocks(mask, q_len=q_len, kv_len=kv_len, q_probe=q_probe, kv_probe=kv_probe)
    if blocks is None and (q_len > q_probe or kv_len > kv_probe) and mask_mod is not None:
        seglens = _locate_document_segments(mask_mod, q_len=q_len, kv_len=kv_len, causal=causal_hint)
        if seglens is not None:
            blocks = {"seglens": seglens, "causal": causal_hint, "window_size": (-1, -1)}
    if blocks is None:
        return None
    return {
        "kind": "document",
        "causal": blocks["causal"],
        "window_size": blocks["window_size"],
        "doc_seglens": blocks["seglens"],
    }


def _verify_full_mask(mask_mod: Callable, *, q_len: int, kv_len: int) -> bool:
    """Check that ``mask_mod`` is True everywhere, past the probed corner.

    Exact (row-block by row-block) while the comparison stays affordable; beyond that
    it falls back to sampled rows -- each sampled row is still checked over the *whole*
    key range, so any restriction that touches a sampled query position is caught.
    """
    if q_len <= _DOC_EXACT_VERIFY_LIMIT and kv_len <= _DOC_EXACT_VERIFY_LIMIT:
        kv_idx = torch.arange(kv_len).view(1, kv_len)
        for lo in range(0, q_len, _DOC_VERIFY_CHUNK):
            hi = min(q_len, lo + _DOC_VERIFY_CHUNK)
            q_rows = torch.arange(lo, hi)
            try:
                raw = mask_mod(0, 0, q_rows.view(-1, 1), kv_idx)
                block = torch.as_tensor(raw, dtype=torch.bool)
                if block.shape != (hi - lo, kv_len):
                    block = block.broadcast_to((hi - lo, kv_len))
            except Exception:
                block = torch.stack([_probe_mask_row(mask_mod, int(q), kv_len) for q in q_rows])
            if not bool(block.all().item()):
                return False
        return True

    rows = {0, q_len - 1, q_len // 2, q_len // 4, (3 * q_len) // 4, min(q_len - 1, _MASK_PROBE_LIMIT)}
    for q_pos in sorted(rows):
        if not bool(_probe_mask_row(mask_mod, int(q_pos), kv_len).all().item()):
            return False
    return True


def _classify_probed_mask(
    mask: torch.Tensor,
    *,
    mask_mod: Optional[Callable],
    q_len: int,
    kv_len: int,
    q_probe: int,
    kv_probe: int,
) -> Dict[str, Any]:
    probe_truncated = q_len > q_probe or kv_len > kv_probe
    if bool(mask.all().item()):
        if not probe_truncated or mask_mod is None:
            return {"kind": "full", "causal": False, "window_size": (-1, -1)}
        # The probe only saw the top-left corner, and that corner was all-visible. That
        # is *not* evidence of full attention: a bidirectionally packed sequence whose
        # first document is longer than the probe, or a symmetric band wider than the
        # probe, looks exactly like this. Answering "full" here would silently run dense
        # attention for a config that asked for something else, so confirm against
        # mask_mod over the whole sequence before believing the corner.
        seglens = _locate_document_segments(mask_mod, q_len=q_len, kv_len=kv_len, causal=False)
        if seglens is not None:
            return {
                "kind": "document",
                "causal": False,
                "window_size": (-1, -1),
                "doc_seglens": seglens,
            }
        if _verify_full_mask(mask_mod, q_len=q_len, kv_len=kv_len):
            return {"kind": "full", "causal": False, "window_size": (-1, -1)}
        raise NotImplementedError(
            f"Turbo flex compat layer: the probed {q_probe}x{kv_probe} corner of this "
            f"block_mask is fully visible, but the mask is not fully visible over the "
            f"whole {q_len}x{kv_len} sequence and does not match bidirectional document "
            "packing. Patterns that only begin to restrict past the probe limit "
            f"{_MASK_PROBE_LIMIT} cannot be verified, and guessing 'full attention' here "
            "would train the wrong model. Please express the pattern with is_causal / "
            "window_size / cu_seqlens on the direct entry points."
        )

    q_idx = torch.arange(q_probe).view(q_probe, 1)
    kv_idx = torch.arange(kv_probe).view(1, kv_probe)
    delta = q_idx - kv_idx  # q - kv; >= 0 is the causal (lower-triangular) region
    causal = delta >= 0

    if bool((mask & (~causal)).any().item()):
        # Something is visible above the diagonal, so this is not causal in any form.
        # One non-causal shape still maps exactly onto the kernels: a *band* around the
        # diagonal, ``-R <= q - kv <= L``. flash_attn's ``window_size=(left, right)`` is
        # precisely that band, and the csrc/CK entry forwards both edges unchanged, so
        # bidirectional local attention (the symmetric ``|q - kv| <= W`` that image and
        # video models use) is dispatchable rather than codegen-only. Everything that is
        # not exactly a band still raises below.
        band = _classify_band_mask(
            mask,
            delta=delta,
            mask_mod=mask_mod,
            q_len=q_len,
            kv_len=kv_len,
            q_probe=q_probe,
            kv_probe=kv_probe,
        )
        if band is not None:
            return band
        # The other non-causal shape the kernels express exactly is *bidirectional
        # document packing*: same_doc(q, kv) with no causal term, optionally with a
        # within-document window. That is how non-autoregressive models (image / video
        # diffusion) batch samples of different resolution or duration into one
        # sequence, and it lowers onto flash_attn_varlen_func's block-diagonal
        # cu_seqlens with causal=False -- no approximation anywhere.
        doc = _classify_document_mask(
            mask,
            mask_mod=mask_mod,
            q_len=q_len,
            kv_len=kv_len,
            q_probe=q_probe,
            kv_probe=kv_probe,
            causal_hint=False,
        )
        if doc is not None:
            return doc
        raise NotImplementedError(
            "Turbo flex compat layer does not support this block_mask: visible positions were found "
            "above the causal diagonal, and the pattern is neither a band around the diagonal "
            "(-R <= q - kv <= L) nor bidirectional document packing (same_doc(q, kv), optionally "
            "windowed). This is an arbitrary mask_mod and requires the codegen path."
        )
    if not bool(mask.any().item()):
        raise NotImplementedError(
            "Turbo flex compat layer does not support a fully-empty block_mask (every position "
            "masked out); it cannot be mapped onto flash_attn."
        )

    truncated = q_len > q_probe or kv_len > kv_probe

    if torch.equal(mask, causal):
        # Within the probed corner it is exactly causal. For longer sequences we
        # still have to rule out a window whose boundary sits beyond the probe.
        if truncated and mask_mod is not None:
            far_visible = _call_mask_mod(mask_mod, 0, 0, q_len - 1, 0)
            if not far_visible:
                # A window exists but its edge is past the probe grid. Locate W by a
                # binary search on the last row and verify it is a clean left window;
                # only fall back to raising if it is not standard/translation-invariant.
                window = _locate_left_window(mask_mod, q_len=q_len, kv_len=kv_len)
                if window is not None:
                    return {
                        "kind": "sliding_window_causal",
                        "causal": True,
                        "window_size": (window, 0),
                    }
                # Not a window: the other pattern that looks causal in the corner yet
                # hides the far-left position is document packing whose first document
                # is longer than the probe grid. Recovered (and exactly verified) from
                # mask_mod over the full sequence.
                doc_seglens = _locate_document_segments(mask_mod, q_len=q_len, kv_len=kv_len)
                if doc_seglens is not None:
                    return {
                        "kind": "document_causal",
                        "causal": True,
                        "window_size": (-1, -1),
                        "doc_seglens": doc_seglens,
                    }
                raise NotImplementedError(
                    "Turbo flex compat layer detected a probable sliding window but could not confirm "
                    "it is standard left-window causal (q>=kv)&(q-kv<=W): the window boundary exceeds "
                    f"the probe limit {_MASK_PROBE_LIMIT} and sampled verification failed. Please use "
                    "create_block_mask to express the window explicitly."
                )
        return {"kind": "causal", "causal": True, "window_size": (-1, -1)}

    inferred_w = int(delta[mask].max().item())
    swa = causal & (delta <= inferred_w)
    if not torch.equal(mask, swa):
        # Before giving up, try the document-packing pattern (block-diagonal + causal).
        # Recognised only via exact reconstruction, so a false match is impossible.
        doc_seglens = _detect_document_causal_segments(
            mask, q_len=q_len, kv_len=kv_len, q_probe=q_probe, kv_probe=kv_probe
        )
        if doc_seglens is None and truncated and mask_mod is not None:
            # The probe only saw the first corner of a longer packed sequence; recover
            # the boundaries from mask_mod itself (exactly verified over the full S).
            doc_seglens = _locate_document_segments(mask_mod, q_len=q_len, kv_len=kv_len)
        if doc_seglens is not None:
            return {
                "kind": "document_causal",
                "causal": True,
                "window_size": (-1, -1),
                "doc_seglens": doc_seglens,
            }
        # Plain document-causal packing is handled above. The generalised detector adds
        # the windowed variant -- packing plus a local window *inside* each document,
        # which the varlen kernel applies per segment for free.
        doc = _classify_document_mask(
            mask,
            mask_mod=mask_mod,
            q_len=q_len,
            kv_len=kv_len,
            q_probe=q_probe,
            kv_probe=kv_probe,
            causal_hint=True,
        )
        if doc is not None:
            return doc
        raise NotImplementedError(
            "Turbo flex compat layer does not support this block_mask: the pattern is not standard "
            "left-window causal (q>=kv) & (q-kv<=W), nor document packing (with or without a "
            "within-document window); it is an arbitrary/data-dependent mask and requires the "
            "codegen path."
        )

    if inferred_w >= q_probe - 1 and truncated:
        raise NotImplementedError(
            f"Turbo flex compat layer detected a sliding window that may be >= {q_probe}, exceeding "
            "the probe limit; the window size could not be verified."
        )

    # Confirm the window is translation invariant at a far query position too,
    # so we do not accept a corner-only coincidence on long sequences.
    if truncated and mask_mod is not None and inferred_w + 1 <= q_len - 1:
        far_q = q_len - 1
        inside = _call_mask_mod(mask_mod, 0, 0, far_q, far_q - inferred_w)
        outside = _call_mask_mod(mask_mod, 0, 0, far_q, far_q - inferred_w - 1)
        if not (inside and not outside):
            raise NotImplementedError(
                "Turbo flex compat layer does not support this block_mask: the window is not "
                "translation-invariant over long sequences, so a single window size cannot be "
                "determined."
            )

    return {"kind": "sliding_window_causal", "causal": True, "window_size": (inferred_w, 0)}


def _classify_block_mask_uncached(
    block_mask: Any,
    *,
    B: int,
    H: int,
    q_len: int,
    kv_len: int,
) -> Dict[str, Any]:
    """Recover full / causal / sliding-window-causal semantics from a BlockMask.

    Returns a dict with ``kind``, ``causal`` (bool) and ``window_size``
    ``(left, right)`` suitable for ``flash_attn_func``. Raises
    ``NotImplementedError`` for anything Turbo's fixed kernels cannot express.
    """
    mask_mod = getattr(block_mask, "mask_mod", None)
    if not callable(mask_mod):
        raise NotImplementedError(
            "Turbo flex compat layer does not support this block_mask: the object has no callable "
            "`mask_mod`, so it cannot be classified into a Turbo-executable pattern. Please build "
            "it with create_block_mask."
        )

    q_probe = min(q_len, _MASK_PROBE_LIMIT)
    kv_probe = min(kv_len, _MASK_PROBE_LIMIT)
    if q_probe <= 0 or kv_probe <= 0:
        raise ValueError("Turbo flex compat layer requires positive q/kv sequence lengths.")

    base = _probe_mask_grid(mask_mod, 0, 0, q_probe, kv_probe)
    if _mask_is_bh_dependent(mask_mod, base, B, H, q_probe, kv_probe):
        raise NotImplementedError(
            "Turbo flex compat layer does not support batch/head-dependent block_mask; the current "
            "backend only supports globally uniform full/causal/sliding-window patterns."
        )
    return _classify_probed_mask(
        base,
        mask_mod=mask_mod,
        q_len=q_len,
        kv_len=kv_len,
        q_probe=q_probe,
        kv_probe=kv_probe,
    )


def _classify_block_mask(
    block_mask: Any,
    *,
    B: int,
    H: int,
    q_len: int,
    kv_len: int,
) -> Dict[str, Any]:
    """Cached wrapper over :func:`_classify_block_mask_uncached`.

    ``None`` short-circuits to full attention. Otherwise the result is memoised by
    ``block_mask`` object identity and ``(B, H, q_len, kv_len)`` (a different object
    or shape re-probes). Caching is a pure speedup -- it returns exactly what the
    uncached classifier would (only successful classifications are stored; a mask
    that raises simply re-raises next time, identical behaviour).
    """
    if block_mask is None:
        return {"kind": "full", "causal": False, "window_size": (-1, -1)}

    key = (B, H, q_len, kv_len)
    cached = _cache_get(_CLASSIFY_CACHE, block_mask, key)
    if cached is not _CACHE_MISS:
        return cached
    cfg = _classify_block_mask_uncached(block_mask, B=B, H=H, q_len=q_len, kv_len=kv_len)
    _cache_put(_CLASSIFY_CACHE, block_mask, key, cfg)
    return cfg
