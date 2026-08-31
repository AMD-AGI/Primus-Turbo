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

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from primus_turbo.common.logger import logger

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


def _recover_document_boundaries(mask_mod: Callable, n: int) -> Optional[Tuple[List[int], torch.Tensor]]:
    """Read packed-document boundaries off the diagonal and sub-diagonal of ``mask_mod``.

    Two vectorised probes, ``O(S)`` elements, ``O(1)`` python-level ``mask_mod`` calls.
    Token ``i`` starts a new document iff it may not attend token ``i-1``, and the
    diagonal must be fully visible. That reading is only a *hypothesis*: nothing here is
    safe until :func:`_verify_packed_pattern` rebuilds the whole mask from it and
    compares bit for bit.

    Returns ``(seg_lens, doc_id)``, or ``None`` if the shape is not packing at all.
    """
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
    return seg_lens, doc_id


def _verify_packed_pattern(
    mask_mod: Callable,
    *,
    n: int,
    doc_id: torch.Tensor,
    causal: bool,
    window: Tuple[int, int] = (-1, -1),
) -> bool:
    """Compare ``mask_mod`` against a reconstructed packed pattern, exactly.

    The reconstruction is ``same_doc`` intersected with the causal half-plane and/or the
    ``(left, right)`` diagonal band -- i.e. precisely what ``flash_attn_varlen_func``
    computes from ``cu_seqlens`` + ``causal`` + ``window_size``, so a match means the
    kernel reproduces the mask rather than approximates it.

    Compared row-block by row-block (one vectorised call per ``_DOC_VERIFY_CHUNK`` rows)
    so peak memory stays at a few MB instead of ``S^2``. Never samples: a single
    differing bit returns ``False`` and the caller refuses to route.
    """
    left, right = window
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
        delta = q_idx - kv_idx
        expected = doc_id[q_rows].view(-1, 1) == doc_id.view(1, n)
        if causal:
            expected = expected & (delta >= 0)
        if left >= 0:
            expected = expected & (delta <= left)
        if right >= 0:
            expected = expected & (delta >= -right)
        if not torch.equal(block, expected):
            return False
    return True


def _document_window_candidates(mask_mod: Callable, *, n: int, seg_lens: List[int]) -> List[Tuple[int, int]]:
    """Guess within-document window edges from three whole rows of the longest document.

    Once a packed sequence outgrows the probe grid the window edge cannot be read off
    the probed corner -- but it can be read off a full row, and guessing is harmless
    here because :func:`_verify_packed_pattern` rebuilds the entire mask from the guess
    before anything is routed.

    Three rows, because no single one of them shows both edges at full extent. A
    document truncates every reading at its own ends, so each edge has to be read where
    that truncation cannot reach it: the **last** token of the document for the left
    edge, the **first** token for the right edge, and the **middle** token for the
    common case where the window is narrow enough that one row shows both. The pair
    ``(max left seen, max right seen)`` is offered as a candidate alongside the raw
    per-row readings -- that is the one that recovers a wide bidirectional window, e.g.
    +-640 inside an 800-token document, where the middle row reports a clipped
    ``(400, 399)`` and each end row reports one real edge and one zero.

    A window wider than the document itself clips every reading; the clipped candidates
    then fail to verify and we refuse, rather than route a half-sized window and
    silently drop the rest of the context.
    """
    longest = max(range(len(seg_lens)), key=lambda i: seg_lens[i])
    start = sum(seg_lens[:longest])
    end = start + seg_lens[longest]
    out: List[Tuple[int, int]] = []
    max_left = max_right = -1
    for q_star in dict.fromkeys((start + (end - start) // 2, end - 1, start)):
        row = _probe_mask_row(mask_mod, int(q_star), n)
        visible = row.nonzero().flatten()
        if visible.numel() == 0:
            continue
        left = q_star - int(visible.min().item())
        right = int(visible.max().item()) - q_star
        if left < 0 or right < 0:
            continue
        max_left = max(max_left, left)
        max_right = max(max_right, right)
        if (left, right) not in out:
            out.append((left, right))
    if max_left >= 0 and (max_left, max_right) not in out:
        out.append((max_left, max_right))
    return out


def _locate_document_blocks(
    mask_mod: Callable, *, q_len: int, kv_len: int, causal_hint: bool = True
) -> Optional[Dict[str, Any]]:
    """Recover packing *and* the within-document window for a sequence past the probe.

    :func:`_detect_document_blocks` recovers both, but only from a *fully* probed mask,
    so it gives up as soon as ``S > _MASK_PROBE_LIMIT``. Until now the truncated-probe
    fallback could only recover unwindowed packing and hardcoded ``window_size=(-1, -1)``
    -- so a long packed sequence with a local window inside each document was reported as
    an "arbitrary" mask and refused, even though ``flash_attn_varlen_func`` expresses it
    exactly (``cu_seqlens`` + ``causal`` + ``window_size``, applied per segment).

    Boundaries come from :func:`_recover_document_boundaries`, window edges from
    :func:`_document_window_candidates`, and every combination is then checked by exact
    chunked reconstruction. Unbounded candidates are tried before windowed ones so a
    plain packed mask is never described as a window, and ``causal_hint`` only decides
    which of the two half-plane variants is tried first -- never which one is accepted.

    Returns ``{"seglens", "causal", "window_size"}`` or ``None``.
    """
    if q_len != kv_len:
        return None
    n = q_len
    # Beyond this the full comparison costs O(S^2), so we decline rather than downgrade
    # to sampled verification -- an unverifiable mask must never be routed.
    if n <= 1 or n > _DOC_EXACT_VERIFY_LIMIT:
        return None

    found = _recover_document_boundaries(mask_mod, n)
    if found is None:
        return None
    seg_lens, doc_id = found

    candidates: List[Tuple[bool, Tuple[int, int]]] = [(causal_hint, (-1, -1)), (not causal_hint, (-1, -1))]
    for left, right in _document_window_candidates(mask_mod, n=n, seg_lens=seg_lens):
        for causal in (causal_hint, not causal_hint):
            # A causal window is one-sided by construction: the right edge is the
            # diagonal, so pass (left, 0) and let the kernel's causal flag own the rest.
            window = (left, 0) if causal else (left, right)
            if (causal, window) not in candidates:
                candidates.append((causal, window))

    for causal, window in candidates:
        if _verify_packed_pattern(mask_mod, n=n, doc_id=doc_id, causal=causal, window=window):
            return {"seglens": seg_lens, "causal": causal, "window_size": window}
    return None


def _locate_document_segments(
    mask_mod: Callable, *, q_len: int, kv_len: int, causal: bool = True
) -> Optional[list]:
    """Recover document lengths for an *unwindowed* packed mask that exceeds the probe.

    The narrow special case of :func:`_locate_document_blocks`, kept separate because
    three callers want exactly one question answered -- "is this plain packing, with the
    half-plane I already believe in?" -- and must not be handed a window they are not
    prepared to forward. ``causal`` selects which pattern to verify against:
    ``same_doc(q,kv) & (q>=kv)`` (autoregressive packing) or plain ``same_doc(q,kv)``
    (bidirectional packing, i.e. what diffusion / encoder models pack). Both lower onto
    the same block-diagonal ``cu_seqlens``; only the ``causal`` flag handed to the varlen
    kernel differs.

    Recognition is by exact reconstruction over the full sequence, never by sampling;
    anything that deviates returns ``None`` and the caller raises.
    """
    if q_len != kv_len:
        return None
    n = q_len
    if n <= 1 or n > _DOC_EXACT_VERIFY_LIMIT:
        return None
    found = _recover_document_boundaries(mask_mod, n)
    if found is None:
        return None
    seg_lens, doc_id = found
    if not _verify_packed_pattern(mask_mod, n=n, doc_id=doc_id, causal=causal):
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


def _locate_band_edges(mask_mod: Callable, *, q_len: int, kv_len: int) -> Optional[Tuple[int, int]]:
    """Recover a band ``-R <= q - kv <= L`` whose edges lie *past* the probe grid.

    The probed corner cannot show an edge it does not contain: a band of +-640 on a
    1024-long sequence fills the whole 512x512 corner, so the corner says "full" and a
    band of L=640, R=64 says "left edge somewhere beyond 511". Both are exactly
    ``window_size``-expressible, and both used to be refused.

    Each edge is read where it has the most room to show itself, which is not the same
    row for the two of them: the **last** query row is the only place a wide *left*
    edge is not clipped by the start of the sequence, and the **first** row is the only
    place a wide *right* edge is not clipped by its end. Within a genuine band each row
    is one contiguous run, so a binary search on the True->False flip finds the edge in
    ~log2(S) probes; an edge that never flips is unbounded and is returned as ``-1``,
    the kernel's own convention.

    That reading is a hypothesis and nothing more. The caller must hand it to
    :func:`_verify_packed_pattern`, which rebuilds the band over the whole sequence and
    compares it bit for bit -- so a mask that merely *starts* like a band, or one whose
    band drifts with position, still ends up refused rather than approximated.
    """
    if q_len != kv_len:
        return None  # a translation-invariant band is square self-attention
    n = q_len
    if n <= 1 or n > _DOC_EXACT_VERIFY_LIMIT:
        return None
    last = n - 1
    if not _call_mask_mod(mask_mod, 0, 0, last, last):
        return None  # a hole on the diagonal: not a band at all

    def _edge(q_pos: int, sign: int) -> int:
        """Largest ``d >= 0`` with ``mask_mod(q_pos, q_pos + sign*d)`` visible, or -1
        if the run reaches the end of the sequence (unbounded on that side)."""
        far = q_pos if sign < 0 else last - q_pos
        if far <= 0:
            return -1
        if _call_mask_mod(mask_mod, 0, 0, q_pos, q_pos + sign * far):
            return -1  # visible all the way out: no edge inside this sequence
        lo, hi = 0, far  # visible at lo, invisible at hi
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if _call_mask_mod(mask_mod, 0, 0, q_pos, q_pos + sign * mid):
                lo = mid
            else:
                hi = mid
        return lo

    left = _edge(last, -1)  # reaching back from the final query
    right = _edge(0, +1)  # reaching forward from the first query
    if left == 0 and right == 0:
        return None  # diagonal only; not something window_size describes usefully
    return left, right


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
      **and** the band is rebuilt from ``mask_mod`` over the whole sequence and compared
      bit for bit (four far-position point probes only where that is unaffordable:
      cross attention, or ``S > _DOC_EXACT_VERIFY_LIMIT``). A band that is not
      translation invariant cannot be expressed as one ``window_size``, and a corner
      that merely *looks* like a band -- bidirectionally packed documents with a window
      inside each -- is handed to the document path instead of being flattened.
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
        # clipped by the grid. The corner cannot settle it either way, so go back to
        # mask_mod: read each edge from the row where it is not clipped, then rebuild
        # the whole band and compare bit for bit. Only if that fails do we refuse.
        if left >= q_probe - 1 or right >= kv_probe - 1:
            edges = _locate_band_edges(mask_mod, q_len=q_len, kv_len=kv_len)
            if edges is not None:
                wide_left, wide_right = edges
                single_doc = torch.zeros(q_len, dtype=torch.int64)
                if _verify_packed_pattern(
                    mask_mod, n=q_len, doc_id=single_doc, causal=False, window=(wide_left, wide_right)
                ):
                    return {
                        "kind": "sliding_window",
                        "causal": False,
                        "window_size": (wide_left, wide_right),
                    }
            blocks = _locate_document_blocks(mask_mod, q_len=q_len, kv_len=kv_len, causal_hint=False)
            if blocks is not None:
                return {
                    "kind": "document",
                    "causal": blocks["causal"],
                    "window_size": blocks["window_size"],
                    "doc_seglens": blocks["seglens"],
                }
            raise NotImplementedError(
                f"Turbo flex compat layer detected a band mask whose edge exceeds the probe "
                f"limit {_MASK_PROBE_LIMIT}, and the edges recovered from mask_mod over the whole "
                f"{q_len}x{kv_len} sequence do not rebuild it exactly (nor is it document "
                "packing). Refusing rather than running the corner's band everywhere. Please "
                "express the window explicitly via window_size on the direct entry points, or "
                "use the codegen path."
            )
        if q_len == kv_len and q_len <= _DOC_EXACT_VERIFY_LIMIT:
            # Affordable: rebuild the band over the whole sequence and compare bit for
            # bit. The corner alone is not evidence -- *bidirectionally packed documents
            # with a window inside each document* reproduce it exactly whenever the
            # first document outlives the probe, and calling that a plain band would
            # drop every document boundary and let queries attend across samples for
            # the entire run, silently. One document spanning everything is exactly
            # "not packed", so the same verifier answers both questions.
            single_doc = torch.zeros(q_len, dtype=torch.int64)
            if not _verify_packed_pattern(
                mask_mod, n=q_len, doc_id=single_doc, causal=False, window=(left, right)
            ):
                blocks = _locate_document_blocks(mask_mod, q_len=q_len, kv_len=kv_len, causal_hint=False)
                if blocks is not None:
                    return {
                        "kind": "document",
                        "causal": blocks["causal"],
                        "window_size": blocks["window_size"],
                        "doc_seglens": blocks["seglens"],
                    }
                raise NotImplementedError(
                    f"Turbo flex compat layer does not support this block_mask: the probed "
                    f"{q_probe}x{kv_probe} corner is exactly the band -{right} <= q-kv <= {left}, "
                    f"but that reconstruction does not hold over the whole {q_len}x{kv_len} "
                    "sequence, and the deviation is not document packing either. Refusing rather "
                    "than running the corner's band everywhere."
                )
        else:
            # Cross attention, or too long to rebuild. Four point probes at the far end
            # are all we can afford; they catch a band that is not translation
            # invariant, and nothing finer.
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

    A truncated probe is not a downgrade: the boundaries, the half-plane *and* the
    within-document window are then recovered from ``mask_mod`` over the full sequence by
    :func:`_locate_document_blocks` and verified by exact chunked reconstruction, so a
    long packed sequence with a local window inside each document is recognised rather
    than refused. ``causal_hint`` only sets the order candidates are tried in.
    """
    blocks = _detect_document_blocks(mask, q_len=q_len, kv_len=kv_len, q_probe=q_probe, kv_probe=kv_probe)
    if blocks is None and (q_len > q_probe or kv_len > kv_probe) and mask_mod is not None:
        blocks = _locate_document_blocks(mask_mod, q_len=q_len, kv_len=kv_len, causal_hint=causal_hint)
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
        blocks = _locate_document_blocks(mask_mod, q_len=q_len, kv_len=kv_len, causal_hint=False)
        if blocks is not None:
            return {
                "kind": "document",
                "causal": blocks["causal"],
                "window_size": blocks["window_size"],
                "doc_seglens": blocks["seglens"],
            }
        if _verify_full_mask(mask_mod, q_len=q_len, kv_len=kv_len):
            return {"kind": "full", "causal": False, "window_size": (-1, -1)}
        # Not full and not packed. One shape still fits: a band so wide that both of
        # its edges fall outside the probed corner (|q-kv| <= 640 on S=1024 fills a
        # 512x512 corner completely). Its edges are recoverable from mask_mod and the
        # band is verified over the whole sequence before it is believed.
        edges = _locate_band_edges(mask_mod, q_len=q_len, kv_len=kv_len)
        if edges is not None:
            wide_left, wide_right = edges
            single_doc = torch.zeros(q_len, dtype=torch.int64)
            if _verify_packed_pattern(
                mask_mod, n=q_len, doc_id=single_doc, causal=False, window=(wide_left, wide_right)
            ):
                return {
                    "kind": "sliding_window",
                    "causal": False,
                    "window_size": (wide_left, wide_right),
                }
        raise NotImplementedError(
            f"Turbo flex compat layer: the probed {q_probe}x{kv_probe} corner of this "
            f"block_mask is fully visible, but the mask is not fully visible over the "
            f"whole {q_len}x{kv_len} sequence, does not match bidirectional document "
            "packing, and is not a wide band around the diagonal either. Patterns that "
            "only begin to restrict past the probe limit "
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
                # Not a plain window: the other patterns that look causal in the corner
                # yet hide the far-left position are document packing whose first
                # document is longer than the probe grid, and document packing that also
                # carries a window inside each document -- for the latter the binary
                # search above reads the *document* edge, not the window edge, which is
                # why it could not confirm. Both are recovered (and exactly verified)
                # from mask_mod over the full sequence.
                blocks = _locate_document_blocks(mask_mod, q_len=q_len, kv_len=kv_len, causal_hint=True)
                if blocks is not None:
                    return {
                        "kind": "document",
                        "causal": blocks["causal"],
                        "window_size": blocks["window_size"],
                        "doc_seglens": blocks["seglens"],
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
            f"Turbo flex compat layer does not support this block_mask: over the probed "
            f"{q_probe}x{kv_probe} corner of a {q_len}x{kv_len} mask the pattern reproduced neither "
            "standard left-window causal (q>=kv) & (q-kv<=W) nor document packing (with or without "
            "a within-document window), and every candidate was checked by exact reconstruction. "
            "Either it is genuinely arbitrary / data-dependent -- which needs the codegen path -- or "
            "it is a regular pattern whose defining edge lies past the verification limit "
            f"({_DOC_EXACT_VERIFY_LIMIT}), in which case it is expressible but not confirmable here. "
            "Refusing either way: routing an unverified mask would silently train a different model."
        )

    if inferred_w >= q_probe - 1 and truncated:
        raise NotImplementedError(
            f"Turbo flex compat layer detected a sliding window that may be >= {q_probe}, exceeding "
            "the probe limit; the window size could not be verified."
        )

    # The probed corner matched a sliding window -- but on a truncated probe the corner
    # is not evidence about the rest of the sequence. Document packing with a window
    # inside each document reproduces the corner *exactly* whenever the first document
    # outlives the probe, and accepting it as a plain window would drop every document
    # boundary: queries would attend across documents for the whole run, silently.
    if truncated and mask_mod is not None:
        if q_len == kv_len and q_len <= _DOC_EXACT_VERIFY_LIMIT:
            # Affordable: rebuild (q>=kv)&(q-kv<=W) over the whole sequence and compare
            # bit for bit. One document spanning everything is exactly "not packed".
            single_doc = torch.zeros(q_len, dtype=torch.int64)
            if not _verify_packed_pattern(
                mask_mod, n=q_len, doc_id=single_doc, causal=True, window=(inferred_w, 0)
            ):
                blocks = _locate_document_blocks(mask_mod, q_len=q_len, kv_len=kv_len, causal_hint=True)
                if blocks is not None:
                    return {
                        "kind": "document",
                        "causal": blocks["causal"],
                        "window_size": blocks["window_size"],
                        "doc_seglens": blocks["seglens"],
                    }
                raise NotImplementedError(
                    f"Turbo flex compat layer does not support this block_mask: the probed "
                    f"{q_probe}x{kv_probe} corner is exactly a left-window causal mask with "
                    f"W={inferred_w}, but that reconstruction does not hold over the whole "
                    f"{q_len}x{kv_len} sequence, and the deviation is not document packing "
                    "either. Refusing rather than running the corner's window everywhere."
                )
        elif inferred_w + 1 <= q_len - 1:
            # q_len != kv_len (cross attention), or too long to rebuild. Two point
            # probes at the far end are all we can afford; they catch a window that is
            # not translation invariant, and nothing finer.
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
    # A cache miss is exactly "a mask/shape this process has not classified before",
    # so this fires once per unique (block_mask, shape) rather than once per call.
    #
    # Without it the run is unauditable: the only flex-related lines in a training
    # log are Primus's own config echo, and that echo is byte-identical whether the
    # compat layer ran or was silently bypassed -- which is how a whole round of
    # end-to-end measurements was once collected against plain TE attention without
    # anyone noticing. Say out loud what the mask was classified as.
    #
    # rank=0 so 8 ranks do not print the same line 8 times; once=True dedupes on the
    # fully-formatted message, so a *different* classification still gets its own
    # line. Formatted eagerly for that reason -- this is the cold path, and the
    # probe it follows costs milliseconds, so the f-string is free.
    logger.info(
        f"flex: classified block_mask as kind={cfg.get('kind')!r} "
        f"causal={cfg.get('causal')} window_size={cfg.get('window_size')} "
        f"(B={B}, H={H}, q_len={q_len}, kv_len={kv_len})",
        once=True,
        rank=0,
    )
    return cfg
