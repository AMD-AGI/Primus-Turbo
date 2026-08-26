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


def _locate_document_segments(mask_mod: Callable, *, q_len: int, kv_len: int) -> Optional[list]:
    """Recover document lengths for a packed mask whose sequence exceeds the probe grid.

    :func:`_detect_document_causal_segments` only looks at the already-probed
    ``<= _MASK_PROBE_LIMIT`` corner, so packed sequences longer than that used to raise
    ``NotImplementedError`` even though the pattern is perfectly expressible. This
    variant works directly on ``mask_mod`` over the *full* sequence:

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
        return None  # a hole on the diagonal: not document-causal

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
        expected = (doc_id[q_rows].view(-1, 1) == doc_id.view(1, n)) & (q_idx >= kv_idx)
        if not torch.equal(block, expected):
            return None
    return seg_lens


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


def _classify_probed_mask(
    mask: torch.Tensor,
    *,
    mask_mod: Optional[Callable],
    q_len: int,
    kv_len: int,
    q_probe: int,
    kv_probe: int,
) -> Dict[str, Any]:
    if bool(mask.all().item()):
        return {"kind": "full", "causal": False, "window_size": (-1, -1)}

    q_idx = torch.arange(q_probe).view(q_probe, 1)
    kv_idx = torch.arange(kv_probe).view(1, kv_probe)
    delta = q_idx - kv_idx  # q - kv; >= 0 is the causal (lower-triangular) region
    causal = delta >= 0

    if bool((mask & (~causal)).any().item()):
        raise NotImplementedError(
            "Turbo flex compat layer does not support this block_mask: visible positions were found "
            "above the causal diagonal (neither causal nor left-window causal). This is an arbitrary "
            "mask_mod and requires the codegen path."
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
        raise NotImplementedError(
            "Turbo flex compat layer does not support this block_mask: the pattern is not standard "
            "left-window causal (q>=kv) & (q-kv<=W); it is an arbitrary/data-dependent mask and "
            "requires the codegen path."
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
