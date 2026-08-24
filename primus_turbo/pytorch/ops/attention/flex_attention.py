###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Torch-compatible ``flex_attention`` entry point backed by Primus-Turbo.

This module exposes a drop-in replacement for
``torch.nn.attention.flex_attention.flex_attention``. Common variants that
Turbo already accelerates (full / causal / sliding-window-causal masks, optional
ALiBi score bias, GQA/MQA) are recognised at runtime and dispatched to the
high-performance ``flash_attn_func`` backend (FlyDSL/AITER on gfx950). Anything
that cannot be mapped to those fixed kernels raises ``NotImplementedError`` with
an explanation, or is routed to the (currently stub) custom fast-path hook.

For variable-length / document-packed batches this module also exposes an
explicit :func:`flex_attention_varlen` entry point: a thin THD-layout wrapper over
``flash_attn_varlen_func`` (caller supplies ``cu_seqlens`` directly). In addition,
the dense :func:`flex_attention` recognises a block-diagonal document-causal
``block_mask`` (``same_doc(q,kv) & (q>=kv)``, verified by exact reconstruction) and
routes it through the varlen backend rather than a cross-document dense causal call.

Design notes
------------
* torch flex uses the ``bhsd`` (``[B, H, S, D]``) layout; ``flash_attn_func`` is
  ``bshd`` (``[B, S, H, D]``). We transpose in/out around the backend call.
* Mask semantics are recovered by probing ``block_mask.mask_mod`` on a small
  index grid, then matching it against the full / causal / sliding-window
  templates. Data-dependent / batch-or-head dependent masks are rejected.
* ALiBi detection is *exact-representability* checking: a ``score_mod`` is only
  mapped to ``alibi_slopes`` when it is verified to be ``score + slope[h] *
  (kv_idx - q_idx)`` (additive in score, translation invariant, batch
  independent). On this build (rocm/primus:v26.5, primus_turbo 0.3.2.dev48) the
  Turbo ``alibi_slopes`` sign convention matches flex's ``+slope*(kv-q)`` form
  (empirically resolved, see ``bench/bench_results_ext2.md``: ``alibi_sign=1.0``).
* Turbo-extension explicit args (a *superset* of the torch signature; all default
  to off -- ``None`` / ``0.0`` -- so a torch-style call is byte-for-byte unchanged
  and stays a drop-in replacement): ``alibi_slopes`` lets the caller pass per-head
  slopes directly -- bypassing the ``score_mod`` detector and connecting straight to
  ``flash_attn_func`` (live now); ``softcap`` reserves the logits soft-cap interface
  (currently gated, see the softcap note below); ``dropout_p`` (attention dropout),
  ``sink`` (per-query-head attention-sink logits) and ``bias`` (a shared ``[Sq,Skv]``
  additive logits bias) are threaded straight to ``flash_attn_func`` (all live now --
  the backend already supports them; ``bias`` needs q's dtype + ``[Sq,Skv]`` shape).
* Backend routing: after a variant is recognised (mask classified + score_mod
  mapped) it still passes through :func:`choose_backend`, a thin performance
  routing layer. By default it returns ``"turbo"`` (dispatch to
  ``flash_attn_func``), so behaviour is identical to a direct dispatch. A tuner
  or test can bias specific shapes/kinds towards the ``_dispatch_custom`` hook
  via :func:`register_backend_override` without touching the classifier.
* Softcap (logits soft-cap, ``cap*tanh(score/cap)``, Gemma2/Grok) is *detected*
  (:func:`_detect_softcap`) and can also be requested via the explicit ``softcap``
  arg, but is currently **blocked at the kernel layer**: the aiter dense
  forward/backward on this build (rocm/primus:v26.5) expose no softcap parameter,
  so it cannot be threaded through ``flash_attn_func`` without patching/rebuilding
  aiter. Both a detected and an explicit ``softcap > 0`` therefore hit a *single*
  gate (see :func:`flex_attention`) that raises ``NotImplementedError`` rather than
  silently dropping the cap (which the ALiBi detector alone would do). See
  FLEX_COMPAT_STATUS.md for the aiter-signature evidence and options.
"""

import math
import warnings
import weakref
from typing import Any, Callable, Dict, Optional, Tuple

import torch

try:
    from torch.nn.attention.flex_attention import create_block_mask as _torch_create_block_mask
except Exception:  # pragma: no cover - depends on torch version
    _torch_create_block_mask = None

__all__ = [
    "flex_attention",
    "flex_attention_varlen",
    "create_block_mask",
    "SUPPORT_STATUS",
    "choose_backend",
    "register_backend_override",
    "clear_backend_overrides",
    "clear_classification_cache",
]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
# Upper bound on the probe grid edge. 512 comfortably covers the sliding-window
# sizes we can express directly; larger windows on longer sequences are located by
# a binary search on the last query row (see _locate_left_window) rather than
# silently mis-classified.
_MASK_PROBE_LIMIT = 512
_ALIBI_TOL = 5e-3
# Relative tolerance for recognising a logits soft-cap (cap*tanh(score/cap)).
_SOFTCAP_TOL = 1e-2


# =============================================================================
# Classification / detection caches (perf: avoid re-probing on reuse)
# =============================================================================
#
# Probing a ``block_mask.mask_mod`` (up to 512x512) and re-running the ALiBi /
# soft-cap ``score_mod`` detectors on *every* call is a fixed ~1-3 ms per-call cost
# that dominates small/medium shapes. In real use the same ``block_mask`` /
# ``score_mod`` object is reused across layers and steps, so we memoise the
# classification / detection *by object identity* (``weakref.WeakKeyDictionary`` so
# entries vanish when the mask/score_mod is garbage-collected -- no leak, no
# lifetime surprises). This is a pure speedup: a different object re-probes, and a
# cache entry is exactly what the uncached path would have returned (identical
# result / error semantics). A sentinel distinguishes a cached ``None`` result from
# a miss; objects that cannot be weakly referenced simply skip the cache.
_CACHE_MISS = object()

# block_mask obj -> {(B, H, q_len, kv_len): cfg dict}
_CLASSIFY_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
# score_mod obj -> {(B, Hq, q_len, kv_len): slopes tensor or None}
_ALIBI_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
# score_mod obj -> {(B, Hq, q_len, kv_len): cap float or None}
_SOFTCAP_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _cache_get(cache: "weakref.WeakKeyDictionary", obj: Any, key: Tuple) -> Any:
    """Return the cached value for ``(obj, key)`` or ``_CACHE_MISS``.

    Tolerates objects that cannot be weakly referenced (returns a miss) so caching
    never changes behaviour -- only speed.
    """
    try:
        inner = cache.get(obj)
    except TypeError:
        return _CACHE_MISS
    if inner is None:
        return _CACHE_MISS
    return inner.get(key, _CACHE_MISS)


def _cache_put(cache: "weakref.WeakKeyDictionary", obj: Any, key: Tuple, value: Any) -> None:
    """Memoise ``value`` for ``(obj, key)``; silently skip non-weak-referenceable objects."""
    try:
        inner = cache.get(obj)
        if inner is None:
            inner = {}
            cache[obj] = inner
    except TypeError:
        return
    inner[key] = value


def clear_classification_cache() -> None:
    """Clear all mask-classification / score_mod-detection caches.

    Rarely needed (entries are keyed by object identity and drop automatically when
    the mask / score_mod is collected), but handy for tests / benchmarks that want a
    cold-cache measurement or full determinism.
    """
    _CLASSIFY_CACHE.clear()
    _ALIBI_CACHE.clear()
    _SOFTCAP_CACHE.clear()


SUPPORT_STATUS: Dict[str, Any] = {
    "supported_now": {
        "mask": ["full", "causal", "sliding_window_causal", "document_causal"],
        "score_mod": ["none", "alibi_detected", "alibi_explicit"],
        "gqa_mqa": True,
        "return_lse": True,
        # A block_mask recognised (exactly) as same_doc(q,kv) & (q>=kv) is routed
        # through the varlen backend (block-diagonal cu_seqlens) rather than a dense
        # causal call. Requires S <= _MASK_PROBE_LIMIT and no bias / return_lse (use
        # flex_attention_varlen for those); recognition is exact so it never misfires.
        "document_causal_dense_recognition": "block_mask_same_doc_and_causal_routed_to_varlen",
        # Sliding-window-causal with a window LARGER than the 512 probe grid (e.g.
        # W=1024/2048/4096 on long sequences) is located by a binary search on the
        # last query row + exact per-row verification (see _locate_left_window), so a
        # big window on S>512 is now classified instead of raising NotImplementedError.
        "sliding_window_large": "window_gt_probe_located_by_binary_search_on_last_row",
    },
    # Classification (block_mask probe) and score_mod (ALiBi/soft-cap) detection are
    # memoised by object identity (weakref) so reusing the same block_mask / score_mod
    # across layers & steps skips the ~1-3 ms per-call probe. Pure speedup; identical
    # results. clear_classification_cache() resets it (tests / cold-cache benchmarks).
    "classification_cache": "memoised_by_block_mask_and_score_mod_object_identity_weakref",
    # Turbo-extension explicit args (superset of the torch signature; both default
    # None so a torch-style call is unchanged and remains a drop-in replacement).
    "turbo_extension_args": {
        # Explicit per-head fp32 slopes -> flash_attn_func(alibi_slopes=...);
        # bypasses the score_mod detector and is live on this build.
        "alibi_slopes": "live_explicit_bypasses_score_mod_detection",
        # Interface is in place but gated: softcap>0 raises NotImplementedError
        # (aiter dense fwd/bwd lack the param); 0/None means disabled (no-op).
        "softcap": "interface_ready_but_gated_positive_softcap_raises",
        # Attention dropout probability (0<=p<1) -> flash_attn_func(dropout_p=...);
        # live on this build. 0.0 (default) disables it (drop-in, no-op).
        "dropout_p": "live_explicit_passthrough_0_disables",
        # Per-query-head attention-sink logits (1D fp32, len==Hq) ->
        # flash_attn_func(sink=...); live on this build. None (default) disables it.
        # Sink kernel path requires head_dim_qk==head_dim_v and power-of-two head dim.
        "sink": "live_explicit_passthrough_none_disables",
        # Additive logits bias -> flash_attn_func(bias=...); live on this build.
        # aiter dense needs a single [Sq,Skv] bias in q's dtype (fp16/bf16) shared
        # across batch/heads (fp32 -> NaN, per-head 4D -> rejected by kernel). Verified
        # fwd+bwd correct. None (default) disables it.
        "bias": "live_explicit_passthrough_needs_Sq_Skv_qdtype_none_disables",
    },
    "unsupported_paths": {
        "arbitrary_score_mod": "path_b_codegen_stub_only",
        "arbitrary_mask_mod": "path_b_codegen_stub_only",
        # Recognised (via _detect_softcap) or requested explicitly, but blocked at
        # the kernel layer: the aiter dense fwd/bwd on this build expose no softcap
        # parameter, so a soft-cap hard-errors instead of silently ignoring the cap.
        "softcap": "detected_or_explicit_but_blocked_aiter_dense_kernel_has_no_softcap_param",
    },
    # Explicit variable-length / document-packing entry point (THD layout). A
    # superset-free thin wrapper around ``flash_attn_varlen_func``: the caller
    # supplies cu_seqlens directly, so there is no mask/score_mod probing here.
    "varlen": {
        "entry": "flex_attention_varlen",
        "layout": "thd_[total_tokens,H,D]",
        "supported": [
            "full",
            "causal (document-internal, block-diagonal via cu_seqlens)",
            "sliding_window_causal (per-segment window_size)",
            "gqa_mqa",
            "alibi_explicit",
            "dropout_p",
            "sink",
            "return_lse",
        ],
        "document_masking": "explicit_cu_seqlens_block_diagonal_plus_causal_true",
        "unsupported": [
            "arbitrary_score_mod_no_such_arg",
            "softcap_gt_0_gated",
            "bias",
        ],
    },
    # The empirically resolved ALiBi sign for this build; see module docstring.
    "alibi_sign_convention": "+slope*(kv-q)",
    # A recognised variant is routed through choose_backend before dispatch.
    # Default policy is "turbo" for everything; register_backend_override lets a
    # tuner steer specific shapes/kinds to the (currently stub) custom hook.
    "backend_routing": {
        "selector": "choose_backend",
        "default": "turbo",
        "backends": ["turbo", "custom"],
        "override_api": ["register_backend_override", "clear_backend_overrides"],
    },
}


def create_block_mask(
    mask_mod,
    B,
    H,
    Q_LEN,
    KV_LEN,
    device="cuda",
    BLOCK_SIZE=128,
    _compile=False,
):
    """Thin passthrough to ``torch``'s ``create_block_mask``.

    We reuse torch's implementation so the returned ``BlockMask`` keeps its
    ``.mask_mod`` attribute (which our dispatcher probes) and stays byte-for-byte
    compatible with code that also feeds the mask to torch's own flex kernel.
    """
    if _torch_create_block_mask is None:
        raise NotImplementedError(
            "Turbo flex compat layer cannot provide create_block_mask: the current torch build "
            "lacks torch.nn.attention.flex_attention.create_block_mask."
        )
    try:
        return _torch_create_block_mask(
            mask_mod=mask_mod,
            B=B,
            H=H,
            Q_LEN=Q_LEN,
            KV_LEN=KV_LEN,
            device=device,
            BLOCK_SIZE=BLOCK_SIZE,
            _compile=_compile,
        )
    except TypeError:
        # Older torch builds may not expose ``_compile``.
        warnings.warn(
            "The current torch build's create_block_mask lacks the `_compile` argument; ignoring it.",
            stacklevel=2,
        )
        return _torch_create_block_mask(
            mask_mod=mask_mod,
            B=B,
            H=H,
            Q_LEN=Q_LEN,
            KV_LEN=KV_LEN,
            device=device,
            BLOCK_SIZE=BLOCK_SIZE,
        )


# =============================================================================
# Low-level probing helpers
# =============================================================================


def _to_scalar_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"mask_mod must return a scalar boolean, got shape={tuple(value.shape)}")
        return bool(value.item())
    return bool(value)


def _to_scalar_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"score_mod must return a scalar score, got shape={tuple(value.shape)}")
        return float(value.item())
    return float(value)


def _call_mask_mod(mask_mod: Callable, b: int, h: int, q_idx: int, kv_idx: int) -> bool:
    """Call a mask_mod at a single index, tolerating int- or tensor-only signatures."""
    try:
        return _to_scalar_bool(mask_mod(b, h, q_idx, kv_idx))
    except (TypeError, RuntimeError):
        return _to_scalar_bool(
            mask_mod(
                torch.tensor(b),
                torch.tensor(h),
                torch.tensor(q_idx),
                torch.tensor(kv_idx),
            )
        )


def _call_score_mod(score_mod: Callable, score: float, b: int, h: int, q_idx: int, kv_idx: int) -> float:
    try:
        return _to_scalar_float(score_mod(score, b, h, q_idx, kv_idx))
    except (TypeError, RuntimeError):
        return _to_scalar_float(
            score_mod(
                torch.tensor(float(score)),
                torch.tensor(b),
                torch.tensor(h),
                torch.tensor(q_idx),
                torch.tensor(kv_idx),
            )
        )


def _probe_mask_grid(mask_mod: Callable, b: int, h: int, q_probe: int, kv_probe: int) -> torch.Tensor:
    """Evaluate ``mask_mod`` over the top-left ``q_probe x kv_probe`` grid.

    Tries a single vectorised broadcast call first (torch mask_mods used through
    ``create_block_mask`` are vectorisable by construction); falls back to an
    element-wise loop for scalar-only lambdas (e.g. those using python ``and``).
    """
    q_idx = torch.arange(q_probe).view(q_probe, 1)
    kv_idx = torch.arange(kv_probe).view(1, kv_probe)
    try:
        raw = mask_mod(b, h, q_idx, kv_idx)
        mask = torch.as_tensor(raw, dtype=torch.bool)
        if mask.shape != (q_probe, kv_probe):
            mask = mask.broadcast_to((q_probe, kv_probe))
        return mask.contiguous()
    except Exception:
        out = torch.empty((q_probe, kv_probe), dtype=torch.bool)
        for qi in range(q_probe):
            for ki in range(kv_probe):
                out[qi, ki] = _call_mask_mod(mask_mod, b, h, qi, ki)
        return out


def _probe_mask_row(mask_mod: Callable, q_pos: int, kv_len: int) -> torch.Tensor:
    """Evaluate ``mask_mod`` for a single query row ``q_pos`` over ``kv in [0, kv_len)``.

    Returns a 1D bool tensor of length ``kv_len``. Vectorised first (one broadcast
    call), falling back to an element-wise loop for scalar-only lambdas. Used by the
    sliding-window locator to verify a candidate window exactly on sampled rows.
    """
    q_idx = torch.tensor([[q_pos]])
    kv_idx = torch.arange(kv_len).view(1, kv_len)
    try:
        raw = mask_mod(0, 0, q_idx, kv_idx)
        row = torch.as_tensor(raw, dtype=torch.bool)
        if row.shape != (1, kv_len):
            row = row.broadcast_to((1, kv_len))
        return row.reshape(kv_len).contiguous()
    except Exception:
        out = torch.empty(kv_len, dtype=torch.bool)
        for ki in range(kv_len):
            out[ki] = _call_mask_mod(mask_mod, 0, 0, q_pos, ki)
        return out


def _mask_is_bh_dependent(
    mask_mod: Callable, base: torch.Tensor, B: int, H: int, q_probe: int, kv_probe: int
) -> bool:
    candidates = []
    if B > 1:
        candidates.append((B - 1, 0))
    if H > 1:
        candidates.append((0, H - 1))
    if B > 1 and H > 1:
        candidates.append((B - 1, H - 1))
    for b, h in candidates:
        other = _probe_mask_grid(mask_mod, b, h, q_probe, kv_probe)
        if not torch.equal(other, base):
            return True
    return False


# =============================================================================
# Mask classification
# =============================================================================


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
            "mask_mod and requires the codegen path (see FLEX_COMPAT_STATUS.md)."
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


# =============================================================================
# score_mod (ALiBi) detection
# =============================================================================


def _alibi_sample_points(q_len: int, kv_len: int) -> Tuple[Tuple[int, int], ...]:
    pts = set()
    for d in (0, 1, 2, 3, 5, 7):
        if d < kv_len:
            pts.add((0, d))  # delta = +d
        if d < q_len:
            pts.add((d, 0))  # delta = -d
    if q_len >= 2 and kv_len >= 2:
        # Far-corner points reuse small deltas at large absolute positions and so
        # catch score_mods that depend on absolute position, not just (kv - q).
        pts.add((q_len - 1, kv_len - 1))
        pts.add((q_len - 1, kv_len - 2))
        pts.add((q_len - 2, kv_len - 1))
    mq, mk = q_len // 2, kv_len // 2
    pts.add((mq, mk))
    if mk + 1 < kv_len:
        pts.add((mq, mk + 1))
    if mq + 1 < q_len:
        pts.add((mq + 1, mk))
    return tuple(sorted(pts))


def _detect_alibi_slopes(
    score_mod: Callable,
    *,
    B: int,
    Hq: int,
    q_len: int,
    kv_len: int,
    tol: float = _ALIBI_TOL,
) -> Optional[torch.Tensor]:
    """Return per-head ALiBi slopes iff ``score_mod`` is exactly representable as
    ``score + slope[h] * (kv_idx - q_idx)`` (additive, translation invariant,
    batch independent); otherwise ``None`` (caller routes to the custom path).
    """
    if q_len <= 0 or kv_len <= 0:
        return None

    pts = _alibi_sample_points(q_len, kv_len)
    nonzero = [(q, k) for (q, k) in pts if (k - q) != 0]

    slopes = []
    for h in range(Hq):
        # (1) additive in score with unit coefficient: score_mod(s) - s independent of s.
        c0 = _call_score_mod(score_mod, 0.0, 0, h, 0, 0)
        c1 = _call_score_mod(score_mod, 1.0, 0, h, 0, 0)
        if abs(c1 - c0 - 1.0) > tol:
            return None
        # bias at delta 0 must be ~0 for a pure (kv-q) bias.
        if abs(c0) > tol:
            return None

        if not nonzero:
            slopes.append(0.0)
            continue

        ref_q, ref_k = min(nonzero, key=lambda qk: abs(qk[1] - qk[0]))
        ref_delta = float(ref_k - ref_q)
        slope = _call_score_mod(score_mod, 0.0, 0, h, ref_q, ref_k) / ref_delta

        # (2) exact linear-in-(kv-q) fit across the sample grid.
        for q_i, k_i in pts:
            got = _call_score_mod(score_mod, 0.0, 0, h, q_i, k_i)
            expect = slope * float(k_i - q_i)
            if abs(got - expect) > tol * (1.0 + abs(expect)):
                return None

        # (3) batch independence (ALiBi slopes are per-head only).
        if B > 1:
            alt = _call_score_mod(score_mod, 0.0, B - 1, h, ref_q, ref_k)
            if abs(alt - _call_score_mod(score_mod, 0.0, 0, h, ref_q, ref_k)) > tol:
                return None

        slopes.append(slope)

    return torch.tensor(slopes, dtype=torch.float32)


# =============================================================================
# score_mod (softcap) detection
# =============================================================================


def _detect_softcap(
    score_mod: Callable,
    *,
    B: int,
    Hq: int,
    q_len: int,
    kv_len: int,
    tol: float = _SOFTCAP_TOL,
) -> Optional[float]:
    """Return ``cap > 0`` iff ``score_mod`` is exactly a logits soft-cap
    ``cap * tanh(score / cap)`` (a function of the score alone, independent of
    batch/head/positions); otherwise ``None``.

    Detected explicitly for two reasons on this build:

    * The fixed Turbo dense kernels expose **no** softcap parameter (aiter dense
      ``mha_fwd``/``fmha_v3_fwd``/``mha_bwd`` lack it; see FLEX_COMPAT_STATUS.md),
      so a soft-cap cannot be silently dropped.
    * :func:`_detect_alibi_slopes` only ever probes ``score=0``, where
      ``cap*tanh(0)=0``; a soft-cap would therefore be misread as a zero-slope
      (no-op) bias and handed to Turbo *without* the cap -- a silently wrong
      result. Recognising the soft-cap shape first prevents that.
    """
    if q_len <= 0 or kv_len <= 0:
        return None

    # (1) A pure soft-cap depends on the score only: constant across batch / head
    #     / query / key position. Any dependence rules it out (e.g. ALiBi, which
    #     varies with kv-q, falls through here to the ALiBi detector).
    ctx_points = [(0, 0, 0, 0)]
    if q_len > 1:
        ctx_points.append((0, 0, q_len - 1, 0))
    if kv_len > 1:
        ctx_points.append((0, 0, 0, kv_len - 1))
    if q_len > 1 and kv_len > 1:
        ctx_points.append((0, 0, q_len - 1, kv_len - 1))
    if Hq > 1:
        ctx_points.append((0, Hq - 1, 0, 0))
    if B > 1:
        ctx_points.append((B - 1, 0, 0, 0))
    for s in (0.5, 1.7, -1.3):
        base = _call_score_mod(score_mod, s, 0, 0, 0, 0)
        for b, h, qi, ki in ctx_points[1:]:
            if abs(_call_score_mod(score_mod, s, b, h, qi, ki) - base) > tol * (1.0 + abs(base)):
                return None

    def f(s: float) -> float:
        return _call_score_mod(score_mod, s, 0, 0, 0, 0)

    # (2) tanh passes through the origin.
    if abs(f(0.0)) > tol:
        return None

    # (3) Estimate the positive asymptote (the cap) from the saturating tail.
    cap = None
    prev = None
    for s in (4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0):
        y = f(s)
        if y <= 0.0:
            return None  # a soft-cap is strictly increasing / positive for s>0
        if prev is not None and abs(y - prev) <= tol * (1.0 + abs(y)):
            cap = y
            break
        prev = y
    if cap is None or cap <= tol:
        return None  # never saturated (identity/linear) or degenerate cap

    # Refine the asymptote where tanh is fully saturated, so the returned cap is
    # accurate rather than the first plateau sample.
    refined = f(64.0 * cap)
    if refined > 0.0:
        cap = refined

    # (4) Full-shape verification against cap*tanh(s/cap): near-linear region,
    #     knee, saturating tail, and odd symmetry f(-s) == -f(s).
    for frac in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
        for s in (frac * cap, -frac * cap):
            got = f(s)
            expect = cap * math.tanh(s / cap)
            if abs(got - expect) > tol * (1.0 + abs(expect)):
                return None

    return float(cap)


# =============================================================================
# Cached score_mod detection (perf: avoid re-probing on reuse)
# =============================================================================


def _cached_detect_alibi_slopes(
    score_mod: Callable, *, B: int, Hq: int, q_len: int, kv_len: int
) -> Optional[torch.Tensor]:
    """Memoised :func:`_detect_alibi_slopes` keyed by ``score_mod`` object identity.

    Returns exactly what the uncached detector would (including ``None``); the cached
    slopes tensor is never mutated in place by callers (they ``.to(device)`` / read
    it), so sharing it is safe.
    """
    key = (B, Hq, q_len, kv_len)
    cached = _cache_get(_ALIBI_CACHE, score_mod, key)
    if cached is not _CACHE_MISS:
        return cached
    val = _detect_alibi_slopes(score_mod, B=B, Hq=Hq, q_len=q_len, kv_len=kv_len)
    _cache_put(_ALIBI_CACHE, score_mod, key, val)
    return val


def _cached_detect_softcap(
    score_mod: Callable, *, B: int, Hq: int, q_len: int, kv_len: int
) -> Optional[float]:
    """Memoised :func:`_detect_softcap` keyed by ``score_mod`` object identity."""
    key = (B, Hq, q_len, kv_len)
    cached = _cache_get(_SOFTCAP_CACHE, score_mod, key)
    if cached is not _CACHE_MISS:
        return cached
    val = _detect_softcap(score_mod, B=B, Hq=Hq, q_len=q_len, kv_len=kv_len)
    _cache_put(_SOFTCAP_CACHE, score_mod, key, val)
    return val


# =============================================================================
# Explicit Turbo-extension args (alibi_slopes / softcap)
# =============================================================================


def _validate_explicit_alibi_slopes(
    alibi_slopes: Any,
    *,
    hq: int,
    device: Any,
) -> torch.Tensor:
    """Validate a caller-supplied explicit ``alibi_slopes`` and align its device.

    Requirements (matches ``flash_attn_func``'s per-head convention): a 1D fp32
    tensor of length ``Hq`` (query heads). Returns the tensor moved onto ``device``
    (q's device). Raises ``ValueError`` with a clear message otherwise.
    """
    if not isinstance(alibi_slopes, torch.Tensor):
        raise ValueError(
            "Turbo flex compat layer requires explicit alibi_slopes to be a torch.Tensor, "
            f"got {type(alibi_slopes).__name__}."
        )
    if alibi_slopes.ndim != 1:
        raise ValueError(
            "Turbo flex compat layer requires explicit alibi_slopes to be a 1D tensor, "
            f"got ndim={alibi_slopes.ndim} (shape={tuple(alibi_slopes.shape)})."
        )
    if alibi_slopes.shape[0] != hq:
        raise ValueError(
            "Turbo flex compat layer requires len(alibi_slopes) to equal the query head count "
            f"Hq={hq}, got length={alibi_slopes.shape[0]}."
        )
    if alibi_slopes.dtype != torch.float32:
        raise ValueError(
            "Turbo flex compat layer requires explicit alibi_slopes to be fp32 (torch.float32), "
            f"got dtype={alibi_slopes.dtype}."
        )
    return alibi_slopes.to(device=device)


def _normalise_explicit_softcap(softcap: Any) -> float:
    """Coerce the explicit ``softcap`` arg to a non-negative float (0.0 == disabled).

    ``None`` -> ``0.0`` (disabled). A negative or NaN cap is rejected. The
    ``softcap > 0`` gating itself lives at a single point in :func:`flex_attention`.
    """
    if softcap is None:
        return 0.0
    try:
        cap = float(softcap)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Turbo flex compat layer requires explicit softcap to be a float or None; "
            f"cannot convert {softcap!r}: {exc}"
        ) from exc
    if math.isnan(cap):
        raise ValueError("Turbo flex compat layer's explicit softcap cannot be NaN.")
    if cap < 0.0:
        raise ValueError(
            f"Turbo flex compat layer requires explicit softcap >= 0 (0/None disables it), got {cap}."
        )
    return cap


def _is_power_of_two(n: int) -> bool:
    """Local copy of the backend's power-of-two check (see attention_aiter_impl).

    Kept here so validating ``sink``'s head-dim constraint does not force the heavy
    kernel module to import during pure classification / this module's import.
    """
    return n > 0 and (n & (n - 1)) == 0


def _validate_dropout_p(dropout_p: Any) -> float:
    """Validate the Turbo-extension ``dropout_p`` (attention dropout probability).

    Requires ``0 <= p < 1`` (``p == 0`` disables dropout, the drop-in default). The value
    is threaded straight to ``flash_attn_func(dropout_p=...)``; as in flash-attn / torch
    ``scaled_dot_product_attention`` it takes effect whenever ``p > 0`` (the training
    convention -- dropout is applied unconditionally on the score matrix, so callers must
    pass ``0`` for eval). ``dropout_p > 0`` composes with ``return_lse`` (the LSE is still
    returned) and the compat layer always dispatches with ``deterministic=False``, so there
    is no dropout/determinism conflict to reject. Returns the validated float.
    """
    try:
        p = float(dropout_p)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Turbo flex compat layer requires dropout_p to be a float; cannot convert {dropout_p!r}: {exc}"
        ) from exc
    if math.isnan(p):
        raise ValueError("Turbo flex compat layer's dropout_p cannot be NaN.")
    if not (0.0 <= p < 1.0):
        raise ValueError(
            f"Turbo flex compat layer requires dropout_p in [0, 1) (0 disables dropout), got {p}."
        )
    return p


def _validate_explicit_sink(
    sink: Any,
    *,
    hq: int,
    head_dim_qk: int,
    head_dim_v: int,
    device: Any,
) -> torch.Tensor:
    """Validate a caller-supplied attention ``sink`` and align its device.

    Mirrors the backend sink constraints (see
    ``attention_aiter_impl.AttnFwdAiterBackend.can_handle`` and ``tests/.../test_attention.py``):
    a 1D fp32 tensor of length ``Hq`` (query heads -- one learned sink logit per query head),
    and the sink kernel path additionally requires ``head_dim_qk == head_dim_v`` with a
    power-of-two head dim. The value is threaded straight to ``flash_attn_func(sink=...)``.
    Returns the tensor moved onto ``device`` (q's device); raises ``ValueError`` with a clear
    message otherwise.
    """
    if not isinstance(sink, torch.Tensor):
        raise ValueError(
            f"Turbo flex compat layer requires sink to be a torch.Tensor, got {type(sink).__name__}."
        )
    if sink.ndim != 1:
        raise ValueError(
            "Turbo flex compat layer requires sink to be a 1D tensor (one sink value per query head), "
            f"got ndim={sink.ndim} (shape={tuple(sink.shape)})."
        )
    if sink.shape[0] != hq:
        raise ValueError(
            "Turbo flex compat layer requires len(sink) to equal the query head count "
            f"Hq={hq}, got length={sink.shape[0]}."
        )
    if sink.dtype != torch.float32:
        raise ValueError(
            f"Turbo flex compat layer requires sink to be fp32 (torch.float32), got dtype={sink.dtype}."
        )
    if head_dim_qk != head_dim_v:
        raise ValueError(
            "Turbo flex compat layer's sink path requires head_dim_qk == head_dim_v (backend "
            f"constraint), got head_dim_qk={head_dim_qk}, head_dim_v={head_dim_v}."
        )
    if not _is_power_of_two(head_dim_qk):
        raise ValueError(
            "Turbo flex compat layer's sink path requires head_dim to be a power of two (backend "
            f"constraint), got head_dim={head_dim_qk}."
        )
    return sink.to(device=device)


def _validate_and_adapt_bias(
    bias: Any,
    *,
    sq: int,
    skv: int,
    dtype: Any,
    device: Any,
) -> torch.Tensor:
    """Validate a caller-supplied additive attention ``bias`` and adapt it to the backend.

    The aiter dense kernel accepts a single **2D** bias of shape ``[Sq, Skv]`` in q's
    dtype (fp16/bf16), added to the pre-softmax logits and *shared across batch and
    heads*. This is an empirically pinned constraint (see FLEX_COMPAT_STATUS.md): a 4D /
    per-head bias raises ``RuntimeError: bias shape should be [sq, sk]`` and an fp32 bias
    yields ``NaN``; only a ``[Sq, Skv]`` bias in q's dtype is numerically correct (fwd &
    bwd). We therefore accept ``[Sq, Skv]`` or a leading-singleton broadcast of it
    (``[1, Sq, Skv]`` / ``[1, 1, Sq, Skv]``) and reject a genuine per-batch / per-head
    bias with a clear message; the value is cast to q's dtype and moved onto q's device.
    Returns the adapted contiguous ``[Sq, Skv]`` tensor.
    """
    if not isinstance(bias, torch.Tensor):
        raise ValueError(
            f"Turbo flex compat layer requires bias to be a torch.Tensor, got {type(bias).__name__}."
        )
    b = bias
    if b.ndim == 4:
        if b.shape[0] != 1 or b.shape[1] != 1:
            raise ValueError(
                "Turbo flex compat layer's bias backend only supports a single [Sq,Skv] shared across "
                "batch/head (AITER dense constraint); a bias that varies per batch/head is not "
                f"supported. Got shape={tuple(bias.shape)} (for per-head/per-sample bias use the "
                "codegen path, see FLEX_COMPAT_STATUS.md)."
            )
        b = b.reshape(b.shape[2], b.shape[3])
    elif b.ndim == 3:
        if b.shape[0] != 1:
            raise ValueError(
                "Turbo flex compat layer's bias backend only supports a single shared [Sq,Skv] "
                f"(AITER dense constraint), got 3D shape={tuple(bias.shape)} (leading dim must be 1)."
            )
        b = b.reshape(b.shape[1], b.shape[2])
    elif b.ndim != 2:
        raise ValueError(
            "Turbo flex compat layer requires bias to be 2D [Sq,Skv] (or a broadcastable shape with "
            f"leading singletons: [1,Sq,Skv]/[1,1,Sq,Skv]), got ndim={b.ndim} "
            f"(shape={tuple(bias.shape)})."
        )
    if tuple(b.shape) != (sq, skv):
        raise ValueError(
            "Turbo flex compat layer requires the last two dims of bias to be "
            "[Sq={0}, Skv={1}] (AITER dense constraint), got {2}.".format(sq, skv, tuple(b.shape))
        )
    if not b.is_floating_point():
        raise ValueError(
            "Turbo flex compat layer requires bias to be a floating-point tensor (it will be cast "
            f"to q's dtype), got dtype={b.dtype}."
        )
    # Adapt precision to q's dtype: the kernel needs bias in q's dtype (fp16/bf16); an
    # fp32 bias produces NaN. Cast + move to q's device, contiguous for the kernel.
    return b.to(dtype=dtype, device=device).contiguous()


def _is_identity_score_mod(
    score_mod: Callable,
    *,
    B: int,
    Hq: int,
    q_len: int,
    kv_len: int,
    tol: float = _ALIBI_TOL,
) -> bool:
    """Return ``True`` iff ``score_mod`` leaves the score unchanged on a probe grid.

    Used only for the explicit-``alibi_slopes`` conflict check: a no-op / identity
    ``score_mod`` may coexist with explicit slopes, whereas *any* non-trivial
    ``score_mod`` alongside explicit slopes is ambiguous and rejected. The probe
    spans several scores and (batch, head, q, kv) positions -- including non-zero
    ``kv-q`` deltas -- so e.g. a (non-zero) ALiBi ``score_mod`` reads as non-identity.
    """
    b_pts = sorted({0, B - 1}) if B > 0 else [0]
    h_pts = sorted({0, Hq - 1}) if Hq > 0 else [0]
    q_pts = sorted({0, q_len // 2, q_len - 1}) if q_len > 0 else [0]
    kv_pts = sorted({0, kv_len // 2, kv_len - 1}) if kv_len > 0 else [0]
    for s in (0.0, 0.7, -1.3, 2.5):
        for b in b_pts:
            for h in h_pts:
                for qi in q_pts:
                    for ki in kv_pts:
                        got = _call_score_mod(score_mod, s, b, h, qi, ki)
                        if abs(got - s) > tol * (1.0 + abs(s)):
                            return False
    return True


def _dispatch_custom(*_args, **_kwargs):
    """Hook for the future high-performance arbitrary score_mod/mask_mod kernel.

    Not implemented yet: this is intentionally a hard stop so callers never get a
    silently wrong result from an unrecognised programmable modification. It is
    the shared landing point for (a) arbitrary score_mod/mask_mod and (b) a variant
    explicitly routed away from Turbo by :func:`choose_backend`. A recognised /
    explicit logits soft-cap is gated *before* this hook (single point in
    :func:`flex_attention`), so ``softcap`` never reaches here as ``> 0``.
    """
    raise NotImplementedError(
        "custom/arbitrary score_mod+mask_mod fast path not implemented yet "
        "(planned: codegen 'path B'); see FLEX_COMPAT_STATUS.md"
    )


# =============================================================================
# Backend selection (performance routing layer)
# =============================================================================

# Valid return values / backend identifiers for the routing layer.
#   "turbo"  -> dispatch the recognised variant to ``flash_attn_func``.
#   "custom" -> route to the ``_dispatch_custom`` hook (currently a stub).
_VALID_BACKENDS: Tuple[str, ...] = ("turbo", "custom")

# Ordered registry of ``(matcher, backend)`` overrides. ``matcher(ctx) -> bool``
# inspects a routing-context dict; the first match wins. An empty registry means
# ``choose_backend`` always returns "turbo", i.e. the historical behaviour where
# every recognised variant goes straight to Turbo.
_BACKEND_OVERRIDES: list = []


def register_backend_override(matcher: Callable[[Dict[str, Any]], bool], backend: str) -> None:
    """Register a performance-routing override consulted by :func:`choose_backend`.

    ``matcher`` receives the routing context dict (keys: ``kind``, ``causal``,
    ``window_size``, ``shape``, ``dtype``, ``has_alibi``, ``has_softcap``,
    ``has_dropout``, ``has_sink``, ``has_bias`` and the full ``mask_cfg``) and
    returns ``True`` to force ``backend`` for that call.
    Overrides are evaluated in registration order and the first match wins, so a
    tuner can steer specific shapes/kinds to the custom hook without disturbing
    the classifier. Registering nothing keeps the default all-Turbo routing.
    """
    if not callable(matcher):
        raise TypeError("register_backend_override: matcher must be a callable(ctx)->bool.")
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"register_backend_override: backend must be one of {_VALID_BACKENDS}, got {backend!r}."
        )
    _BACKEND_OVERRIDES.append((matcher, backend))


def clear_backend_overrides() -> None:
    """Remove all registered overrides, restoring the default all-Turbo routing."""
    _BACKEND_OVERRIDES.clear()


def choose_backend(
    mask_cfg: Dict[str, Any],
    *,
    shape: Tuple[int, ...],
    dtype: Any,
    has_alibi: bool,
    has_softcap: bool = False,
    has_dropout: bool = False,
    has_sink: bool = False,
    has_bias: bool = False,
) -> str:
    """Pick the execution backend for an already-recognised (mask, score_mod) combo.

    Returns ``"turbo"`` (dispatch to ``flash_attn_func``) or ``"custom"`` (route to
    the ``_dispatch_custom`` hook). The default is always ``"turbo"`` so every
    variant the compat layer accelerates keeps the fast path and existing
    behaviour is unchanged. Registered overrides (see
    :func:`register_backend_override`) are consulted first, in order; the first
    matching one decides. ``has_dropout`` / ``has_sink`` / ``has_bias`` reflect the
    Turbo-extension passthrough args and are surfaced in the routing context
    alongside ``has_alibi`` / ``has_softcap`` so a tuner can key on them too.
    """
    ctx: Dict[str, Any] = {
        "kind": mask_cfg.get("kind"),
        "causal": mask_cfg.get("causal"),
        "window_size": mask_cfg.get("window_size"),
        "shape": tuple(shape),
        "dtype": dtype,
        "has_alibi": bool(has_alibi),
        "has_softcap": bool(has_softcap),
        "has_dropout": bool(has_dropout),
        "has_sink": bool(has_sink),
        "has_bias": bool(has_bias),
        "mask_cfg": mask_cfg,
    }
    for matcher, backend in _BACKEND_OVERRIDES:
        try:
            hit = matcher(ctx)
        except Exception as exc:  # a broken matcher must fail loud, never silently reroute
            raise RuntimeError(f"choose_backend: backend override matcher raised: {exc!r}") from exc
        if hit:
            return backend
    return "turbo"


# =============================================================================
# Public entry point
# =============================================================================


def _validate_qkv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            "Turbo flex compat layer requires q/k/v to be 4D [B,H,S,D] (bhsd) tensors; "
            f"got ndim=({query.ndim},{key.ndim},{value.ndim})."
        )
    for name, t in (("query", query), ("key", key), ("value", value)):
        if t.dtype not in _SUPPORTED_DTYPES:
            raise NotImplementedError(
                "Turbo flex compat layer currently supports only fp16/bf16 inputs; "
                f"{name}.dtype={t.dtype}, fall back to torch flex_attention for other dtypes."
            )
    if query.device != key.device or query.device != value.device:
        raise ValueError("Turbo flex compat layer requires q/k/v to be on the same device.")
    if not (query.shape[0] == key.shape[0] == value.shape[0]):
        raise ValueError("Turbo flex compat layer requires q/k/v to share the same batch dim.")
    if key.shape[1] != value.shape[1] or key.shape[2] != value.shape[2]:
        raise ValueError("Turbo flex compat layer requires key/value to share head count and seqlen.")


def _dispatch_document_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seg_lens: list,
    *,
    scale: Optional[float],
    alibi_slopes: Optional[torch.Tensor],
    dropout_p: float,
    sink: Optional[torch.Tensor],
    deterministic: bool = False,
) -> torch.Tensor:
    """Run a recognised document-causal *dense* call through the varlen backend.

    ``query``/``key``/``value`` are bhsd ``[B, H, S, D]`` sharing the same per-batch
    document structure ``seg_lens`` (``sum(seg_lens) == S``; batch/head independence of
    the mask is already verified by the classifier). They are packed to THD, dispatched
    block-diagonally (``causal=True``) via ``flash_attn_varlen_func`` -- which honours
    document boundaries through ``cu_seqlens`` rather than attending across them -- and
    the packed output is unpacked back to bhsd. ``sink`` is threaded only when supplied
    (newer-backend feature; a no-op default otherwise).
    """
    bsz, hq, sq, _ = query.shape
    dv = value.shape[-1]

    def _pack(t: torch.Tensor) -> torch.Tensor:  # (B,H,S,D) -> (B*S, H, D)
        b, h, s, d = t.shape
        return t.transpose(1, 2).reshape(b * s, h, d).contiguous()

    q_thd = _pack(query)
    k_thd = _pack(key)
    v_thd = _pack(value)

    # Replicate the per-batch document boundaries across the packed batch dimension.
    seglens_all = list(seg_lens) * bsz
    cu = torch.zeros(len(seglens_all) + 1, dtype=torch.int32, device=query.device)
    cu[1:] = torch.tensor(seglens_all, dtype=torch.int32, device=query.device).cumsum(0)
    max_s = int(max(seg_lens))

    from primus_turbo.pytorch.ops.attention.flash_attn_interface import flash_attn_varlen_func

    call_kwargs: Dict[str, Any] = dict(
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_lse=False,
    )
    if sink is not None:
        call_kwargs["sink"] = sink

    out_thd = flash_attn_varlen_func(q_thd, k_thd, v_thd, cu, cu, max_s, max_s, **call_kwargs)
    # (B*S, Hq, Dv) -> (B, S, Hq, Dv) -> (B, Hq, S, Dv)
    return out_thd.reshape(bsz, sq, hq, dv).transpose(1, 2).contiguous()


def flex_attention(
    query,
    key,
    value,
    score_mod=None,
    block_mask=None,
    scale=None,
    enable_gqa=False,
    return_lse=False,
    kernel_options=None,
    alibi_slopes: Optional[torch.Tensor] = None,
    softcap: Optional[float] = None,
    dropout_p: float = 0.0,
    sink: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
):
    """Drop-in compatible ``flex_attention`` dispatching to Turbo fast paths.

    Mirrors ``torch.nn.attention.flex_attention.flex_attention``. Recognised
    (mask, score_mod) combinations run on ``flash_attn_func``; everything else
    raises ``NotImplementedError``.

    Turbo extension (superset of the torch signature)
    -------------------------------------------------
    ``alibi_slopes``, ``softcap``, ``dropout_p``, ``sink`` and ``bias`` are optional
    Turbo-only additions placed *after* the torch parameters. All default to off
    (``None`` / ``0.0``), so a torch-style call is byte-for-byte unchanged and this
    stays a drop-in replacement for
    ``torch.nn.attention.flex_attention.flex_attention``.

    * ``alibi_slopes`` (``Optional[torch.Tensor]``, default ``None``): explicit
      per-head ALiBi slopes, a 1D fp32 tensor of length ``Hq`` (query heads).
      When given it **bypasses** the ``score_mod`` ALiBi auto-detector and is
      threaded straight to ``flash_attn_func(alibi_slopes=...)`` -- so this works
      **now** on this build (it sidesteps the detector's conservative limits).
      Passing both an explicit ``alibi_slopes`` *and* a non-trivial (non-identity)
      ``score_mod`` is ambiguous and raises ``ValueError`` (choose one); a
      ``None``/identity ``score_mod`` alongside explicit slopes is allowed. Works
      with causal / sliding-window masks (ALiBi is commonly paired with causal).
    * ``softcap`` (``Optional[float]``, default ``None``): logits soft-cap
      (``cap*tanh(score/cap)``, Gemma2/Grok). ``None`` or ``0`` means disabled
      (no-op). A positive value currently raises ``NotImplementedError`` -- the
      interface is in place, but it is blocked at the kernel layer (the aiter
      dense fwd/bwd on this build expose no softcap parameter; see
      FLEX_COMPAT_STATUS.md). It will take effect once the upstream kernel adds the
      parameter.
    * ``dropout_p`` (``float``, default ``0.0``): attention-dropout probability
      threaded straight to ``flash_attn_func(dropout_p=...)``. Requires
      ``0 <= p < 1`` (``0`` disables dropout, the drop-in default). As in
      flash-attn / torch ``scaled_dot_product_attention`` it is applied whenever
      ``p > 0`` (training convention -- pass ``0`` for eval); it composes with
      ``return_lse`` and the layer always dispatches ``deterministic=False``.
    * ``sink`` (``Optional[torch.Tensor]``, default ``None``): attention-sink
      logits (one learned value per query head), threaded straight to
      ``flash_attn_func(sink=...)``. Requires a 1D fp32 tensor of length ``Hq``;
      the sink kernel path also requires ``head_dim_qk == head_dim_v`` with a
      power-of-two head dim (backend constraint). ``None`` disables it (no-op).
    * ``bias`` (``Optional[torch.Tensor]``, default ``None``): additive attention
      bias on the pre-softmax logits, threaded to ``flash_attn_func(bias=...)``.
      The aiter dense kernel accepts a single ``[Sq, Skv]`` bias in q's dtype
      shared across batch/heads (an fp32 bias yields NaN, a per-head 4D bias is
      rejected by the kernel; see FLEX_COMPAT_STATUS.md). This entry accepts
      ``[Sq, Skv]`` (or a leading-singleton broadcast ``[1,Sq,Skv]`` /
      ``[1,1,Sq,Skv]``), casts it to q's dtype and moves it to q's device; a
      genuine per-batch/per-head bias raises ``ValueError``. Verified numerically
      correct fwd+bwd (rel-L2 ~2e-3). ``None`` disables it (no-op).
    """
    _validate_qkv(query, key, value)

    if kernel_options:
        warnings.warn(
            "Turbo flex compat layer does not support kernel_options yet; ignoring: "
            f"{sorted(kernel_options.keys())}",
            stacklevel=2,
        )

    bsz, hq, sq, dq = query.shape
    _, hkv, skv, _ = key.shape
    if hq != hkv:
        if hkv <= 0 or hq % hkv != 0:
            raise ValueError(
                f"Turbo flex compat layer requires Hq to be divisible by Hkv, got Hq={hq}, Hkv={hkv}."
            )
        if not enable_gqa:
            raise ValueError(
                "Turbo flex compat layer detected Hq!=Hkv; please pass enable_gqa=True explicitly "
                "(matching torch flex)."
            )

    # ------------------------------------------------------------------
    # Turbo-extension passthrough args: dropout_p / sink (superset of the torch
    # signature). Both default to off (0.0 / None), so with neither supplied this
    # is inert and the historical behaviour is reproduced exactly (zero regression).
    # Validate the cheap scalar/1D args up front so illegal values fail fast.
    # ------------------------------------------------------------------
    dropout_p = _validate_dropout_p(dropout_p)
    if sink is not None:
        sink = _validate_explicit_sink(
            sink, hq=hq, head_dim_qk=dq, head_dim_v=value.shape[-1], device=query.device
        )
    effective_bias: Optional[torch.Tensor] = None
    if bias is not None:
        effective_bias = _validate_and_adapt_bias(
            bias, sq=sq, skv=skv, dtype=query.dtype, device=query.device
        )
    has_dropout = dropout_p > 0.0
    has_sink = sink is not None
    has_bias = effective_bias is not None

    mask_cfg = _classify_block_mask(block_mask, B=bsz, H=hq, q_len=sq, kv_len=skv)

    # ------------------------------------------------------------------
    # Turbo-extension explicit args (superset of the torch signature). Both
    # default to None, so with neither supplied this whole block is inert and the
    # historical behaviour is reproduced exactly (zero regression).
    # ------------------------------------------------------------------
    explicit_alibi = alibi_slopes is not None
    if explicit_alibi:
        alibi_slopes = _validate_explicit_alibi_slopes(alibi_slopes, hq=hq, device=query.device)
    explicit_softcap = _normalise_explicit_softcap(softcap)

    # ``effective_*`` are what actually reaches the backend / routing / gating,
    # folding in both the explicit args and (only when no explicit ALiBi is given)
    # the score_mod auto-detection.
    effective_alibi_slopes: Optional[torch.Tensor] = None
    effective_softcap = explicit_softcap

    if explicit_alibi:
        # Explicit slopes bypass the score_mod ALiBi detector entirely. A
        # non-trivial score_mod alongside explicit slopes is ambiguous -> reject.
        if score_mod is not None:
            if not callable(score_mod):
                raise ValueError("Turbo flex compat layer requires score_mod to be callable or None.")
            if not _is_identity_score_mod(score_mod, B=bsz, Hq=hq, q_len=sq, kv_len=skv):
                raise ValueError(
                    "Turbo flex compat layer: both an explicit alibi_slopes and a non-trivial "
                    "score_mod were provided, which is semantically ambiguous; please pick one "
                    "(either pass explicit alibi_slopes, or express ALiBi via score_mod and let "
                    "auto-detection handle it)."
                )
        effective_alibi_slopes = alibi_slopes
    elif score_mod is not None:
        if not callable(score_mod):
            raise ValueError("Turbo flex compat layer requires score_mod to be callable or None.")
        # Detect a pure logits soft-cap (cap*tanh(score/cap)) first. It cannot run
        # on the fixed dense kernels (no softcap param on this build), and the
        # ALiBi detector below only probes score=0 -- where a soft-cap looks like
        # a zero-slope no-op -- so probing it first avoids silently dropping the
        # cap. A soft-cap+ALiBi (or other) composition is not a pure soft-cap and
        # falls through to the ALiBi detector, which rejects it -> custom.
        detected_softcap = _cached_detect_softcap(score_mod, B=bsz, Hq=hq, q_len=sq, kv_len=skv) or 0.0
        if detected_softcap > 0.0:
            # Unify with the explicit softcap gate below (no double-handling): a
            # detected soft-cap simply raises the effective cap and hits one gate.
            effective_softcap = max(effective_softcap, detected_softcap)
        else:
            detected_alibi = _cached_detect_alibi_slopes(score_mod, B=bsz, Hq=hq, q_len=sq, kv_len=skv)
            if detected_alibi is None:
                # Unrecognised programmable score modification -> custom (stub) path.
                return _dispatch_custom(score_mod=score_mod, block_mask=block_mask)
            if bool(torch.all(detected_alibi.abs() < _ALIBI_TOL).item()):
                # A no-op / identity score_mod: keep the plain (fastest) path.
                effective_alibi_slopes = None
            else:
                effective_alibi_slopes = detected_alibi.to(device=query.device)

    # Performance routing: even a recognised variant passes through the backend
    # selector so a tuner can override specific shapes/kinds. With no overrides
    # registered this always returns "turbo", i.e. behaviour is identical to a
    # direct dispatch (the historical default). Both explicit and detected
    # alibi/softcap are folded into the routing context.
    backend = choose_backend(
        mask_cfg,
        shape=(bsz, hq, sq, dq),
        dtype=query.dtype,
        has_alibi=effective_alibi_slopes is not None,
        has_softcap=effective_softcap > 0.0,
        has_dropout=has_dropout,
        has_sink=has_sink,
        has_bias=has_bias,
    )

    # ---- single softcap enablement point --------------------------------------
    # softcap (explicit or detected) is blocked at the kernel layer on this build:
    # the aiter dense fwd/bwd expose no softcap parameter (see FLEX_COMPAT_STATUS.md).
    # Centralising the "softcap>0 -> raise" here (never silently dropping the cap)
    # keeps a single switch to flip once the kernel supports it.
    # TODO(softcap): once upstream aiter dense fwd+bwd supports it, thread through to
    #   flash_attn_func(softcap=...): drop this guard and pass effective_softcap at the
    #   flash_attn_func call below -- a one-line switch to enable.
    if effective_softcap > 0.0:
        raise NotImplementedError(
            "Turbo flex compat layer: the softcap interface is in place "
            f"(cap~={effective_softcap:.4g}), but it is blocked by this build's aiter dense fwd/bwd "
            "kernels, which lack a softcap parameter (see FLEX_COMPAT_STATUS.md); it will take "
            "effect once upstream kernels support it. To avoid silently dropping the cap and "
            "producing wrong results, both an explicit softcap and a soft-cap detected from "
            "score_mod raise here -- we never degrade to a path that ignores the cap."
        )

    if backend != "turbo":
        # A variant explicitly routed away from Turbo by a tuner override lands on
        # the (currently stub) custom hook instead of a silently wrong result.
        return _dispatch_custom(
            score_mod=score_mod,
            block_mask=block_mask,
            mask_cfg=mask_cfg,
            alibi_slopes=effective_alibi_slopes,
            softcap=effective_softcap,
            dropout_p=dropout_p,
            sink=sink,
            bias=effective_bias,
            backend=backend,
        )

    # ---- document packing (block-diagonal + within-doc causal) ----------------
    # A block_mask recognised as ``same_doc(q,kv) & (q>=kv)`` is dispatched through
    # the varlen backend (packed cu_seqlens) instead of a dense causal call, which
    # would attend across document boundaries. The recognition is exact (see
    # _detect_document_causal_segments), so this only fires on genuine doc packing.
    if mask_cfg.get("kind") == "document_causal":
        if has_bias:
            raise NotImplementedError(
                "Turbo flex compat layer: document-causal block_mask combined with a shared bias is "
                "not supported yet (a [Sq,Skv] bias cannot be aligned under packed varlen); use the "
                "codegen path if you need it."
            )
        if return_lse:
            raise NotImplementedError(
                "Turbo flex compat layer: document-causal block_mask does not currently support "
                "return_lse (packed LSE does not align with the dense [B,H,S] layout); if you need "
                "LSE, use the explicit flex_attention_varlen entry point."
            )
        return _dispatch_document_varlen(
            query,
            key,
            value,
            mask_cfg["doc_seglens"],
            scale=scale,
            alibi_slopes=effective_alibi_slopes,
            dropout_p=dropout_p,
            sink=sink,
        )

    # Lazy import so pure classification (and this module's import) does not force
    # the heavy backend kernels to load.
    from primus_turbo.pytorch.ops.attention.flash_attn_interface import flash_attn_func

    q_bshd = query.transpose(1, 2).contiguous()
    k_bshd = key.transpose(1, 2).contiguous()
    v_bshd = value.transpose(1, 2).contiguous()

    out = flash_attn_func(
        q_bshd,
        k_bshd,
        v_bshd,
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=mask_cfg["causal"],
        window_size=mask_cfg["window_size"],
        bias=effective_bias,
        alibi_slopes=effective_alibi_slopes,
        deterministic=False,
        return_lse=return_lse,
        sink=sink,
        # TODO(softcap): once upstream aiter dense fwd+bwd supports it, pass softcap=effective_softcap
        #   here and delete the softcap guard above -- a one-line enable (effective_softcap is
        #   currently always 0.0).
    )

    if return_lse:
        out_bshd, lse = out
        return out_bshd.transpose(1, 2).contiguous(), lse
    return out.transpose(1, 2).contiguous()


# =============================================================================
# Varlen / document-packing entry point (THD layout)
# =============================================================================


def _validate_qkv_varlen(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    """Validate packed THD ``[total_tokens, H, D]`` q/k/v for the varlen entry.

    Unlike the dense entry (``bhsd``), the varlen backend consumes the THD packed
    layout directly (sequences concatenated along dim 0), so no transpose happens
    here. GQA/MQA is supported natively (``Hq % Hkv == 0``); q/k share the
    ``head_dim_qk`` while v may carry a different ``head_dim_v``.
    """
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError(
            "Turbo flex varlen entry requires q/k/v to be 3D [total_tokens, H, D] (THD packed) "
            f"tensors; got ndim=({query.ndim},{key.ndim},{value.ndim})."
        )
    for name, t in (("query", query), ("key", key), ("value", value)):
        if t.dtype not in _SUPPORTED_DTYPES:
            raise NotImplementedError(
                "Turbo flex varlen entry currently supports only fp16/bf16 inputs; "
                f"{name}.dtype={t.dtype}, fall back to the torch reference implementation."
            )
    if query.device != key.device or query.device != value.device:
        raise ValueError("Turbo flex varlen entry requires q/k/v to be on the same device.")
    if key.shape[0] != value.shape[0]:
        raise ValueError(
            "Turbo flex varlen entry requires key/value to have the same total_tokens, "
            f"got key={key.shape[0]}, value={value.shape[0]}."
        )
    if key.shape[1] != value.shape[1]:
        raise ValueError(
            "Turbo flex varlen entry requires key/value to have the same head count, "
            f"got Hk={key.shape[1]}, Hv={value.shape[1]}."
        )
    if query.shape[2] != key.shape[2]:
        raise ValueError(
            "Turbo flex varlen entry requires query/key to have the same head_dim (head_dim_qk), "
            f"got Dq={query.shape[2]}, Dk={key.shape[2]}."
        )
    hq, hkv = query.shape[1], key.shape[1]
    if hq != hkv and (hkv <= 0 or hq % hkv != 0):
        raise ValueError(
            f"Turbo flex varlen entry requires Hq divisible by Hkv (GQA/MQA), got Hq={hq}, Hkv={hkv}."
        )


def _validate_window_size(window_size: Any) -> Tuple[int, int]:
    """Coerce/validate a ``(left, right)`` window into a 2-int tuple.

    ``(-1, -1)`` means full attention; a left window ``(W, 0)`` mirrors the dense
    classifier's sliding-window-causal mapping (per segment in the varlen case).
    """
    if isinstance(window_size, torch.Tensor) or not isinstance(window_size, (tuple, list)):
        raise ValueError(
            "Turbo flex varlen entry requires window_size to be a length-2 (left, right) tuple/list, "
            f"got {type(window_size).__name__}."
        )
    if len(window_size) != 2:
        raise ValueError(
            "Turbo flex varlen entry requires window_size to have length 2 (left, right), "
            f"got length={len(window_size)}."
        )
    left, right = window_size
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (left, right)):
        raise ValueError(
            f"Turbo flex varlen entry requires both window_size elements to be int, got {window_size!r}."
        )
    return (int(left), int(right))


def _validate_max_seqlen(name: str, value: Any) -> int:
    """Validate a ``max_seqlen_*`` argument (a positive Python int)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Turbo flex varlen entry requires {name} to be a Python int, "
            f"got {type(value).__name__}={value!r}."
        )
    if value <= 0:
        raise ValueError(f"Turbo flex varlen entry requires {name} to be a positive int, got {value}.")
    return int(value)


def _validate_cu_seqlens(
    cu_seqlens_q: Any,
    cu_seqlens_k: Any,
    *,
    total_q: int,
    total_k: int,
    max_seqlen_q: Any,
    max_seqlen_k: Any,
    causal: bool,
    device: Any,
) -> Tuple[int, int]:
    """Validate the varlen cumulative-sequence-length descriptors.

    Requirements (matching ``flash_attn_varlen_func``): both are 1D **int32**
    tensors on q's device, starting at ``0``, monotonically non-decreasing, with a
    final element equal to ``total_q`` / ``total_k`` (the packed token counts) and a
    matching number of segments (``len(cu_seqlens_q) == len(cu_seqlens_k)``). The
    longest per-segment length must not exceed the supplied ``max_seqlen``. When
    ``causal`` is set, document-internal causal masking is only well-defined when
    every segment has ``q_len == k_len`` (the kernel's bottom-right alignment would
    otherwise silently shift the mask), so ``cu_seqlens_q`` must equal
    ``cu_seqlens_k``. Raises ``ValueError`` with a clear message on any violation.
    Returns the validated ``(max_seqlen_q, max_seqlen_k)`` ints.
    """
    max_seqlen_q = _validate_max_seqlen("max_seqlen_q", max_seqlen_q)
    max_seqlen_k = _validate_max_seqlen("max_seqlen_k", max_seqlen_k)

    for name, cu in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
        if not isinstance(cu, torch.Tensor):
            raise ValueError(
                f"Turbo flex varlen entry requires {name} to be a torch.Tensor, got {type(cu).__name__}."
            )
        if cu.dtype != torch.int32:
            raise ValueError(f"Turbo flex varlen entry requires {name} to be int32, got dtype={cu.dtype}.")
        if cu.ndim != 1:
            raise ValueError(
                f"Turbo flex varlen entry requires {name} to be a 1D [num_seqs+1] tensor, "
                f"got ndim={cu.ndim} (shape={tuple(cu.shape)})."
            )
        if cu.numel() < 2:
            raise ValueError(
                f"Turbo flex varlen entry requires {name} to have at least 2 elements ([0, total]), "
                f"got numel={cu.numel()}."
            )
        if cu.device != device:
            raise ValueError(
                f"Turbo flex varlen entry requires {name} to be on the same device as q, "
                f"got {cu.device} vs {device}."
            )
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError(
            "Turbo flex varlen entry requires cu_seqlens_q and cu_seqlens_k to have the same number "
            f"of segments (equal len), got {cu_seqlens_q.numel()} vs {cu_seqlens_k.numel()}."
        )

    # These descriptors are tiny; inspect on CPU in int64 (avoids int32 overflow in
    # the diff and keeps the value reads off the classifier's hot path).
    q_cpu = cu_seqlens_q.detach().to(device="cpu", dtype=torch.int64)
    k_cpu = cu_seqlens_k.detach().to(device="cpu", dtype=torch.int64)

    if int(q_cpu[0]) != 0 or int(k_cpu[0]) != 0:
        raise ValueError(
            "Turbo flex varlen entry requires the first cu_seqlens element to be 0, "
            f"got cu_seqlens_q[0]={int(q_cpu[0])}, cu_seqlens_k[0]={int(k_cpu[0])}."
        )

    q_seg = q_cpu[1:] - q_cpu[:-1]
    k_seg = k_cpu[1:] - k_cpu[:-1]
    if bool((q_seg < 0).any()) or bool((k_seg < 0).any()):
        raise ValueError(
            "Turbo flex varlen entry requires cu_seqlens to be monotonically non-decreasing (every "
            f"segment length >= 0), got cu_seqlens_q={q_cpu.tolist()}, cu_seqlens_k={k_cpu.tolist()}."
        )

    if int(q_cpu[-1]) != int(total_q):
        raise ValueError(
            "Turbo flex varlen entry requires the last cu_seqlens_q element to equal query's "
            f"total_tokens, got cu_seqlens_q[-1]={int(q_cpu[-1])}, total_q={int(total_q)}."
        )
    if int(k_cpu[-1]) != int(total_k):
        raise ValueError(
            "Turbo flex varlen entry requires the last cu_seqlens_k element to equal key/value's "
            f"total_tokens, got cu_seqlens_k[-1]={int(k_cpu[-1])}, total_k={int(total_k)}."
        )

    q_max_seg = int(q_seg.max()) if q_seg.numel() > 0 else 0
    k_max_seg = int(k_seg.max()) if k_seg.numel() > 0 else 0
    if max_seqlen_q < q_max_seg:
        raise ValueError(
            "Turbo flex varlen entry requires max_seqlen_q >= the longest query segment length, "
            f"got max_seqlen_q={max_seqlen_q}, actual longest segment={q_max_seg}."
        )
    if max_seqlen_k < k_max_seg:
        raise ValueError(
            "Turbo flex varlen entry requires max_seqlen_k >= the longest key segment length, "
            f"got max_seqlen_k={max_seqlen_k}, actual longest segment={k_max_seg}."
        )

    if causal and not torch.equal(q_cpu, k_cpu):
        raise ValueError(
            "Turbo flex varlen entry with causal=True requires q_len == k_len per segment "
            "(in-document causal is bottom-right aligned, so mismatched segment lengths silently "
            "misalign); make cu_seqlens_q == cu_seqlens_k, or use causal=False for cross-attention."
        )

    return max_seqlen_q, max_seqlen_k


def flex_attention_varlen(
    query,
    key,
    value,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    *,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    scale: Optional[float] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    sink: Optional[torch.Tensor] = None,
    softcap: Optional[float] = None,
    return_lse: bool = False,
):
    """Explicit variable-length ("varlen" / document-packing) flex entry point.

    A thin, correctness-first wrapper over
    :func:`primus_turbo.pytorch.ops.attention.flash_attn_varlen_func`. Sequences are
    packed back-to-back along dim 0 (THD ``[total_tokens, H, D]``, *no* transpose),
    exactly the layout the varlen backend consumes, and boundaries are described by
    ``cu_seqlens_q`` / ``cu_seqlens_k`` (int32 ``[num_seqs+1]`` prefix sums). This is
    the packing counterpart to the dense :func:`flex_attention` entry.

    Parameters
    ----------
    query, key, value : torch.Tensor
        Packed THD tensors ``[total_tokens, H, D]``, fp16/bf16. ``query``/``key``
        share ``head_dim_qk``; ``value`` may carry a different ``head_dim_v``.
        GQA/MQA is supported natively (``Hq % Hkv == 0``) with no extra flag.
    cu_seqlens_q, cu_seqlens_k : torch.Tensor
        1D int32 cumulative sequence lengths ``[num_seqs+1]`` on q's device
        (``[0, s0, s0+s1, ..., total]``). Validated (dtype / monotonicity / final
        element / matching lengths) with a clear ``ValueError`` on any violation.
    max_seqlen_q, max_seqlen_k : int
        The longest per-batch query / key segment length (positive ints).
    causal : bool, default False
        Document-internal causal masking. ``True`` gives the standard block-diagonal
        + within-segment causal (``same_doc(q,kv) & (q>=kv)``) packing; it requires
        ``cu_seqlens_q == cu_seqlens_k`` (per-segment ``q_len == k_len``).
    window_size : (int, int), default (-1, -1)
        Per-segment sliding window ``(left, right)``. ``(-1, -1)`` is full/plain
        block-causal; ``(W, 0)`` is a left window of ``W`` (mirrors the dense
        sliding-window-causal mapping).
    scale : float, optional
        QK^T softmax scale; defaults to ``1/sqrt(head_dim_qk)``.
    alibi_slopes : torch.Tensor, optional
        Explicit per-head ALiBi slopes (1D fp32, ``len == Hq``), threaded straight to
        the backend. There is no ``score_mod`` probing on this entry, so ALiBi is
        expressed explicitly here (reuses the dense validator).
    dropout_p : float, default 0.0
        Attention-dropout probability (``0 <= p < 1``; ``0`` disables it).
    sink : torch.Tensor, optional
        Per-query-head attention-sink logits (1D fp32, ``len == Hq``); the sink
        kernel path also needs ``head_dim_qk == head_dim_v`` with a power-of-two head
        dim (reuses the dense validator).
    softcap : float, optional
        Logits soft-cap. ``None``/``0`` disables it (no-op); a positive value raises
        ``NotImplementedError`` (blocked at the kernel layer on this build -- the
        varlen backward has no softcap parameter -- exactly like the dense entry, so
        the cap is never silently dropped).
    return_lse : bool, default False
        Also return the backend's ``softmax_lse`` (returns ``(out, lse)``).

    Returns
    -------
    torch.Tensor or (torch.Tensor, torch.Tensor)
        The packed THD output ``[total_q, Hq, head_dim_v]`` (and the backend LSE when
        ``return_lse`` is set).
    """
    _validate_qkv_varlen(query, key, value)

    total_q, hq, dq = query.shape
    total_k = key.shape[0]
    dv = value.shape[-1]

    window_size = _validate_window_size(window_size)
    max_seqlen_q, max_seqlen_k = _validate_cu_seqlens(
        cu_seqlens_q,
        cu_seqlens_k,
        total_q=total_q,
        total_k=total_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
        device=query.device,
    )

    dropout_p = _validate_dropout_p(dropout_p)

    effective_alibi_slopes: Optional[torch.Tensor] = None
    if alibi_slopes is not None:
        effective_alibi_slopes = _validate_explicit_alibi_slopes(alibi_slopes, hq=hq, device=query.device)

    effective_sink: Optional[torch.Tensor] = None
    if sink is not None:
        effective_sink = _validate_explicit_sink(
            sink, hq=hq, head_dim_qk=dq, head_dim_v=dv, device=query.device
        )

    # softcap (explicit) is blocked at the kernel layer on this build exactly like the
    # dense entry: the aiter varlen backward exposes no softcap parameter (see
    # FLEX_COMPAT_STATUS.md), so a positive cap hard-errors rather than being silently
    # dropped. None/0 is a no-op (drop-in default).
    effective_softcap = _normalise_explicit_softcap(softcap)
    if effective_softcap > 0.0:
        raise NotImplementedError(
            "Turbo flex varlen entry: the softcap interface is in place "
            f"(cap~={effective_softcap:.4g}), but it is blocked by this build's aiter kernels (the "
            "varlen backward lacks a softcap parameter, see FLEX_COMPAT_STATUS.md). To avoid "
            "silently dropping the cap and producing wrong results, softcap>0 raises here -- we "
            "never degrade to a path that ignores the cap."
        )

    # Lazy import so pure classification / validation (and this module's import) does
    # not force the heavy backend kernels to load.
    from primus_turbo.pytorch.ops.attention.flash_attn_interface import flash_attn_varlen_func

    # ``sink`` is a newer varlen-backend feature: only thread it through when the
    # caller actually supplies one, so this entry stays compatible with backend
    # builds whose ``flash_attn_varlen_func`` predates the ``sink`` parameter (a
    # ``sink=None`` default is a no-op either way). ``bias`` is not exposed by this
    # varlen entry, so it is left to the backend default rather than passed.
    call_kwargs: Dict[str, Any] = dict(
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=effective_alibi_slopes,
        deterministic=False,
        return_lse=return_lse,
    )
    if effective_sink is not None:
        call_kwargs["sink"] = effective_sink

    return flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        **call_kwargs,
    )
