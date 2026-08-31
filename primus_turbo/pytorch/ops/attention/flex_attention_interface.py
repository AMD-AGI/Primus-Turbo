###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Public API of the Turbo ``flex_attention`` compatibility layer.

This module is the **only** entry point callers should import. It exposes a
drop-in replacement for ``torch.nn.attention.flex_attention.flex_attention``:
common variants that Turbo accelerates (full / causal / sliding-window-causal /
bidirectional band / document-causal masks, optional ALiBi score bias, GQA/MQA) are recognised at
runtime and dispatched onto the high-performance ``flash_attn_func`` backend
(FlyDSL/AITER on gfx950). Anything that cannot be mapped onto those fixed kernels
raises ``NotImplementedError`` with an explanation -- it is never silently dropped.

Layering
--------
Only the signatures, the argument contract and the top-level control flow live
here. The machinery sits in the :mod:`primus_turbo.pytorch.ops.attention.flex`
subpackage, one concern per module::

    flex/_config.py         tunable constants and tolerances
    flex/_cache.py          identity-keyed classification / detection caches
    flex/_probe.py          primitives that call ``mask_mod`` / ``score_mod``
    flex/_mask_classify.py  block_mask  -> kernel mask parameters
    flex/_score_mod.py      score_mod   -> alibi_slopes / softcap
    flex/_validate.py       argument validation
    flex/_routing.py        backend selection (performance routing layer)
    flex/_dispatch.py       lowering onto the varlen (THD) backend
    flex/_layout.py         bhsd <-> backend layout, without needless copies

The dependency graph is strictly layered and acyclic: ``_config`` / ``_cache`` /
``_probe`` / ``_routing`` / ``_dispatch`` / ``_layout`` are leaves, ``_mask_classify``
and ``_score_mod`` build on them, and only this module depends on everything.

Design notes
------------
* torch flex uses the ``bhsd`` (``[B, H, S, D]``) layout; ``flash_attn_func`` takes a
  ``[B, S, H, D]``-shaped tensor and reads the real memory order out of its strides.
  ``transpose(1, 2)`` supplies that shape as a *view*, and bhsd is a layout the backend
  addresses natively, so the round-trip normally costs no copies at all -- see
  ``flex/_layout.py`` for the one case that still materialises.
* Mask semantics are recovered by probing ``block_mask.mask_mod`` on an index grid
  and matching it against fixed templates (see ``flex/_mask_classify.py``).
  Data-dependent / batch-or-head dependent masks are rejected.
* Turbo-extension explicit args are a *superset* of the torch signature and all
  default to off (``None`` / ``0.0``), so a torch-style call is byte-for-byte
  unchanged and this stays a drop-in replacement.
* Softcap is detected and accepted in the signature but currently **blocked at the
  kernel layer** (the aiter dense fwd/bwd on this build expose no softcap
  parameter), so it raises rather than dropping the cap.


Relationship to PyTorch
-----------------------
This module is an **independent implementation**, not a port or a copy of
``torch/nn/attention/flex_attention.py``. It matches torch's *interface* -- the
module name, the ``flex_attention`` / ``create_block_mask`` entry points, and the
``score_mod`` / ``mask_mod`` calling convention -- because it is meant to be a
drop-in replacement. It shares none of torch's *implementation*: torch lowers
``score_mod`` / ``mask_mod`` into generated Triton kernels through Inductor,
whereas this module classifies them at runtime and dispatches onto Turbo's fixed
aiter kernels. There is no codegen here at all (see :func:`_dispatch_custom`).

That is why this file carries only the AMD copyright header. The repo's
convention for adapted third-party code is the dual-copyright + SPDX +
``Adapted from ...`` banner that ``tools/check_license.py`` emits (the
``primus_turbo/flydsl`` tree is a live example); that banner deliberately does
*not* apply here, and adding a PyTorch (BSD-3-Clause) notice would misattribute
AMD-authored code.

The only PyTorch code this module uses is used **by import**, never by copying:

* ``torch.nn.attention.flex_attention.create_block_mask`` -- imported at module
  scope and re-exported through the thin :func:`create_block_mask` passthrough, so
  the returned object is a genuine torch ``BlockMask`` and stays compatible with
  code that feeds the same mask to torch's own kernel.
* ``BlockMask`` is never reimplemented; we consume the instances torch returns and
  only read their ``.mask_mod`` attribute.

Torch helpers deliberately *not* reused, with the reason:

* ``create_mask`` cannot replace :func:`_probe_mask_grid`. It is built on
  ``torch.vmap``, which raises ``RuntimeError`` ("data-dependent control flow")
  for scalar-style ``mask_mod`` callables written with Python ``and``; the probe
  helper falls back to an element-wise loop for exactly those. ``create_mask``
  also materialises the whole ``[B, H, Q_LEN, KV_LEN]`` mask, while
  :func:`_probe_mask_row` and :func:`_locate_left_window` sample single rows on
  purpose, keeping classification of a long-sequence mask O(S log S) rather than
  O(S**2).
* ``and_masks`` / ``or_masks`` / ``noop_mask`` are not reimplemented here either;
  callers that want them should import them from torch directly.
"""

import warnings
from typing import Any, Dict, Optional, Tuple

import torch

try:
    from torch.nn.attention.flex_attention import create_block_mask as _torch_create_block_mask
except Exception:  # pragma: no cover - depends on torch version
    _torch_create_block_mask = None

from .flex._config import _ALIBI_TOL
from .flex._dispatch import _dispatch_document_varlen, require_varlen_sink_support
from .flex._layout import from_backend_layout, to_backend_layout_qkv
from .flex._mask_classify import _classify_block_mask
from .flex._routing import _dispatch_custom, choose_backend
from .flex._score_mod import _cached_detect_alibi_slopes, _cached_detect_softcap, _is_identity_score_mod
from .flex._support_status import SUPPORT_STATUS
from .flex._validate import (
    _normalise_explicit_softcap,
    _validate_and_adapt_bias,
    _validate_cu_seqlens,
    _validate_dropout_p,
    _validate_explicit_alibi_slopes,
    _validate_explicit_sink,
    _validate_qkv,
    _validate_qkv_varlen,
    _validate_window_size,
)

# Public API of the flex-attention compatibility layer.
#
# Deliberately small: the four entry points plus the capability manifest. Everything
# else in this package is an implementation detail under ``flex/`` -- the backend
# routing registry and the classification caches are reachable from
# ``primus_turbo.pytorch.ops.attention.flex._routing`` / ``._cache`` for tests and
# tuners, but they are not part of the supported surface.
__all__ = [
    "flex_attention",
    "flex_attention_bshd",
    "flex_attention_varlen",
    "create_block_mask",
    "SUPPORT_STATUS",
]


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


# torch flex_attention kernel_options that only steer Triton codegen for the kernel
# torch would have generated. Turbo routes onto fixed backend kernels instead, so these
# have no analogue here; dropping them costs (at most) performance, never correctness.
_IGNORABLE_KERNEL_OPTIONS = frozenset(
    {
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_M1",
        "BLOCK_N1",
        "BLOCK_M2",
        "BLOCK_N2",
        "num_stages",
        "num_warps",
        "PRESCALE_QK",
        "ROWS_GUARANTEED_SAFE",
        "BLOCKS_ARE_CONTIGUOUS",
        "WRITE_DQ",
        "FORCE_USE_FLEX_ATTENTION",
        "OUTPUT_LOGSUMEXP",
        "USE_TMA",
    }
)


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
    deterministic: bool = False,
    fp8: bool = False,
    fp8_config: Optional[Any] = None,
    _return_bshd: bool = False,
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
      dense fwd/bwd on this build expose no softcap parameter). It will take
      effect once the upstream kernel adds the
      parameter.
    * ``dropout_p`` (``float``, default ``0.0``): attention-dropout probability
      threaded straight to ``flash_attn_func(dropout_p=...)``. Requires
      ``0 <= p < 1`` (``0`` disables dropout, the drop-in default). As in
      flash-attn / torch ``scaled_dot_product_attention`` it is applied whenever
      ``p > 0`` (training convention -- pass ``0`` for eval); it composes with
      ``return_lse`` and with the ``deterministic`` flag below.
    * ``sink`` (``Optional[torch.Tensor]``, default ``None``): attention-sink
      logits (one learned value per query head), threaded straight to
      ``flash_attn_func(sink=...)``. Requires a 1D fp32 tensor of length ``Hq``;
      the sink kernel path also requires ``head_dim_qk == head_dim_v`` with a
      power-of-two head dim (backend constraint). ``None`` disables it (no-op).
    * ``bias`` (``Optional[torch.Tensor]``, default ``None``): additive attention
      bias on the pre-softmax logits, threaded to ``flash_attn_func(bias=...)``.
      The aiter dense kernel accepts a single ``[Sq, Skv]`` bias in q's dtype
      shared across batch/heads (an fp32 bias yields NaN, a per-head 4D bias is
      rejected by the kernel). This entry accepts
      ``[Sq, Skv]`` (or a leading-singleton broadcast ``[1,Sq,Skv]`` /
      ``[1,1,Sq,Skv]``), casts it to q's dtype and moves it to q's device; a
      genuine per-batch/per-head bias raises ``ValueError``. Verified numerically
      correct fwd+bwd (rel-L2 ~2e-3). ``None`` disables it (no-op).
    * ``deterministic`` (``bool``, default ``False``): threaded straight to
      ``flash_attn_func(deterministic=...)`` -- the same knob, with the same meaning
      and the same backend guarantees, that a direct ``flash_attn_func`` caller gets.
      It selects the backward's deterministic dQ accumulation where the backend
      offers one. ``False`` (the default) reproduces the historical behaviour
      byte-for-byte, so this stays a drop-in replacement; torch's own
      ``flex_attention`` has no such parameter, so this is a Turbo extension.
      Note this is *not* a promise of bit-reproducibility: what it does is
      exactly what the underlying kernel does with the flag, no more. Previously
      this layer hard-coded ``deterministic=False``, which meant a caller asking
      for determinism silently did not get it.
    * ``fp8`` (``bool``, default ``False``) / ``fp8_config``
      (``Float8QuantConfig``, default ``None``): run the attention in fp8 via
      :func:`primus_turbo.pytorch.ops.attention.flash_attn_fp8_func` instead of the
      bf16/fp16 ``flash_attn_func``. Passing ``fp8_config`` implies ``fp8=True``;
      passing ``fp8=True`` alone uses the backend default (BLOCKWISE, block_size=64).
      ``False`` / ``None`` (the defaults) leave the historical path untouched.

      fp8 lands on a *different kernel* (aiter's Triton attention) than the bf16 path
      (aiter's CK attention), and that kernel supports strictly less. Every feature it
      cannot do is rejected here rather than dropped: ``sink`` (``flash_attn_fp8_func``
      has no such parameter), ``bias`` (its backward asserts ``bias is None``),
      a sliding window (both its fwd and bwd assert ``window_size == (-1, -1)``),
      ``dropout_p > 0`` (the forward applies dropout but the backward has no
      ``dropout_p`` parameter at all, so the gradients would not match the forward),
      ``deterministic=True`` (accepted by the signature and never read),
      ``return_lse`` (the Triton LSE buffer is ``[B, H, 2*Sq]``, a different convention
      from the dense path's ``[B, H, Sq]``), a document-causal ``block_mask`` (there is
      no fp8 varlen entry to lower it onto), and a non-bf16 dtype (the backward asserts
      the incoming grad is bfloat16). This is the whole point of the gate: an fp8 run
      that quietly ignored the window or the sink would train a different model than
      the config asked for.
    """
    _validate_qkv(query, key, value)

    if kernel_options:
        # torch's flex_attention takes kernel_options as a bag of Triton autotuning
        # knobs for the kernel it generates. Turbo does not generate a kernel -- it
        # routes onto a fixed aiter/CK or Triton entry -- so none of them can be
        # honoured. The known ones are safe to drop: BLOCK_M/BLOCK_N/num_stages/
        # num_warps only pick a tile shape and occupancy (performance, not numerics),
        # and PRESCALE_QK/ROWS_GUARANTEED_SAFE/BLOCKS_ARE_CONTIGUOUS/WRITE_DQ are
        # opt-in fast paths whose default is the *more* conservative, more accurate
        # behaviour, which is what the backend already does. FORCE_USE_FLEX_ATTENTION
        # only disables torch's own decode shortcut, which this layer never takes.
        # An unrecognised key is a different matter: it is either a typo or a knob
        # that does change results, and silently ignoring it is exactly the
        # silent-drop failure this layer exists to prevent -- so raise instead.
        unknown = sorted(set(kernel_options) - _IGNORABLE_KERNEL_OPTIONS)
        if unknown:
            raise NotImplementedError(
                "Turbo flex compat layer: unrecognised kernel_options "
                f"{unknown}. Turbo dispatches onto fixed backend kernels instead of "
                "generating one, so it cannot honour them, and ignoring an option it "
                "does not recognise could silently change what gets computed. Known "
                "performance-only options that are safe to ignore: "
                f"{sorted(_IGNORABLE_KERNEL_OPTIONS)}."
            )
        warnings.warn(
            "Turbo flex compat layer ignores kernel_options (it dispatches onto fixed "
            "backend kernels rather than generating one); these are performance-only "
            f"knobs and dropping them does not change results: {sorted(kernel_options)}",
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

    # ---- bidirectional band window + sink -------------------------------------
    # A non-causal band mask lowers to window_size=(left, right). The csrc/CK entry
    # forwards both edges to the forward *and* the backward, so it is exact there. The
    # sink route is not: its backward calls triton_flash_attn_onekernel_backward with
    # sliding_window=window_size_left only -- the right edge is dropped, so the backward
    # would differentiate a wider mask than the forward computed and return silently
    # wrong gradients. Refuse the combination instead.
    if has_sink and mask_cfg["window_size"][1] > 0:
        raise NotImplementedError(
            "Turbo flex compat layer: a bidirectional (non-causal) window "
            f"{mask_cfg['window_size']} cannot be combined with a sink -- the sink backward takes "
            "only a left window (sliding_window=window_size_left), so it would compute gradients "
            "for a wider mask than the forward used. Drop the sink, or use a causal left-window "
            "mask."
        )

    # ---- single softcap enablement point --------------------------------------
    # softcap (explicit or detected) is blocked at the kernel layer on this build.
    # Surveyed rather than assumed -- the state is asymmetric between fwd and bwd:
    #   forward   CK ck_tile implements it (ops/fmha/block/variants.hpp,
    #             LogitsSoftCapParams; tanh is the compile-time default, softsign
    #             selectable via CK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT) and
    #             aiter's dense binding passes a literal `0.0, // logits_soft_cap`
    #             (csrc/py_itfs_ck/mha_fwd_kernels.cu). Binding-only work.
    #   backward  nothing implements it. CK's fmha_bwd_kernel.hpp has no soft cap at
    #             all, nor do aiter's mha_bwd_kernels.cu / mha_varlen_bwd_kernels.cu;
    #             dS needs the tanh derivative and that math is not written. aiter's
    #             own varlen python fwd DOES take logits_soft_cap while
    #             FlashAttnVarlenFunc.backward neither passes it nor returns a
    #             gradient for it -- capped forward, uncapped gradients.
    # So this cannot be unblocked by exposing one forward argument: doing that alone
    # would train capped logits against uncapped gradients, which is precisely the
    # silent-drop failure this layer exists to prevent. Both an explicit softcap and
    # one detected from score_mod raise here; we never degrade to a path that ignores
    # the cap, and we do not accept a forward-only cap either.
    # TODO(softcap): once a trainable aiter fwd+bwd PAIR supports it, thread through to
    #   flash_attn_func(softcap=...): drop this guard and pass effective_softcap at the
    #   flash_attn_func call below -- a one-line switch to enable. Enabling on a
    #   forward-only kernel is NOT sufficient; check the backward before flipping.
    if effective_softcap > 0.0:
        raise NotImplementedError(
            "Turbo flex compat layer: the softcap interface is in place "
            f"(cap~={effective_softcap:.4g}), but no trainable aiter fwd+bwd pair on this build "
            "implements it. The dense forward could expose it (CK has the tanh variant; aiter "
            "hardcodes logits_soft_cap=0.0 in the binding), but no backward -- CK's fmha_bwd, "
            "aiter's mha_bwd/mha_varlen_bwd, and the trainable triton backward kernels all lack "
            "it, so gradients would be computed without the cap. To avoid silently dropping the "
            "cap -- or, worse, capping the forward while leaving the backward uncapped -- both an "
            "explicit softcap and a soft-cap detected from score_mod raise here."
        )

    # ---- fp8 gate --------------------------------------------------------------
    # fp8 is a different kernel family with a strictly smaller feature set; see the
    # docstring. Reject every combination it cannot honour instead of silently
    # dropping the feature.
    use_fp8 = bool(fp8) or fp8_config is not None
    if use_fp8:
        _reject = []
        if has_sink:
            _reject.append("sink (flash_attn_fp8_func has no 'sink' parameter)")
        if has_bias:
            _reject.append("bias (the fp8 backward asserts bias is None)")
        if mask_cfg["window_size"] != (-1, -1):
            _reject.append(
                f"sliding window {mask_cfg['window_size']} (the Triton fp8 fwd and bwd both "
                "assert window_size == (-1, -1))"
            )
        if has_dropout:
            _reject.append(
                "dropout_p > 0 (the Triton fp8 forward applies dropout but its backward takes "
                "no dropout_p at all, so the gradients would not match the forward)"
            )
        if deterministic:
            _reject.append("deterministic=True (flash_attn_fp8_func accepts the flag and never reads it)")
        if return_lse:
            _reject.append(
                "return_lse (the Triton fp8 LSE buffer is [B, H, 2*Sq], a different convention "
                "from the dense path's [B, H, Sq])"
            )
        if mask_cfg.get("kind") in ("document_causal", "document"):
            _reject.append(
                "a document-packed block_mask (there is no fp8 varlen entry to lower packing onto)"
            )
        if query.dtype is not torch.bfloat16:
            _reject.append(
                f"dtype {query.dtype} (the Triton fp8 backward asserts the incoming grad is "
                "bfloat16, so fp8 here is bf16-only)"
            )
        if _reject:
            raise NotImplementedError(
                "Turbo flex compat layer: fp8 attention cannot honour "
                + "; ".join(_reject)
                + ". fp8 runs on aiter's Triton attention kernel, which supports strictly less "
                "than the bf16/fp16 CK kernel the rest of this layer dispatches to. These raise "
                "rather than being ignored, because an fp8 run that silently dropped one of them "
                "would compute a different attention than the one that was configured."
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
            deterministic=deterministic,
            backend=backend,
        )

    # ---- document packing (block-diagonal, any within-document pattern) -------
    # A block_mask recognised as ``same_doc(q,kv)`` -- optionally intersected with a
    # causal term and/or a within-document window -- is dispatched through the varlen
    # backend (packed cu_seqlens) instead of a dense call, which would attend across
    # document boundaries. Recognition is by exact reconstruction (see
    # _detect_document_blocks), so this only fires on genuine doc packing. "causal" is
    # the historical kind for the autoregressive shape; "document" carries the recovered
    # causal flag and within-document window for the general case (notably bidirectional
    # packing, which is what non-autoregressive image / video models use).
    if mask_cfg.get("kind") in ("document_causal", "document"):
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
        if has_sink:
            # Order matters. First: can this build's varlen wrapper carry a sink at all?
            # If not, no arrangement of the documents helps, and blaming their lengths
            # would send the caller off padding them for nothing. Checked here as well as
            # at the attach point so the diagnosis stays attached to the block_mask the
            # caller actually passed.
            from primus_turbo.pytorch.ops.attention.flash_attn_interface import (
                flash_attn_varlen_func,
            )

            require_varlen_sink_support(flash_attn_varlen_func)
            if len(set(mask_cfg["doc_seglens"])) != 1:
                # Only then: on a build that does forward a sink, ragged segments still
                # have no eligible backend -- the FlyDSL varlen backend is the one that
                # carries a sink and it requires uniform lengths, while the aiter varlen
                # path declines (VarlenAttnFwdAiterBackend.can_handle returns False,
                # which is what keeps the sink from being silently dropped). Without this
                # the caller gets a generic "No compatible backend found" from deep
                # inside kernel selection.
                raise NotImplementedError(
                    "Turbo flex compat layer: a sink combined with document packing requires "
                    "equal document lengths (only the FlyDSL varlen backend carries a sink), "
                    f"got {sorted(set(mask_cfg['doc_seglens']))}. Drop the sink, or pad the "
                    "documents to a common length."
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
            deterministic=deterministic,
            return_bshd=_return_bshd,
            causal=mask_cfg["causal"],
            window_size=mask_cfg["window_size"],
        )

    # ---- dense route: deterministic + sink ------------------------------------
    # ``FlashAttnFunc.forward`` asserts these two are never on together (the sink
    # backward has no deterministic dQ accumulation path). Reaching that bare
    # assertion from here would name the wrong culprit, and it only became reachable
    # once this layer started threading ``deterministic`` instead of hard-coding it
    # to False. Document packing is exempt on purpose: it lands on the varlen entry,
    # which carries no such assertion, and it has already returned above.
    if deterministic and has_sink:
        raise NotImplementedError(
            "Turbo flex compat layer: deterministic=True cannot be combined with a sink on the "
            "dense route -- the backend asserts the same thing (flash_attn_interface.py: "
            '"deterministic and sink cannot be enabled together currently"), because the sink '
            "backward has no deterministic dQ accumulation. Set deterministic=False, or drop the "
            "sink."
        )

    # Lazy import so pure classification (and this module's import) does not force
    # the heavy backend kernels to load.
    if use_fp8:
        from primus_turbo.pytorch.ops.attention.flash_attn_interface import flash_attn_fp8_func

        # The fp8 entry lowers onto the Triton kernel, which hardcodes layout="bshd"
        # and asserts q/k/v are contiguous. The zero-copy bhsd passthrough that the
        # bf16 path uses (to_backend_layout_qkv) is deliberately not used here: it hands
        # the backend a [B,S,H,D]-shaped *view* over bhsd bytes, which
        # ``_infer_qkv_format`` reports as "bhsd" and ``flash_attn_fp8_func`` then
        # permutes again -- a second transpose that the Triton path does not want.
        # Materialising a plain bshd-contiguous buffer takes the unambiguous branch,
        # and costs nothing extra: the fp8 entry calls ``.contiguous()`` regardless.
        q_be = query.transpose(1, 2).contiguous()
        k_be = key.transpose(1, 2).contiguous()
        v_be = value.transpose(1, 2).contiguous()

        # The Triton kernel behind the fp8 entry addresses the slopes as
        # ``off_z * stride_az + off_h * stride_ah`` and reads ``alibi_slopes.stride(1)``,
        # i.e. it wants a 2D ``[B, Hq]`` tensor. Every other backend in this layer takes
        # the aiter 1D ``[Hq]`` convention, and _validate enforces 1D, so handing the 1D
        # tensor straight through raises a bare IndexError from inside the kernel launch.
        # Broadcasting over the batch is the faithful translation rather than a guess:
        # ALiBi slopes are per-head and batch-independent by construction, which is
        # exactly what _detect_alibi_slopes verifies before returning them.
        fp8_alibi_slopes = effective_alibi_slopes
        if fp8_alibi_slopes is not None and fp8_alibi_slopes.dim() == 1:
            fp8_alibi_slopes = fp8_alibi_slopes.unsqueeze(0).expand(q_be.shape[0], -1).contiguous()

        out = flash_attn_fp8_func(
            q_be,
            k_be,
            v_be,
            dropout_p=dropout_p,  # gated to 0.0 above
            softmax_scale=scale,
            causal=mask_cfg["causal"],
            window_size=mask_cfg["window_size"],  # gated to (-1, -1) above
            bias=None,  # gated to None above
            alibi_slopes=fp8_alibi_slopes,  # 1D [Hq] -> 2D [B, Hq] for the Triton kernel
            deterministic=deterministic,  # gated to False above
            return_lse=return_lse,  # gated to False above
            fp8_config=fp8_config,  # None -> backend default (BLOCKWISE, block_size=64)
        )
        if _return_bshd:
            return out
        return from_backend_layout(out)

    from primus_turbo.pytorch.ops.attention.flash_attn_interface import flash_attn_func

    # bhsd -> the backend's [B,S,H,D] logical shape. transpose(1, 2) is a view; the
    # backend reads the real memory order out of the strides and addresses bhsd
    # natively, so this normally copies nothing (see flex/_layout.py).
    # Converted as a triple, not one by one: the backend asserts that q, k and v all
    # report the same memory order, so the passthrough decision has to be joint.
    q_be, k_be, v_be = to_backend_layout_qkv(query, key, value)

    out = flash_attn_func(
        q_be,
        k_be,
        v_be,
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=mask_cfg["causal"],
        window_size=mask_cfg["window_size"],
        bias=effective_bias,
        alibi_slopes=effective_alibi_slopes,
        deterministic=deterministic,
        return_lse=return_lse,
        sink=sink,
        # TODO(softcap): once a trainable upstream aiter fwd+bwd PAIR supports it, pass
        #   softcap=effective_softcap here and delete the softcap guard above -- a one-line
        #   enable (effective_softcap is currently always 0.0). A forward-only kernel does
        #   not qualify: see the survey in the guard above.
    )

    # ``_return_bshd`` (private, used by flex_attention_bshd) hands the backend's native
    # bshd output straight back instead of restoring the bhsd view.
    if return_lse:
        out_bshd, lse = out
        if _return_bshd:
            return out_bshd, lse
        return from_backend_layout(out_bshd), lse
    if _return_bshd:
        return out
    return from_backend_layout(out)


def flex_attention_bshd(
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
    deterministic: bool = False,
    fp8: bool = False,
    fp8_config: Optional[Any] = None,
):
    """Layout-native entry: ``[B, S, H, D]`` in, ``[B, S, H, D]`` out (no transposes).

    Semantically identical to :func:`flex_attention` -- same classification, same
    validation, same errors, same numerics -- but it speaks the backend's own **bshd**
    layout, so the layout plumbing disappears entirely:

    * A bshd caller going through :func:`flex_attention` would have to transpose into
      bhsd first and take a bhsd-shaped result back, which forces the shape bookkeeping
      onto the caller even where no bytes move.
    * This entry hands the backend the tensor it already has and returns the backend's
      own output buffer: q/k/v pass straight through when they are bshd-contiguous, so
      **zero layout copies and no transposes on either side**.
    * :func:`flex_attention` itself is no longer the copying path it once was -- a
      bhsd-contiguous batch is handed through as a view as well (``flex/_layout.py``).
      The remaining reason to pick this entry is that bshd in / bshd out needs no
      transposes at all, and that a ``B == 1`` bhsd input does still get materialised.

    Not a drop-in for ``torch.nn.attention.flex_attention.flex_attention`` (its layout
    differs) -- that contract belongs to :func:`flex_attention`, which is unchanged. Use
    this one deliberately when your tensors are bshd.

    ``score_mod`` / ``block_mask`` index semantics are unaffected: ``q_idx`` / ``kv_idx``
    are sequence positions in both layouts.
    """
    # transpose(1, 2) on a bshd tensor is a *view* (no copy). The result is the bhsd
    # tensor flex_attention expects, and its own `.transpose(1, 2).contiguous()` then
    # collapses back to the original bshd-contiguous buffer -- torch's contiguous() is a
    # no-op on an already-contiguous tensor, so nothing is copied on the way in either.
    out = flex_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        return_lse=return_lse,
        kernel_options=kernel_options,
        alibi_slopes=alibi_slopes,
        softcap=softcap,
        dropout_p=dropout_p,
        sink=sink,
        bias=bias,
        deterministic=deterministic,
        fp8=fp8,
        fp8_config=fp8_config,
        _return_bshd=True,
    )
    return out


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
    deterministic: bool = False,
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
    deterministic : bool, default False
        Threaded straight to ``flash_attn_varlen_func(deterministic=...)``; same knob,
        same meaning and same backend guarantees as a direct varlen call. ``False``
        (the default) is the historical behaviour.
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
    # dense entry: the aiter varlen backward exposes no softcap parameter, so a
    # positive cap hard-errors rather than being silently
    # dropped. None/0 is a no-op (drop-in default).
    effective_softcap = _normalise_explicit_softcap(softcap)
    if effective_softcap > 0.0:
        raise NotImplementedError(
            "Turbo flex varlen entry: the softcap interface is in place "
            f"(cap~={effective_softcap:.4g}), but it is blocked by this build's aiter kernels (the "
            "varlen backward lacks a softcap parameter). To avoid "
            "silently dropping the cap and producing wrong results, softcap>0 raises here -- we "
            "never degrade to a path that ignores the cap."
        )

    # Lazy import so pure classification / validation (and this module's import) does
    # not force the heavy backend kernels to load.
    from primus_turbo.pytorch.ops.attention.flash_attn_interface import flash_attn_varlen_func

    # ``sink`` is threaded only when the caller supplies one, and only after
    # ``require_varlen_sink_support`` confirms this build's ``flash_attn_varlen_func``
    # has somewhere to put it (a ``sink=None`` default is a no-op either way).
    # ``bias`` is deliberately not passed, and the reason is not "the varlen entry has
    # no such parameter" -- it does. aiter's _flash_attn_varlen_forward takes ``bias``
    # while _flash_attn_varlen_backward has none, which is the same fwd-only shape as
    # logits_soft_cap: wiring it up would give a correct forward and a silently wrong
    # gradient. Left at the backend default until the backward exists.
    call_kwargs: Dict[str, Any] = dict(
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=effective_alibi_slopes,
        deterministic=deterministic,
        return_lse=return_lse,
    )
    if effective_sink is not None:
        require_varlen_sink_support(flash_attn_varlen_func)
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
