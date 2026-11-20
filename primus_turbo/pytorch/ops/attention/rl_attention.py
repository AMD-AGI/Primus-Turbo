###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Batch-invariant attention for RL: rollout (paged KV cache) and training (varlen) run one kernel
family with one reduction order, so a token's forward output is bitwise the same whether it was
computed in a rollout prefill, a rollout decode step, or the training forward, and whatever else
shares its batch.

* ``rl_paged_attention_func``: rollout, vLLM-layout paged KV cache, forward only.
* ``rl_flash_attn_varlen_func`` / ``rl_flash_attn_func``: training, THD / BSHD, forward + backward.

All three are causal (bottom-right aligned) by default, take an optional sliding window
(``window_left``), per-head attention sinks (``sinks``), an optional visibility mask built by
``create_block_mask``, and ``kernel_options``.

Kernel options
--------------
``kernel_options`` (a dict, as in FlexAttention) sets kernel parameters per call; the environment
variable PRIMUS_TURBO_RL_ATTN_KERNEL_OPTIONS ("KEY=VALUE,KEY=VALUE") sets process-wide defaults
that a call overrides key by key. 0 or "auto" means the backend default.

==============  ===========  ===================================================================
Key             Bits change  Meaning
==============  ===========  ===================================================================
TILE_SIZE       yes          KV tile per inner step (default 64, 32 for head size > 128)
SEGMENT_SIZE    yes          KV segment length, the split-KV unit (default 1024)
BACKEND         yes          "TRITON" or "FLYDSL" (gfx950)
MODE            no           "auto", "fused" (one pass) or "split" (split-KV + reduce)
SPLIT_SEGS      no           FLYDSL: segments per split-KV program (default 1)
PACK            no           FLYDSL: "auto", "on" or "off" — (token, head) rows of one query group
BLOCK_M         no           query rows per program (a multiple of the GQA group)
num_warps       no           clamped to BLOCK_M / 16 when BLOCK_M > 16 to keep the bits
num_stages      no           software pipelining depth
waves_per_eu    no           occupancy hint
==============  ===========  ===================================================================

Keys that change bits must be the same in the rollout and the training process (set them in the
environment of both, or pass the same dict); the others may differ per call. The backward always
uses its own fixed configuration.

Visibility masks
----------------
Each query token sees one contiguous, inclusive range [lo, hi] of keys of its own sequence, given
in global key indices: keys are numbered across the batch in sequence order, so sequence s owns
keys cu_seqlens_k[s] .. cu_seqlens_k[s + 1] - 1 (for a paged cache, cu_seqlens_k is the exclusive
cumulative sum of seq_lens). With i the global index of a query token in self-attention:

======================================  =============================  ===========
Mask                                    lo                             hi
======================================  =============================  ===========
Causal                                  start of i's sequence          i
Bidirectional span (e.g. an image)      span start                     span end
Sliding window of W keys                max(seq start, i - W + 1)      i
No keys (output 0, LSE +inf)            hi + 1                         any
======================================  =============================  ===========

Keys outside [lo, hi] are exact no-ops, and tiles are walked in increasing key position whatever the
mask, so a token's output depends only on its own interval. The backward prunes its walks to the
intervals.
"""

import functools
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import torch

from primus_turbo.common import envs
from primus_turbo.pytorch.kernels.attention import rl_attention_impl

__all__ = [
    "KERNEL_OPTIONS",
    "RLBlockMask",
    "create_block_mask",
    "rl_flash_attn_func",
    "rl_flash_attn_varlen_func",
    "rl_paged_attention_func",
]

_INT_OPTIONS = (
    "TILE_SIZE",
    "SEGMENT_SIZE",
    "BLOCK_M",
    "num_warps",
    "num_stages",
    "waves_per_eu",
    "SPLIT_SEGS",
)
KERNEL_OPTIONS = _INT_OPTIONS + ("MODE", "PACK", "BACKEND")


@dataclass(frozen=True)
class _Options:
    tile_size: int = 0
    segment_size: int = 0
    block_m: int = 0
    num_warps: int = 0
    num_stages: int = 0
    waves_per_eu: int = 0
    split_segs: int = 0
    mode: str = "auto"
    pack: str = "auto"
    backend: str = "TRITON"

    def fwd_args(self) -> tuple:
        """In the order of rl_attention_fwd_impl's trailing arguments, after ``return_lse``."""
        return (
            self.mode,
            self.tile_size,
            self.segment_size,
            self.block_m,
            self.num_warps,
            self.num_stages,
            self.waves_per_eu,
            self.split_segs,
            self.pack,
            self.backend,
        )


def _merge_options(options: dict, source: str, into: dict) -> dict:
    unknown = set(options) - set(KERNEL_OPTIONS)
    if unknown:
        raise ValueError(f"unsupported {source} keys {sorted(unknown)}; supported: {KERNEL_OPTIONS}")
    for key, value in options.items():
        if key in _INT_OPTIONS:
            try:
                value = int(value)
            except (TypeError, ValueError):
                raise ValueError(f"{source} {key} must be an integer, got {value!r}") from None
            if value < 0:
                raise ValueError(f"{source} {key} must be >= 0, got {value}")
        elif key == "MODE":
            value = str(value).lower()
            if value not in ("auto", "fused", "split"):
                raise ValueError(f"{source} MODE must be auto, fused or split, got {value!r}")
        elif key == "PACK":
            value = str(value).lower()
            if value not in ("auto", "on", "off"):
                raise ValueError(f"{source} PACK must be auto, on or off, got {value!r}")
        else:
            value = str(value).upper()
            if value not in rl_attention_impl.BACKEND_NAMES:
                raise ValueError(
                    f"{source} BACKEND must be one of {rl_attention_impl.BACKEND_NAMES}, got {value!r}"
                )
        into[key] = value
    return into


@functools.lru_cache(maxsize=8)
def _env_options(raw: Optional[str]) -> Tuple[Tuple[str, object], ...]:
    if raw is None:
        return ()
    pairs = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"PRIMUS_TURBO_RL_ATTN_KERNEL_OPTIONS: expected KEY=VALUE, got {item!r}")
        pairs[key.strip()] = value.strip()
    return tuple(_merge_options(pairs, "PRIMUS_TURBO_RL_ATTN_KERNEL_OPTIONS", {}).items())


def _resolve_options(kernel_options: Optional[dict]) -> _Options:
    merged = dict(_env_options(envs.PRIMUS_TURBO_RL_ATTN_KERNEL_OPTIONS))
    _merge_options(dict(kernel_options or {}), "kernel_options", merged)
    return _Options(
        tile_size=merged.get("TILE_SIZE", 0),
        segment_size=merged.get("SEGMENT_SIZE", 0),
        block_m=merged.get("BLOCK_M", 0),
        num_warps=merged.get("num_warps", 0),
        num_stages=merged.get("num_stages", 0),
        waves_per_eu=merged.get("waves_per_eu", 0),
        split_segs=merged.get("SPLIT_SEGS", 0),
        mode=merged.get("MODE", "auto"),
        pack=merged.get("PACK", "auto"),
        backend=merged.get("BACKEND", "TRITON"),
    )


@dataclass(frozen=True)
class RLBlockMask:
    """Per query token visible key range [lo, hi], inclusive, in global key indices."""

    lo: torch.Tensor  # [num_tokens] int32
    hi: torch.Tensor  # [num_tokens] int32
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def num_tokens(self) -> int:
        return self.lo.numel()

    def with_window(self, window_left: int) -> "RLBlockMask":
        """Narrow each row to [max(lo, hi - window_left), hi]. Cached, so every layer of a step
        reuses one narrowed mask."""
        if window_left is None or window_left < 0:
            return self
        key = ("window", window_left)
        if key not in self._cache:
            self._cache[key] = RLBlockMask(
                torch.maximum(self.lo, self.hi - window_left).contiguous(), self.hi
            )
        return self._cache[key]

    def key_starts(self, seq_lens: torch.Tensor) -> torch.Tensor:
        """cu_seqlens_k of a paged cache: the exclusive cumulative sum of seq_lens. Cached per
        seq_lens tensor and version, so every layer of a step reuses one."""
        cached = self._cache.get("starts")
        if cached is None or cached[0] is not seq_lens or cached[1] != seq_lens._version:
            starts = torch.nn.functional.pad(torch.cumsum(seq_lens, 0, dtype=torch.int32), (1, 0))
            cached = self._cache["starts"] = (seq_lens, seq_lens._version, starts)
        return cached[2]

    def validate(self, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor) -> None:
        """Check every non-empty interval lies inside its token's own keys. Syncs with the device."""
        q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).long()
        first = torch.repeat_interleave(cu_seqlens_k[:-1].long(), q_lens)
        last = torch.repeat_interleave(cu_seqlens_k[1:].long(), q_lens) - 1
        bad = (self.lo <= self.hi) & ((self.lo < first) | (self.hi > last))
        if bool(bad.any()):
            idx = int(bad.nonzero()[0])
            raise ValueError(
                f"visibility of token {idx} is [{int(self.lo[idx])}, {int(self.hi[idx])}], outside its "
                f"sequence's keys [{int(first[idx])}, {int(last[idx])}]"
            )


def create_block_mask(
    bounds: torch.Tensor, hi: Optional[torch.Tensor] = None, *, window_left: int = -1
) -> RLBlockMask:
    """Build a visibility mask from a [2, num_tokens] tensor of (lo, hi), or from lo and hi.

    Unlike ``torch.nn.attention.flex_attention.create_block_mask`` this takes the bounds directly,
    not a mask_mod function. ``window_left`` >= 0 narrows each row to [max(lo, hi - window_left), hi].
    """
    if hi is None:
        if bounds.ndim != 2 or bounds.shape[0] != 2:
            raise ValueError(
                f"expected a [2, num_tokens] tensor of (lo, hi), got shape {tuple(bounds.shape)}"
            )
        lo, hi = bounds[0], bounds[1]
    else:
        lo = bounds
    if lo.shape != hi.shape or lo.ndim != 1:
        raise ValueError(
            f"lo and hi must be 1D tensors of equal length, got {tuple(lo.shape)} and {tuple(hi.shape)}"
        )
    return RLBlockMask(lo.to(torch.int32).contiguous(), hi.to(torch.int32).contiguous()).with_window(
        window_left
    )


def _prepare(num_tokens, num_q_heads, window_left, block_mask, sinks):
    """Fold the window into the mask, and bring sinks to fp32."""
    if block_mask is not None:
        if block_mask.num_tokens != num_tokens:
            raise ValueError(f"block_mask has {block_mask.num_tokens} rows for {num_tokens} query tokens")
        block_mask = block_mask.with_window(window_left)
        window_left = -1
    if sinks is not None:
        if sinks.numel() != num_q_heads:
            raise ValueError(f"sinks must hold one logit per query head ({num_q_heads}), got {sinks.numel()}")
        sinks = sinks.to(torch.float32).contiguous()
    return window_left, block_mask, sinks


def rl_paged_attention_func(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    window_left: int = -1,
    block_mask: Optional[RLBlockMask] = None,
    sinks: Optional[torch.Tensor] = None,
    kernel_options: Optional[dict] = None,
) -> torch.Tensor:
    """Rollout forward over a paged KV cache.

    q [num_tokens, Hq, D]; k_cache, v_cache [num_blocks, block_size, Hkv, D]; block_table
    [num_seqs, max_blocks] int32; seq_lens [num_seqs] int32 (keys per sequence, including this
    step's queries); cu_seqlens_q [num_seqs + 1] int32. Returns out [num_tokens, Hq, Dv].
    """
    opts = _resolve_options(kernel_options)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    window_left, block_mask, sinks = _prepare(q.shape[0], q.shape[1], window_left, block_mask, sinks)
    out, _ = rl_attention_impl.rl_attention_fwd_impl(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        block_mask.key_starts(seq_lens) if block_mask is not None else None,
        seq_lens,
        block_table,
        int(max_seqlen_q),
        int(max_seqlen_k),
        float(softmax_scale),
        window_left,
        block_mask.lo if block_mask is not None else None,
        block_mask.hi if block_mask is not None else None,
        sinks,
        False,
        *opts.fwd_args(),
    )
    return out


class RLFlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, window_left, vis_lo,
        vis_hi, sinks, opts,
    ):  # fmt: skip
        sinks32 = sinks.to(torch.float32).contiguous() if sinks is not None else None
        out, lse = rl_attention_impl.rl_attention_fwd_impl(
            q, k, v, cu_seqlens_q, cu_seqlens_k, None, None, max_seqlen_q, max_seqlen_k, softmax_scale,
            window_left, vis_lo, vis_hi, sinks32, True, *opts.fwd_args(),
        )  # fmt: skip
        ctx.backend = opts.backend
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, vis_lo, vis_hi, sinks32)
        ctx.has_sinks = sinks is not None
        ctx.sinks_dtype = sinks.dtype if sinks is not None else None
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.window_left = window_left
        return out, lse

    @staticmethod
    def backward(ctx, dout, _dlse):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, vis_lo, vis_hi, sinks = ctx.saved_tensors
        dq, dk, dv, dsink = rl_attention_impl.rl_attention_bwd_impl(
            dout.contiguous(), q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, ctx.max_seqlen_q,
            ctx.max_seqlen_k, ctx.softmax_scale, ctx.window_left, vis_lo, vis_hi, sinks, ctx.backend,
        )  # fmt: skip
        dsink = dsink.to(ctx.sinks_dtype) if ctx.has_sinks else None
        return dq, dk, dv, None, None, None, None, None, None, None, None, dsink, None


def rl_flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    window_left: int = -1,
    block_mask: Optional[RLBlockMask] = None,
    sinks: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    kernel_options: Optional[dict] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Training forward + backward, THD layout: q [total_q, Hq, D], k, v [total_k, Hkv, D],
    cu_seqlens_q / cu_seqlens_k [num_seqs + 1] int32. Returns out [total_q, Hq, Dv], plus the
    natural-log LSE [total_q, Hq] fp32 with ``return_lse``."""
    opts = _resolve_options(kernel_options)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    window_left, block_mask, _ = _prepare(q.shape[0], q.shape[1], window_left, block_mask, sinks)
    out, lse = RLFlashAttnVarlenFunc.apply(
        q, k, v, cu_seqlens_q, cu_seqlens_k, int(max_seqlen_q), int(max_seqlen_k), float(softmax_scale),
        window_left, block_mask.lo if block_mask is not None else None,
        block_mask.hi if block_mask is not None else None, sinks, opts,
    )  # fmt: skip
    return (out, lse) if return_lse else out


def rl_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    window_left: int = -1,
    block_mask: Optional[RLBlockMask] = None,
    sinks: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    kernel_options: Optional[dict] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Training forward + backward, BSHD layout: q [B, Sq, Hq, D], k, v [B, Sk, Hkv, D]. A
    ``block_mask`` has B * Sq rows in batch-major order, in global key indices (row b's keys start
    at b * Sk)."""
    batch, seqlen_q, num_q_heads, _ = q.shape
    seqlen_k = k.shape[1]
    cu_q = torch.arange(0, (batch + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=q.device)
    cu_k = torch.arange(0, (batch + 1) * seqlen_k, seqlen_k, dtype=torch.int32, device=q.device)
    res = rl_flash_attn_varlen_func(
        q.reshape(batch * seqlen_q, num_q_heads, q.shape[-1]),
        k.reshape(batch * seqlen_k, k.shape[2], k.shape[-1]),
        v.reshape(batch * seqlen_k, v.shape[2], v.shape[-1]),
        cu_q,
        cu_k,
        seqlen_q,
        seqlen_k,
        softmax_scale,
        window_left,
        block_mask,
        sinks,
        return_lse,
        kernel_options,
    )
    if return_lse:
        out, lse = res
        return out.view(batch, seqlen_q, num_q_heads, -1), lse.view(batch, seqlen_q, num_q_heads)
    return res.view(batch, seqlen_q, num_q_heads, -1)


# ----------------------------------------------------------------------------- flash_attn_* routing
#
# flash_attn_func / flash_attn_varlen_func run this kernel when the caller asks for batch
# invariance (batch_invariant=True or PRIMUS_TURBO_ATTN_BATCH_INVARIANT=1), or passes an argument
# only this kernel implements (block_mask, kernel_options, a paged block_table). Their other
# backends are untouched.


def use_rl_attention(batch_invariant=False, block_mask=None, kernel_options=None, block_table=None) -> bool:
    return (
        bool(batch_invariant)
        or block_mask is not None
        or kernel_options is not None
        or block_table is not None
        or envs.PRIMUS_TURBO_ATTN_BATCH_INVARIANT
    )


def _flash_args(dropout_p, causal, window_size, bias, alibi_slopes, return_attn_probs, block_mask):
    """Map flash_attn_* arguments onto this kernel: returns (window_left, full), where ``full``
    asks for non-causal attention over each sequence's keys."""
    if dropout_p != 0.0 or bias is not None or alibi_slopes is not None or return_attn_probs:
        raise ValueError("batch-invariant attention does not implement dropout, bias, alibi or attn probs")
    left, right = window_size if window_size is not None else (-1, -1)
    if block_mask is not None:
        if (left, right) != (-1, -1):
            raise ValueError("pass the window through block_mask (create_block_mask(..., window_left=W))")
        return -1, False
    if causal:
        if right not in (-1, 0):
            raise ValueError(f"batch-invariant attention supports window_size (left, 0), got {window_size}")
        return left, False
    if (left, right) != (-1, -1):
        raise ValueError("batch-invariant attention supports sliding windows only with causal=True")
    return -1, True


def _full_mask(cu_seqlens_q, cu_seqlens_k):
    q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).long()
    lo = torch.repeat_interleave(cu_seqlens_k[:-1], q_lens)
    hi = torch.repeat_interleave(cu_seqlens_k[1:] - 1, q_lens)
    return create_block_mask(lo, hi)


def flash_attn_rl_varlen(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal,
    window_size, bias, alibi_slopes, return_lse, return_attn_probs, sink, block_mask, kernel_options,
    block_table, seqused_k,
):  # fmt: skip
    """flash_attn_varlen_func on this kernel. With ``block_table``, k / v are a paged cache
    [num_blocks, block_size, Hkv, D] and ``seqused_k`` [num_seqs] gives each sequence's key count
    (forward only). LSE is returned as [Hq, total_q], like the other varlen backends."""
    window_left, full = _flash_args(
        dropout_p, causal, window_size, bias, alibi_slopes, return_attn_probs, block_mask
    )
    if block_table is not None:
        if seqused_k is None:
            raise ValueError("a paged block_table needs seqused_k (keys per sequence)")
        if return_lse:
            raise ValueError("return_lse is not supported with a paged block_table")
        if full:
            starts = torch.nn.functional.pad(torch.cumsum(seqused_k, 0, dtype=torch.int32), (1, 0))
            block_mask = _full_mask(cu_seqlens_q, starts)
        return rl_paged_attention_func(
            q, k, v, cu_seqlens_q, seqused_k, block_table, max_seqlen_q, max_seqlen_k, softmax_scale,
            window_left, block_mask, sink, kernel_options,
        )  # fmt: skip
    if full:
        block_mask = _full_mask(cu_seqlens_q, cu_seqlens_k)
    res = rl_flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, window_left,
        block_mask, sink, return_lse, kernel_options,
    )  # fmt: skip
    if return_lse:
        out, lse = res
        return out, lse.t()
    return res


def flash_attn_rl_dense(
    q, k, v, dropout_p, softmax_scale, causal, window_size, bias, alibi_slopes, return_lse,
    return_attn_probs, sink, block_mask, kernel_options,
):  # fmt: skip
    """flash_attn_func on this kernel: q, k, v [B, S, H, D]; LSE is returned as [B, Hq, Sq]."""
    window_left, full = _flash_args(
        dropout_p, causal, window_size, bias, alibi_slopes, return_attn_probs, block_mask
    )
    if full:
        batch, seqlen_q, seqlen_k = q.shape[0], q.shape[1], k.shape[1]
        cu_q = torch.arange(0, (batch + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=q.device)
        cu_k = torch.arange(0, (batch + 1) * seqlen_k, seqlen_k, dtype=torch.int32, device=q.device)
        block_mask = _full_mask(cu_q, cu_k)
    res = rl_flash_attn_func(
        q, k, v, softmax_scale, window_left, block_mask, sink, return_lse, kernel_options
    )  # fmt: skip
    if return_lse:
        out, lse = res
        return out, lse.permute(0, 2, 1)
    return res
