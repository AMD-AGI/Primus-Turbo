###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Host-side adapter for the vendored one-kernel fused MHA backward.

The Triton kernels live in ``primus_turbo.triton.attention.fused_mha_bwd_kernel``
and were vendored from AITER; see that file's header and
``primus_turbo/triton/attention/FUSED_MHA_PROVENANCE.md``.

Two layers live here:

``flash_attn_onekernel_backward``
    The launch wrapper, adapted from
    ``aiter/ops/triton/attention/mha_onekernel_bwd.py``. Same signature and
    same semantics as upstream, so it stays diffable against AITER.

``dense_fused_backward``
    A thin Primus-Turbo-shaped entry point with the same arguments as
    ``attention_triton_impl.dense_backward`` takes, returning ``(dq, dk, dv)``.

Nothing here is wired into the dispatcher; ``attention_impl.py`` is untouched.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import triton

from primus_turbo.triton.attention.attention_kernel import FIXED_BLOCK_M
from primus_turbo.triton.attention.fused_mha_bwd_kernel import (
    _bwd_preprocess,
    bwd_kernel_causal,
    bwd_kernel_noncausal,
    get_fused_bwd_config,
)

# CHANGED vs upstream: aiter's wrapper calls `_is_fp8(q)` imported from
# aiter.ops.triton.utils.types, which additionally asks arch_info whether the
# device supports fp8 -- an import-time device probe we deliberately do not
# vendor. The predicate itself is just this dtype set.
_FP8_DTYPES = frozenset(
    dtype
    for name in ("float8_e4m3fnuz", "float8_e4m3fn", "float8_e5m2", "float8_e5m2fnuz")
    for dtype in (getattr(torch, name, None),)
    if dtype is not None
)


def _is_fp8(x: torch.Tensor) -> bool:
    return x.dtype in _FP8_DTYPES


def flash_attn_onekernel_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dbias: Optional[torch.Tensor],
    sm_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    causal: bool,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    philox_seed: Optional[int] = 0,
    philox_offset: Optional[int] = 0,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    descale_v: Optional[torch.Tensor] = None,
    descale_do: Optional[torch.Tensor] = None,
    USE_INT64_STRIDES: Optional[bool] = False,
    sink: Optional[torch.Tensor] = None,
    dsink: Optional[torch.Tensor] = None,
    config: Optional[Dict[str, Any]] = None,
    sliding_window: int = 0,
) -> torch.Tensor:
    """Flash Attention one-kernel backward. Computes dQ, dK, dV without atomics.

    Adapted from aiter/ops/triton/attention/mha_onekernel_bwd.py. Deviations
    from upstream, all marked inline below:
      - the AiterTritonLogger f-string call is dropped (it was built on every
        call whether or not logging was enabled);
      - `_get_config()` becomes `get_fused_bwd_config()`, a hard-coded table;
      - `_is_fp8` is the local dtype predicate above.

    Args:
        do: output gradient, (batch, seqlen_q, num_q_heads, v_head_dim), or
            (total_tokens, num_q_heads, v_head_dim) for varlen.
        q: (batch, seqlen_q, num_q_heads, qk_head_dim). Layout is **bshd**, not
            bhsd. qk_head_dim may exceed v_head_dim (positional encoding).
        k: (batch, seqlen_k, num_k_heads, qk_head_dim).
        v: (batch, seqlen_k, num_k_heads, v_head_dim).
        o: forward output, same shape as `do`.
        softmax_lse: **natural-log** log-sum-exp from the forward, shape
            (batch, num_q_heads, seqlen_q), or (total_tokens, num_q_heads) for
            varlen. The kernel applies 1/ln2 itself (USE_EXP2=True below).
        dq, dk, dv: pre-allocated gradients, same shapes as q, k, v. They are
            written in place; this function returns only `delta`.
        dbias: not supported, must be None.
        sm_scale: softmax scale, typically 1/sqrt(head_dim).
        alibi_slopes: (num_q_heads,) or None.
        causal: apply causal masking.
        cu_seqlens_q / cu_seqlens_k: (batch + 1,) to select varlen mode.
        max_seqlen_q / max_seqlen_k: sequence length bounds.
        dropout_p: 0.0 disables dropout.
        sink / dsink: attention-sink logits and their gradient, (num_q_heads,).
        config: override for get_fused_bwd_config().
        sliding_window: left window size in tokens; 0 disables.

    Returns:
        delta, the rowsum(dO * O) tensor, shaped like softmax_lse.
    """
    # CHANGED vs upstream: the `_LOGGER.info(f"...")` call was removed here.
    if dbias is not None:
        raise ValueError("Bias is not supported yet in the Triton Backend")

    use_alibi, (stride_az, stride_ah) = (
        (True, alibi_slopes.stride()) if alibi_slopes is not None else (False, (0, 0))
    )

    # CHANGED vs upstream: local `_is_fp8`, no arch_info probe.
    IS_FP8 = _is_fp8(q)
    if IS_FP8:
        FP8_MAX = torch.finfo(q.dtype).max
        descale_strides = (
            descale_q.stride(0),
            descale_k.stride(0),
            descale_v.stride(0),
            descale_do.stride(0),
        )
    else:
        FP8_MAX = None
        descale_strides = (None, None, None, None)

    IS_VARLEN = cu_seqlens_q is not None

    # get strides and shape
    if IS_VARLEN:
        # Layout is thd. q/k are [total_tokens, num_head, head_dim_qk],
        # v is [total_tokens, num_head, head_dim_v].
        batch, _seqlen_q, num_q_heads = (
            len(cu_seqlens_q) - 1,
            max_seqlen_q,
            q.shape[1],
        )
        _, num_k_heads = max_seqlen_k, k.shape[1]
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
        dq_strides = (0, dq.stride(1), dq.stride(0), dq.stride(2))
        dk_strides = (0, dk.stride(1), dk.stride(0), dk.stride(2))
        dv_strides = (0, dv.stride(1), dv.stride(0), dv.stride(2))
        do_strides = (0, do.stride(1), do.stride(0), do.stride(2))
    else:
        # Layout is bshd. q/k are [batch, seq_len, num_head, head_dim_qk],
        # v is [batch, seq_len, num_head, head_dim_v]. The kernel wants
        # (b, h, s, d) order, hence the reshuffle.
        batch, _seqlen_q, num_q_heads = q.shape[:-1]
        _, num_k_heads = k.shape[1], k.shape[2]
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
        dq_strides = (dq.stride(0), dq.stride(2), dq.stride(1), dq.stride(3))
        dk_strides = (dk.stride(0), dk.stride(2), dk.stride(1), dk.stride(3))
        dv_strides = (dv.stride(0), dv.stride(2), dv.stride(1), dv.stride(3))
        do_strides = (do.stride(0), do.stride(2), do.stride(1), do.stride(3))

    qk_head_dim = q.shape[-1]
    v_head_dim = v.shape[-1]
    pe_head_dim = qk_head_dim - v_head_dim
    # padding for head_dim: power of 2, at least 16
    BLOCK_D_MODEL_POW2 = max(triton.next_power_of_2(v_head_dim), 16)
    BLOCK_D_MODEL_PE_POW2 = 0 if pe_head_dim == 0 else max(triton.next_power_of_2(pe_head_dim), 16)
    assert (pe_head_dim == 0 and BLOCK_D_MODEL_PE_POW2 == 0) or (
        v_head_dim == BLOCK_D_MODEL_POW2 and pe_head_dim == BLOCK_D_MODEL_PE_POW2
    ), "Positional encoding support requires NOPE and PE head sizes to be unpadded powers of 2."
    assert (not IS_FP8) or (IS_FP8 and pe_head_dim == 0), "Positional encoding doesn't support FP8."

    assert (sink is None) or (
        sink is not None and sink.dim() == 1 and sink.shape[0] == num_q_heads
    ), "Sink must be 1D and have one element per query head."
    assert (dsink is None) or (
        dsink is not None and dsink.dim() == 1 and dsink.shape[0] == num_q_heads
    ), "Sink gradient must be 1D and have one element per query head."
    assert (sink is None) == (dsink is None), "Sink and its gradient must be both present or absent."

    # CHANGED vs upstream: `_get_config()` (JSON discovery) -> hard-coded table.
    if config is None:
        config = get_fused_bwd_config()

    # init delta
    delta = torch.zeros_like(softmax_lse)
    if IS_VARLEN:
        # [total_tokens, num_q_heads]
        delta_strides = (0, delta.stride(1), delta.stride(0))
    else:
        # [batch, num_q_heads, seqlen_q]
        delta_strides = delta.stride()

    # preprocess: delta = rowsum(dO * O), element-wise product.
    pre_grid = (
        triton.cdiv(max_seqlen_q, config["preprocess_kernel"]["PRE_BLOCK"]),
        batch,
        num_q_heads,
    )
    _bwd_preprocess[pre_grid](
        o,
        do,
        delta,
        *o_strides,
        *do_strides,
        *delta_strides,
        descale_strides[3],
        cu_seqlens_q,
        max_seqlen_q,
        descale_do,
        BLOCK_M=config["preprocess_kernel"]["PRE_BLOCK"],
        BLOCK_D_MODEL=v_head_dim,
        BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
        IS_VARLEN=IS_VARLEN,
        IS_FP8=IS_FP8,
    )

    use_dropout = dropout_p > 0.0
    if use_dropout:
        dropout_mask = torch.zeros(
            (batch, num_q_heads, max_seqlen_q, max_seqlen_k),
            device=q.device,
            dtype=torch.float32,
        )
        dropout_strides = dropout_mask.stride()
    else:
        dropout_mask = None
        dropout_strides = (0, 0, 0, 0)

    seqlen = max(max_seqlen_q, max_seqlen_k)

    # "onekernel_pe" is the positional-encoding causal variant, used when present.
    config_onekernel = (
        config["onekernel_pe"] if (pe_head_dim > 0 and causal and "onekernel_pe" in config) else config["onekernel"]
    )
    grid = (
        num_k_heads,
        triton.cdiv(seqlen, config_onekernel["BLOCK_N1"]),
        batch,
    )

    kernel = bwd_kernel_causal if causal else bwd_kernel_noncausal
    kernel[grid](
        q,
        k,
        v,
        sink,
        sm_scale,
        do,
        dq,
        dk,
        dv,
        dsink,
        softmax_lse,
        delta,
        *q_strides,
        *k_strides,
        *v_strides,
        *dq_strides,
        *dk_strides,
        *dv_strides,
        *delta_strides,
        *do_strides,
        *dropout_strides,
        *descale_strides,
        stride_az,
        stride_ah,
        num_q_heads,
        num_k_heads,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_mask,
        dropout_p,
        philox_seed,
        philox_offset,
        alibi_slopes,
        descale_q,
        descale_k,
        descale_v,
        descale_do,
        HEAD_DIM=BLOCK_D_MODEL_POW2,
        ACTUAL_HEAD_DIM=v_head_dim,
        PE_HEAD_DIM=pe_head_dim,
        ENABLE_DROPOUT=use_dropout,
        IS_VARLEN=IS_VARLEN,
        USE_ALIBI=use_alibi,
        USE_EXP2=True,
        IS_FP8=IS_FP8,
        FP8_MAX=FP8_MAX,
        DEBUG_TRITON=False,
        DEBUG_TRITON_DETAIL=False,
        USE_INT64_STRIDES=USE_INT64_STRIDES,
        ENABLE_SINK=sink is not None,
        SLIDING_WINDOW=sliding_window,
        **config_onekernel,
    )

    return delta


# ---------------------------------------------------------------------------
# Primus-Turbo LSE/delta scratch <-> aiter softmax_lse
# ---------------------------------------------------------------------------
# THIS IS THE PART MOST LIKELY TO BE SILENTLY WRONG, so it is spelled out.
#
# Primus-Turbo's own forward (primus_turbo/triton/attention/attention_kernel.py,
# attn_fwd) does NOT write a plain [B, Hq, Sq] LSE. It allocates a [B, Hq, 2*Sq]
# scratch and writes LSE for block m at element offset `m * BLOCK_M * 2`; the
# in-tree backward preprocess later writes delta at `+ BLOCK_M` of that same
# block. LSE and delta therefore interleave every FIXED_BLOCK_M (=64) rows, not
# every row. attention_triton_impl._lse_delta_views reconstructs that on the
# host; the same arithmetic is repeated here rather than imported, because
# that helper is private to a module this file must not depend on.
#
# The vendored aiter kernel expects the flat convention instead: softmax_lse is
# [B, Hq, Sq] row-major, delta is a separate tensor of the same shape, and both
# are addressed with plain strides. So we gather in and scatter out.
#
# Units: both agree on NATURAL log. attention_kernel.py:1093-1099 computes
# `m_i/ln2 + log2(l_i)` then multiplies by ln2, and the aiter kernel multiplies
# the loaded value by 1/ln2 itself (USE_EXP2=True). No conversion is needed.
# If turbo's forward is ever switched to the USE_EXP2=False branch it still
# writes natural log (`m_i + log(l_i)`), so that branch is safe too.


def _packed_lse_index(seqlen_q: int, device: torch.device) -> torch.Tensor:
    """Row -> element offset of that row's LSE inside the [B, Hq, 2*Sq] scratch.

    Indexing rather than reshaping: the scratch is allocated as `2 * seqlen_q`,
    which need not be a whole number of `2 * FIXED_BLOCK_M` blocks, so a view
    would be wrong on a ragged tail.
    """
    row = torch.arange(seqlen_q, device=device)
    return (row // FIXED_BLOCK_M) * (2 * FIXED_BLOCK_M) + (row % FIXED_BLOCK_M)


def dense_fused_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale: Optional[float],
    causal: bool,
    window_size: Tuple[int, int] = (-1, -1),
    write_back_delta: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense bshd backward on the vendored fused kernel. Returns (dq, dk, dv).

    ``write_back_delta`` scatters the computed delta into the delta half of a packed
    ``softmax_lse``. It defaults to **False** because that write MUTATES a tensor autograd
    has saved: `FlashAttnFunc` stashes the scratch in ``ctx.saved_tensors``, and an in-place
    write bumps its version counter, so the next backward raises "one of the variables
    needed for gradient computation has been modified by an inplace operation". The in-tree
    ``dense_backward`` performs the same mutation without tripping this only because it goes
    through a ``torch.library.custom_op`` declaring ``mutates_args=()``, which hides the
    write from the version counter. Turn this on only for a caller that needs delta for a
    sink gradient AND owns the scratch.

    Argument-compatible with ``attention_triton_impl.dense_backward`` minus its
    ``sink`` parameter (the fused kernel supports sinks, but plumbing dsink
    through is a separate change and is deliberately not done here).

    ``softmax_lse`` is accepted in either shape:
      * ``[B, Hq, 2*Sq]`` -- Primus-Turbo's packed LSE/delta scratch. The LSE
        rows are gathered out for the kernel and the computed delta is
        scattered back into the delta half, so a subsequent
        ``dense_sink_grad(softmax_lse, ...)`` still sees what it expects.
      * ``[B, Hq, Sq]`` -- a plain natural-log LSE. Nothing is written back.
    Any other trailing dimension raises rather than guessing.
    """
    if q.dim() != 4:
        raise ValueError(f"dense_fused_backward expects bshd 4-D q, got shape {tuple(q.shape)}")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    batch, seqlen_q, nheads_q, _ = q.shape
    seqlen_k = k.shape[1]

    if softmax_lse.dim() != 3 or softmax_lse.shape[:2] != (batch, nheads_q):
        raise ValueError(
            f"softmax_lse must be [B, Hq, Sq] or [B, Hq, 2*Sq] with B={batch} Hq={nheads_q}, "
            f"got {tuple(softmax_lse.shape)}"
        )
    packed = softmax_lse.shape[2] == 2 * seqlen_q
    if not packed and softmax_lse.shape[2] != seqlen_q:
        raise ValueError(
            f"softmax_lse trailing dim must be {seqlen_q} (plain) or {2 * seqlen_q} (packed "
            f"Primus-Turbo scratch), got {softmax_lse.shape[2]}"
        )

    if packed:
        lse_idx = _packed_lse_index(seqlen_q, softmax_lse.device)
        # .contiguous(): the gather already materialises a new tensor, but be
        # explicit -- the kernel indexes it with plain strides.
        lse = softmax_lse[:, :, lse_idx].contiguous().float()
    else:
        lse = softmax_lse.contiguous().float()

    # window_size is (left, right); the fused kernel only has a left window.
    left, right = window_size
    if right not in (-1, 0):
        raise ValueError(f"fused backward supports no right window, got window_size={window_size}")
    if left >= 0 and not causal:
        raise ValueError("fused backward's sliding window is only defined together with causal=True")
    sliding_window = left if left > 0 else 0

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    delta = flash_attn_onekernel_backward(
        do,
        q,
        k,
        v,
        o,
        lse,
        dq,
        dk,
        dv,
        None,  # dbias -- unsupported upstream, must be None
        softmax_scale,
        None,  # alibi_slopes
        causal,
        None,  # cu_seqlens_q -- dense path
        None,  # cu_seqlens_k
        seqlen_q,
        seqlen_k,
        0.0,  # dropout_p
        sink=None,
        dsink=None,
        sliding_window=sliding_window,
    )

    if packed and write_back_delta:
        # Put delta where the rest of Primus-Turbo expects to find it, so
        # dense_sink_grad() and anything else reading the scratch keep working.
        softmax_lse[:, :, lse_idx + FIXED_BLOCK_M] = delta.to(softmax_lse.dtype)

    return dq, dk, dv


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------
# The fused backward is faster than the in-tree two-kernel one by a wide margin at the
# sequence lengths training actually uses, and SLOWER at very short ones. Measured on
# gfx1250, vendored path, forward num_stages=2, total fwd+bwd ms (lower is better):
#
#   shape              N1=32    N1=64   N1=128   N1=256
#   b1 s1024 h8         1.000    1.013    1.197    1.787      <- short: small tile wins
#   b4 s4096           24.407   12.983    8.888    7.876
#   b2 s8192           47.338   25.302   16.310   12.817
#   b4 s8192           93.449   49.472   31.642   24.282      <- long: N1=256 wins by 3.8x
#
# The crossover is monotone in sequence length and the shipped tile is N1=256, so the gate
# is a sequence-length threshold. At s=1024 that tile leaves only four K blocks and most of
# it is wasted, which is why it loses there.
_MIN_SEQLEN_FOR_FUSED = 2048


def fused_backward_eligible(
    q: torch.Tensor,
    seqlen_k: int,
    sink: Optional[torch.Tensor] = None,
) -> bool:
    """Whether the vendored fused backward should be preferred over the in-tree one.

    Narrow by construction: anything this declines falls back to the path that was already
    shipping, so a wrong 'no' costs performance while a wrong 'yes' could cost correctness.
    """
    # No sink support: the fused kernel has the machinery, but dsink is not plumbed through
    # this adapter, and returning None for a gradient the caller asked for is worse than
    # being slower.
    if sink is not None:
        return False
    if q.dtype not in (torch.bfloat16, torch.float16):
        return False
    if q.dim() != 4:
        return False
    # Below this the shipped tile is a pessimisation -- see the table above.
    return seqlen_k >= _MIN_SEQLEN_FOR_FUSED
