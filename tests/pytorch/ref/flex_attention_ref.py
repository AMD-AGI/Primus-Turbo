###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Dense fp32 eager references for flex attention (differentiable).

``score_fn(score, b, h, q_idx, kv_idx)`` and ``keep_fn(b, h, q_idx, kv_idx)`` are the
PyTorch twins of the ``@triton.jit`` score_mod / mask_mod under test, evaluated on
broadcast index tensors.
"""

import torch


def flex_attention_ref(q, k, v, score_fn=None, keep_fn=None, scale=None):
    """q ``[B, Hq, Sq, D]``, k/v ``[B, Hkv, Skv, D]`` -> (out ``[B, Hq, Sq, Dv]``, lse ``[B, Hq, Sq]``)."""
    B, Hq, Sq, D = q.shape
    Skv = k.shape[2]
    group = Hq // k.shape[1]
    scale = D**-0.5 if scale is None else scale
    kk = k.float().repeat_interleave(group, dim=1)
    vv = v.float().repeat_interleave(group, dim=1)
    s = (q.float() @ kk.transpose(-1, -2)) * scale
    dev = q.device
    b = torch.arange(B, device=dev).view(B, 1, 1, 1)
    h = torch.arange(Hq, device=dev).view(1, Hq, 1, 1)
    q_idx = torch.arange(Sq, device=dev).view(1, 1, Sq, 1)
    kv_idx = torch.arange(Skv, device=dev).view(1, 1, 1, Skv)
    if score_fn is not None:
        s = score_fn(s, b, h, q_idx, kv_idx)
    if keep_fn is not None:
        s = s.masked_fill(~keep_fn(b, h, q_idx, kv_idx), float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    p = torch.softmax(s, dim=-1).nan_to_num(0.0)  # fully masked rows attend nothing
    return p @ vv, lse


def flex_attention_varlen_ref(q, k, v, cu_seqlens_q, cu_seqlens_k, score_fn=None, keep_fn=None, scale=None):
    """Packed ``[T, H, D]`` inputs: one dense reference per sequence, sequence-local
    indices, and ``b`` = the sequence index. Returns (out ``[T, Hq, Dv]``, lse ``[Hq, T]``)."""
    outs, lses = [], []
    bounds = zip(
        cu_seqlens_q.tolist(), cu_seqlens_q[1:].tolist(), cu_seqlens_k.tolist(), cu_seqlens_k[1:].tolist()
    )
    for seq, (q0, q1, k0, k1) in enumerate(bounds):

        def seq_score(s, b, h, qi, ki, seq=seq):
            return score_fn(s, torch.full_like(b, seq), h, qi, ki)

        def seq_keep(b, h, qi, ki, seq=seq):
            return keep_fn(torch.full_like(b, seq), h, qi, ki)

        out, lse = flex_attention_ref(
            *(t.transpose(0, 1).unsqueeze(0) for t in (q[q0:q1], k[k0:k1], v[k0:k1])),
            score_fn=None if score_fn is None else seq_score,
            keep_fn=None if keep_fn is None else seq_keep,
            scale=scale,
        )
        outs.append(out[0].transpose(0, 1))
        lses.append(lse[0])
    return torch.cat(outs, dim=0), torch.cat(lses, dim=1)


def rel_bias_ref(q, k, v, rel_logits, cu_seqlens, scale=None, window_left=None):
    """Causal varlen attention plus ``rel_logits[t, h, q_idx - kv_idx]`` for
    ``0 <= q_idx - kv_idx < RE`` (``rel_logits`` ``[T, Hq, RE]``), keeping keys with
    ``q_idx - kv_idx <= window_left`` when a window is given."""
    RE = rel_logits.shape[-1]
    starts = cu_seqlens[:-1]

    def score_fn(s, b, h, qi, ki):
        row, hh, rd = torch.broadcast_tensors(starts[b] + qi, h, qi - ki)
        bias = rel_logits[row, hh, rd.clamp(0, RE - 1)]
        return torch.where((rd >= 0) & (rd < RE), s + bias.float(), s)

    def keep_fn(b, h, qi, ki):
        keep = qi >= ki
        return keep if window_left is None else keep & (qi - ki <= window_left)

    return flex_attention_varlen_ref(q, k, v, cu_seqlens, cu_seqlens, score_fn, keep_fn, scale)[0]
