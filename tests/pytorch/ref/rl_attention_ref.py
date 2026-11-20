###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch


def rl_attention_ref(q, k, v, cu_q, cu_k, scale, lo=None, hi=None, sinks=None, window_left=-1):
    """fp32 THD attention. With lo / hi, query token t sees keys [lo[t], hi[t]] (global key
    indices, inclusive); otherwise causal, bottom-right aligned, narrowed by ``window_left`` >= 0.
    ``sinks`` [Hq] adds one logit per head to each softmax denominator. Rows with no key give 0."""
    out = torch.zeros(q.shape[0], q.shape[1], v.shape[-1], dtype=torch.float32, device=q.device)
    group = q.shape[1] // k.shape[1]
    for s in range(cu_q.numel() - 1):
        q0, q1 = int(cu_q[s]), int(cu_q[s + 1])
        k0, k1 = int(cu_k[s]), int(cu_k[s + 1])
        qs = q[q0:q1].float().transpose(0, 1)
        ks = k[k0:k1].float().repeat_interleave(group, dim=1).transpose(0, 1)
        vs = v[k0:k1].float().repeat_interleave(group, dim=1).transpose(0, 1)
        kpos = torch.arange(k0, k1, device=q.device)[None, :]
        if lo is not None:
            allowed = (kpos >= lo[q0:q1, None].long()) & (kpos <= hi[q0:q1, None].long())
        else:
            qpos = torch.arange(q1 - q0, device=q.device)[:, None] + k1 - (q1 - q0)
            allowed = kpos <= qpos
            if window_left >= 0:
                allowed &= qpos - kpos <= window_left
        scores = (qs @ ks.transpose(1, 2) * scale).masked_fill(~allowed, float("-inf"))
        if sinks is not None:
            sink_col = sinks.float().view(-1, 1, 1).expand(-1, scores.shape[1], 1)
            probs = torch.softmax(torch.cat([scores, sink_col], -1), dim=-1)[..., :-1]
        else:
            probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        out[q0:q1] = (probs @ vs).transpose(0, 1)
    return out
