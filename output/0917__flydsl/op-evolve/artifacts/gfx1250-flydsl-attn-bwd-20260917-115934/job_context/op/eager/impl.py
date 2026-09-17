"""The precision reference: plain torch, fp32, no library attention call anywhere.

This is the ONLY thing anything in this job is aligned against. It is never checked
against a library implementation, and never against the code under optimization.

Written from op.reference.logic, not transcribed from a fused kernel:

    s     = q @ k.T * scale                 (fp32)
    mask  = causal, BOTTOM-RIGHT
    p     = exp(s - lse)                    lse comes from the forward, natural log
    delta = rowsum(dO * O)
    ds    = p * (dO @ v.T - delta) * scale
    dq    = ds @ k ;  dk = ds.T @ q ;  dv = p.T @ dO

`torch.nn.functional.scaled_dot_product_attention(is_causal=True)` is NOT used and would
be wrong here even as a cross-check: its causal mask is TOP-LEFT aligned, which differs
from op.config.causal = bottom-right whenever seqlen_q != seqlen_kv. The mask below is
built explicitly for that reason.

Chunked over (batch, q head) and query blocks only to fit in memory -- a dense
[4, 32, 8192, 8192] fp32 score tensor is 34 GB. Chunking changes nothing but the
summation order of dk/dv, which is done in fp32.
"""
import math

import torch

Q_CHUNK = 1024


def eager_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True):
    """Same signature as op.reference.api. Returns (dq, dk, dv) in fp32."""
    b, sq, hq, d = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    assert hq % hkv == 0
    g = hq // hkv
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    scale = float(softmax_scale)
    shift = skv - sq          # bottom-right: query i attends keys j <= i + shift

    dq = torch.zeros((b, sq, hq, d), device=q.device, dtype=torch.float32)
    dk = torch.zeros((b, skv, hkv, d), device=k.device, dtype=torch.float32)
    dv = torch.zeros((b, skv, hkv, d), device=v.device, dtype=torch.float32)
    lse = lse.float()

    for bi in range(b):
        for h in range(hq):
            hk = h // g
            kb = k[bi, :, hk, :].float()          # [skv, d]
            vb = v[bi, :, hk, :].float()
            for q0 in range(0, sq, Q_CHUNK):
                q1 = min(q0 + Q_CHUNK, sq)
                qb = q[bi, q0:q1, h, :].float()   # [n, d]
                ob = o[bi, q0:q1, h, :].float()
                dob = do[bi, q0:q1, h, :].float()

                s = (qb @ kb.T) * scale           # [n, skv]
                if causal:
                    qi = torch.arange(q0, q1, device=s.device).unsqueeze(1)
                    kj = torch.arange(skv, device=s.device).unsqueeze(0)
                    s = s.masked_fill(kj > qi + shift, float("-inf"))
                p = torch.exp(s - lse[bi, h, q0:q1].unsqueeze(1))
                delta = (dob * ob).sum(-1, keepdim=True)          # [n, 1]
                ds = p * ((dob @ vb.T) - delta) * scale           # [n, skv]

                dq[bi, q0:q1, h, :] = ds @ kb
                dk[bi, :, hk, :] += ds.T @ qb
                dv[bi, :, hk, :] += p.T @ dob
    return dq, dk, dv


attn_bwd = eager_attn_bwd   # uniform name every loader uses
