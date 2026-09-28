"""The precision reference: plain torch, fp32, no library attention call anywhere.

This is the ONLY thing anything in this job is aligned against. It is never checked
against a library implementation, and never against the code under optimization.

`forward_reference` is copied VERBATIM (body unchanged) from the backward job's
artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op/ut/common.py:53-82,
as op.reference.impl directs.

    s = q @ k.T * scale (fp32); bottom-right causal mask; lse = logsumexp(s, -1)  (natural log)
    p = exp(s - lse);  o = p @ v  (cast to bf16)

The causal mask is built EXPLICITLY: query i attends keys j <= i + (skv - sq).
`scaled_dot_product_attention(is_causal=True)` is NOT used, even as a cross-check: its mask
is TOP-LEFT aligned, which differs from bottom-right whenever seqlen_q != seqlen_kv.

Chunked over (batch, q head) and query blocks only to fit in memory -- a dense
[4, 32, 8192, 8192] fp32 score tensor is 34 GB.
"""
import os

# ASSIGNED, not setdefault (sitecustomize already setdefault'ed it to "0"). With "0" the fp32
# reference GEMMs took the Tensile path that faulted this card; see the spec's runtime.env.
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
os.environ["HIPBLASLT_TENSILE_LIBPATH"] = "/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250"

import math  # noqa: E402

import torch  # noqa: E402


def forward_reference(q, k, v, causal=True, softmax_scale=None, q_chunk=1024):
    """fp32 forward producing o (bf16) and lse (fp32, NATURAL log), bottom-right causal.

    Written here rather than taken from a library: the backward under test consumes o and
    lse, so a library forward would put a library in the reference chain.
    """
    b, sq, hq, d = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    g = hq // hkv
    scale = float(softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d))
    shift = skv - sq
    o = torch.empty((b, sq, hq, d), device=q.device, dtype=torch.bfloat16)
    lse = torch.empty((b, hq, sq), device=q.device, dtype=torch.float32)
    for bi in range(b):
        for h in range(hq):
            hk = h // g
            kb = k[bi, :, hk, :].float()
            vb = v[bi, :, hk, :].float()
            for q0 in range(0, sq, q_chunk):
                q1 = min(q0 + q_chunk, sq)
                s = (q[bi, q0:q1, h, :].float() @ kb.T) * scale
                if causal:
                    qi = torch.arange(q0, q1, device=s.device).unsqueeze(1)
                    kj = torch.arange(skv, device=s.device).unsqueeze(0)
                    s = s.masked_fill(kj > qi + shift, float("-inf"))
                l = torch.logsumexp(s, dim=-1)
                p = torch.exp(s - l.unsqueeze(1))
                o[bi, q0:q1, h, :] = (p @ vb).to(torch.bfloat16)
                lse[bi, h, q0:q1] = l
    return o, lse


def eager_attn_fwd(q, k, v, softmax_scale=None, causal=True):
    """op.reference.api. Returns (o bf16 [B,Sq,Hq,D], lse fp32 [B,Hq,Sq])."""
    assert q.shape[2] % k.shape[2] == 0
    return forward_reference(q, k, v, causal=causal, softmax_scale=softmax_scale)


attn_fwd = eager_attn_fwd   # uniform name every loader uses
