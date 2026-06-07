###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL flash-attention BACKWARD for gfx1250 (RDNA4), D_qk=192 / D_v=128 bf16.

There is no upstream FlyDSL attention backward to vendor (the gfx1250 FMHA
reference is forward-only), so this is authored net-new.

Math (standard flash-attention-2 backward, recompute from Q/K/V + LSE):

    S   = scale * Q @ K^T                  [Sq, Sk]   (causal mask -> -inf)
    P   = exp(S - LSE[:, None])            [Sq, Sk]
    D   = rowsum(O * dO)                   [Sq]
    dV  = P^T @ dO                         [Sk, D_v]
    dP  = dO @ V^T                         [Sq, Sk]
    dS  = P * (dP - D[:, None])            [Sq, Sk]
    dQ  = scale * dS @ K                   [Sq, D_qk]
    dK  = scale * dS^T @ Q                 [Sk, D_qk]

The five matmuls (the compute-dominant part) run on the gfx1250 bf16 WMMA
GEMM (``wmma_gemm_bf16.gemm_nt_bf16``); every product is cast to the NT form
``A @ B^T`` via transposed contiguous operands. The pointwise / reduction glue
(mask, exp, row-dot, the dS combine) currently runs in torch -- a
correctness-first v1; a fully fused FlyDSL backward kernel is a later perf
step. All gradients are returned in fp32-accumulated bf16, matching the
forward's THD varlen layout.
"""

from __future__ import annotations

import torch

from .wmma_gemm_bf16 import gemm_nt_bf16

HEAD_DIM_QK = 192
HEAD_DIM_V = 128


def _bwd_one_head(q, k, v, o, do, lse, scale, causal):
    """Backward for a single (sequence, head). All tensors 2-D, q/k [S*, 192],
    v/o/do [S*, 128], lse [Sq]. Returns dq [Sq,192], dk [Sk,192], dv [Sk,128]."""
    Sq = q.shape[0]
    Sk = k.shape[0]

    # S = scale * Q @ K^T  ->  NT(Q[Sq,192], K[Sk,192])
    s = gemm_nt_bf16(q, k) * scale  # [Sq, Sk] f32
    if causal:
        qi = torch.arange(Sq, device=q.device).view(-1, 1)
        ki = torch.arange(Sk, device=q.device).view(1, -1)
        s = s.masked_fill(ki > (qi + (Sk - Sq)), float("-inf"))

    # P = exp(S - LSE)
    p = torch.exp(s - lse.view(-1, 1))  # [Sq, Sk] f32
    p_bf = p.to(torch.bfloat16)

    # D = rowsum(O * dO)
    d = (o.float() * do.float()).sum(dim=-1)  # [Sq]

    # dV = P^T @ dO  ->  NT(P^T[Sk,Sq], dO^T[128,Sq])
    dv = gemm_nt_bf16(p_bf.t().contiguous(), do.t().contiguous())  # [Sk, 128]

    # dP = dO @ V^T  ->  NT(dO[Sq,128], V[Sk,128])
    dp = gemm_nt_bf16(do, v)  # [Sq, Sk] f32

    # dS = P * (dP - D)
    ds = p * (dp - d.view(-1, 1))  # [Sq, Sk] f32
    ds_bf = ds.to(torch.bfloat16)

    # dQ = scale * dS @ K  ->  NT(dS[Sq,Sk], K^T[192,Sk])
    dq = gemm_nt_bf16(ds_bf, k.t().contiguous()) * scale  # [Sq, 192]

    # dK = scale * dS^T @ Q  ->  NT(dS^T[Sk,Sq], Q^T[192,Sq])
    dk = gemm_nt_bf16(ds_bf.t().contiguous(), q.t().contiguous()) * scale  # [Sk, 192]

    return dq, dk, dv


def flash_attn_varlen_bwd_d192_gfx1250(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale=None,
    causal=False,
):
    """Varlen THD backward. q/k: [total, H, 192], v/o/do: [total, Hv?, 128].

    lse: [total_q, H] (forward's return_lse layout). Returns (dq, dk, dv) in
    the same THD layout / dtype as q, k, v.
    """
    assert q.shape[-1] == HEAD_DIM_QK and v.shape[-1] == HEAD_DIM_V
    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_DIM_QK**0.5)

    H = q.shape[1]
    Hk = k.shape[1]
    gqa = H // Hk
    B = cu_seqlens_q.numel() - 1

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    for b in range(B):
        qs, qe = cu_q[b], cu_q[b + 1]
        ks, ke = cu_k[b], cu_k[b + 1]
        for h in range(H):
            hk = h // gqa
            dq_h, dk_h, dv_h = _bwd_one_head(
                q[qs:qe, h, :],
                k[ks:ke, hk, :],
                v[ks:ke, hk, :],
                o[qs:qe, h, :],
                do[qs:qe, h, :],
                lse[qs:qe, h],
                softmax_scale,
                causal,
            )
            dq[qs:qe, h, :] = dq_h.to(dq.dtype)
            # GQA: several q-heads map to one kv-head -> accumulate.
            dk[ks:ke, hk, :] += dk_h.to(dk.dtype)
            dv[ks:ke, hk, :] += dv_h.to(dv.dtype)

    return dq, dk, dv
