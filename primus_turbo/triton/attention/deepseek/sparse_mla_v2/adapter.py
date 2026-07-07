###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pool -> fused single-latent (K==V) sparse-MLA bridge for the ``triton_v2`` CSA.

Ported from upstream Primus ``v4_sparse_mla_adapter.py`` (``_V4SparseMLACSAFn``).
Maps Primus-Turbo's separate-K/V CSA-from-pool representation

    ``(q[B,H,S,D], k_local[B,H,S,D], v_local[B,H,S,D], pool[B,P,D],
       topk_idxs[B,S,K], sink[H], swa_window, scale)``

onto the fused single-latent sparse-MLA kernel-pair contract

    ``fwd(q[T,H,D_qk], kv[num_kv,1,D_qk], topk[T,W+K], attn_sink, kv_lora_rank, scale)``
    ``bwd(q, kv, o, do, topk, lse, attn_sink, kv_lora_rank, scale) -> (dq, dkv, d_sink)``

and maps gradients back once. In V4 the local K/V is a single MQA latent
(K == V, RoPE baked in-place over the 512 latent), so ``k_local``/``v_local`` are
``.expand`` views of one latent; the kv buffer is ``[latent(S) ++ pool(P)]`` and
the flat topk is ``[SWA window ++ sparse pool]`` over that per-batch buffer.
"""

from __future__ import annotations

from typing import Optional

import torch

from .dsa_bwd import sparse_mla_bwd_v4_triton
from .dsa_fwd import sparse_mla_fwd_v4_triton

# The kernels require a nonzero rope block (D_ROPE > 0) even though V4 bakes RoPE
# in-place over the 512 latent (the rope term is provably zero -> skipped inside).
_ROPE_PAD = 64


def _pad_topk_64(topk: torch.Tensor) -> torch.Tensor:
    """Pad the topk width to a multiple of 64 with -1 (kept for parity with the
    gluon dKV tiling; harmless for the triton_v2 kernels)."""
    tk = topk.shape[1]
    pad = ((tk + 63) // 64) * 64 - tk
    if pad > 0:
        topk = torch.cat(
            [topk, torch.full((topk.shape[0], pad), -1, device=topk.device, dtype=topk.dtype)], dim=1
        )
    return topk.contiguous()


def _build_csa_topk(topk_idxs: torch.Tensor, S: int, P: int, W: int) -> torch.Tensor:
    """Flat topk ``[B*S, W+K]`` over the per-batch ``[local(S) ++ pool(P)]`` buffer.

    ``topk_idxs`` ``[B, S, K]`` holds pool indices in ``[0, P)`` (or -1). Batch
    ``b`` occupies rows ``[b*(S+P) : (b+1)*(S+P))`` (local 0..S-1, pool S..S+P-1).
    The local block is the sliding window ``[m-W+1 .. m]`` (causal, clamped).
    """
    B, _, K = topk_idxs.shape
    device = topk_idxs.device
    base = (torch.arange(B, device=device) * (S + P)).view(B, 1, 1)

    win_pos = torch.arange(S, device=device).view(S, 1) - W + 1 + torch.arange(W, device=device).view(1, W)
    win_valid = win_pos >= 0
    win_idx = base + win_pos.view(1, S, W)
    win_idx = torch.where(win_valid.view(1, S, W), win_idx, torch.full_like(win_idx, -1))

    pool_valid = topk_idxs >= 0
    pool_idx = torch.where(pool_valid, base + S + topk_idxs, torch.full_like(topk_idxs, -1))

    return torch.cat([win_idx, pool_idx], dim=2).reshape(B * S, W + K).to(torch.int32).contiguous()


class _CSAPoolSparseMLATritonV2Fn(torch.autograd.Function):
    """Autograd wrapper: triton_v2 sparse-MLA FWD/BWD for the V4 CSA (cr=4) layer."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q_bh: torch.Tensor,  # [B, H, S, D]
        k_local_bh: torch.Tensor,  # [B, H, S, D] (single MQA latent, head-broadcast)
        v_local_bh: torch.Tensor,  # [B, H, S, D] (== k_local in V4)
        pool: torch.Tensor,  # [B, P, D]
        topk_idxs: torch.Tensor,  # [B, S, K] pool indices, -1 = invalid
        sink: Optional[torch.Tensor],  # [H] or None
        swa_window: int,
        scale: float,
    ) -> torch.Tensor:
        B, H, S, D = q_bh.shape
        P = pool.shape[1]
        W = int(swa_window)
        assert q_bh.dtype == torch.bfloat16, "triton_v2 sparse-MLA requires bf16"
        assert W > 0, "triton_v2 sparse-MLA requires swa_window > 0"

        latent = k_local_bh[:, 0, :, :]  # [B, S, D]

        z_q = torch.zeros(B * S, H, _ROPE_PAD, device=q_bh.device, dtype=q_bh.dtype)
        q_g = torch.cat([q_bh.permute(0, 2, 1, 3).reshape(B * S, H, D), z_q], dim=-1).contiguous()

        kv512 = torch.cat([latent, pool], dim=1).reshape(B * (S + P), 1, D)
        z_kv = torch.zeros(B * (S + P), 1, _ROPE_PAD, device=q_bh.device, dtype=q_bh.dtype)
        kv_g = torch.cat([kv512, z_kv], dim=-1).contiguous()

        topk_g = _pad_topk_64(_build_csa_topk(topk_idxs, S, P, W))

        sink_arg = sink.float().contiguous() if sink is not None else None
        o_g, lse = sparse_mla_fwd_v4_triton(
            q_g, kv_g, topk_g, attn_sink=sink_arg, kv_lora_rank=D, scale=float(scale)
        )

        ctx.save_for_backward(q_g, kv_g, o_g, lse, topk_g, sink_arg if sink is not None else q_g.new_empty(0))
        ctx.shapes = (B, H, S, D, P, W)
        ctx.scale = float(scale)
        ctx.sink_was_none = sink is None
        return o_g.reshape(B, S, H, D).permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def backward(ctx, grad_o_bh: torch.Tensor):  # type: ignore[override]
        q_g, kv_g, o_g, lse, topk_g, sink_saved = ctx.saved_tensors
        B, H, S, D, P, W = ctx.shapes
        sink_arg = None if ctx.sink_was_none else sink_saved

        grad_o_g = grad_o_bh.permute(0, 2, 1, 3).reshape(B * S, H, D).contiguous()
        dq_g, dkv_g, dsink = sparse_mla_bwd_v4_triton(
            q_g, kv_g, o_g, grad_o_g, topk_g, lse, attn_sink=sink_arg, kv_lora_rank=D, scale=ctx.scale
        )

        dq_bh = dq_g[:, :, :D].reshape(B, S, H, D).permute(0, 2, 1, 3).contiguous()
        dkv512 = dkv_g[:, 0, :D].reshape(B, S + P, D)
        dlatent = dkv512[:, :S, :]
        dpool = dkv512[:, S:, :].contiguous()

        dk_local = torch.zeros(B, H, S, D, device=dq_bh.device, dtype=dq_bh.dtype)
        dk_local[:, 0, :, :] = dlatent.to(dq_bh.dtype)
        # V4 single-latent (K == V): the kernel returns one combined ``dkv`` routed
        # entirely through ``dk_local``; the V-branch gradient is structurally zero.
        dv_local = None

        dsink_out = None
        if not ctx.sink_was_none and dsink is not None:
            dsink_out = dsink.to(sink_saved.dtype)

        # forward args: (q, k_local, v_local, pool, topk_idxs, sink, swa_window, scale)
        return dq_bh, dk_local, dv_local, dpool.to(dq_bh.dtype), None, dsink_out, None, None


def csa_pool_attention_triton_v2(
    q: torch.Tensor,  # [B, H, S, D]
    k_local: torch.Tensor,  # [B, H, S, D] (or [B, 1, S, D] MQA)
    v_local: torch.Tensor,  # [B, H, S, D] (== k_local in V4)
    pool: torch.Tensor,  # [B, P, D]
    topk_idxs: torch.Tensor,  # [B, S, K]
    sink: Optional[torch.Tensor],
    swa_window: int,
    scale: float,
) -> torch.Tensor:
    """CSA-from-pool via the fused single-latent triton_v2 sparse-MLA kernels.

    ``k_local``/``v_local`` may be MQA ``[B, 1, S, D]`` or MHA ``[B, H, S, D]``;
    only the first head (the shared latent) is used (V4 single-latent K==V).
    Returns ``[B, H, S, D]`` in ``q.dtype``.
    """
    if k_local.shape[1] == 1:
        H = q.shape[1]
        k_local = k_local.expand(-1, H, -1, -1)
        v_local = v_local.expand(-1, H, -1, -1)
    return _CSAPoolSparseMLATritonV2Fn.apply(
        q, k_local, v_local, pool, topk_idxs, sink, int(swa_window), float(scale)
    )
