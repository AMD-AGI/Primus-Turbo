###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Ulysses for MLA through the latents: all-gather the compressed q / kv rows, up-project locally.

Per-head Ulysses exchanges q / k / v after MLA's up-projections, which expand a 1536 / 576 wide
latent into 128 heads x 192 / 256 columns at little FLOP cost: the exchange outweighs the GEMM it
could hide behind. Here every rank gathers the latents of the whole sequence instead and runs the
up-projections only for its own heads (same FLOPs as the per-head path):

    forward   all-gather [q_latent | kv_latent]  ->  q = Lq @ Wq_b[r]^T,  [k_nope | v] = Lc @ Wkv_b[r]^T
    backward  [dLq | dLc | dk_pe] = per-rank partial sums  ->  reduce-scatter

Sequence order matches ``flash_attn_usp_func``: rank r holds the r-th contiguous chunk. The weight
gradients only fill the rank's head rows (zero elsewhere); the data/context-parallel gradient
all-reduce sums them into the full gradient.
"""

from typing import Optional, Tuple

import torch
import torch.distributed as dist

__all__ = ["ulysses_mla_latent_up_proj"]

_QKV_FORMATS = ("sbhd", "bshd")


def _gather_rows(local, group, qkv_format):
    """``[s/n, b, C]`` / ``[b, s/n, C]`` of every rank -> the whole sequence as ``[s * b, C]`` rows in
    the output layout (``(s, b)`` for sbhd, ``(b, s)`` for bshd)."""
    n = group.size()
    rows = torch.empty(
        n * local.numel() // local.shape[-1], local.shape[-1], dtype=local.dtype, device=local.device
    )
    dist.all_gather_into_tensor(rows, local.contiguous().view(-1, local.shape[-1]), group=group)
    if qkv_format == "bshd":
        b, s_local, c = local.shape
        rows = rows.view(n, b, s_local, c).transpose(0, 1).reshape(-1, c)
    return rows


def _scatter_rows(full, group, qkv_format, local_shape):
    """Sum ``[s * b, C]`` partials over the ranks and keep this rank's chunk, shaped ``local_shape``."""
    n, c = group.size(), local_shape[-1]
    if qkv_format == "bshd":
        b, s_local = local_shape[:2]
        full = full.view(b, n, s_local, c).transpose(0, 1)
    full = full.contiguous().view(-1, c)
    out = torch.empty(local_shape, dtype=full.dtype, device=full.device)
    dist.reduce_scatter_tensor(out.view(-1, c), full, group=group)
    return out


def _rank_rows(w, n, rank):
    return w.view(n, w.shape[0] // n, w.shape[1])[rank]


_SPLIT_KV_WEIGHTS = {}


def _split_kv_rows(w_kv_b, n, rank, heads, dn, dv):
    """The rank's ``kv_b`` rows as ``[k_nope of every local head; v of every local head]``, rebuilt
    only when the weight changes."""
    key = (w_kv_b.data_ptr(), n, rank, heads, dn, dv)
    cached = _SPLIT_KV_WEIGHTS.get(key)
    if cached is not None and cached[0] == w_kv_b._version:
        return cached[1]
    w = _rank_rows(w_kv_b.detach(), n, rank).view(heads, dn + dv, -1)
    split = torch.cat((w[:, :dn].reshape(heads * dn, -1), w[:, dn:].reshape(heads * dv, -1)))
    _SPLIT_KV_WEIGHTS[key] = (w_kv_b._version, split)
    return split


def _fused_supported(qkv_format, heads, cq, ckv, dn, dr, dv):
    """Shapes the FlyDSL AgGemm covers (``ag_gemm_kernel``)."""
    return (
        qkv_format == "sbhd"
        and cq % 64 == 0
        and ckv % 64 == 0
        and dn % 32 == 0
        and 256 % dn == 0
        and dr % 8 == 0
        and (heads * dn) % 256 == 0
        and (heads * dv) % 256 == 0
    )


def _forward_fused(q_latent, kv_latent, w_q_b, w_kv_b, group, heads, dn, dr, dv, save):
    from primus_turbo.flydsl.ulysses.ag_gemm_kernel import (
        ag_gemm_bank_bytes,
        ulysses_mla_ag_gemm,
    )
    from primus_turbo.flydsl.ulysses.workspace import (
        REGION_MLA_AG_GEMM,
        get_ulysses_workspace,
    )

    n, rank = group.size(), group.rank()
    cq, ckv = q_latent.shape[-1], kv_latent.shape[-1] - dr
    xq, xkv = q_latent.reshape(-1, cq).contiguous(), kv_latent.reshape(-1, ckv + dr).contiguous()
    R = n * xq.shape[0]
    q, k = (torch.empty(R, heads, dn + dr, dtype=xq.dtype, device=xq.device) for _ in range(2))
    v = torch.empty(R, heads, dv, dtype=xq.dtype, device=xq.device)
    lq = torch.empty(R, cq, dtype=xq.dtype, device=xq.device) if save else None
    lc = torch.empty(R, ckv, dtype=xq.dtype, device=xq.device) if save else None
    workspace = get_ulysses_workspace(group, ag_gemm_bank_bytes(R, cq, ckv, dr))
    ulysses_mla_ag_gemm(
        xq,
        xkv,
        _rank_rows(w_q_b, n, rank).contiguous(),
        _split_kv_rows(w_kv_b, n, rank, heads, dn, dv),
        q,
        k,
        v,
        lq,
        lc,
        workspace,
        dn=dn,
        dr=dr,
        dv=dv,
        flag_region=REGION_MLA_AG_GEMM,
    )
    return q, k, v, lq, lc


def _forward_unfused(q_latent, kv_latent, w_q_b, w_kv_b, group, qkv_format, heads, dn, dr, dv):
    n, rank = group.size(), group.rank()
    cq, ckv = q_latent.shape[-1], kv_latent.shape[-1] - dr
    gathered = _gather_rows(torch.cat((q_latent, kv_latent), dim=-1), group, qkv_format)
    lq, lkv = gathered[:, :cq], gathered[:, cq:]
    q = lq @ _rank_rows(w_q_b, n, rank).t()
    kv = (lkv[:, :ckv] @ _rank_rows(w_kv_b, n, rank).t()).view(-1, heads, dn + dv)
    k = torch.empty(kv.shape[0], heads, dn + dr, dtype=kv.dtype, device=kv.device)
    k[..., :dn] = kv[..., :dn]
    k[..., dn:] = lkv[:, None, ckv:]
    return q, k, kv[..., dn:].contiguous(), lq, lkv[:, :ckv]


class UlyssesMLALatentUpProjFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_latent,
        kv_latent,
        w_q_b,
        w_kv_b,
        num_heads,
        qk_nope_dim,
        qk_rope_dim,
        v_dim,
        group,
        qkv_format,
        fused,
    ):
        n = group.size()
        cq, ckv = q_latent.shape[-1], kv_latent.shape[-1] - qk_rope_dim
        dn, dr, dv = qk_nope_dim, qk_rope_dim, v_dim
        if qkv_format == "sbhd":
            s_local, b = q_latent.shape[:2]
            lead = (s_local * n, b)
        else:
            b, s_local = q_latent.shape[:2]
            lead = (b, s_local * n)
        heads = num_heads // n
        if fused is None:
            fused = _fused_supported(qkv_format, heads, cq, ckv, dn, dr, dv)
        if fused:
            assert _fused_supported(qkv_format, heads, cq, ckv, dn, dr, dv), "shape not covered by the AgGemm"
            save = w_q_b.requires_grad or w_kv_b.requires_grad
            q, k, v, lq, lc = _forward_fused(
                q_latent, kv_latent, w_q_b, w_kv_b, group, heads, dn, dr, dv, save
            )
        else:
            q, k, v, lq, lc = _forward_unfused(
                q_latent, kv_latent, w_q_b, w_kv_b, group, qkv_format, heads, dn, dr, dv
            )

        ctx.save_for_backward(lq, lc, w_q_b, w_kv_b)
        ctx.dims = (heads, cq, ckv, dn, dr, dv)
        ctx.group, ctx.qkv_format = group, qkv_format
        ctx.local_shape = q_latent.shape
        return q.view(*lead, heads, dn + dr), k.view(*lead, heads, dn + dr), v.view(*lead, heads, dv)

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        lq, lc, w_q_b, w_kv_b = ctx.saved_tensors
        heads, cq, ckv, dn, dr, dv = ctx.dims
        group, qkv_format = ctx.group, ctx.qkv_format
        n, rank = group.size(), group.rank()
        rows = grad_q.numel() // (heads * (dn + dr))

        gq = grad_q.reshape(rows, heads * (dn + dr))
        gk = grad_k.reshape(rows, heads, dn + dr)
        gkv = torch.cat((gk[..., :dn], grad_v.reshape(rows, heads, dv)), dim=-1).view(rows, -1)
        wq_r, wkv_r = _rank_rows(w_q_b, n, rank), _rank_rows(w_kv_b, n, rank)

        partial = torch.empty(rows, cq + ckv + dr, dtype=grad_q.dtype, device=grad_q.device)
        torch.matmul(gq, wq_r, out=partial[:, :cq])
        torch.matmul(gkv, wkv_r, out=partial[:, cq : cq + ckv])
        partial[:, cq + ckv :] = gk[..., dn:].float().sum(dim=1)
        local_shape = (*ctx.local_shape[:-1], cq + ckv + dr)
        local = _scatter_rows(partial, group, qkv_format, local_shape)
        d_q_latent, d_kv_latent = local[..., :cq].contiguous(), local[..., cq:].contiguous()

        d_w_q_b = d_w_kv_b = None
        if ctx.needs_input_grad[2]:
            d_w_q_b = torch.zeros_like(w_q_b)
            torch.matmul(gq.t(), lq, out=_rank_rows(d_w_q_b, n, rank))
        if ctx.needs_input_grad[3]:
            d_w_kv_b = torch.zeros_like(w_kv_b)
            torch.matmul(gkv.t(), lc, out=_rank_rows(d_w_kv_b, n, rank))
        return d_q_latent, d_kv_latent, d_w_q_b, d_w_kv_b, None, None, None, None, None, None, None


def ulysses_mla_latent_up_proj(
    q_latent: torch.Tensor,
    kv_latent: torch.Tensor,
    w_q_b: torch.Tensor,
    w_kv_b: torch.Tensor,
    num_heads: int,
    group: dist.ProcessGroup,
    qk_nope_dim: int = 128,
    qk_rope_dim: int = 64,
    v_dim: int = 128,
    qkv_format: str = "sbhd",
    fused: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MLA up-projections of the local heads over the full sequence, from this rank's latents.

    Args:
        q_latent: normalized q latent of this rank's tokens, ``[s/n, b, q_lora_rank]`` (sbhd) or
            ``[b, s/n, q_lora_rank]`` (bshd).
        kv_latent: ``[normalized c_kv | k_pe after RoPE]`` of this rank's tokens,
            ``[..., kv_lora_rank + qk_rope_dim]``.
        w_q_b: ``[num_heads * (qk_nope_dim + qk_rope_dim), q_lora_rank]``, head-major rows.
        w_kv_b: ``[num_heads * (qk_nope_dim + v_dim), kv_lora_rank]``, rows ``[k_nope | v]`` per head.
        fused: forward through the FlyDSL AgGemm (all-gather fused into the GEMMs, sbhd only) or
            RCCL all-gather + hipBLASLt; ``None`` picks the AgGemm wherever it covers the shape.

    Returns:
        q, k, v for the local heads over the full sequence, ``[s, b, h/n, d]`` (sbhd) or
        ``[b, s, h/n, d]``. q's last ``qk_rope_dim`` columns are not rotated yet (RoPE on the full
        sequence is the caller's); k carries the shared ``k_pe`` in its last ``qk_rope_dim`` columns.
    """
    assert qkv_format in _QKV_FORMATS, f"qkv_format must be one of {_QKV_FORMATS}"
    n = group.size()
    assert num_heads % n == 0, f"num_heads {num_heads} must divide by the group size {n}"
    assert q_latent.shape[:-1] == kv_latent.shape[:-1], "q_latent and kv_latent cover different tokens"
    assert w_q_b.shape == (num_heads * (qk_nope_dim + qk_rope_dim), q_latent.shape[-1])
    assert w_kv_b.shape == (num_heads * (qk_nope_dim + v_dim), kv_latent.shape[-1] - qk_rope_dim)
    return UlyssesMLALatentUpProjFunction.apply(
        q_latent,
        kv_latent,
        w_q_b,
        w_kv_b,
        num_heads,
        qk_nope_dim,
        qk_rope_dim,
        v_dim,
        group,
        qkv_format,
        fused,
    )
