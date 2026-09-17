"""Eager reference: plain PyTorch, transcribed from `op.reference.logic`.

THIS IS THE PRECISION REFERENCE, AND IT IS THE ONLY ONE. Nothing in this job is
ever aligned against a library -- not against `op/beat/`, not against the
installed Primus-Turbo, and never against the code under optimization. If this
file and a kernel disagree, this file is right until someone proves otherwise
against the maths below, not against another kernel.

Same signature as every other implementation here:

    attention(q, k, v, causal=True, softmax_scale=None) -> out

Written for clarity, not for speed. Everything is fp32 and the softmax is the
definitional two-pass form, not the online recurrence -- an online softmax is an
optimization, and an optimization in the oracle is a way to share a bug with the
thing it is checking.

Two points where getting it wrong is quiet rather than loud:

  * dk and dv accumulate over ALL FOUR query heads of a GQA group. The loop below
    is written per KV head with the group as an explicit leading axis so the sum
    over the group is a visible `.sum(0)` rather than something a broadcast might
    or might not have done. `op.reference.logic` calls this out: forgetting it is
    wrong by a factor a sampled check will not localise.

  * `op.config.causal` is `bottom-right`, so the mask is `j <= i + (Skv - Sq)`.
    At Sq == Skv that is identical to top-left, which is why the square shape
    cannot tell the two apart -- and why `op/ut/` carries a shape where the two
    sequence axes differ. Note torch SDPA's `is_causal=True` is TOP-LEFT and
    therefore is NOT this function for a non-square mask; that is exactly the
    disagreement `op.reference.logic` says to resolve in favour of the logic.

P is recomputed from the SAVED lse, not from a fresh row max, because that is
what a flash backward does and the two differ in floating point.

Memory: the full [Sq, Skv] score matrix is materialised, so the loop is over
(batch, kv head) -- 32 iterations at the job shape, each holding a
[4, Sq, Skv] fp32 block. The maths is exact; only the loop is a concession.
"""

import torch

NAME = "eager"


def _causal_mask(seqlen_q, seqlen_kv, device):
    """live(i, j) for bottom-right causal: the last query attends to all of K."""
    i = torch.arange(seqlen_q, device=device).unsqueeze(1)
    j = torch.arange(seqlen_kv, device=device).unsqueeze(0)
    return j <= i + (seqlen_kv - seqlen_q)


def reference_forward(q, k, v, causal=True, softmax_scale=None):
    """Returns (out fp32, lse fp32 [B, Hq, Sq]). All maths in fp32."""
    b, sq, hq, d = q.shape
    _, skv, hkv, _ = k.shape
    g = hq // hkv
    if softmax_scale is None:
        softmax_scale = d**-0.5

    qf, kf, vf = q.float(), k.float(), v.float()
    mask = _causal_mask(sq, skv, q.device) if causal else None

    out = torch.empty(b, sq, hq, d, device=q.device, dtype=torch.float32)
    lse = torch.empty(b, hq, sq, device=q.device, dtype=torch.float32)

    for bi in range(b):
        for h in range(hkv):
            # [G, Sq, D] and [Skv, D] -- the group is an explicit axis.
            qg = qf[bi, :, h * g : (h + 1) * g, :].permute(1, 0, 2)
            kk, vv = kf[bi, :, h, :], vf[bi, :, h, :]

            s = torch.einsum("gid,jd->gij", qg, kk) * softmax_scale
            if mask is not None:
                s = s.masked_fill(~mask, float("-inf"))

            # A fully masked row has row_max = -inf, and exp(-inf - -inf) is NaN.
            # Under bottom-right causal with Sq > Skv the first (Sq - Skv) rows are
            # exactly that. Their softmax has no defined value; the convention every
            # flash implementation uses -- and the one this reference adopts -- is
            # p = 0, out = 0, lse = -inf. Handled with `where`, not with a clamp,
            # so a live row is never perturbed.
            row_max = s.max(dim=-1, keepdim=True).values
            dead = torch.isinf(row_max) & (row_max < 0)
            safe_max = torch.where(dead, torch.zeros_like(row_max), row_max)
            e = torch.exp(s - safe_max)
            e = torch.where(dead, torch.zeros_like(e), e)
            row_sum = e.sum(dim=-1, keepdim=True)
            p = torch.where(dead, torch.zeros_like(e), e / torch.where(dead, torch.ones_like(row_sum), row_sum))

            out[bi, :, h * g : (h + 1) * g, :] = torch.einsum("gij,jd->gid", p, vv).permute(1, 0, 2)
            lse[bi, h * g : (h + 1) * g, :] = torch.where(
                dead, torch.full_like(row_max, float("-inf")), safe_max + torch.log(row_sum)
            ).squeeze(-1)

    return out, lse


def reference_backward(dout, q, k, v, out, lse, causal=True, softmax_scale=None):
    """Returns (dq, dk, dv), all fp32. Transcribed line for line from op.reference.logic."""
    b, sq, hq, d = q.shape
    _, skv, hkv, _ = k.shape
    g = hq // hkv
    if softmax_scale is None:
        softmax_scale = d**-0.5

    qf, kf, vf = q.float(), k.float(), v.float()
    dof, of = dout.float(), out.float()
    mask = _causal_mask(sq, skv, q.device) if causal else None

    dq = torch.zeros(b, sq, hq, d, device=q.device, dtype=torch.float32)
    dk = torch.zeros(b, skv, hkv, d, device=q.device, dtype=torch.float32)
    dv = torch.zeros(b, skv, hkv, d, device=q.device, dtype=torch.float32)

    for bi in range(b):
        for h in range(hkv):
            qg = qf[bi, :, h * g : (h + 1) * g, :].permute(1, 0, 2)   # [G, Sq, D]
            dog = dof[bi, :, h * g : (h + 1) * g, :].permute(1, 0, 2)  # [G, Sq, D]
            og = of[bi, :, h * g : (h + 1) * g, :].permute(1, 0, 2)    # [G, Sq, D]
            kk, vv = kf[bi, :, h, :], vf[bi, :, h, :]                  # [Skv, D]
            lg = lse[bi, h * g : (h + 1) * g, :]                       # [G, Sq]

            s = torch.einsum("gid,jd->gij", qg, kk) * softmax_scale
            # p from the SAVED lse. Masked entries are zeroed rather than -inf'd,
            # because exp(-inf - lse) is 0 but exp(-inf - -inf) is NaN on a row
            # that is entirely masked.
            # `where`, not a multiply: on a fully masked row lse is -inf, so
            # exp(s - lse) is +inf and 0 * inf would be NaN. `where` selects the
            # dead branch instead of arithmetically cancelling it.
            p = torch.exp(s - lg.unsqueeze(-1))
            if mask is not None:
                p = torch.where(mask, p, torch.zeros_like(p))

            delta = (dog * og).sum(dim=-1)                             # [G, Sq]
            dp = torch.einsum("gid,jd->gij", dog, vv)                  # [G, Sq, Skv]
            ds = p * (dp - delta.unsqueeze(-1))
            if mask is not None:
                ds = torch.where(mask, ds, torch.zeros_like(ds))

            # The group sum: dk/dv take contributions from all G query heads.
            dv[bi, :, h, :] = torch.einsum("gij,gid->jd", p, dog)
            dk[bi, :, h, :] = torch.einsum("gij,gid->jd", ds, qg) * softmax_scale
            dq[bi, :, h * g : (h + 1) * g, :] = (
                torch.einsum("gij,jd->gid", ds, kk) * softmax_scale
            ).permute(1, 0, 2)

    return dq, dk, dv


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, softmax_scale):
        out, lse = reference_forward(q, k, v, causal, softmax_scale)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal, ctx.softmax_scale = causal, softmax_scale
        return out.to(q.dtype)

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = reference_backward(dout, q, k, v, out, lse, ctx.causal, ctx.softmax_scale)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


def attention(q, k, v, causal=True, softmax_scale=None):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    return _Attention.apply(q, k, v, causal, softmax_scale)
