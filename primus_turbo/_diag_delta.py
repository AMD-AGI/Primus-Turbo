"""Diagnostic: identity-delta vs consistent-delta dq accuracy, per-row.

Reproduces the harness correctness setup (bf16 GQA 64/8 hd64 causal, fp32 ref via
autograd) and, for whatever delta path is active (env FLYDSL_BWD_IDENTITY_DELTA),
reports for dq/dk/dv:
  - min per-row cosine (the harness gate metric) and WHERE it occurs,
  - the ref-norm rank/percentile of that worst row (is it a tiny-signal row?),
  - global relative L2,
  - cosine percentiles so we can see if 0.954 is one outlier row or a broad drop.
Run twice (env 0 / env 1) and compare.
"""
import os
import torch
import torch.nn.functional as F

BF16 = torch.bfloat16
DEV = "cuda"
H_Q, H_KV, D = 64, 8, 64
CASES = [(2, 512), (2, 1024), (1, 2048)]

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    attention_flydsl_forward_impl as fwd,
    attention_flydsl_backward_impl as bwd,
)


def ref_fwd_fp32(q, k, v, scale, causal=True):
    b, s, hq, d = q.shape
    hkv = k.shape[2]
    qf = q.float().transpose(1, 2)
    kf = k.float().transpose(1, 2).repeat_interleave(hq // hkv, 1)
    vf = v.float().transpose(1, 2).repeat_interleave(hq // hkv, 1)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    if causal:
        idx = torch.arange(s, device=q.device)
        scores = scores.masked_fill(idx[None, None, :, None] < idx[None, None, None, :], float("-inf"))
    p = scores.softmax(-1)
    o = torch.matmul(p, vf)
    return o.transpose(1, 2)


def per_row(x, ref):
    x = x.float().reshape(-1, x.shape[-1])
    ref = ref.float().reshape(-1, ref.shape[-1])
    l2 = (x - ref).norm().item() / (ref.norm().item() + 1e-12)
    ref_norm = ref.norm(dim=1)
    mask = ref_norm > 1e-6 * (ref_norm.max() + 1e-12)
    cos = F.cosine_similarity(x[mask], ref[mask], dim=1)
    order = cos.argsort()
    worst = order[0].item()
    # map worst (index into masked rows) back to global row
    global_idx = mask.nonzero(as_tuple=True)[0]
    grow = global_idx[worst].item()
    rn = ref_norm[mask]
    # percentile of the worst row's ref-norm among kept rows
    pct = (rn < rn[worst]).float().mean().item() * 100
    qs = torch.tensor([0.0, 0.001, 0.01, 0.05, 0.5], device=cos.device)
    cq = torch.quantile(cos, qs).tolist()
    return {
        "l2": round(l2, 5),
        "cos_min": round(cos.min().item(), 5),
        "cos_p0.1%": round(cq[1], 5),
        "cos_p1%": round(cq[2], 5),
        "cos_p5%": round(cq[3], 5),
        "cos_med": round(cq[4], 5),
        "worst_refnorm_pctile": round(pct, 2),
        "worst_refnorm": round(rn[worst].item(), 6),
        "max_refnorm": round(ref_norm.max().item(), 4),
        "n_below_cos0.98": int((cos < 0.98).sum().item()),
        "n_rows": int(cos.numel()),
    }


def main():
    mode = "IDENTITY" if os.environ.get("FLYDSL_BWD_IDENTITY_DELTA") == "1" else "CONSISTENT"
    scale = 1.0 / (D ** 0.5)
    print(f"==== delta mode = {mode} ====")
    for b, s in CASES:
        torch.manual_seed(0)
        q = torch.randn(b, s, H_Q, D, device=DEV, dtype=BF16)
        k = torch.randn(b, s, H_KV, D, device=DEV, dtype=BF16)
        v = torch.randn(b, s, H_KV, D, device=DEV, dtype=BF16)
        dout = torch.randn(b, s, H_Q, D, device=DEV, dtype=BF16)

        qg = q.float().clone().requires_grad_(True)
        kg = k.float().clone().requires_grad_(True)
        vg = v.float().clone().requires_grad_(True)
        ref_out = ref_fwd_fp32(qg, kg, vg, scale, True)
        ref_out.backward(dout.float())

        out, lse = fwd(q, k, v, scale, True)
        dq, dk, dv = bwd(dout, q, k, v, out, lse, scale, True)

        print(f"-- b{b} s{s} --")
        for nm, x, r in (("dq", dq, qg.grad), ("dk", dk, kg.grad), ("dv", dv, vg.grad)):
            print(f"  {nm}: {per_row(x, r)}")


if __name__ == "__main__":
    main()
