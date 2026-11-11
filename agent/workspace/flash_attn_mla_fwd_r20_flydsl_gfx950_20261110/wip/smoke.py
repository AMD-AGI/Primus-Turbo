"""Small-shape correctness smoke: python wip/smoke.py PKG_DIR [B,S,H ...] [--base DIR]."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ab_bench as ab  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("pkg")
p.add_argument("shapes", nargs="*", default=["1,1024,4"])
p.add_argument("--base")
a = p.parse_args()
cand = ab.load_pkg(a.pkg, "mla_cand")
base = ab.load_pkg(a.base, "mla_base") if a.base else None
for spec in a.shapes:
    B, S, H = (int(x) for x in spec.split(","))
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(S, B, H, d, generator=g, device="cuda", dtype=torch.bfloat16) for d in (ab.DQK, ab.DQK, ab.DV))
    o, lse = cand.mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)
    torch.cuda.synchronize()
    lv = lse.view(B, S, H)
    rows = torch.arange(S, device="cuda") if S <= 2048 else ab.sample_rows(S, 0)
    worst_o = worst_l = 0.0
    for bi in range(B):
        ref_o, ref_lse = ab.exact_rows(q, k, v, rows, bi)
        worst_o = max(worst_o, ab.rel(o[rows, bi], ref_o))
        worst_l = max(worst_l, (lv[bi, rows] - ref_lse).abs().max().item())
        if worst_o > 3e-3:
            err = (o[rows, bi].float() - ref_o).norm(dim=-1) / ref_o.norm(dim=-1)
            bad = (err > 1e-2).nonzero()
            print(f"  b{bi}: {bad.shape[0]} bad (row, head) pairs, first: {bad[:8].tolist()}")
    line = f"B={B} S={S} H={H}: o_rel {worst_o:.3e} lse_max {worst_l:.3e} finite {bool(torch.isfinite(o).all())}"
    if base is not None:
        ob, lb = base.mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)
        line += f" | bitwise_vs_base {bool(torch.equal(ob, o) and torch.equal(lb, lse))}"
    print(line, flush=True)
