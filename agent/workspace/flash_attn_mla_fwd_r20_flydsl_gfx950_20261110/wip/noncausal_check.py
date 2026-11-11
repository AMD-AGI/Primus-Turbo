"""Non-causal correctness of a package against an fp32 reference: python wip/noncausal_check.py PKG [B,S,H ...]"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ab_bench as ab  # noqa: E402

pkg = ab.load_pkg(sys.argv[1], "mla_nc")
causal = os.environ.get("CAUSAL", "0") == "1"
for spec in sys.argv[2:] or ["1,1024,4"]:
    B, S, H = (int(x) for x in spec.split(","))
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(S, B, H, d, generator=g, device="cuda", dtype=torch.bfloat16) for d in (ab.DQK, ab.DQK, ab.DV))
    o, lse = pkg.mla_fwd_sbhd(q, k, v, causal=causal, return_lse=True)
    torch.cuda.synchronize()
    qf, kf, vf = (t.float().permute(1, 2, 0, 3) for t in (q, k, v))
    s = qf @ kf.transpose(-1, -2) / ab.DQK**0.5
    if causal:
        s = s.masked_fill(torch.ones(S, S, device="cuda", dtype=torch.bool).triu(1), float("-inf"))
    ref_lse = torch.logsumexp(s, dim=-1)
    ref_o = (torch.softmax(s, dim=-1) @ vf).permute(2, 0, 1, 3)
    o_rel = ((o.float() - ref_o).norm() / ref_o.norm()).item()
    lse_err = (lse.view(B, S, H).permute(0, 2, 1) - ref_lse).abs().max().item()
    print(f"non-causal B={B} S={S} H={H}: o_rel {o_rel:.3e} lse_max {lse_err:.3e} finite {bool(torch.isfinite(o).all())}")
