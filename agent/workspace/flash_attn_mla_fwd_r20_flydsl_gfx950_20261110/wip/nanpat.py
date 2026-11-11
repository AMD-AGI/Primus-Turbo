import os, sys, torch
sys.path.insert(0, "/home/xiaompen/Turbo/agent/workspace/flash_attn_mla_fwd_r20_flydsl_gfx950_20261110")
import ab_bench as ab
m = ab.load_pkg(sys.argv[1], "mc")
b = ab.load_pkg("rounds/round-1/kernel_snapshot/branch_mla", "mb")
B,S,H = 1, 1024, 2
g = torch.Generator(device="cuda").manual_seed(0)
q, k, v = (torch.randn(S, B, H, d, generator=g, device="cuda", dtype=torch.bfloat16) for d in (192,192,128))
o, l = m.mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)
ob, lb = b.mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)
bad = ~torch.isfinite(o)
print("nonfinite", bad.sum().item(), "of", o.numel())
rows = bad.any(-1).any(-1).any(-1).nonzero().flatten()
print("bad rows", rows[:40].tolist(), "count", rows.numel())
print("bad cols (h0)", bad[:, 0, 0].nonzero().flatten()[:10].tolist())
fin = torch.isfinite(o)
d = (o.float()-ob.float()).abs()
d[~fin] = 0
print("max diff finite", d.max().item(), "lse max diff", (l-lb).abs().max().item())
r = rows[0].item() if rows.numel() else 0
print("row", r, o[r,0,0,:8].tolist(), ob[r,0,0,:8].tolist())
