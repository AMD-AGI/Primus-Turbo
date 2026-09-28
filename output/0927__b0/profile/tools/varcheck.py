"""Correctness of fwd variants vs r6 at one shape (VC_SHAPE=toy|prod): max|diff| of o and lse, SQNR of o."""
import os, sys, json
from pathlib import Path
E2E = Path("/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/e2e"); PROF = E2E.parent / "profile"
sys.path.insert(0, str(E2E / "attn_backends")); sys.path.insert(0, "/home/lihuzhan/.local/flydsl0341")
os.environ["E2E_ATTN"] = "asm"
V = {v: {"fwd": str(PROF / "arms" / f"fwd_r6_{v}"), "bwd": str(E2E / "arms" / "bwd_r20_0341")} for v in ("nospec", "nodefer")}
os.environ["E2E_FLY_TREES"] = json.dumps(V)
import torch
from e2e_attn import arms
toy = os.environ.get("VC_SHAPE", "toy") == "toy"
b, s, hq, hkv, d = (1, 256, 8, 2, 128) if toy else (4, 8192, 32, 8, 128)
torch.manual_seed(0)
if toy or os.environ.get("VC_SET", "randn") == "randn":
    q = torch.randn(b, s, hq, d, device="cuda", dtype=torch.bfloat16) * 4; k = torch.randn(b, s, hkv, d, device="cuda", dtype=torch.bfloat16); v = torch.randn_like(k)
else:
    t = torch.load("/home/lihuzhan/_prof_dump/qkv_call0680.pt"); q, k, v = (t[n].cuda() for n in ("q", "k", "v"))
ref_o, ref_l = arms.get_fwd("fly")(q, k, v, d ** -0.5)
for name in ("nospec", "nodefer"):
    o, l = arms.get_fwd(name)(q, k, v, d ** -0.5); torch.cuda.synchronize()
    e = (o.float() - ref_o.float()); sq = 10 * torch.log10(ref_o.float().pow(2).sum() / e.pow(2).sum().clamp_min(1e-30))
    print(f"VC {name} shape={'toy' if toy else 'prod'} finite={torch.isfinite(o).all().item()} max|do|={e.abs().max().item():.3g} "
          f"max|dlse|={(l - ref_l).abs().max().item():.3g} sqnr_vs_r6={sq.item():.1f}dB", flush=True)
