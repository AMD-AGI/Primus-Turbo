"""Per-row-segment SNR of the MLA backward vs the fp32 reference, to localize errors in a partial
last kv band. Usage: python tail_band_check.py B S [key=value ...] (dkdv builder overrides)."""

import ast
import sys

sys.path.insert(0, "/perf_apps/xiaoming/Primus-Turbo/agent/workspace/flash_attn_bwd_mla_flydsl_gfx950_20261109")
import quick_test_bench as qb  # noqa: E402
import torch  # noqa: E402

B, S = int(sys.argv[1]), int(sys.argv[2])
over = {k: ast.literal_eval(v) for k, v in (a.split("=", 1) for a in sys.argv[3:])}
bk = over.get("block_kv", 192)
g = torch.Generator(device="cuda").manual_seed(1)
q, k = (torch.randn((S, B, 128, 192), device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(2))
v = torch.randn((S, B, 128, 128), device="cuda", dtype=torch.bfloat16, generator=g)
do = torch.randn((S, B, 128, 128), device="cuda", dtype=torch.bfloat16, generator=g)
o, lse = qb.mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)
dq, dk, dv = qb.mla_bwd_sbhd(do, q, k, v, o, lse, **over)
torch.cuda.synchronize()


def snr(x, r):
    x, r = x.float(), r.float()
    return (10 * torch.log10(r.pow(2).sum() / (x - r).pow(2).sum().clamp_min(1e-30))).item()


last = (S - 1) // bk * bk
segs = [(0, bk, "band0"), (max(0, last - bk), last, "band-2"), (last, S, f"last[{last}:{S})")]
worst = 1e9
for b in range(B):
    for h in (0, 63, 127):
        rq, rk, rv = qb.ref_head(q[:, b, h], k[:, b, h], v[:, b, h], do[:, b, h])
        cells = []
        for lo, hi, name in segs:
            s3 = [snr(dq[lo:hi, b, h], rq[lo:hi]), snr(dk[lo:hi, b, h], rk[lo:hi]), snr(dv[lo:hi, b, h], rv[lo:hi])]
            worst = min(worst, *s3)
            cells.append(f"{name} {'/'.join('%.1f' % x for x in s3)}")
        print(f"b{b} h{h:3d}  " + "  ".join(cells), flush=True)
print(f"worst segment SNR {worst:.2f} dB; finite {all(bool(torch.isfinite(t).all()) for t in (dq, dk, dv))}")
