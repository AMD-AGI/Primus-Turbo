"""Quick timing of packages on one shape: python wip/tq.py PKG... [--shape B,S,H]"""
import sys
sys.argv, pk = sys.argv[:1], sys.argv[1:]
shape = (1, 16384, 128)
if "--shape" in pk:
    i = pk.index("--shape"); shape = tuple(int(x) for x in pk[i + 1].split(",")); del pk[i:i + 2]
sys.path.insert(0, "/home/xiaompen/Turbo/agent/workspace/flash_attn_mla_fwd_r20_flydsl_gfx950_20261110")
import ab_bench as ab, torch
B, S, H = shape
g = torch.Generator(device="cuda").manual_seed(0)
q, k, v = (torch.randn(S, B, H, d, generator=g, device="cuda", dtype=torch.bfloat16) for d in (192, 192, 128))
mods = [(p, ab.load_pkg(p, f"m{i}")) for i, p in enumerate(pk)]
fl = B * S * (S + 1) * H * 320
for rep in range(2):
    for p, m in mods:
        t = ab.time_ms(lambda: m.mla_fwd_sbhd(q, k, v, causal=True, return_lse=True))
        print(f"{p:45s} {t:8.3f} ms {fl / t / 1e9:8.1f} TF", flush=True)
