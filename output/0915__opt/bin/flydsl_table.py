"""Build the offline FlyDSL TN config table that nkfix rule 3 reads.

Run this on an idle card, NOT during training. It calls FlyDSL's autotune, and autotune's
measurement loop cannot survive a candidate whose launch faults: the fault surfaces at its
torch.cuda.synchronize(), the `except Exception: continue` swallows it, and the loop keeps
going against a dead HIP context. Done here, the blast radius is this process. Done inside a
training step -- which is how it was tried first -- it takes the training and the card with it.

Each shape is tuned in its OWN subprocess for the same reason, so one faulting shape costs that
shape and not the table. Shapes are keyed on the full (M, N, K): FlyDSL's own cache omits M
because "M is the token count", which is true for NT/NN and inverted for TN, where M is
out_features and K is the token count.

    python3 flydsl_table.py --out /home/lihuzhan/_dbg_l8b/flydsl_tn.json
"""
import argparse, json, os, subprocess, sys

T = 32768
# (M = out_features, K = tokens, N = in_features) for every wgrad of the 8-layer llama3.1-8b
SHAPES = [(4096, T, 4096), (1024, T, 4096), (6144, T, 4096),
          (14336, T, 4096), (4096, T, 14336), (128256, T, 4096)]

CHILD = r'''
import json, sys, torch
M, K, N = (int(x) for x in sys.argv[1:4])
from primus_turbo.flydsl.gemm.gemm_gfx1250_kernel import autotune, gemm_gfx1250, can_run
a_t = torch.randn(K, M, device="cuda", dtype=torch.bfloat16)
b   = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
if not can_run(a_t, b, layout="tn", out_dtype=torch.bfloat16):
    print("DECLINED"); raise SystemExit(0)
ref = torch.mm(a_t.t(), b).float()
tile, mw, nw, nb = autotune(a_t, b, layout="tn", out_dtype=torch.bfloat16)
out = gemm_gfx1250(a_t, b, layout="tn", tile=tile, m_warp=mw, n_warp=nw,
                   num_buffers=nb, out_dtype=torch.bfloat16)
torch.cuda.synchronize()
d = out.float() - ref
sqnr = 10 * torch.log10((ref**2).mean() / (d**2).mean().clamp_min(1e-30)).item()
# The gate is on the table, not on the ranking: a config that is fast and wrong must never be
# written to a file that training will read without re-checking.
if sqnr < 50:
    print("SQNR_FAIL %.1f" % sqnr); raise SystemExit(0)
print("OK " + json.dumps({"tile": list(tile), "m_warp": mw, "n_warp": nw,
                          "num_buffers": nb, "sqnr_db": sqnr}))
'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    table = {}
    for M, K, N in SHAPES:
        key = f"{M},{N},{K}"
        try:
            p = subprocess.run([sys.executable, "-c", CHILD, str(M), str(K), str(N)],
                               capture_output=True, text=True, timeout=a.timeout)
        except subprocess.TimeoutExpired:
            print(f"  {key:<24} TIMEOUT after {a.timeout}s -- excluded")
            table[key] = None
            continue
        line = next((l for l in reversed(p.stdout.splitlines()) if l.startswith(("OK ", "DECLINED", "SQNR_FAIL"))), "")
        if line.startswith("OK "):
            cfg = json.loads(line[3:])
            table[key] = cfg
            print(f"  {key:<24} tile={cfg['tile']} mw={cfg['m_warp']} nw={cfg['n_warp']} "
                  f"nb={cfg['num_buffers']}  SQNR {cfg['sqnr_db']:.1f} dB")
        else:
            table[key] = None
            tail = (p.stderr.strip().splitlines() or ["(no stderr)"])[-1]
            print(f"  {key:<24} EXCLUDED rc={p.returncode} {line or tail[:90]}")
    with open(a.out, "w") as f:
        json.dump(table, f, indent=1)
    good = sum(1 for v in table.values() if v)
    print(f"\n{good}/{len(table)} shapes usable -> {a.out}")
    print("Shapes with no entry fall through to rule 2; that is the intended degradation.")

main()
