#!/usr/bin/env python3
"""R26 probe: BF16 grouped fwd RCR — 4 families × 4 shapes = 16 shapes.

R25 falsified the single-family Qwen3-GateUP rule (kernel +1.68 %
but only +0.5 score on its own — sub-noise). Per R25's
recommendation, R26 builds the multi-family aggregate to clear
the +5 metric noise floor (R20 / R24 pattern).

Probe ALL non-gpt_oss fwd RCR cells (gpt_oss is heavily R10/R21
tuned per-shape with its own H4 reroute path; non-gpt_oss
families either fall through to the binding default or to
broad/old rules from R5/R7/R10):

  family            current dispatch     shapes covered
  Qwen3-GateUP        DEFAULT (4, 8)     all 4 (R25 data reused)
  Qwen3-Down M=2048   DEFAULT (4, 8)     R25 confirmed sub-noise
  Qwen3-Down M=4096   cube (2, 32)       UNTESTED in R25 — probe
  DSV3-GateUP M=2048  R10 (1, 4)         re-probe (post-BUFFER)
  DSV3-GateUP M=4096  cube (2, 32)       UNTESTED — probe
  DSV3-Down all 4     R10 (16, 2)        re-probe (post-BUFFER)

This is 10 fresh shapes plus 4 R25-reused = 14 of 16 metric
non-gpt_oss fwd shapes covered (Qwen3-Down M=2048 already
characterized as sub-noise dead). Bundle all uniform-positive
winners → R20-style aggregate.
"""

import os
import sys
import statistics

sys.path.insert(0, "/workspace/code/Primus-Turbo")

import torch  # noqa: E402

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
os.environ.setdefault("PRIMUS_TURBO_GROUPED_GEMM_BACKEND", "HIPKITTEN")

from primus_turbo.pytorch.kernels.hipkitten import loader as hipkitten  # noqa: E402

hk = hipkitten.load_bf16()
fwd_fn = hk.grouped_rcr
assert fwd_fn is not None

device = "cuda"


def warmup():
    """Light kernel-only warmup — no K-tail in non-gpt_oss families."""
    for B, M_per, N_fwd, K_fwd in [
        (16, 2048, 4096, 7168),  # DSV3-GateUP
        (16, 2048, 7168, 2048),  # DSV3-Down
        (16, 2048, 4096, 1536),  # Qwen3-Down
        (16, 2048, 3072, 4096),  # Qwen3-GateUP
    ]:
        M_total = B * M_per
        a = torch.randn(M_total, K_fwd, dtype=torch.bfloat16, device=device)
        b = torch.randn(B, N_fwd, K_fwd, dtype=torch.bfloat16, device=device)
        out = torch.empty(M_total, N_fwd, dtype=torch.bfloat16, device=device)
        offs = torch.tensor([i * M_per for i in range(B + 1)],
                            dtype=torch.int64, device=device)
        for _ in range(5):
            fwd_fn(a, b, out, offs, 4, 8, M_per)
        torch.cuda.synchronize()


def make_tensors(B, M_per, N_fwd, K_fwd):
    M_total = B * M_per
    torch.manual_seed(42)
    a = torch.randn(M_total, K_fwd, dtype=torch.bfloat16, device=device)
    b = torch.randn(B, N_fwd, K_fwd, dtype=torch.bfloat16, device=device)
    out = torch.empty(M_total, N_fwd, dtype=torch.bfloat16, device=device)
    offs = torch.tensor([i * M_per for i in range(B + 1)],
                        dtype=torch.int64, device=device)
    return a, b, out, offs, M_per


def time_one(a, b, out, offs, m_per_group, gm, xcd, iters=200):
    for _ in range(20):
        fwd_fn(a, b, out, offs, gm, xcd, m_per_group)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fwd_fn(a, b, out, offs, gm, xcd, m_per_group)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def correctness(a, b, offs, m_per_group, prod_gm, prod_xcd, gm_t, xcd_t):
    M_total, K_fwd = a.shape
    N_fwd = b.shape[1]
    out_p = torch.empty(M_total, N_fwd, dtype=a.dtype, device=device)
    fwd_fn(a, b, out_p, offs, prod_gm, prod_xcd, m_per_group)
    out_t = torch.empty_like(out_p)
    fwd_fn(a, b, out_t, offs, gm_t, xcd_t, m_per_group)
    max_abs = (out_p.float() - out_t.float()).abs().max().item()
    bit_eq = torch.equal(out_p.view(torch.int16), out_t.view(torch.int16))
    return max_abs, bit_eq


def sweep_shape(B, M_per, N_fwd, K_fwd, prod_gm, prod_xcd, label):
    a, b, out, offs, m_per_group = make_tensors(B, M_per, N_fwd, K_fwd)
    M_total = B * M_per
    flops = 2 * M_total * N_fwd * K_fwd

    candidates = [
        (prod_gm, prod_xcd),
        (1, 4),  (2, 4),  (4, 4),  (8, 4), (16, 4), (32, 4),
        (4, 8),  (4, 2),  (2, 32), (16, 2),
    ]
    seen = set(); uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c); seen.add(c)
    candidates = uniq

    print(f"\n=== {label} ===")
    print(f"  m={M_per} n={N_fwd} k={K_fwd} mt={M_total} "
          f"tiles_m={M_per//256} tiles_n={N_fwd//256} prod=(gm={prod_gm}, xcds={prod_xcd})")

    ma, be = correctness(a, b, offs, m_per_group, prod_gm, prod_xcd, 4, 4)
    print(f"  correctness ((prod) vs (4,4)): max_abs={ma:.6e} bit_eq={be}")

    results = []
    for (gm, xcd) in candidates:
        ms_trials = [time_one(a, b, out, offs, m_per_group, gm, xcd, iters=200)
                     for _ in range(5)]
        med_ms = statistics.median(ms_trials)
        tflops = flops / (med_ms * 1e9)
        results.append((gm, xcd, med_ms, tflops))

    prod_tf = next(r[3] for r in results
                   if r[0] == prod_gm and r[1] == prod_xcd)
    results.sort(key=lambda r: r[3], reverse=True)

    print(f"  {'(gm,xcd)':>10s}  {'med_ms':>8s}  {'tflops':>7s}  Δ vs prod")
    for r in results:
        d = (r[3] - prod_tf) / prod_tf * 100
        marker = "  *PROD*" if (r[0] == prod_gm and r[1] == prod_xcd) else ""
        print(f"  gm={r[0]:>2d} xcd={r[1]:>2d}  {r[2]:7.4f}   "
              f"{r[3]:6.1f}   {d:+5.2f}%{marker}")

    return [(gm, xcd, tf) for (gm, xcd, _, tf) in results]


def family_aggregate(family_name, by_shape, prod_cell):
    print("\n" + "-" * 92)
    print(f"AGG {family_name}  vs prod={prod_cell}")
    print("-" * 92)
    sh_keys = list(by_shape.keys())
    cells = sorted(by_shape[sh_keys[0]].keys())
    prod_by_shape = {sh: by_shape[sh][prod_cell] for sh in sh_keys}
    hdr = f"  {'cell':>10s}  " + "  ".join(f"{sh:>9s}" for sh in sh_keys) + "  " + \
          f"{'avg':>7s}  {'min':>7s}  {'max':>7s}  uniform"
    print(hdr)
    best_uniform = None
    for cell in cells:
        gm, xcd = cell
        deltas = []
        for sh in sh_keys:
            d = (by_shape[sh][cell] - prod_by_shape[sh]) / prod_by_shape[sh] * 100
            deltas.append(d)
        avg = sum(deltas) / len(deltas)
        mn, mx = min(deltas), max(deltas)
        u = "  +" if mn > 0 else ("  ~" if mn >= -0.3 else "  -")
        marker = "  *PROD*" if cell == prod_cell else ""
        ds = "  ".join(f"{d:+8.2f}%" for d in deltas)
        print(f"  gm={gm:>2d} xc={xcd:>2d}  {ds}  {avg:+6.2f}%  "
              f"{mn:+6.2f}%  {mx:+6.2f}%  {u}{marker}")
        if mn > 0 and (best_uniform is None or avg > best_uniform[1]):
            best_uniform = (cell, avg, mn, mx)
    if best_uniform:
        cell, avg, mn, mx = best_uniform
        print(f"  >> UNIFORM-POSITIVE WINNER: gm={cell[0]} xcds={cell[1]}  "
              f"avg={avg:+.2f}% min={mn:+.2f}% max={mx:+.2f}%")
    else:
        print("  >> NO uniform-positive cell.")
    return best_uniform


def main():
    print("=" * 92)
    print("R26 BF16 fwd RCR multi-family probe — DSV3 + Qwen3-Down M=4096")
    print("=" * 92)
    print("Warmup...")
    warmup()
    print("Warmup OK.")

    families = [
        # DSV3-GateUP — N_fwd=4096 K_fwd=7168
        # M=2048 hits R10 (gm=1, xcds=4); M=4096 hits cube (gm=2, xcds=32).
        # Probe each separately so we can tell if the existing rules are
        # still optimal under the post-R19/R20 BUFFER kernel.
        ("DSV3-GateUP-M2k", 4096, 7168, (1, 4), [
            ("B16", 16, 2048),
            ("B32", 32, 2048),
        ]),
        ("DSV3-GateUP-M4k", 4096, 7168, (2, 32), [
            ("B16", 16, 4096),
            ("B32", 32, 4096),
        ]),
        # DSV3-Down — N_fwd=7168 K_fwd=2048. R10 rule (gm=16, xcds=2)
        # for tiles_n=28 ∧ 8<=tiles_m<=16 ∧ k<=4096.
        ("DSV3-Down", 7168, 2048, (16, 2), [
            ("B16-M2k", 16, 2048),
            ("B16-M4k", 16, 4096),
            ("B32-M2k", 32, 2048),
            ("B32-M4k", 32, 4096),
        ]),
        # Qwen3-Down M=4096 — N_fwd=4096 K_fwd=1536. Cube rule (gm=2, xcds=32).
        ("Qwen3-Down-M4k", 4096, 1536, (2, 32), [
            ("B16", 16, 4096),
            ("B32", 32, 4096),
        ]),
    ]

    winners = {}
    for fam, N_fwd, K_fwd, prod, shapes in families:
        print("\n" + "#" * 92)
        print(f"# Family: {fam}  N_fwd={N_fwd} K_fwd={K_fwd}  prod={prod}")
        print("#" * 92)
        by_shape = {}
        for sh_name, B, M_per in shapes:
            res = sweep_shape(B, M_per, N_fwd, K_fwd, prod[0], prod[1],
                              f"{fam} {sh_name}")
            by_shape[sh_name] = {(g, x): tf for (g, x, tf) in res}
        winners[fam] = family_aggregate(fam, by_shape, prod)

    print("\n" + "=" * 92)
    print("R26 SUMMARY — uniform-positive winners by family")
    print("=" * 92)
    for fam, w in winners.items():
        if w:
            cell, avg, mn, mx = w
            print(f"  {fam:18s} -> (gm={cell[0]:>2d}, xcds={cell[1]:>2d})  "
                  f"avg={avg:+.2f}%  range=[{mn:+.2f}%, {mx:+.2f}%]")
        else:
            print(f"  {fam:18s} NO uniform-positive cell")


if __name__ == "__main__":
    main()
