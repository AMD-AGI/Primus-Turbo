#!/usr/bin/env python3
"""R24 probe: BF16 var-K (dB CRR) for the 4 still-on-DEFAULT families.

Per R23's recommendation: probe the 4 dB var-K families currently
falling through to BF16 default ``(gm=4, xcds=8)``:

  family            tiles_m  tiles_n  k_arg
  DSV3-GateUP            16       28  M_per
  DSV3-Down              28        8  M_per
  Qwen3-GateUP           12       16  M_per
  Qwen3-Down             16        6  M_per

(R23 already probed gpt_oss-Down — `(gm=1, xcds=4)` won uniformly,
+1.52% avg vs production cube-rule `(gm=2, xcds=32)`. R1 already
tuned gpt_oss-GateUP. Both excluded here.)

For each family, sweep 11 (gm, xcds) cells × 5 trials × 120 iters
on all 4 metric shapes (B ∈ {16, 32}, M_per ∈ {2048, 4096}). For
each family, identify the cell that is uniformly positive on all 4
shapes (if any) — those become R24's multi-rule aggregate
candidates (R20-style: stack 4 family-specific rules in one
commit so the combined wall delta crosses the metric noise floor).

Also re-anchors gpt_oss-Down and gpt_oss-GateUP with the same
methodology so all 6 dB var-K families are characterized in one
log file.
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
var_k_fn = hk.grouped_variable_k_crr
assert var_k_fn is not None

device = "cuda"


def warmup_bf16_runtime():
    """K-tail cold-start workaround (R22). Mirror metric's iteration
    order: DSV3 → gpt_oss-B4 → gpt_oss-Down-B32 fwd+bwd via autograd
    so the K-tail kernel state is initialized."""
    import primus_turbo.pytorch as turbo
    from primus_turbo.pytorch.core.backend import (
        BackendType, GlobalBackendManager, PrecisionType,
    )
    GlobalBackendManager.set_grouped_gemm_backend(
        BackendType.HIPKITTEN, PrecisionType.BF16_FP16_FP32
    )
    def warm_one(B, M_per, N, K):
        M_total = B * M_per
        a = torch.randn(M_total, K, dtype=torch.bfloat16, device=device,
                        requires_grad=True)
        b = torch.randn(B, N, K, dtype=torch.bfloat16, device=device,
                        requires_grad=True)
        gl = torch.full((B,), M_per, dtype=torch.int64, device=device)
        out = turbo.ops.grouped_gemm(a, b, gl, trans_b=True)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        torch.cuda.synchronize()
    warm_one(16, 2048, 4096, 7168)   # DSV3-GateUP B=16
    warm_one(16, 2048, 7168, 2048)   # DSV3-Down   B=16
    warm_one(16, 2048, 3072, 4096)   # Qwen3-GateUP B=16
    warm_one(16, 2048, 4096, 1536)   # Qwen3-Down   B=16
    warm_one(4,  2048, 5760, 2880)   # gpt_oss-GateUP B=4
    warm_one(4,  2048, 2880, 2880)   # gpt_oss-Down   B=4
    warm_one(32, 2048, 2880, 2880)   # gpt_oss-Down   B=32
    torch.cuda.synchronize()


def make_tensors(B, M_per, N_fwd, K_fwd):
    M_total = B * M_per
    torch.manual_seed(42)
    grad_out = torch.randn(M_total, N_fwd, dtype=torch.bfloat16, device=device)
    x = torch.randn(M_total, K_fwd, dtype=torch.bfloat16, device=device)
    grad_b = torch.empty(B, N_fwd, K_fwd, dtype=torch.bfloat16, device=device)
    offs = torch.tensor([i * M_per for i in range(B + 1)],
                        dtype=torch.int64, device=device)
    return grad_out, x, grad_b, offs


def time_one(grad_out, x, grad_b, offs, gm, xcd, iters=120):
    for _ in range(15):
        var_k_fn(grad_out, x, grad_b, offs, gm, xcd)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        var_k_fn(grad_out, x, grad_b, offs, gm, xcd)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def correctness_check(grad_out, x, offs, gm_p, xcd_p, gm_t, xcd_t,
                      B, N_fwd, K_fwd):
    """Bit-equivalence vs production cfg (gm_p, xcd_p)."""
    c_prod = torch.zeros(B, N_fwd, K_fwd, dtype=grad_out.dtype, device=device)
    var_k_fn(grad_out, x, c_prod, offs, gm_p, xcd_p)
    c_test = torch.zeros_like(c_prod)
    var_k_fn(grad_out, x, c_test, offs, gm_t, xcd_t)
    max_abs = (c_prod.float() - c_test.float()).abs().max().item()
    bit_eq = torch.equal(c_prod.view(torch.int16), c_test.view(torch.int16))
    return max_abs, bit_eq


def sweep_shape(B, M_per, N_fwd, K_fwd, prod_gm, prod_xcd, label):
    M_total = B * M_per
    grad_out, x, grad_b, offs = make_tensors(B, M_per, N_fwd, K_fwd)
    flops = 2 * N_fwd * K_fwd * M_total

    candidates = [
        (prod_gm, prod_xcd),  # production (1st position to anchor)
        (1, 0),  (2, 0),  (4, 0),  (8, 0), (16, 0),
        (4, 2),  (4, 4),  (2, 4),  (8, 4), (16, 4),  (1, 4),
    ]
    # dedupe (prod might repeat in standard list)
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c); seen.add(c)
    candidates = uniq

    print(f"\n=== {label} ===")
    print(f"  fwd=(B={B}, M_per={M_per}, N={N_fwd}, K={K_fwd}), "
          f"M_total={M_total}, tiles_m={N_fwd//256}, tiles_n={K_fwd//256}, "
          f"k_arg={M_per}, prod=(gm={prod_gm}, xcds={prod_xcd})")

    ma, be = correctness_check(grad_out, x, offs, prod_gm, prod_xcd,
                                4, 4, B, N_fwd, K_fwd)
    print(f"  correctness ((prod) vs (4,4)): max_abs={ma:.6e} bit_eq={be}")

    results = []
    for (gm, xcd) in candidates:
        ms_trials = [
            time_one(grad_out, x, grad_b, offs, gm, xcd, iters=120)
            for _ in range(5)
        ]
        median_ms = statistics.median(ms_trials)
        tflops = flops / (median_ms * 1e9)
        results.append((gm, xcd, median_ms, tflops))

    prod_tflops = next(r[3] for r in results
                       if r[0] == prod_gm and r[1] == prod_xcd)
    results.sort(key=lambda r: r[3], reverse=True)

    print(f"  {'(gm,xcd)':>10s}  {'med_ms':>8s}  {'tflops':>7s}  Δ vs prod({prod_gm},{prod_xcd})")
    for r in results:
        delta = (r[3] - prod_tflops) / prod_tflops * 100
        d = "  *PROD*" if (r[0] == prod_gm and r[1] == prod_xcd) else ""
        print(f"  gm={r[0]:>2d} xcd={r[1]:>2d}  {r[2]:7.4f}   "
              f"{r[3]:6.1f}   {delta:+5.2f}%{d}")

    return [(gm, xcd, tf) for (gm, xcd, _, tf) in results]


def family_aggregate(family_name, by_shape, prod_cell):
    """Print uniform-positive cell aggregate for one family."""
    print("\n" + "-" * 90)
    print(f"AGG {family_name}  vs prod={prod_cell}")
    print("-" * 90)
    prod_by_shape = {sh: by_shape[sh][prod_cell] for sh in by_shape}
    cells = sorted(by_shape[next(iter(by_shape))].keys())
    sh_keys = list(by_shape.keys())
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
        mn = min(deltas)
        mx = max(deltas)
        u = "  +" if mn > 0 else ("  ~" if mn >= -0.3 else "  -")
        marker = "  *PROD*" if cell == prod_cell else ""
        delta_strs = "  ".join(f"{d:+8.2f}%" for d in deltas)
        print(f"  gm={gm:>2d} xc={xcd:>2d}  {delta_strs}  "
              f"{avg:+6.2f}%  {mn:+6.2f}%  {mx:+6.2f}%  {u}{marker}")
        if mn > 0 and (best_uniform is None or avg > best_uniform[1]):
            best_uniform = (cell, avg, mn, mx)
    if best_uniform:
        cell, avg, mn, mx = best_uniform
        print(f"  >> UNIFORM-POSITIVE WINNER: gm={cell[0]} xcds={cell[1]}  "
              f"avg={avg:+.2f}% min={mn:+.2f}% max={mx:+.2f}%")
    else:
        print(f"  >> NO uniform-positive cell.")
    return best_uniform


def main():
    print("=" * 90)
    print("R24 multi-family dB var-K probe — 4 still-on-DEFAULT families")
    print("=" * 90)
    print("Warming up HK BF16 runtime via autograd fwd+bwd...")
    warmup_bf16_runtime()
    print("Warmup OK.")

    # All families: (label, B, M_per, N_fwd, K_fwd, prod_gm, prod_xcd)
    families_meta = [
        # DSV3-Down (tiles_m=28, tiles_n=8) — DEFAULT
        ("DSV3-Down", 7168, 2048, (4, 8), [
            ("B16-M2k", 16, 2048),
            ("B16-M4k", 16, 4096),
            ("B32-M2k", 32, 2048),
            ("B32-M4k", 32, 4096),
        ]),
        # DSV3-GateUP (tiles_m=16, tiles_n=28) — DEFAULT
        ("DSV3-GateUP", 4096, 7168, (4, 8), [
            ("B16-M2k", 16, 2048),
            ("B16-M4k", 16, 4096),
            ("B32-M2k", 32, 2048),
            ("B32-M4k", 32, 4096),
        ]),
        # Qwen3-Down (tiles_m=16, tiles_n=6) — DEFAULT
        ("Qwen3-Down", 4096, 1536, (4, 8), [
            ("B16-M2k", 16, 2048),
            ("B16-M4k", 16, 4096),
            ("B32-M2k", 32, 2048),
            ("B32-M4k", 32, 4096),
        ]),
        # Qwen3-GateUP (tiles_m=12, tiles_n=16) — DEFAULT
        ("Qwen3-GateUP", 3072, 4096, (4, 8), [
            ("B16-M2k", 16, 2048),
            ("B16-M4k", 16, 4096),
            ("B32-M2k", 32, 2048),
            ("B32-M4k", 32, 4096),
        ]),
    ]

    family_winners = {}

    for fam_name, N_fwd, K_fwd, (prod_gm, prod_xcd), shapes in families_meta:
        print("\n" + "#" * 90)
        print(f"# Family: {fam_name}  N_fwd={N_fwd}  K_fwd={K_fwd}  "
              f"prod=(gm={prod_gm}, xcds={prod_xcd})")
        print("#" * 90)
        by_shape = {}
        for sh_name, B, M_per in shapes:
            res = sweep_shape(B, M_per, N_fwd, K_fwd, prod_gm, prod_xcd,
                              f"{fam_name} {sh_name}")
            by_shape[sh_name] = {(g, x): tf for (g, x, tf) in res}
        winner = family_aggregate(fam_name, by_shape, (prod_gm, prod_xcd))
        family_winners[fam_name] = winner

    print("\n" + "=" * 90)
    print("R24 SUMMARY — uniform-positive winners by family")
    print("=" * 90)
    for fam, winner in family_winners.items():
        if winner:
            cell, avg, mn, mx = winner
            print(f"  {fam:14s}  -> (gm={cell[0]:>2d}, xcds={cell[1]:>2d})  "
                  f"avg={avg:+.2f}%  range=[{mn:+.2f}%, {mx:+.2f}%]")
        else:
            print(f"  {fam:14s}  NO uniform-positive cell")


if __name__ == "__main__":
    main()
