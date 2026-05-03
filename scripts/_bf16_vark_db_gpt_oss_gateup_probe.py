#!/usr/bin/env python3
"""R26 add-on: BF16 dB var-K (CRR) probe for gpt_oss-GateUP.

R24's multi-family probe covered:
  - gpt_oss-Down (cube rule misfire → (gm=1, xcds=4))
  - DSV3-GateUP/Down, Qwen3-GateUP/Down (default → uniform winners)

But gpt_oss-GateUP dB var-K — current R1 rule (gm=4, xcds=4) for
``tiles_n == 11 and 8 <= tiles_m <= 24 and k <= 4096`` — was NEVER
re-probed against the post-R19/R20 BUFFER kernel. R24's per-family
probes found 3 of 4 newly-tuned families converged on (gm=1, xcds=4)
— same cell that wins for gpt_oss-Down. If gpt_oss-GateUP also
prefers a different cell now, that's 4 shapes × 3 weight = 12
weight units of lift potential.

Geometry: m=N_fwd=5760, n=K_fwd=2880, k=M_per ∈ {2048, 4096};
m_total = B * M_per; tiles_m_disp=22, tiles_n_disp=11, k_disp=M_per.
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
    """K-tail cold-start prophylaxis (R22). Mirror metric's iter
    order via autograd fwd+bwd."""
    import primus_turbo.pytorch as turbo
    from primus_turbo.pytorch.core.backend import (
        BackendType, GlobalBackendManager, PrecisionType,
    )
    GlobalBackendManager.set_grouped_gemm_backend(
        BackendType.HIPKITTEN, PrecisionType.BF16_FP16_FP32
    )

    def warm(B, M_per, N, K):
        M_total = B * M_per
        a = torch.randn(M_total, K, dtype=torch.bfloat16, device=device,
                        requires_grad=True)
        b = torch.randn(B, N, K, dtype=torch.bfloat16, device=device,
                        requires_grad=True)
        gl = torch.full((B,), M_per, dtype=torch.int64, device=device)
        out = turbo.ops.grouped_gemm(a, b, gl, trans_b=True)
        grad = torch.randn_like(out)
        out.backward(grad)
        torch.cuda.synchronize()

    warm(16, 2048, 4096, 7168)   # DSV3-GateUP B=16
    warm(16, 2048, 7168, 2048)   # DSV3-Down   B=16
    warm(16, 2048, 3072, 4096)   # Qwen3-GateUP B=16
    warm(16, 2048, 4096, 1536)   # Qwen3-Down   B=16
    warm(4,  2048, 5760, 2880)   # gpt_oss-GateUP B=4
    warm(4,  2048, 2880, 2880)   # gpt_oss-Down   B=4
    warm(32, 2048, 5760, 2880)   # gpt_oss-GateUP B=32
    warm(32, 2048, 2880, 2880)   # gpt_oss-Down   B=32
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


def correctness(grad_out, x, offs, prod_gm, prod_xcd, gm_t, xcd_t,
                B, N_fwd, K_fwd):
    c_p = torch.zeros(B, N_fwd, K_fwd, dtype=grad_out.dtype, device=device)
    var_k_fn(grad_out, x, c_p, offs, prod_gm, prod_xcd)
    c_t = torch.zeros_like(c_p)
    var_k_fn(grad_out, x, c_t, offs, gm_t, xcd_t)
    max_abs = (c_p.float() - c_t.float()).abs().max().item()
    bit_eq = torch.equal(c_p.view(torch.int16), c_t.view(torch.int16))
    return max_abs, bit_eq


def sweep_shape(B, M_per, prod_gm, prod_xcd, label):
    N_fwd = 5760
    K_fwd = 2880
    M_total = B * M_per
    grad_out, x, grad_b, offs = make_tensors(B, M_per, N_fwd, K_fwd)
    flops = 2 * N_fwd * K_fwd * M_total

    candidates = [
        (prod_gm, prod_xcd),
        (1, 4),  (2, 4),  (4, 4),  (8, 4), (16, 4),
        (1, 0),  (2, 0),  (4, 0),  (8, 0), (16, 0),
        (4, 8),  (4, 2),
    ]
    seen = set(); uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c); seen.add(c)
    candidates = uniq

    print(f"\n=== {label} ===")
    print(f"  fwd=(B={B}, M_per={M_per}, N={N_fwd}, K={K_fwd}), "
          f"M_total={M_total}, prod=(gm={prod_gm}, xcds={prod_xcd})")

    ma, be = correctness(grad_out, x, offs, prod_gm, prod_xcd, 4, 4,
                          B, N_fwd, K_fwd)
    print(f"  correctness ((prod) vs (4,4)): max_abs={ma:.6e} bit_eq={be}")

    results = []
    for (gm, xcd) in candidates:
        ms_trials = [time_one(grad_out, x, grad_b, offs, gm, xcd, iters=120)
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


def main():
    print("=" * 92)
    print("R26 BF16 dB var-K probe — gpt_oss-GateUP (R1 rule re-probe)")
    print("=" * 92)
    print("Warmup...")
    warmup_bf16_runtime()
    print("Warmup OK.")

    shapes = [
        ("B4-M2k",  4,  2048),
        ("B4-M4k",  4,  4096),
        ("B32-M2k", 32, 2048),
        ("B32-M4k", 32, 4096),
    ]
    by_shape = {}
    for sh_name, B, M_per in shapes:
        res = sweep_shape(B, M_per, 4, 4, f"gpt_oss-GateUP {sh_name}")
        by_shape[sh_name] = {(g, x): tf for (g, x, tf) in res}

    print("\n" + "-" * 92)
    print("AGG gpt_oss-GateUP dB var-K  vs prod=(4, 4)")
    print("-" * 92)
    sh_keys = list(by_shape.keys())
    cells = sorted(by_shape[sh_keys[0]].keys())
    prod_by_shape = {sh: by_shape[sh][(4, 4)] for sh in sh_keys}
    hdr = f"  {'cell':>10s}  " + "  ".join(f"{sh:>9s}" for sh in sh_keys) + "  " + \
          f"{'avg':>7s}  {'min':>7s}  {'max':>7s}  uniform"
    print(hdr)
    best_uniform = None
    for cell in cells:
        gm, xcd = cell
        deltas = [(by_shape[sh][cell] - prod_by_shape[sh]) / prod_by_shape[sh] * 100
                  for sh in sh_keys]
        avg = sum(deltas) / len(deltas)
        mn, mx = min(deltas), max(deltas)
        u = "  +" if mn > 0 else ("  ~" if mn >= -0.3 else "  -")
        marker = "  *PROD*" if cell == (4, 4) else ""
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


if __name__ == "__main__":
    main()
