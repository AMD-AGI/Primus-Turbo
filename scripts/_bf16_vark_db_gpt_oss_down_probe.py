#!/usr/bin/env python3
"""R23 probe: BF16 var-K (dB CRR) for gpt_oss-Down family vs production cfg.

R22 falsification revealed an oversight: gpt_oss-Down dB var-K
(tiles_m=11, tiles_n=11, k <= 4096) currently dispatches via the
LAYOUT-AGNOSTIC cube rule at config.py line 691
(``tiles_m <= 16 and tiles_m == tiles_n and k <= 12288``) which
fires BEFORE the gpt_oss-specific CRR rule at line 1080. The cube
rule returns ``(gm=2, xcds=32)`` — a config tuned for dense LLaMA
FORWARD cubes (R1+R5), never validated for grouped var-K backward.

This probe sweeps the 11 standard (gm, xcds) cells on all 4
gpt_oss-Down shapes with ``(gm=2, xcds=32)`` (the actual production
cfg) as the comparison baseline, NOT ``(gm=4, xcds=4)`` (R22's
incorrect baseline for these shapes).

If a uniform-positive cell emerges and the win is substantial,
land a CRR-specific rule that fires BEFORE the cube rule for
``tiles_m == tiles_n and tiles_n == 11 and k <= 4096 and m_total
is not None`` — narrowed to grouped var-K (which always passes
m_total) so dense LLaMA cube forward is untouched.
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
    """K-tail cold-start workaround (R22 finding): direct var_k calls
    on cold gpt_oss-GateUP-B=32 fault. Run autograd fwd+bwd through
    the K%128==0 path and the K%128==64 path progressively."""
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


def correctness_check(grad_out, x, offs, gm, xcd, B, N_fwd, K_fwd):
    """Bit-equivalence vs current production cfg (gm=2, xcds=32)."""
    c_prod = torch.zeros(B, N_fwd, K_fwd, dtype=grad_out.dtype, device=device)
    var_k_fn(grad_out, x, c_prod, offs, 2, 32)
    c_test = torch.zeros_like(c_prod)
    var_k_fn(grad_out, x, c_test, offs, gm, xcd)
    max_abs = (c_prod.float() - c_test.float()).abs().max().item()
    bit_eq = torch.equal(c_prod.view(torch.int16), c_test.view(torch.int16))
    return max_abs, bit_eq


def sweep_shape(B, M_per, N_fwd, K_fwd, label):
    M_total = B * M_per
    grad_out, x, grad_b, offs = make_tensors(B, M_per, N_fwd, K_fwd)
    flops = 2 * N_fwd * K_fwd * M_total

    # 11 cells: production (2,32) + the 11 from R22 probe (deduped) → 12 total.
    # Place production first to anchor baseline.
    candidates = [
        (2, 32),   # current production (cube rule)
        (1, 0),  (2, 0),  (4, 0),  (8, 0), (16, 0),
        (4, 2),  (4, 4),  (2, 4),  (8, 4), (16, 4),  (1, 4),
    ]

    print(f"\n=== {label} ===")
    print(f"  fwd=(B={B}, M_per={M_per}, N={N_fwd}, K={K_fwd}), "
          f"M_total={M_total}, tiles_m={N_fwd//256}, tiles_n={K_fwd//256}, "
          f"k_arg={M_per}")

    ma, be = correctness_check(grad_out, x, offs, 4, 4, B, N_fwd, K_fwd)
    print(f"  correctness ((2,32) vs (4,4)): max_abs={ma:.6e} bit_eq={be}")

    results = []
    for (gm, xcd) in candidates:
        ms_trials = [
            time_one(grad_out, x, grad_b, offs, gm, xcd, iters=120)
            for _ in range(5)
        ]
        median_ms = statistics.median(ms_trials)
        tflops = flops / (median_ms * 1e9)
        results.append((gm, xcd, median_ms, tflops))

    prod_tflops = next(r[3] for r in results if r[0] == 2 and r[1] == 32)
    results.sort(key=lambda r: r[3], reverse=True)

    print(f"  {'(gm,xcd)':>10s}  {'med_ms':>8s}  {'tflops':>7s}  Δ vs prod(2,32)")
    for r in results:
        delta = (r[3] - prod_tflops) / prod_tflops * 100
        d = "  *PROD*" if (r[0] == 2 and r[1] == 32) else ""
        print(f"  gm={r[0]:>2d} xcd={r[1]:>2d}  {r[2]:7.4f}   "
              f"{r[3]:6.1f}   {delta:+5.2f}%{d}")

    return [(gm, xcd, tf) for (gm, xcd, _, tf) in results]


def main():
    print("=" * 70)
    print("gpt_oss-Down var-K (dB CRR) — full B={4,32} family")
    print("Production cfg: (gm=2, xcds=32) via cube rule (config.py:691)")
    print("=" * 70)
    print("Warming HK BF16 runtime via autograd fwd+bwd (K-tail cold-start workaround)...")
    warmup_bf16_runtime()
    print("Warmup OK.")

    s_b4_m2k = sweep_shape(4,  2048, 2880, 2880,
                           "gpt_oss-Down B=4  M=2048 (m_total=8192)")
    s_b4_m4k = sweep_shape(4,  4096, 2880, 2880,
                           "gpt_oss-Down B=4  M=4096 (m_total=16384)")
    s_b32_m2k = sweep_shape(32, 2048, 2880, 2880,
                            "gpt_oss-Down B=32 M=2048 (m_total=65536)")
    s_b32_m4k = sweep_shape(32, 4096, 2880, 2880,
                            "gpt_oss-Down B=32 M=4096 (m_total=131072)")

    print("\n" + "=" * 70)
    print("Aggregate per-cell deltas vs production (2,32)")
    print("=" * 70)

    by_shape = {
        "B4-M2k":  {(g, x): tf for (g, x, tf) in s_b4_m2k},
        "B4-M4k":  {(g, x): tf for (g, x, tf) in s_b4_m4k},
        "B32-M2k": {(g, x): tf for (g, x, tf) in s_b32_m2k},
        "B32-M4k": {(g, x): tf for (g, x, tf) in s_b32_m4k},
    }
    prod_by_shape = {
        sh: by_shape[sh][(2, 32)] for sh in by_shape
    }

    cells = sorted(by_shape["B4-M2k"].keys())
    print(f"  {'cell':>10s}  {'B4-M2k':>9s}  {'B4-M4k':>9s}  "
          f"{'B32-M2k':>9s}  {'B32-M4k':>9s}  {'avg':>7s}  "
          f"{'min':>7s}  {'max':>7s}  uniform")
    for cell in cells:
        gm, xcd = cell
        deltas = []
        for sh in ("B4-M2k", "B4-M4k", "B32-M2k", "B32-M4k"):
            d = (by_shape[sh][cell] - prod_by_shape[sh]) / prod_by_shape[sh] * 100
            deltas.append(d)
        avg = sum(deltas) / len(deltas)
        mn = min(deltas)
        mx = max(deltas)
        u = "  +" if mn > 0 else ("  ~" if mn >= -0.3 else "  -")
        marker = "  *PROD*" if (gm == 2 and xcd == 32) else ""
        print(f"  gm={gm:>2d} xc={xcd:>2d}  {deltas[0]:+8.2f}%  "
              f"{deltas[1]:+8.2f}%  {deltas[2]:+8.2f}%  {deltas[3]:+8.2f}%  "
              f"{avg:+6.2f}%  {mn:+6.2f}%  {mx:+6.2f}%  {u}{marker}")


if __name__ == "__main__":
    main()
