#!/usr/bin/env python3
"""R22 probe: BF16 var-K (dB CRR) config sweep on gpt_oss B=32 family.

Per R21 plan ("(A) BF16 var-K (dB CRR) gpt_oss B=32 split"):
clone scripts/_bf16_rrr_da_probe.py to a var-K probe; sweep 11 cells
× 5 trials × 100 iters on the 4 gpt_oss B=32 var-K shapes. If a
uniform-positive cell emerges, split the existing single-cfg rule
(`tiles_n==11 and 8<=tiles_m<=24 and k<=4096 -> (gm=4, xcds=4)`) by
m_total>=65536 (B=32) into a B=32-specific cell.

Kernel signature (BF16 grouped var-K CRR):
  grouped_variable_k_crr(a, b, c, group_offs, group_m, num_xcds)
where:
  a = grad_out_2d [M_total, N_fwd]    (dispatcher's `b`)
  b = x_2d       [M_total, K_fwd]    (dispatcher's `a`)
  c = grad_b     [B, N_fwd, K_fwd]   output dB

R21 baseline (per metric): all 4 gpt_oss B=32 shapes ratio 1.044-1.090
(3x weight). Default rule (gm=4, xcds=4) is shared with B=4 family
(uniform-positive at R1). Split candidate: m_total>=65536.
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
assert var_k_fn is not None, "BF16 binding lacks grouped_variable_k_crr"

device = "cuda"


def warmup_bf16_runtime():
    """Workaround for HK BF16 K-tail cold-start sync-fault bug.

    The metric mirrors the same workaround at the suite level: it runs
    DSV3 (K_fwd=2048, K%128==0) fwd+bwd before gpt_oss (K_fwd=2880,
    K%128==64) so the K-tail kernel has a baseline runtime state.
    Direct var-K calls on cold gpt_oss-GateUP-B=32 (the largest
    geometry) memory-fault without this path warm-up — confirmed by
    bisection (DSV3 var-K alone was insufficient; full autograd
    fwd+bwd through the suite is required).

    Done here via the autograd dispatcher (mirrors the metric's
    iteration order: DSV3 → gpt_oss-B4 → gpt_oss-B32-Down → ...).
    """
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

    # DSV3 first (K%128==0 fast path); then small gpt_oss
    # (K%128==64 K-tail path) progressively up to B=32.
    warm_one(16, 2048, 4096, 7168)   # DSV3-GateUP B=16
    warm_one(16, 2048, 7168, 2048)   # DSV3-Down   B=16
    warm_one(4,  2048, 5760, 2880)   # gpt_oss-GateUP B=4
    warm_one(4,  2048, 2880, 2880)   # gpt_oss-Down   B=4
    warm_one(32, 2048, 2880, 2880)   # gpt_oss-Down   B=32
    torch.cuda.synchronize()


def make_tensors(B, M_per_group, N_fwd, K_fwd):
    M_total = B * M_per_group
    torch.manual_seed(42)
    # var-K dispatcher passes in (b=grad_out [M_total,N_fwd], a=x [M_total,K_fwd])
    # and the kernel signature is (a=grad_out, b=x, c=dB[B,N_fwd,K_fwd], ...)
    grad_out = torch.randn(M_total, N_fwd, dtype=torch.bfloat16, device=device)
    x = torch.randn(M_total, K_fwd, dtype=torch.bfloat16, device=device)
    grad_b = torch.empty(B, N_fwd, K_fwd, dtype=torch.bfloat16, device=device)
    group_offs = torch.tensor(
        [i * M_per_group for i in range(B + 1)], dtype=torch.int64, device=device
    )
    return grad_out, x, grad_b, group_offs


def time_one(grad_out, x, grad_b, group_offs, group_m, num_xcds, iters=100):
    for _ in range(10):
        var_k_fn(grad_out, x, grad_b, group_offs, group_m, num_xcds)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        var_k_fn(grad_out, x, grad_b, group_offs, group_m, num_xcds)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def correctness_check(grad_out, x, group_offs, group_m, num_xcds, B, N_fwd, K_fwd):
    """Verify (group_m, num_xcds) is bit-identical to default (4, 4)."""
    c_default = torch.zeros(B, N_fwd, K_fwd, dtype=grad_out.dtype, device=device)
    var_k_fn(grad_out, x, c_default, group_offs, 4, 4)
    c_test = torch.zeros_like(c_default)
    var_k_fn(grad_out, x, c_test, group_offs, group_m, num_xcds)
    max_abs = (c_default.float() - c_test.float()).abs().max().item()
    bit_eq = torch.equal(c_default.view(torch.int16), c_test.view(torch.int16))
    return max_abs, bit_eq


def sweep_shape(B, M_per_group, N_fwd, K_fwd, label):
    M_total = B * M_per_group
    grad_out, x, grad_b, group_offs = make_tensors(B, M_per_group, N_fwd, K_fwd)
    # Per-call FLOPs: dB = grad_out^T @ x summed across B groups, where each
    # group is (N_fwd, K_fwd, M_per_group). 2 * N_fwd * K_fwd * M_total flops.
    flops = 2 * N_fwd * K_fwd * M_total

    # Same 11 cells as the dA RRR probe (R20).
    candidates = [
        (1, 0),  (2, 0),  (4, 0),  (8, 0), (16, 0),
        (4, 2),  (4, 4),  (2, 4),  (8, 4), (16, 4),  (1, 4),
    ]

    print(f"\n=== {label} ===")
    print(f"  fwd=(B={B}, M_per={M_per_group}, N={N_fwd}, K={K_fwd}), "
          f"M_total={M_total}, "
          f"tiles_m={N_fwd // 256}, tiles_n={K_fwd // 256}")

    # Bit-equivalence check: (16, 4) vs default (4, 4)
    ma, be = correctness_check(grad_out, x, group_offs, 16, 4, B, N_fwd, K_fwd)
    print(f"  correctness ((4,4) vs (16,4)): max_abs={ma:.6e} bit_eq={be}")

    results = []
    for (gm, xcd) in candidates:
        ms_trials = [
            time_one(grad_out, x, grad_b, group_offs, gm, xcd, iters=100)
            for _ in range(5)
        ]
        median_ms = statistics.median(ms_trials)
        tflops = flops / (median_ms * 1e9)
        results.append((gm, xcd, median_ms, tflops))

    default_tflops = next(r[3] for r in results if r[0] == 4 and r[1] == 4)
    results.sort(key=lambda r: r[3], reverse=True)

    print(f"  {'(gm,xcd)':>10s}  {'med_ms':>8s}  {'tflops':>7s}  Δ vs default(4,4)")
    for r in results:
        delta = (r[3] - default_tflops) / default_tflops * 100
        d = " *def*" if (r[0] == 4 and r[1] == 4) else ""
        print(f"  gm={r[0]:>2d} xcd={r[1]:>2d}  {r[2]:7.4f}   "
              f"{r[3]:6.1f}   {delta:+5.2f}%{d}")

    return [(gm, xcd, tf) for (gm, xcd, _, tf) in
            sorted(results, key=lambda r: (-r[3]))]


def main():
    print("=" * 70)
    print("gpt_oss var-K (dB CRR) — B=32 family (m_total=65536, 131072)")
    print("=" * 70)
    print("Warming up HK BF16 runtime on K%128==0 (DSV3-like) shapes "
          "to dodge the cold-start sync-fault bug ...")
    warmup_bf16_runtime()
    print("Warmup OK.")

    s1 = sweep_shape(B=32, M_per_group=2048, N_fwd=5760, K_fwd=2880,
                     label="gpt_oss-GateUP B=32 M=2048 (m_total=65536)")
    s2 = sweep_shape(B=32, M_per_group=4096, N_fwd=5760, K_fwd=2880,
                     label="gpt_oss-GateUP B=32 M=4096 (m_total=131072)")
    s3 = sweep_shape(B=32, M_per_group=2048, N_fwd=2880, K_fwd=2880,
                     label="gpt_oss-Down   B=32 M=2048 (m_total=65536)")
    s4 = sweep_shape(B=32, M_per_group=4096, N_fwd=2880, K_fwd=2880,
                     label="gpt_oss-Down   B=32 M=4096 (m_total=131072)")

    # Aggregate: which cell wins on all 4?
    print("\n" + "=" * 70)
    print("Aggregate per-cell deltas vs default (4,4) — uniform-positive search")
    print("=" * 70)

    def_tf_by_shape = {
        "GateUP-M2048": next(tf for (gm, xcd, tf) in s1 if gm == 4 and xcd == 4),
        "GateUP-M4096": next(tf for (gm, xcd, tf) in s2 if gm == 4 and xcd == 4),
        "Down-M2048":   next(tf for (gm, xcd, tf) in s3 if gm == 4 and xcd == 4),
        "Down-M4096":   next(tf for (gm, xcd, tf) in s4 if gm == 4 and xcd == 4),
    }
    by_shape = {
        "GateUP-M2048": {(gm, xcd): tf for (gm, xcd, tf) in s1},
        "GateUP-M4096": {(gm, xcd): tf for (gm, xcd, tf) in s2},
        "Down-M2048":   {(gm, xcd): tf for (gm, xcd, tf) in s3},
        "Down-M4096":   {(gm, xcd): tf for (gm, xcd, tf) in s4},
    }

    cells = sorted(by_shape["GateUP-M2048"].keys())
    print(f"  {'cell':>10s}  {'GateUP-M2048':>14s}  {'GateUP-M4096':>14s}  "
          f"{'Down-M2048':>12s}  {'Down-M4096':>12s}  {'avg':>7s}  "
          f"{'min':>7s}  {'max':>7s}  uniform")
    for cell in cells:
        gm, xcd = cell
        deltas = []
        for sh in ("GateUP-M2048", "GateUP-M4096", "Down-M2048", "Down-M4096"):
            d = (by_shape[sh][cell] - def_tf_by_shape[sh]) / def_tf_by_shape[sh] * 100
            deltas.append(d)
        avg = sum(deltas) / len(deltas)
        mn = min(deltas)
        mx = max(deltas)
        u = "  +" if mn > 0 else ("  ~" if mn >= -0.3 else "  -")
        marker = "  *def*" if (gm == 4 and xcd == 4) else ""
        print(f"  gm={gm:>2d} xc={xcd:>2d}  {deltas[0]:+13.2f}%  "
              f"{deltas[1]:+13.2f}%  {deltas[2]:+11.2f}%  {deltas[3]:+11.2f}%  "
              f"{avg:+6.2f}%  {mn:+6.2f}%  {mx:+6.2f}%  {u}{marker}")


if __name__ == "__main__":
    main()
