#!/usr/bin/env python3
"""R42 probe: sweep group_m / num_xcds for grouped_rrr (FP8 dA backward kernel).

Rationale
---------
Primus-side FP8 RRR dispatch at grouped_gemm_fp8_impl.py:470-495 passes
``cfg.group_m`` + ``cfg.num_xcds`` from ``select_default_config``. Tracing
the FP8 branch in config.py (line 849+), **ALL ``layout == "rrr"`` rules
are gated behind ``if dtype == "bf16"``** (line 210); the FP8 section has
only ``layout == "rcr"`` rules. FP8 RRR thus ALWAYS falls through to the
default ``(group_m=4, num_xcds=None→0→kernel fallback 8)`` for every
grouped FP8 dA call.

This is the R39 playbook applied to the dA path:
  R39 fixed var_k (dB) which had the same gap (always defaults).
  R42 now probes RRR (dA) which has the same gap.

Coverage
--------
The dA path via grouped_rrr fires when ``trans_b=False`` AND ``K_RRR %
128 == 0`` AND ``N_RRR % 256 == 0`` (otherwise the R13/R18 H4 reroute
transposes to RCR). In the 24 MoE metric suite:
  * DSV3-GateUP (N_fwd=4096, K_fwd=7168) — dA RRR (k=4096, n=7168) ✓
  * DSV3-Down   (N_fwd=7168, K_fwd=2048) — dA RRR (k=7168, n=2048) ✓
  * Qwen3-GateUP(N_fwd=3072, K_fwd=4096) — dA RRR (k=3072, n=4096) ✓
  * Qwen3-Down  (N_fwd=4096, K_fwd=1536) — dA RRR (k=4096, n=1536) ✓
  * gpt_oss-*   — K_RRR=5760/2880 misaligned → reroutes via transpose, NOT grouped_rrr

So 16/24 shapes (all DSV3 + all Qwen3) hit grouped_rrr.

Methodology
-----------
For each (group_m, num_xcds) in an 11-candidate set, time 100 iters via
torch.cuda.Event. Report median + min across 5 repeats. Also run a
bit-identical correctness check against the current default (4, 0).
"""

import os
import sys
import statistics

sys.path.insert(0, "/workspace/code/Primus-Turbo")

import torch  # noqa: E402

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
os.environ.setdefault("PRIMUS_TURBO_GROUPED_GEMM_BACKEND", "HIPKITTEN")

from primus_turbo.pytorch.kernels.hipkitten import loader as hipkitten  # noqa: E402

hk = hipkitten.load_fp8()
rrr_fn = hk.module.grouped_rrr
rrr_dscale_fn = getattr(hk.module, "grouped_rrr_dscale", None)

device = "cuda"


def make_tensors(G, M_per_group, N_rrr, K_rrr):
    """Build inputs for grouped_rrr: C[M,N] = A[M,K] @ B_g[K,N] per group.

    For FP8 dA: K_rrr = N_fwd (reduction), N_rrr = K_fwd (output col).
    """
    M_total = G * M_per_group
    torch.manual_seed(42)
    a_bf16 = torch.randn(M_total, K_rrr, dtype=torch.bfloat16, device=device)
    # B is 3D grouped [G, K, N] for RRR (trans_b=False semantics)
    b_bf16 = torch.randn(G, K_rrr, N_rrr, dtype=torch.bfloat16, device=device)
    a_fp8 = a_bf16.to(torch.float8_e4m3fnuz)
    b_fp8 = b_bf16.to(torch.float8_e4m3fnuz)
    c = torch.empty(M_total, N_rrr, dtype=torch.bfloat16, device=device)
    group_offs = torch.tensor(
        [i * M_per_group for i in range(G + 1)], dtype=torch.int64, device=device
    )
    sa_d = torch.tensor(1.0, device=device, dtype=torch.float32)
    sb_d = torch.tensor(1.0, device=device, dtype=torch.float32)
    return a_fp8, b_fp8, c, sa_d, sb_d, group_offs


def time_one(a, b, c, sa_d, sb_d, group_offs, group_m, num_xcds, M_per_group, iters=100):
    fn = rrr_dscale_fn if rrr_dscale_fn is not None else rrr_fn
    # Warmup
    for _ in range(10):
        fn(a, b, c, sa_d, sb_d, group_offs,
           group_m=group_m, m_per_group=M_per_group, num_xcds=num_xcds)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(a, b, c, sa_d, sb_d, group_offs,
           group_m=group_m, m_per_group=M_per_group, num_xcds=num_xcds)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def correctness_check(a, b, c, sa_d, sb_d, group_offs, group_m, num_xcds, M_per_group):
    """Confirm bit-identical output between default (4, 0) and (gm, xcd)."""
    fn = rrr_dscale_fn if rrr_dscale_fn is not None else rrr_fn
    c_default = torch.zeros_like(c)
    fn(a, b, c_default, sa_d, sb_d, group_offs,
       group_m=4, m_per_group=M_per_group, num_xcds=0)
    c_test = torch.zeros_like(c)
    fn(a, b, c_test, sa_d, sb_d, group_offs,
       group_m=group_m, m_per_group=M_per_group, num_xcds=num_xcds)
    max_abs = (c_default.float() - c_test.float()).abs().max().item()
    bit_eq = torch.equal(c_default.view(torch.int16), c_test.view(torch.int16))
    return max_abs, bit_eq


def sweep_shape(G, M_per_group, N_fwd, K_fwd, label):
    """N_fwd and K_fwd are the FORWARD dims; the probe maps to RRR space."""
    # For dA: K_rrr = N_fwd (reduction), N_rrr = K_fwd (output column)
    K_rrr = N_fwd
    N_rrr = K_fwd
    M_total = G * M_per_group
    a, b, c, sa_d, sb_d, group_offs = make_tensors(G, M_per_group, N_rrr, K_rrr)
    flops = 2 * M_total * N_rrr * K_rrr

    candidates = [
        (1, 0),  (2, 0),  (4, 0),  (8, 0), (16, 0),  # xcd = default (8)
        (4, 2),  (4, 4),  (2, 4),  (8, 4), (16, 4),  # try xcd=4 variants
        (1, 4),  # skinny-M x wide-xcd
    ]

    print(f"\n=== {label} ===")
    print(f"  fwd=(M_per_g={M_per_group}, N={N_fwd}, K={K_fwd}), "
          f"RRR=(k={K_rrr}, n={N_rrr}), M_total={M_total}")
    print(f"  FLOPs per call = {flops / 1e9:.2f} GFLOPS, "
          f"tiles_m_per_g={M_per_group // 256}, tiles_n={N_rrr // 256}")

    # Correctness check on default (4,0) vs a non-default
    ma, be = correctness_check(a, b, c, sa_d, sb_d, group_offs, 8, 4, M_per_group)
    print(f"  correctness (default vs (8,4)):  max_abs={ma:.6e}  bit_eq={be}")

    results = []
    for (gm, xcd) in candidates:
        ms_trials = []
        for _ in range(5):
            ms_trials.append(time_one(a, b, c, sa_d, sb_d, group_offs,
                                      group_m=gm, num_xcds=xcd,
                                      M_per_group=M_per_group, iters=100))
        median_ms = statistics.median(ms_trials)
        min_ms = min(ms_trials)
        tflops = flops / (median_ms * 1e9)
        results.append((gm, xcd, median_ms, min_ms, tflops))

    default_tflops = next(r[4] for r in results if r[0] == 4 and r[1] == 0)
    results.sort(key=lambda r: r[4], reverse=True)

    print(f"  {'(gm, xcd)':>14s}  {'median_ms':>10s}  {'tflops':>8s}  Δ vs default")
    for r in results:
        delta = (r[4] - default_tflops) / default_tflops * 100
        is_default = " *current default*" if (r[0] == 4 and r[1] == 0) else ""
        print(f"  gm={r[0]:>2d} xcd={r[1]:>2d}   {r[2]:8.4f}   "
              f"{r[4]:6.1f}    {delta:+5.2f}%{is_default}")


def main():
    # All 16 shapes that hit grouped_rrr (DSV3 all 8 + Qwen3 all 8).
    # gpt_oss is excluded (K_RRR misaligned → transpose reroute to RCR).

    # DSV3-Down (tiles_n=8) — narrow-N
    sweep_shape(G=16, M_per_group=2048, N_fwd=7168, K_fwd=2048,
                label="DSV3-Down B=16 M=2048 (m_total=32768)")
    sweep_shape(G=32, M_per_group=2048, N_fwd=7168, K_fwd=2048,
                label="DSV3-Down B=32 M=2048 (m_total=65536)")
    sweep_shape(G=32, M_per_group=4096, N_fwd=7168, K_fwd=2048,
                label="DSV3-Down B=32 M=4096 (m_total=131072)")

    # DSV3-GateUP (tiles_n=28) — wide-N
    sweep_shape(G=16, M_per_group=2048, N_fwd=4096, K_fwd=7168,
                label="DSV3-GateUP B=16 M=2048 (m_total=32768)")
    sweep_shape(G=32, M_per_group=2048, N_fwd=4096, K_fwd=7168,
                label="DSV3-GateUP B=32 M=2048 (m_total=65536)")
    sweep_shape(G=32, M_per_group=4096, N_fwd=4096, K_fwd=7168,
                label="DSV3-GateUP B=32 M=4096 (m_total=131072)")

    # Qwen3-Down (tiles_n=6) — narrowest-N
    sweep_shape(G=16, M_per_group=2048, N_fwd=4096, K_fwd=1536,
                label="Qwen3-Down B=16 M=2048 (m_total=32768)")
    sweep_shape(G=32, M_per_group=2048, N_fwd=4096, K_fwd=1536,
                label="Qwen3-Down B=32 M=2048 (m_total=65536)")
    sweep_shape(G=32, M_per_group=4096, N_fwd=4096, K_fwd=1536,
                label="Qwen3-Down B=32 M=4096 (m_total=131072)")

    # Qwen3-GateUP (tiles_n=16) — mid-N
    sweep_shape(G=16, M_per_group=2048, N_fwd=3072, K_fwd=4096,
                label="Qwen3-GateUP B=16 M=2048 (m_total=32768)")
    sweep_shape(G=32, M_per_group=2048, N_fwd=3072, K_fwd=4096,
                label="Qwen3-GateUP B=32 M=2048 (m_total=65536)")
    sweep_shape(G=32, M_per_group=4096, N_fwd=3072, K_fwd=4096,
                label="Qwen3-GateUP B=32 M=4096 (m_total=131072)")


if __name__ == "__main__":
    main()
