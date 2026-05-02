#!/usr/bin/env python3
"""R39 probe: sweep group_m / num_xcds for var_k backward kernel.

Rationale
---------
Primus-side ``var_k_fn(...)`` at grouped_gemm_fp8_impl.py:647 always uses
binding defaults (group_m=4, num_xcds=0→kernel fallback 8) — forward path
passes cfg-tuned values but var_k has NEVER been tuned.

Before wiring any new dispatcher rule, confirm whether the knob actually
affects wall-time by more than bench noise (~20% seen in R38 on this
shared MI355X).

Methodology
-----------
- Single shape: gpt_oss-Down B=4 M=2048 (worst-bwd TFLOPS in R38 baseline
  = 241). Variable-K kernel gets grad_out[8192, 2880] @ x[8192, 2880] →
  grad_b[4, 2880, 2880]. Grid = 12x12 = 144 tiles/group * 4 groups = 576
  tiles over 256 CUs = 2.25 tiles/CU.
- For each (group_m, num_xcds) in a small candidate set, time 100 iters
  using torch.cuda.Event. Report median + min/max across 3 repeats.
- Bit-identical correctness is GUARANTEED by kernel construction (these
  are pure tile-scheduling knobs).
"""

import os
import sys

sys.path.insert(0, "/workspace/code/Primus-Turbo")

import torch
import statistics

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
os.environ.setdefault("PRIMUS_TURBO_GROUPED_GEMM_BACKEND", "HIPKITTEN")

from primus_turbo.pytorch.kernels.hipkitten import loader as hipkitten  # noqa: E402

hk = hipkitten.load_fp8()
var_k_fn = hk.module.grouped_variable_k_crr

device = "cuda"


def make_tensors(G, M_per_group, N_fwd, K_fwd):
    M_total = G * M_per_group
    torch.manual_seed(42)
    grad_out_bf16 = torch.randn(M_total, N_fwd, dtype=torch.bfloat16, device=device)
    x_bf16 = torch.randn(M_total, K_fwd, dtype=torch.bfloat16, device=device)
    grad_out_fp8 = grad_out_bf16.to(torch.float8_e4m3fnuz)
    x_fp8 = x_bf16.to(torch.float8_e4m3fnuz)
    grad_b = torch.empty(G, N_fwd, K_fwd, dtype=torch.bfloat16, device=device)
    group_offs = torch.tensor(
        [i * M_per_group for i in range(G + 1)], dtype=torch.int64, device=device
    )
    sa = torch.tensor(1.0, device=device, dtype=torch.float32)
    sb = torch.tensor(1.0, device=device, dtype=torch.float32)
    return grad_out_fp8, x_fp8, grad_b, sa, sb, group_offs


def time_one(grad_out, x, grad_b, sa, sb, group_offs, group_m, num_xcds, iters=100):
    # Warmup
    for _ in range(10):
        var_k_fn(grad_out, x, grad_b, sa, sb, group_offs,
                 group_m=group_m, num_xcds=num_xcds)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        var_k_fn(grad_out, x, grad_b, sa, sb, group_offs,
                 group_m=group_m, num_xcds=num_xcds)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def sweep_shape(G, M_per_group, N_fwd, K_fwd, label):
    M_total = G * M_per_group
    grad_out, x, grad_b, sa, sb, group_offs = make_tensors(
        G, M_per_group, N_fwd, K_fwd
    )
    flops = 2 * M_total * N_fwd * K_fwd

    candidates = [
        (1, 0),  # gm=1, xcd=kernel default (8)
        (2, 0),
        (4, 0),  # CURRENT DEFAULT
        (8, 0),
        (4, 2),
        (4, 4),
        (4, 8),
        (2, 2),
        (2, 4),
        (8, 4),
        (1, 4),
    ]

    print(f"\n=== Shape {label}: G={G} M_per_g={M_per_group} N_fwd={N_fwd} K_fwd={K_fwd} ===")
    print(f"FLOPs per call = {flops / 1e9:.2f} GFLOPS")

    results = []
    for (gm, xcd) in candidates:
        ms_trials = []
        for _ in range(5):
            ms_trials.append(time_one(grad_out, x, grad_b, sa, sb, group_offs,
                                       group_m=gm, num_xcds=xcd, iters=100))
        median_ms = statistics.median(ms_trials)
        min_ms = min(ms_trials)
        tflops = flops / (median_ms * 1e9)
        results.append((gm, xcd, median_ms, min_ms, tflops))

    results.sort(key=lambda r: r[4], reverse=True)
    print(f"  {'(gm, xcd)':>12s}  {'median_ms':>10s}  {'tflops':>8s}  Δ vs default")
    default_tflops = next(r[4] for r in results if r[0] == 4 and r[1] == 0)
    for r in results[:6]:
        delta = (r[4] - default_tflops) / default_tflops * 100
        is_default = " *current default*" if (r[0] == 4 and r[1] == 0) else ""
        print(f"  gm={r[0]:>2d} xcd={r[1]:>2d}   {r[2]:8.4f}   {r[4]:6.1f}    {delta:+5.2f}%{is_default}")


def main():
    # Threshold-check shapes (both B=4 sizes to nail down m_total cutoff)
    sweep_shape(G=4, M_per_group=4096, N_fwd=2880, K_fwd=2880, label="gpt_oss-Down  B=4 M=4096 (m_total=16384, R38=1039)")
    sweep_shape(G=4, M_per_group=4096, N_fwd=5760, K_fwd=2880, label="gpt_oss-GateUP B=4 M=4096 (m_total=16384, R38=1048)")
    # Worst-bwd shapes from R38 bench — Down subset (all showed high bwd/fwd ratio)
    sweep_shape(G=4, M_per_group=2048, N_fwd=2880, K_fwd=2880, label="gpt_oss-Down  B=4 M=2048 (R38=241 TFLOPS)")
    sweep_shape(G=32, M_per_group=2048, N_fwd=4096, K_fwd=1536, label="Qwen3-Down   B=32 M=2048 (R38=695)")
    sweep_shape(G=32, M_per_group=2048, N_fwd=2880, K_fwd=2880, label="gpt_oss-Down B=32 M=2048 (R38=716)")
    sweep_shape(G=16, M_per_group=2048, N_fwd=7168, K_fwd=2048, label="DSV3-Down    B=16 M=2048 (R38=814)")
    # GateUP subset — check if the winning config extrapolates or differs
    sweep_shape(G=32, M_per_group=2048, N_fwd=5760, K_fwd=2880, label="gpt_oss-GateUP B=32 M=2048 (R38=988)")
    sweep_shape(G=16, M_per_group=4096, N_fwd=4096, K_fwd=7168, label="DSV3-GateUP  B=16 M=4096 (R38=1362)")
    sweep_shape(G=16, M_per_group=4096, N_fwd=3072, K_fwd=4096, label="Qwen3-GateUP B=16 M=4096 (R38=1317)")
    # Largest work-size — highest absolute TFLOPS (where config should matter most)
    sweep_shape(G=32, M_per_group=4096, N_fwd=4096, K_fwd=7168, label="DSV3-GateUP  B=32 M=4096 (R38=1472)")
    sweep_shape(G=32, M_per_group=4096, N_fwd=5760, K_fwd=2880, label="gpt_oss-GateUP B=32 M=4096 (R38=1280)")


if __name__ == "__main__":
    main()
