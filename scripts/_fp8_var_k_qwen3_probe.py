#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""R15 probe: var-K kernel config sweep for Qwen3 family (8 shapes).

R39 set (gm=8, xcd=4) for m_total >= 16384 across all families. The
R39 sweep had only ONE Qwen3 shape (Qwen3-Down-B32-M2048). This probe
covers all 8 Qwen3 metric shapes (Down/GateUP × M_per_g={2048,4096}
× B={16,32}) to check whether a tighter Qwen3-specific rule wins.

Result (R15, MI355X / GPU 3, 200-iter × 5-trial p20):
  ALL 8 Qwen3 shapes — current (8,4) is at the plateau. Largest alt
  gains within 8 candidates were:
    Qwen3-Down-B16-M4096      (16,4) +0.13 %  marginal
    Qwen3-GateUP-B16-M4096    (1,4)  +0.32 %
    Qwen3-GateUP-B32-M4096    (1,4)  +0.34 %
  Qwen3-Down family: NO alt cell beats baseline by > 0.13 %.
  Qwen3-GateUP M_per_g=4096: (1,4) gains +0.3 %, but Qwen3-GateUP
    M_per_g=2048 has the same (1,4) at only +0.09 % (noise) — the
    pattern doesn't generalise to a clean rule.
  No actionable rule lift; var-K config sweep for Qwen3 is FALSIFIED.
  See analysis/_notes/round-15-fp8-grouped-var-k-qwen3-falsified.md
  for the full data.

Run:
    python3 scripts/_fp8_var_k_qwen3_probe.py
"""
import os
import sys
sys.path.insert(0, "/workspace/code/Primus-Turbo")
os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")

import torch
import statistics
from primus_turbo.pytorch.kernels.hipkitten import loader as hipkitten

hk = hipkitten.load_fp8()
var_k_fn = hk.module.grouped_variable_k_crr

device = "cuda"


def make_var_k(G, M_per_group, N_fwd, K_fwd):
    """For dB var-K: grad_out [M_total, N_fwd] @ x [M_total, K_fwd] -> grad_b [G, N_fwd, K_fwd].

    Internally the binding's CRR layout means grad_out goes as 'a' (lhs, transposed)
    and x goes as 'b'. Per the GroupedGEMMFP8VariableKHipKittenBackend.execute
    swap-on-trans_c logic, when trans_c=True we pass (b=grad_out_2d as 'a',
    a=x_2d as 'b'). But here we directly call var_k_fn with what it expects.
    Reading existing _fp8_var_k_config_probe.py:
      var_k_fn(grad_out, x, grad_b, sa, sb, group_offs, group_m, num_xcds)
    where 'a' arg = grad_out_2d, 'b' arg = x_2d, 'c' arg = grad_b.
    """
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


def time_one(grad_out, x, grad_b, sa, sb, group_offs, gm, xcd, iters=200):
    for _ in range(20):
        var_k_fn(grad_out, x, grad_b, sa, sb, group_offs, group_m=gm, num_xcds=xcd)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        var_k_fn(grad_out, x, grad_b, sa, sb, group_offs, group_m=gm, num_xcds=xcd)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


# Qwen3 family: all 8 metric shapes (B=16, B=32 × M=2048, M=4096 × Down, GateUP)
# Down: N_fwd=4096, K_fwd=1536
# GateUP: N_fwd=3072, K_fwd=4096
QWEN3_SHAPES = [
    ("Qwen3-Down-B16-M2048",   16, 2048, 4096, 1536),
    ("Qwen3-Down-B16-M4096",   16, 4096, 4096, 1536),
    ("Qwen3-Down-B32-M2048",   32, 2048, 4096, 1536),
    ("Qwen3-Down-B32-M4096",   32, 4096, 4096, 1536),
    ("Qwen3-GateUP-B16-M2048", 16, 2048, 3072, 4096),
    ("Qwen3-GateUP-B16-M4096", 16, 4096, 3072, 4096),
    ("Qwen3-GateUP-B32-M2048", 32, 2048, 3072, 4096),
    ("Qwen3-GateUP-B32-M4096", 32, 4096, 3072, 4096),
]

# Candidates: focus around the current rule (8, 4) and a few promising alts
CANDS = [
    (4, 0),   # ~ binding default
    (8, 4),   # current R39 rule
    (8, 8),
    (4, 4),
    (16, 4),
    (16, 8),
    (2, 4),
    (1, 4),
]


def trial(grad_out, x, grad_b, sa, sb, group_offs, gm, xcd, n_trials=5):
    times = [time_one(grad_out, x, grad_b, sa, sb, group_offs, gm, xcd) for _ in range(n_trials)]
    times.sort()
    # p20 (= median of 5 sorted = 3rd; for 5 elements: 5*0.2 = 1, so index 1)
    return times[len(times)//5], min(times), max(times)


for name, G, Mpg, Nfwd, Kfwd in QWEN3_SHAPES:
    grad_out, x, grad_b, sa, sb, group_offs = make_var_k(G, Mpg, Nfwd, Kfwd)
    M_total = G * Mpg
    flops = 2.0 * M_total * Nfwd * Kfwd
    print(f"\n=== {name}  G={G} M_per_g={Mpg} N={Nfwd} K={Kfwd}  M_total={M_total} ===")

    base_p20, base_min, base_max = trial(grad_out, x, grad_b, sa, sb, group_offs, 8, 4)
    base_tf = flops / (base_p20 * 1e6)
    print(f"  baseline (8,4): {base_p20:.4f} ms  ({base_tf:.1f} TF)  min={base_min:.4f} max={base_max:.4f}")

    print(f"  {'cfg':>10s}  {'p20 (ms)':>10s}  {'TF':>9s}  {'Δ vs (8,4)':>11s}")
    for gm, xcd in CANDS:
        if (gm, xcd) == (8, 4):
            continue
        p20, _, _ = trial(grad_out, x, grad_b, sa, sb, group_offs, gm, xcd)
        tf = flops / (p20 * 1e6)
        delta = (base_p20 - p20) / base_p20 * 100  # +ve = faster than baseline
        print(f"  {(gm, xcd)!s:>10s}  {p20:>10.4f}  {tf:>9.1f}  {delta:>+10.2f}%")
