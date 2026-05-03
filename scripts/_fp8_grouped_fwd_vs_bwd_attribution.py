#!/usr/bin/env python3
"""R17 attribution probe: split FP8 grouped GEMM HK-vs-Triton ratio into
forward and backward components on the bottom-ratio metric shapes.

Why this exists
---------------
The metric ``scripts/_metric_grouped_fused_wall.py`` reports a single
fwd+bwd ratio per shape; rounds R12 / R15 / R16 had to guess whether a
given shape's gap was forward-kernel-bound or backward-kernel-bound.
This probe times forward and backward separately and prints both
ratios, identifying which kernel family (forward ``grouped_rcr_kernel``
vs backward ``grouped_rrr`` + ``grouped_variable_k_crr``) drags the
ratio.

Methodology
-----------
* For each of the 5 lowest-ratio metric shapes (per the R17 baseline
  metric @ HEAD ``46d56aa``), instantiate a fresh ``a, b`` pair with
  ``requires_grad=True`` and run 10 warmup fwd+bwd cycles, then 30
  timed cycles where forward and backward are bracketed by separate
  ``cuda.Event`` pairs.
* Take p20 (matches metric convention).
* Repeat for both BackendType.HIPKITTEN and BackendType.TRITON.
* Report per-shape ``fwd_x = trt_fwd / hk_fwd`` and ``bwd_x = trt_bwd /
  hk_bwd``; ``total_x`` matches the metric's per-shape ratio.

Usage
-----
$ python3 scripts/_fp8_grouped_fwd_vs_bwd_attribution.py

R17 (HEAD 46d56aa) findings
---------------------------
+-------------------------------------+-------+-------+----------+
| shape                               | fwd_x | bwd_x | gap_bwd-fwd |
+-------------------------------------+-------+-------+-------------+
| Qwen3-Down-B16-M2048   (K=1536)     | 1.199 | 1.286 |     +0.087  |
| Qwen3-Down-B16-M4096   (K=1536)     | 1.182 | 1.313 |     +0.131  |
| Qwen3-GateUP-B16-M2048 (K=4096)     | 1.203 | 1.286 |     +0.083  |
| gpt_oss-Down-B32-M2048 (K=2880)     | 1.118 | 1.347 |     +0.229  |
| DSV3-Down-B16-M2048    (K=2048)     | 1.232 | 1.336 |     +0.104  |
+-------------------------------------+-------+-------+-------------+

Reading: every bottom-ratio shape has fwd_x < bwd_x. Forward kernel
(``grouped_rcr_kernel`` for K-aligned shapes; transpose + RCR for
K-misaligned gpt_oss) is the universal bottleneck.

Implications for next-round work
--------------------------------
The remaining gap to ratio=1.35 is concentrated in the FORWARD kernel,
not the backward (var-K dB or RRR dA) kernel. Backward is already at
1.28-1.35 on these shapes; var-K config tuning (R15) and RRR config
tuning (R44) found no further actionable rules. Forward kernel
internals are the only remaining lever:

1. **BLOCK_K=64 template specialization** for shallow-K shapes
   (K∈{1536, 2880, 4096}). HK FP8 ``grouped_rcr_kernel`` currently
   uses ``K_BLOCK=128`` exclusively — for Qwen3-Down K=1536 that's
   only 12 K-iters per output tile, leaving the LDS double-buffer
   under-fed. Doubling K-iters to 24 (BLOCK_K=64) might keep LDS warm.
   Multi-round HK source work
   (``analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp``).

2. **gpt_oss K=2880 H4 reroute** is also forward-bound (fwd_x=1.118,
   the worst of all 5 shapes). The current path does a Triton
   ``fp8_transpose_3d`` + RCR. Both transpose + RCR happen in the
   forward critical path. Worth profiling whether the transpose
   itself or the RCR-on-transposed-K=2880 is the slower piece.

The Python-side dispatch / cache levers (R7-R16) are exhausted; the
``per-call host overhead`` measured at R16 is now ~50-100 ns / call
on both forward and backward dispatch paths, well below kernel
wall (HK fwd is 0.30-0.65 ms / call on these shapes).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
sys.path.insert(0, "/workspace/code/Primus-Turbo")

import torch  # noqa: E402

import primus_turbo.pytorch as turbo  # noqa: E402
from primus_turbo.pytorch.core.backend import (  # noqa: E402
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)
from primus_turbo.pytorch.core.low_precision import (  # noqa: E402
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)


def _make_inputs(B, M, N, K):
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    group_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    cfg = Float8QuantConfig(
        format=Format.E4M3,
        granularity=ScalingGranularity.TENSORWISE,
        fuse_act_quant=True,
    )
    return a, b, group_lens, cfg


def time_fwd_bwd(backend, B, M, N, K, iters=30, warmup=10):
    a, b, group_lens, cfg = _make_inputs(B, M, N, K)
    GlobalBackendManager.reset()
    GlobalBackendManager.set_grouped_gemm_backend(backend, PrecisionType.FP8)
    out_warm = turbo.ops.grouped_gemm_fp8(
        a, b, group_lens, trans_b=True, config=cfg
    )
    grad_out = torch.randn_like(out_warm)
    a.grad = None
    b.grad = None
    out_warm.backward(grad_out)
    for _ in range(warmup):
        a.grad = None
        b.grad = None
        out = turbo.ops.grouped_gemm_fp8(
            a, b, group_lens, trans_b=True, config=cfg
        )
        out.backward(grad_out)
    torch.cuda.synchronize()

    fwd_times: list[float] = []
    bwd_times: list[float] = []
    se = torch.cuda.Event(enable_timing=True)
    me = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        a.grad = None
        b.grad = None
        torch.cuda.synchronize()
        se.record()
        out = turbo.ops.grouped_gemm_fp8(
            a, b, group_lens, trans_b=True, config=cfg
        )
        me.record()
        out.backward(grad_out)
        ee.record()
        torch.cuda.synchronize()
        fwd_times.append(se.elapsed_time(me))
        bwd_times.append(me.elapsed_time(ee))
    fwd_times.sort()
    bwd_times.sort()
    GlobalBackendManager.reset()
    return fwd_times[len(fwd_times) // 5], bwd_times[len(bwd_times) // 5]


def main():
    shapes = [
        ("Qwen3-Down-B16-M2048   (K=1536)", 16, 2048, 4096, 1536),
        ("Qwen3-Down-B16-M4096   (K=1536)", 16, 4096, 4096, 1536),
        ("Qwen3-GateUP-B16-M2048 (K=4096)", 16, 2048, 3072, 4096),
        ("gpt_oss-Down-B32-M2048 (K=2880)", 32, 2048, 2880, 2880),
        ("DSV3-Down-B16-M2048    (K=2048)", 16, 2048, 7168, 2048),
    ]

    aw = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
    bw = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
    for _ in range(50):
        cw = aw @ bw
    torch.cuda.synchronize()
    del aw, bw, cw

    print("Per-shape fwd/bwd attribution (HK vs TRT, p20 ms over 30 iters):")
    print()
    print(
        f"  {'shape':<33} {'hk_fwd':>8} {'trt_fwd':>8} {'fwd_x':>6} "
        f"{'hk_bwd':>8} {'trt_bwd':>8} {'bwd_x':>6} {'total_x':>8}"
    )
    print("  " + "-" * 95)
    for label, B, M, N, K in shapes:
        hk_fwd, hk_bwd = time_fwd_bwd(BackendType.HIPKITTEN, B, M, N, K)
        trt_fwd, trt_bwd = time_fwd_bwd(BackendType.TRITON, B, M, N, K)
        fwd_speedup = trt_fwd / hk_fwd if hk_fwd > 0 else 0
        bwd_speedup = trt_bwd / hk_bwd if hk_bwd > 0 else 0
        total_speedup = (
            (trt_fwd + trt_bwd) / (hk_fwd + hk_bwd)
            if (hk_fwd + hk_bwd) > 0
            else 0
        )
        print(
            f"  {label:<33} {hk_fwd:>8.3f} {trt_fwd:>8.3f} {fwd_speedup:>6.3f} "
            f"{hk_bwd:>8.3f} {trt_bwd:>8.3f} {bwd_speedup:>6.3f} "
            f"{total_speedup:>8.3f}"
        )
    print()
    print(
        "  Reading: total_x is the metric ratio. fwd_x / bwd_x show where HK "
        "loses. fwd_x < bwd_x ⇒ forward kernel is the bottleneck."
    )


if __name__ == "__main__":
    main()
