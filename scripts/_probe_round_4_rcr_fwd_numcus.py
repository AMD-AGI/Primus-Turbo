#!/usr/bin/env python3
"""Round-4 (gpt_oss FP8 kernel-only): RCR fwd NUM_CUS sweep.

Sister probe to R2's var-K wgrad sweep, applied to the file-scope
``grouped_rcr_kernel`` (the FP8 fwd persistent kernel). HK build now
exposes ``TK_RCR_NUM_CUS`` env (mirrors R2's ``TK_VARK_NUM_CUS``); the
kernel body uses ``gridDim.x`` for both the chiplet swizzle range and
the persistent loop stride so any ``slots ∈ [1, 256]`` is bit-equiv.

Anchors (sparse fwd + counter):
  Down_B4_M2048    fwd (m_total=8192,  N=2880, K=2880) — anchor
                       tiles_m=8, tiles_n=11+1pad → ~(8*12)*4 = 384 tiles
                       384/256 = 1.5 wave-steps/CU (sparser than var-K
                       Down-B4 at 1.89; expected to benefit from slots
                       reduction).
  GateUP_B32_M4096 fwd (m_total=131072, N=5760, K=2880) — counter
                       tiles_m=16, tiles_n=22+1pad → (16*23)*32 = 11776
                       11776/256 = 46 wave-steps/CU (saturated; expected
                       to LOSE from slots reduction, as a regression
                       guard).
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from typing import Dict, List

SWEEP_VALUES = [128, 160, 192, 208, 224, 240, 256]

ANCHORS = [
    {"label": "Down_B4_M2048_fwd",
     "B": 4, "M": 2048, "N": 2880, "K": 2880},
    {"label": "Down_B4_M4096_fwd",
     "B": 4, "M": 4096, "N": 2880, "K": 2880},
    {"label": "GateUP_B32_M4096_fwd",
     "B": 32, "M": 4096, "N": 5760, "K": 2880},
]

CHILD_SOURCE = r'''
import os, statistics, sys, torch
os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
sys.path.insert(0, "/workspace/code/Primus-Turbo/scripts")
import _metric_hk_ratio as hk_ratio  # noqa
import primus_turbo.pytorch as turbo  # noqa
from primus_turbo.pytorch.core.backend import BackendType, PrecisionType  # noqa
from primus_turbo.pytorch.core.low_precision import ScalingGranularity  # noqa
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (  # noqa
    grouped_gemm_compute_offs, grouped_gemm_fp8_impl,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8


def _make(B, M, N, K, seed):
    torch.manual_seed(seed)
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = grouped_gemm_compute_offs(g_lens)
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    a_fp8, sa = quantize_fp8(a, torch.float8_e4m3fn,
                             ScalingGranularity.TENSORWISE, axis=-2)
    b_fp8, sb = quantize_fp8(b, torch.float8_e4m3fn,
                             ScalingGranularity.TENSORWISE, axis=-2)
    return lambda: grouped_gemm_fp8_impl(
        a_fp8, b_fp8, sa, sb, g_lens, g_offs,
        trans_a=False, trans_b=True,
        out_dtype=torch.bfloat16,
        granularity=ScalingGranularity.TENSORWISE.value, num_cu=None,
        default_backend=BackendType.HIPKITTEN.value,
    )


def time_kernel(fn, n_iters=250, n_trials=7):
    durs = []
    for _ in range(n_trials):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(n_iters):
            fn()
        ev1.record()
        torch.cuda.synchronize()
        durs.append(ev0.elapsed_time(ev1) / n_iters)
    durs.sort()
    return durs[max(0, int(len(durs) * 0.2))], durs


def main():
    B, M, N, K = (int(x) for x in sys.argv[1:5])
    medians = []
    flops = 2.0 * B * M * N * K
    for s in (42, 137, 2024):
        with hk_ratio.force_grouped_gemm_backend(
            BackendType.HIPKITTEN, PrecisionType.FP8
        ):
            fn = _make(B, M, N, K, s)
            ms, _ = time_kernel(fn)
            medians.append(ms)
    med = statistics.median(medians)
    print(f"RESULT: median_ms={med:.4f} tflops={flops/med/1e9:.1f} per-seed={medians}",
          flush=True)


if __name__ == "__main__":
    main()
'''


def _run_one(slots: int, B: int, M: int, N: int, K: int) -> dict:
    env = os.environ.copy()
    env.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
    env["TK_RCR_NUM_CUS"] = str(slots)
    env["METRIC_SKIP_IDLE_CHECK"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", CHILD_SOURCE,
         str(B), str(M), str(N), str(K)],
        env=env, capture_output=True, text=True, timeout=180,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    median_ms = tflops = None
    for line in out.splitlines():
        if line.startswith("RESULT:"):
            for tok in line.split():
                if tok.startswith("median_ms="):
                    median_ms = float(tok.split("=", 1)[1])
                elif tok.startswith("tflops="):
                    tflops = float(tok.split("=", 1)[1])
    return {"slots": slots, "median_ms": median_ms, "tflops": tflops,
            "stderr_tail": out[-300:] if median_ms is None else ""}


def main() -> int:
    all_rows: Dict[str, List[dict]] = {}
    for sh in ANCHORS:
        print(f"\n[round4-rcr-numcus] sweep {sh['label']} "
              f"(B={sh['B']} M={sh['M']} N={sh['N']} K={sh['K']})", flush=True)
        rows = []
        for slots in SWEEP_VALUES:
            r = _run_one(slots, sh["B"], sh["M"], sh["N"], sh["K"])
            rows.append(r)
            print(f"  slots={slots:>3}  TFLOPS={r['tflops'] or 0:>7.1f}  "
                  f"median_ms={r['median_ms'] or 0:.4f}", flush=True)
        base = next((r["tflops"] for r in rows
                     if r["slots"] == 256 and r["tflops"]), None)
        if base:
            print(f"  baseline @ slots=256: {base:.1f} T", flush=True)
            for r in rows:
                if r["tflops"] is None:
                    continue
                d = (r["tflops"] - base) / base * 100.0
                print(f"    slots={r['slots']:>3}  {r['tflops']:>7.1f}  "
                      f"{d:>+7.2f}%", flush=True)
        all_rows[sh["label"]] = rows

    out_path = "/tmp/round_4_rcr_numcus_sweep.json"
    with open(out_path, "w") as fh:
        json.dump(all_rows, fh, indent=2)
    print(f"\n[round4-rcr-numcus] saved {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
