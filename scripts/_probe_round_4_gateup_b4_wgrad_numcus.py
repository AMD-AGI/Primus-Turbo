#!/usr/bin/env python3
"""Round-4 (gpt_oss FP8 kernel-only ceiling): GateUP-B4 wgrad NUM_CUS sweep.

Per R3 commit message follow-up plan: GateUP-B4 wgrad has tile-step
density between R2's anchor (Down-B4: 1.89 wave-steps/CU, slots=192
wins +6%) and counter (GateUP-B32-M4096: 30.25 wave-steps/CU,
slots=192 loses -17%). Per-group output [N_fwd, K_fwd] = [5760, 2880]
-> tiles_n=22, tiles_k=11 = 242 tile-steps/group * 4 groups =
968 tile-steps, 968/256 = 3.78 wave-steps/CU.

Two metric shapes:
  GateUP_B4_M2048 wgrad (m_total=8192,  N=5760, K=2880)
  GateUP_B4_M4096 wgrad (m_total=16384, N=5760, K=2880)

Methodology: same R2 protocol (250-iter * 7-trial p20 * 3 seeds *
kernel-only direct call, subprocess-per-slot due to HK static cache).
Sweep slots in {128, 160, 192, 224, 256, 288} -- 288 will be clamped
to NUM_CUS=256 by the HK env hook (negative-control, should equal 256).
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from typing import Dict, List

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
SWEEP_VALUES = [128, 160, 192, 208, 224, 240, 256]

ANCHORS = [
    {"label": "GateUP_B4_M2048_wgrad",
     "B": 4, "M": 2048, "N": 5760, "K": 2880},
    {"label": "GateUP_B4_M4096_wgrad",
     "B": 4, "M": 4096, "N": 5760, "K": 2880},
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
    grouped_gemm_compute_offs, grouped_gemm_fp8_variable_k_impl,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8


def _make(B, M, N, K, seed):
    torch.manual_seed(seed)
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = grouped_gemm_compute_offs(g_lens)
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    grad = torch.randn((B * M, N), dtype=torch.bfloat16, device="cuda")
    a_col, a_s = quantize_fp8(a, torch.float8_e4m3fn,
                              ScalingGranularity.TENSORWISE, axis=-2)
    g_col, g_s = quantize_fp8(grad, torch.float8_e4m3fn,
                              ScalingGranularity.TENSORWISE, axis=-2)
    return lambda: grouped_gemm_fp8_variable_k_impl(
        a_col, g_col, a_s, g_s, g_lens, g_offs,
        trans_a=True, trans_b=False, trans_c=True,
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
    env["TK_VARK_NUM_CUS"] = str(slots)
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
        print(f"\n[round4-gateup-b4-numcus] sweep {sh['label']} "
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

    out_path = "/tmp/round_4_gateup_b4_numcus_sweep.json"
    with open(out_path, "w") as fh:
        json.dump(all_rows, fh, indent=2)
    print(f"\n[round4-gateup-b4-numcus] saved {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
