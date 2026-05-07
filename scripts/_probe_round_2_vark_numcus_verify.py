#!/usr/bin/env python3
"""Round-2 var-K NUM_CUS env-hook bit-equivalence verifier.

Verifies that ``TK_VARK_NUM_CUS=N`` for N ∈ {32, 64, 128, 256} produces
identical FP8 var-K wgrad output (max_abs_diff = 0.0) on Down-B4-M2048
wgrad. The new env hook in HK ``dispatch_grouped_var_k_fp8`` only changes
the persistent grid stride / chiplet swizzle range, not the math —
should be bit-equivalent at every legal slot count.

Each (slot value) runs in a fresh subprocess because the HK side caches
the env value via ``static const int slots``.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

BASE_SLOTS = 256
TEST_SLOTS = [32, 64, 96, 128, 160, 192, 256]

CHILD = r'''
import os, sys, torch
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

B, M, N, K = 4, 2048, 2880, 2880
torch.manual_seed(7)
g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
g_offs = grouped_gemm_compute_offs(g_lens)
a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
grad = torch.randn((B * M, N), dtype=torch.bfloat16, device="cuda")
a_col, a_s = quantize_fp8(a, torch.float8_e4m3fn, ScalingGranularity.TENSORWISE, axis=-2)
g_col, g_s = quantize_fp8(grad, torch.float8_e4m3fn, ScalingGranularity.TENSORWISE, axis=-2)

with hk_ratio.force_grouped_gemm_backend(BackendType.HIPKITTEN, PrecisionType.FP8):
    out = grouped_gemm_fp8_variable_k_impl(
        a_col, g_col, a_s, g_s, g_lens, g_offs,
        trans_a=True, trans_b=False, trans_c=True,
        out_dtype=torch.bfloat16,
        granularity=ScalingGranularity.TENSORWISE.value, num_cu=None,
        default_backend=BackendType.HIPKITTEN.value,
    )
torch.cuda.synchronize()

# Save flat float32 view to tmpfile for parent compare
out_path = sys.argv[1]
torch.save(out.float().cpu(), out_path)
print(f"DONE slots={os.environ.get('TK_VARK_NUM_CUS','default')} shape={list(out.shape)} dtype={out.dtype} sum={out.float().sum().item():.6f}", flush=True)
'''


def _run_with_slots(slots, out_path):
    env = os.environ.copy()
    env["TK_VARK_NUM_CUS"] = str(slots)
    env["PRIMUS_TURBO_HIPKITTEN_PATH"] = "/workspace/code/HipKittens"
    env["METRIC_SKIP_IDLE_CHECK"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, out_path],
        env=env, capture_output=True, text=True, timeout=180,
    )
    print(f"[verify-slots={slots}] stdout={proc.stdout.strip()}", flush=True)
    if proc.returncode != 0:
        print(f"[verify-slots={slots}] STDERR:\n{proc.stderr[-1500:]}", flush=True)


def main():
    import torch
    with tempfile.TemporaryDirectory() as td:
        # baseline
        base_path = os.path.join(td, "out_base.pt")
        _run_with_slots(BASE_SLOTS, base_path)
        base = torch.load(base_path)
        print(f"[verify] baseline (slots={BASE_SLOTS}): "
              f"shape={list(base.shape)} sum={base.sum().item():.6f}", flush=True)
        for s in TEST_SLOTS:
            if s == BASE_SLOTS:
                continue
            cand_path = os.path.join(td, f"out_{s}.pt")
            _run_with_slots(s, cand_path)
            cand = torch.load(cand_path)
            diff = (cand - base).abs().max().item()
            mismatch = (cand != base).sum().item()
            total = cand.numel()
            print(f"[verify] slots={s:>3}: "
                  f"max_abs_diff={diff:.6e}  mismatch_elems={mismatch} / {total}",
                  flush=True)


if __name__ == "__main__":
    main()
