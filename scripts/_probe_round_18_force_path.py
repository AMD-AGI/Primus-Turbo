#!/usr/bin/env python3
"""R18 part 2 — verify R5 fast-path firing under metric's force_grouped_gemm_backend.

The metric pins HIPKITTEN via ``force_grouped_gemm_backend(HIPKITTEN, FP8)``.
This sets ``GlobalBackendManager._grouped_gemm_backend = {FP8: HIPKITTEN}``,
so ``get_grouped_gemm_backend(FP8) is None`` is FALSE. R5 fast path's
condition ``user_backend is None`` excludes this case → metric falls
through to the slow dispatcher path with kwargs dict + 2 env lookups.

Probe times grouped_gemm_fp8_impl WITHOUT force (R5 fast path) vs
WITH force (R5 slow path) and computes the gap.
"""
import os
import sys
import time

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
sys.path.insert(0, "/workspace/code/Primus-Turbo")
sys.path.insert(0, "/workspace/code/Primus-Turbo/scripts")

import torch  # noqa: E402

import primus_turbo.pytorch as turbo  # noqa: F401  E402
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager, PrecisionType  # noqa: E402
from primus_turbo.pytorch.core.low_precision import ScalingGranularity  # noqa: E402
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (  # noqa: E402
    grouped_gemm_compute_offs,
    grouped_gemm_fp8_impl,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8  # noqa: E402

import _metric_hk_ratio as hkr  # noqa: E402

_FP8_DTYPE = torch.float8_e4m3fn
_GRAN = ScalingGranularity.TENSORWISE


def _bench(fn, warmup=50, iters=2000):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    se = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        se.record()
        fn()
        ee.record()
        torch.cuda.synchronize()
        times.append(se.elapsed_time(ee))
    times.sort()
    return times[len(times) // 5]


def make_call(B=4, M=2048, N=2880, K=2880):
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = grouped_gemm_compute_offs(g_lens)
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    a_fp8, a_s = quantize_fp8(a, _FP8_DTYPE, _GRAN)
    b_fp8, b_s = quantize_fp8(b, _FP8_DTYPE, _GRAN)
    flops = 2.0 * B * M * N * K

    def _call():
        grouped_gemm_fp8_impl(
            a_fp8, b_fp8, a_s, b_s, g_lens, g_offs,
            trans_a=False, trans_b=True, out_dtype=torch.bfloat16,
            granularity=_GRAN.value, num_cu=None,
            default_backend=BackendType.HIPKITTEN.value,
        )
    return _call, flops


def main():
    print(f"[probe] R18 part 2 — R5 fast-path firing under force_grouped_gemm_backend")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")
    print()

    fn, flops = make_call()

    # Path A: NO force — user_backend=None — R5 fast path takes effect.
    GlobalBackendManager.reset()
    assert GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP8) is None
    t_no_force = _bench(fn)
    print(f"  [no force]    grouped_gemm_fp8_impl: {t_no_force*1e3:.2f} µs  {flops/(t_no_force*1e9):.1f} TF")

    # Path B: WITH force — user_backend=HIPKITTEN — R5 fast path SKIPPED.
    with hkr.force_grouped_gemm_backend(BackendType.HIPKITTEN, PrecisionType.FP8):
        assert GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP8) == BackendType.HIPKITTEN
        t_with_force = _bench(fn)
    print(f"  [with force]  grouped_gemm_fp8_impl: {t_with_force*1e3:.2f} µs  {flops/(t_with_force*1e9):.1f} TF")

    print()
    print(f"  Gap (force - no_force): {(t_with_force - t_no_force)*1e6:.2f} µs")
    print(f"  This is the cost of the R5 slow path (kwargs dict + 2 env lookups + dispatcher protocol).")


if __name__ == "__main__":
    main()
