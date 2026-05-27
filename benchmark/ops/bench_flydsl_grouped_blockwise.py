"""Bench FlyDSL persistent grouped blockwise FP8 GEMM vs the Triton reference.

Three timings per shape:
  - Triton ref grouped blockwise GEMM (current fwd backend)
  - FlyDSL kernel only (B pre-shuffled once outside the timed call)
  - FlyDSL e2e incl. per-call shuffle_b_batched (matches today's launcher path
    where callers do not cache pre-shuffled weights)
"""
from __future__ import annotations

import torch

from primus_turbo.flydsl.grouped_gemm import (
    grouped_gemm_fp8_blockwise_flydsl_kernel,
    shuffle_b_batched,
    compile_blockscale_grouped_gemm_persistent,
)
from primus_turbo.flydsl.grouped_gemm.launcher import _select_super_m, _select_tile
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import float8_e4m3
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_fp8_impl,
)
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    quant_fp8_blockwise_for_weight_impl,
    quant_fp8_blockwise_segment_m_row_col_impl,
)

import flydsl.compiler as flyc

DEVICE, DTYPE = "cuda", torch.bfloat16

# (N, K) per model. GPT-OSS K=2880 is skipped (not divisible by tile_k=128).
MODELS = {
    "LFM2-8B-A1B":     (2048, 1792),
    "Qwen3-235B-A22B": (4096, 1536),
    "DeepSeek-V3":     (7168, 2048),
}
B_LIST = [4, 16]
M_LIST = [2048, 4096]


def time_fn(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    trim = times[iters // 5: -iters // 5]
    return sum(trim) / len(trim)


def make_inputs(B, M, N, K):
    M_total = B * M
    group_lens = torch.full((B,), M, device=DEVICE, dtype=torch.int64)
    group_offs = torch.zeros(B + 1, dtype=torch.int64, device=DEVICE)
    group_offs[1:] = torch.cumsum(group_lens, 0)
    torch.manual_seed(0)
    a = torch.randn((M_total, K), dtype=DTYPE, device=DEVICE) * 0.1
    b = torch.randn((B, N, K), dtype=DTYPE, device=DEVICE) * 0.1
    return a, b, group_lens, group_offs, M_total


def run_one(B, M, N, K):
    a, b, group_lens, group_offs, M_total = make_inputs(B, M, N, K)
    a_fp8, _, a_sc, _, _, _ = quant_fp8_blockwise_segment_m_row_col_impl(
        a, float8_e4m3, 128, group_lens, group_offs,
    )
    b_fp8, b_sc = quant_fp8_blockwise_for_weight_impl(b, float8_e4m3, block_size=128)

    fn_triton = lambda: grouped_gemm_fp8_impl(
        a_fp8, b_fp8, a_sc, b_sc, group_lens, group_offs,
        trans_a=False, trans_b=True, out_dtype=DTYPE,
        granularity=3, num_cu=None, default_backend=BackendType.TRITON.value,
    )
    ms_triton = time_fn(fn_triton)

    fn_fly_with_shuf = lambda: grouped_gemm_fp8_blockwise_flydsl_kernel(
        a_fp8, b_fp8, a_sc, b_sc, group_offs, out_dtype=DTYPE,
    )
    ms_fly_with = time_fn(fn_fly_with_shuf)

    # Kernel-only: pre-shuffle weight once, bypass launcher's per-call shuffle.
    b_shuf = shuffle_b_batched(b_fp8)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    out = torch.empty((M_total, N), dtype=DTYPE, device=DEVICE)
    a_sc_flat = a_sc.view(-1)
    b_sc_flat = b_sc.view(-1)
    stream = torch.cuda.current_stream()
    tm, tn, tk = _select_tile(M_total, N, K)
    sm = _select_super_m(N, M_total)
    exe = compile_blockscale_grouped_gemm_persistent(
        Mt=M_total, N=N, K=K, G=B,
        tile_m=tm, tile_n=tn, tile_k=tk, num_sms=num_sms, super_m=sm,
        scale_block_k=128, out_dtype="bf16",
    )
    compiled = flyc.compile(
        exe, out, a_fp8, b_shuf, a_sc_flat, b_sc_flat, group_offs, M_total, N, stream,
    )
    fn_fly_kernel = lambda: compiled(
        out, a_fp8, b_shuf, a_sc_flat, b_sc_flat, group_offs, M_total, N, stream,
    )
    ms_fly_kernel = time_fn(fn_fly_kernel)

    flops = 2 * M_total * N * K
    return {
        "ms_triton": ms_triton,
        "ms_fly_kernel": ms_fly_kernel,
        "ms_fly_with_shuf": ms_fly_with,
        "tf_triton": flops / (ms_triton * 1e-3) / 1e12,
        "tf_fly_kernel": flops / (ms_fly_kernel * 1e-3) / 1e12,
        "tf_fly_with_shuf": flops / (ms_fly_with * 1e-3) / 1e12,
        "kernel_speedup": ms_triton / ms_fly_kernel,
        "e2e_speedup": ms_triton / ms_fly_with,
    }


def main():
    print(f"{'shape':35s} {'tri (ms)':>9s} {'fly k (ms)':>10s} {'fly+s (ms)':>10s} "
          f"{'tri (TF)':>9s} {'fly k (TF)':>10s} {'fly+s (TF)':>10s} "
          f"{'k/tri':>7s} {'e2e/tri':>8s}")
    rows = []
    for model, (N, K) in MODELS.items():
        for B in B_LIST:
            for M in M_LIST:
                r = run_one(B, M, N, K)
                tag = f"{model[:14]} B={B:<2d} M={M}"
                print(f"{tag:35s} "
                      f"{r['ms_triton']:>9.3f} {r['ms_fly_kernel']:>10.3f} {r['ms_fly_with_shuf']:>10.3f} "
                      f"{r['tf_triton']:>9.1f} {r['tf_fly_kernel']:>10.1f} {r['tf_fly_with_shuf']:>10.1f} "
                      f"{r['kernel_speedup']:>7.2f} {r['e2e_speedup']:>8.2f}")
                rows.append((model, B, M, N, K, r))

    kspeed = [r["kernel_speedup"] for *_, r in rows]
    espeed = [r["e2e_speedup"] for *_, r in rows]
    print(f"\nKernel-only speedup vs Triton: avg={sum(kspeed)/len(kspeed):.2f}x  "
          f"min={min(kspeed):.2f}x  max={max(kspeed):.2f}x  "
          f"wins={sum(1 for x in kspeed if x > 1.0)}/{len(kspeed)}")
    print(f"E2E (incl. shuffle) speedup:   avg={sum(espeed)/len(espeed):.2f}x  "
          f"min={min(espeed):.2f}x  max={max(espeed):.2f}x  "
          f"wins={sum(1 for x in espeed if x > 1.0)}/{len(espeed)}")


if __name__ == "__main__":
    main()
