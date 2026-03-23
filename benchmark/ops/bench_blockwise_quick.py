"""Quick blockwise FP8 grouped GEMM benchmark for iteration.

Tests a representative subset of shapes to quickly measure optimization impact.
Usage: HIP_VISIBLE_DEVICES=2 python bench_blockwise_quick.py [--bf16] [--blockwise]
"""

import argparse
import os
import time

import torch
import torch.utils.benchmark as benchmark

os.environ.setdefault("PRIMUS_TURBO_GROUPED_GEMM_BACKEND", "TRITON")

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)
from config import compute_snr, gen_grouped_gemm_group_lens, get_platform_info, grouped_gemm_ref

BLOCKWISE_CONFIG = Float8QuantConfig(
    format=Format.E4M3,
    granularity=ScalingGranularity.BLOCKWISE,
    block_size=128,
)

REPRESENTATIVE_CASES = [
    {"name": "DSv3-GateUP-ep32", "B": 8, "M": 512, "N": 4096, "K": 7168},
    {"name": "DSv3-GateUP-ep32", "B": 8, "M": 2048, "N": 4096, "K": 7168},
    {"name": "DSv3-GateUP-ep32", "B": 8, "M": 8192, "N": 4096, "K": 7168},
    {"name": "DSv3-Down-ep32", "B": 8, "M": 2048, "N": 7168, "K": 2048},
    {"name": "DSv3-Down-ep32", "B": 8, "M": 8192, "N": 7168, "K": 2048},
    {"name": "DSv3-GateUP-ep16", "B": 16, "M": 2048, "N": 4096, "K": 7168},
    {"name": "DSv3-GateUP-ep16", "B": 16, "M": 8192, "N": 4096, "K": 7168},
    {"name": "DSv3-GateUP-ep8", "B": 32, "M": 2048, "N": 4096, "K": 7168},
    {"name": "DSv3-GateUP-ep8", "B": 32, "M": 8192, "N": 4096, "K": 7168},
    {"name": "Mixtral-GateUP", "B": 1, "M": 4096, "N": 28672, "K": 4096},
]

WARMUP = 10
ITERS = 50


def profile_fp8_blockwise(B, M, N, K, dtype=torch.bfloat16):
    device = "cuda"
    group_lens = gen_grouped_gemm_group_lens(B, M, balance=True).to(device)
    a = torch.randn((B * M, K), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((B, N, K), dtype=dtype, device=device, requires_grad=True)

    out = turbo.ops.grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=BLOCKWISE_CONFIG)
    grad_out = torch.randn_like(out)

    out_ref = grouped_gemm_ref(a.detach(), b.detach(),
                                torch.tensor([M]*B, dtype=torch.int64, device=device), trans_b=True)
    snr = compute_snr(out_ref, out.detach())

    fwd_func = lambda: turbo.ops.grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=BLOCKWISE_CONFIG)
    def fwd_bwd_func():
        o = turbo.ops.grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=BLOCKWISE_CONFIG)
        o.backward(grad_out)

    for _ in range(WARMUP):
        fwd_bwd_func()
    torch.cuda.synchronize()

    fwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func})
    fwd_bwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": fwd_bwd_func})
    fwd_m = fwd_timer.timeit(ITERS)
    fwd_bwd_m = fwd_bwd_timer.timeit(ITERS)

    fwd_ms = fwd_m.mean * 1e3
    bwd_ms = (fwd_bwd_m.mean - fwd_m.mean) * 1e3
    fwd_flops = 2 * B * M * N * K
    bwd_flops = 2 * fwd_flops
    fwd_tflops = fwd_flops / (fwd_ms * 1e-3) / 1e12
    bwd_tflops = bwd_flops / (bwd_ms * 1e-3) / 1e12

    return fwd_ms, fwd_tflops, bwd_ms, bwd_tflops, snr


def profile_bf16(B, M, N, K, dtype=torch.bfloat16):
    device = "cuda"
    group_lens = gen_grouped_gemm_group_lens(B, M, balance=True).to(device)
    x = torch.randn((B * M, K), dtype=dtype, device=device, requires_grad=True)
    w = torch.randn((B, N, K), dtype=dtype, device=device, requires_grad=True)

    out = turbo.ops.grouped_gemm(x, w, group_lens, trans_b=True)
    grad_out = torch.randn_like(out)

    fwd_func = lambda: turbo.ops.grouped_gemm(x, w, group_lens, trans_b=True)
    def fwd_bwd_func():
        o = turbo.ops.grouped_gemm(x, w, group_lens, trans_b=True)
        o.backward(grad_out)

    for _ in range(WARMUP):
        fwd_bwd_func()
    torch.cuda.synchronize()

    fwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func})
    fwd_bwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": fwd_bwd_func})
    fwd_m = fwd_timer.timeit(ITERS)
    fwd_bwd_m = fwd_bwd_timer.timeit(ITERS)

    fwd_ms = fwd_m.mean * 1e3
    bwd_ms = (fwd_bwd_m.mean - fwd_m.mean) * 1e3
    fwd_flops = 2 * B * M * N * K
    bwd_flops = 2 * fwd_flops
    fwd_tflops = fwd_flops / (fwd_ms * 1e-3) / 1e12
    bwd_tflops = bwd_flops / (bwd_ms * 1e-3) / 1e12

    return fwd_ms, fwd_tflops, bwd_ms, bwd_tflops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", action="store_true", help="Run BF16 baseline")
    parser.add_argument("--blockwise", action="store_true", help="Run blockwise FP8")
    parser.add_argument("--both", action="store_true", help="Run both and compare")
    args = parser.parse_args()

    if not any([args.bf16, args.blockwise, args.both]):
        args.both = True

    platform, gpu = get_platform_info()
    print(f"Platform: {platform}, GPU: {gpu}")
    print(f"Backend: {os.environ.get('PRIMUS_TURBO_GROUPED_GEMM_BACKEND', 'default')}")
    print()

    bf16_results = {}
    bw_results = {}

    for case in REPRESENTATIVE_CASES:
        B, M, N, K = case["B"], case["M"], case["N"], case["K"]
        tag = f"{case['name']}_B{B}_M{M}"
        print(f"{'='*70}")
        print(f"  {tag}  B={B} M={M} N={N} K={K}")
        print(f"{'='*70}")

        if args.bf16 or args.both:
            try:
                fwd_ms, fwd_tf, bwd_ms, bwd_tf = profile_bf16(B, M, N, K)
                bf16_results[tag] = (fwd_tf, bwd_tf)
                print(f"  BF16      Fwd: {fwd_ms:7.2f}ms  {fwd_tf:8.1f} TFLOPS | Bwd: {bwd_ms:7.2f}ms  {bwd_tf:8.1f} TFLOPS")
            except Exception as e:
                print(f"  BF16      ERROR: {e}")
                bf16_results[tag] = (0, 0)

        if args.blockwise or args.both:
            try:
                fwd_ms, fwd_tf, bwd_ms, bwd_tf, snr = profile_fp8_blockwise(B, M, N, K)
                bw_results[tag] = (fwd_tf, bwd_tf)
                print(f"  Blockwise Fwd: {fwd_ms:7.2f}ms  {fwd_tf:8.1f} TFLOPS | Bwd: {bwd_ms:7.2f}ms  {bwd_tf:8.1f} TFLOPS  SNR={snr:.1f}dB")
            except Exception as e:
                print(f"  Blockwise ERROR: {e}")
                import traceback; traceback.print_exc()
                bw_results[tag] = (0, 0)

    if args.both and bf16_results and bw_results:
        print(f"\n{'='*70}")
        print(f"  SUMMARY: Blockwise / BF16 ratio")
        print(f"{'='*70}")
        print(f"  {'Case':<40} {'Fwd Ratio':>10} {'Bwd Ratio':>10}")
        fwd_ratios, bwd_ratios = [], []
        for tag in bf16_results:
            if tag in bw_results:
                bf_fwd, bf_bwd = bf16_results[tag]
                bw_fwd, bw_bwd = bw_results[tag]
                fr = bw_fwd / bf_fwd if bf_fwd > 0 else 0
                br = bw_bwd / bf_bwd if bf_bwd > 0 else 0
                fwd_ratios.append(fr)
                bwd_ratios.append(br)
                print(f"  {tag:<40} {fr:>9.2f}x {br:>9.2f}x")
        if fwd_ratios:
            print(f"  {'AVG':<40} {sum(fwd_ratios)/len(fwd_ratios):>9.2f}x {sum(bwd_ratios)/len(bwd_ratios):>9.2f}x")
            print(f"\n  Target: >= 1.50x")


if __name__ == "__main__":
    main()
