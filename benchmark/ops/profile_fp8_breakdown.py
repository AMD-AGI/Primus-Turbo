"""
Profile FP8 grouped GEMM forward with a path that matches GroupedGemmFP8TensorFunc.

The script breaks forward into:
  - quant A
  - quant B
  - grouped GEMM only
  - end-to-end grouped_gemm_fp8

It also exposes a few upper-bound experiments:
  - reuse_b_quant
  - known_scale_quant
  - delayed_scaling_stub
"""

import argparse
from contextlib import contextmanager
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, Format, ScalingGranularity
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import grouped_gemm_fp8_impl
from primus_turbo.triton.quantization.quant_fp8_tensorwise import (
    _BufferCache,
    _select_config,
    _use_reduce_and_quant_fuse,
    quantize_fp8_tensorwise as _quant_fp8_tw,
)
from config import (
    expand_grouped_gemm_cases_by_balance,
    filter_grouped_gemm_test_cases,
    gen_grouped_gemm_group_lens,
    get_grouped_gemm_preset_cases,
)

config = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
FP8_BACKEND = BackendType.TRITON

DEFAULT_SHAPES = [
    {"Case": "DeepSeek-V3-GateUP", "B": 8, "M": 512, "N": 4096, "K": 7168, "balance": True},
    {"Case": "DeepSeek-V3-GateUP", "B": 8, "M": 4096, "N": 4096, "K": 7168, "balance": True},
    {"Case": "DeepSeek-V3-GateUP", "B": 8, "M": 16384, "N": 4096, "K": 7168, "balance": True},
    {"Case": "DeepSeek-V3-Down", "B": 8, "M": 4096, "N": 7168, "K": 2048, "balance": True},
    {"Case": "Kimi-K2-GateUP", "B": 48, "M": 512, "N": 4096, "K": 7168, "balance": True},
    {"Case": "Kimi-K2-GateUP", "B": 48, "M": 4096, "N": 4096, "K": 7168, "balance": True},
    {"Case": "Kimi-K2-GateUP", "B": 48, "M": 16384, "N": 4096, "K": 7168, "balance": True},
    {"Case": "MoE-1T-GateUP", "B": 28, "M": 4096, "N": 3840, "K": 8192, "balance": True},
    {"Case": "Grok-2-GateUP", "B": 1, "M": 8192, "N": 32768, "K": 8192, "balance": True},
]


def bench_fn(fn, warmup=10, rep=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep * 1000.0


@contextmanager
def _temporary_grouped_gemm_backend(backend: BackendType | None):
    prev_backend = GlobalBackendManager._grouped_gemm_backend
    GlobalBackendManager.set_grouped_gemm_backend(backend)
    try:
        yield
    finally:
        GlobalBackendManager.set_grouped_gemm_backend(prev_backend)


@contextmanager
def _temporary_env_flag(name: str, enabled: bool):
    prev = os.environ.get(name)
    if enabled:
        os.environ[name] = "1"
    else:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev


def _format_balance(balance: bool) -> str:
    return "balanced" if balance else "imbalanced"


def _quant_path(x: torch.Tensor):
    n = x.numel()
    block_size, _, _ = _select_config(n)
    num_tiles = (n + block_size - 1) // block_size
    path = "2k-fused" if _use_reduce_and_quant_fuse(n, num_tiles) else "3k-separate"
    return path, num_tiles, block_size


def _run_fp8_e2e(a, b, group_lens, reuse_b_quant=False):
    with _temporary_grouped_gemm_backend(FP8_BACKEND), _temporary_env_flag(
        "PRIMUS_TURBO_FP8_TW_REUSE_B", reuse_b_quant
    ):
        return turbo.ops.grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=config)


def _run_fp8_gemm_only(a_fp8, b_fp8, a_si, b_si, group_lens, group_offs):
    with _temporary_grouped_gemm_backend(FP8_BACKEND):
        return grouped_gemm_fp8_impl(
            a_fp8,
            b_fp8,
            a_si,
            b_si,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=True,
            out_dtype=torch.bfloat16,
            granularity=config.granularity.value,
            num_cu=None,
            default_backend=FP8_BACKEND.value,
        )


def _load_shapes(args):
    if args.preset == "default":
        shapes = [dict(shape) for shape in DEFAULT_SHAPES]
    else:
        shapes = get_grouped_gemm_preset_cases(args.preset)

    shapes = expand_grouped_gemm_cases_by_balance(shapes, args.balance)
    shapes = filter_grouped_gemm_test_cases(
        shapes,
        case_names=args.case,
        b_values=args.b_values,
        m_values=args.m_values,
        n_values=args.n_values,
        k_values=args.k_values,
    )
    if not shapes:
        raise ValueError("No shapes selected for FP8 breakdown profiling.")
    return shapes


def profile_shape(shape, args):
    case = shape["Case"]
    B, M, N, K = shape["B"], shape["M"], shape["N"], shape["K"]
    balance = shape.get("balance", True)

    device = "cuda"
    a = torch.randn(B * M, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(B, N, K, dtype=torch.bfloat16, device=device)
    group_lens = gen_grouped_gemm_group_lens(B, M, balance=balance).to(device)
    group_offs = torch.zeros(B + 1, dtype=torch.int64, device=device)
    group_offs[1:] = group_lens.cumsum(0)

    a_dtype = torch.float8_e4m3fn
    b_dtype = torch.float8_e4m3fn
    cache_a = _BufferCache()
    cache_b = _BufferCache()
    cache_a_known = _BufferCache()
    cache_b_known = _BufferCache()

    path_a, tiles_a, _ = _quant_path(a)
    path_b, tiles_b, _ = _quant_path(b)

    def quant_a():
        return _quant_fp8_tw(a, a_dtype, buf_cache=cache_a)

    def quant_b():
        return _quant_fp8_tw(b, b_dtype, buf_cache=cache_b)

    a_fp8, a_si = quant_a()
    b_fp8, b_si = quant_b()

    def gemm_only():
        return _run_fp8_gemm_only(a_fp8, b_fp8, a_si, b_si, group_lens, group_offs)

    out = gemm_only()
    torch.cuda.synchronize()

    ms_qa = bench_fn(quant_a, warmup=args.warmup, rep=args.rep)
    ms_qb = bench_fn(quant_b, warmup=args.warmup, rep=args.rep)
    ms_gemm = bench_fn(gemm_only, warmup=args.warmup, rep=args.rep)
    ms_e2e = bench_fn(
        lambda: _run_fp8_e2e(a, b, group_lens),
        warmup=args.warmup,
        rep=args.rep,
    )
    ms_bf16 = bench_fn(
        lambda: turbo.ops.grouped_gemm(a, b, group_lens, trans_b=True),
        warmup=args.warmup,
        rep=args.rep,
    )

    total_flops = 2 * B * M * N * K
    tflops_fp8 = total_flops / (ms_e2e * 1e-3) / 1e12
    tflops_bf16 = total_flops / (ms_bf16 * 1e-3) / 1e12

    quant_a_pct = ms_qa / ms_e2e * 100.0
    quant_b_pct = ms_qb / ms_e2e * 100.0
    gemm_pct = ms_gemm / ms_e2e * 100.0
    ratio = ms_bf16 / ms_e2e

    result = {
        "case": case,
        "balance": _format_balance(balance),
        "B": B,
        "M": M,
        "N": N,
        "K": K,
        "path_a": path_a,
        "tiles_a": tiles_a,
        "path_b": path_b,
        "tiles_b": tiles_b,
        "ms_qa": ms_qa,
        "ms_qb": ms_qb,
        "ms_gemm": ms_gemm,
        "ms_e2e": ms_e2e,
        "ms_bf16": ms_bf16,
        "quant_a_pct": quant_a_pct,
        "quant_b_pct": quant_b_pct,
        "gemm_pct": gemm_pct,
        "tflops_fp8": tflops_fp8,
        "tflops_bf16": tflops_bf16,
        "ratio": ratio,
    }

    if args.reuse_b_quant:
        result["ms_e2e_reuse_b"] = bench_fn(
            lambda: _run_fp8_e2e(a, b, group_lens, reuse_b_quant=True),
            warmup=args.warmup,
            rep=args.rep,
        )

    if args.known_scale_quant or args.delayed_scaling_stub:
        a_scale = a.abs().max().reshape(1)
        b_scale = b.abs().max().reshape(1)

        def quant_a_known():
            return _quant_fp8_tw(a, a_dtype, scale=a_scale, buf_cache=cache_a_known)

        def quant_b_known():
            return _quant_fp8_tw(b, b_dtype, scale=b_scale, buf_cache=cache_b_known)

        if args.known_scale_quant:
            result["ms_qa_known"] = bench_fn(quant_a_known, warmup=args.warmup, rep=args.rep)
            result["ms_qb_known"] = bench_fn(quant_b_known, warmup=args.warmup, rep=args.rep)

        if args.delayed_scaling_stub:
            def fwd_delayed_stub():
                a_fp8_local, a_si_local = quant_a_known()
                b_fp8_local, b_si_local = quant_b_known()
                return _run_fp8_gemm_only(
                    a_fp8_local, b_fp8_local, a_si_local, b_si_local, group_lens, group_offs
                )

            result["ms_e2e_delayed"] = bench_fn(fwd_delayed_stub, warmup=args.warmup, rep=args.rep)

    del a, b, a_fp8, b_fp8, out
    torch.cuda.empty_cache()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Profile FP8 grouped GEMM breakdown on representative shapes")
    parser.add_argument(
        "--preset",
        type=str,
        choices=["default", "smoke", "tuning", "acceptance"],
        default="default",
        help="Shape preset to profile (default: built-in representative set)",
    )
    parser.add_argument("--case", nargs="*", default=None, help="Filter case names")
    parser.add_argument("--B", dest="b_values", nargs="*", type=int, default=None, help="Filter B values")
    parser.add_argument("--M", dest="m_values", nargs="*", type=int, default=None, help="Filter M values")
    parser.add_argument("--N", dest="n_values", nargs="*", type=int, default=None, help="Filter N values")
    parser.add_argument("--K", dest="k_values", nargs="*", type=int, default=None, help="Filter K values")
    parser.add_argument(
        "--balance",
        type=str,
        choices=["balanced", "imbalanced", "both"],
        default="balanced",
        help="Group length distribution to profile (default: balanced)",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per measurement")
    parser.add_argument("--rep", type=int, default=50, help="Timed iterations per measurement")
    parser.add_argument(
        "--reuse-b-quant",
        action="store_true",
        help="Measure an upper bound where B quantization is reused across iterations",
    )
    parser.add_argument(
        "--known-scale-quant",
        action="store_true",
        help="Measure quant upper bounds using precomputed scales",
    )
    parser.add_argument(
        "--delayed-scaling-stub",
        action="store_true",
        help="Measure an end-to-end upper bound using cached scales for both A and B",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    shapes = _load_shapes(args)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Backend: TRITON")
    print(
        f"Selected {len(shapes)} shapes | preset={args.preset} | balance={args.balance} | "
        f"reuse_b_quant={args.reuse_b_quant} | known_scale_quant={args.known_scale_quant} | "
        f"delayed_scaling_stub={args.delayed_scaling_stub}"
    )
    print()
    hdr = (
        f"{'Case':>22s} | {'Bal':>4s} | {'B':>3s} {'M':>6s} {'N':>6s} {'K':>5s} | "
        f"{'PA':>10s} {'TA':>6s} {'PB':>10s} {'TB':>6s} | "
        f"{'QA':>6s} {'QB':>6s} {'GEMM':>6s} {'E2E':>6s} {'BF16':>6s} | "
        f"{'A%':>5s} {'B%':>5s} {'G%':>5s} | {'Ratio':>5s}"
    )
    print(hdr)
    print("-" * len(hdr))

    for shape in shapes:
        try:
            r = profile_shape(shape, args)
            print(
                f"{r['case']:>22s} | {r['balance']:>4s} | {r['B']:>3d} {r['M']:>6d} {r['N']:>6d} {r['K']:>5d} | "
                f"{r['path_a']:>10s} {r['tiles_a']:6d} {r['path_b']:>10s} {r['tiles_b']:6d} | "
                f"{r['ms_qa']:6.3f} {r['ms_qb']:6.3f} {r['ms_gemm']:6.3f} {r['ms_e2e']:6.3f} {r['ms_bf16']:6.3f} | "
                f"{r['quant_a_pct']:5.1f} {r['quant_b_pct']:5.1f} {r['gemm_pct']:5.1f} | "
                f"{r['ratio']:5.2f}x"
            )
            if args.reuse_b_quant or args.known_scale_quant or args.delayed_scaling_stub:
                extras = []
                if "ms_e2e_reuse_b" in r:
                    extras.append(f"reuseB={r['ms_e2e_reuse_b']:.3f}ms")
                if "ms_qa_known" in r:
                    extras.append(f"QA_known={r['ms_qa_known']:.3f}ms")
                if "ms_qb_known" in r:
                    extras.append(f"QB_known={r['ms_qb_known']:.3f}ms")
                if "ms_e2e_delayed" in r:
                    extras.append(f"delayedE2E={r['ms_e2e_delayed']:.3f}ms")
                print(" " * 8 + "upper-bound: " + " | ".join(extras))
        except Exception as e:
            print(
                f"{shape['Case']:>22s} | {_format_balance(shape.get('balance', True)):>4s} | "
                f"{shape['B']:>3d} {shape['M']:>6d} {shape['N']:>6d} {shape['K']:>5d} | ERROR: {e}"
            )


if __name__ == "__main__":
    main()
