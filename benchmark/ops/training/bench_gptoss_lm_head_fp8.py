###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Validate the GPT-OSS-20B LM head with tensorwise E4M3 HipBLASLt GEMMs.

The default performance shape is taken from the target training trace:

    [32768, 2880] @ [128256, 2880].T -> [32768, 128256]

Correctness uses the same wide vocabulary and hidden dimension with a smaller
token count so a BF16 forward/backward reference does not dominate runtime.
The performance loop includes operand quantization and backward quantization;
it therefore measures the public autograd operation rather than a prequantized
kernel in isolation.
"""

import argparse
import json
from dataclasses import asdict, dataclass

import torch

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)


@dataclass
class Timing:
    forward_ms: float
    backward_ms: float
    iteration_ms: float
    peak_memory_gib: float


def _snr(ref: torch.Tensor, actual: torch.Tensor) -> float:
    ref = ref.float()
    actual = actual.float()
    signal = torch.sum(ref * ref, dtype=torch.float64)
    error = torch.sum((ref - actual) ** 2, dtype=torch.float64)
    return float(10.0 * torch.log10(signal / (error + 1.0e-12)))


def _inputs(m: int, n: int, k: int):
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
    return a, b, grad


def _fp8_config() -> Float8QuantConfig:
    return Float8QuantConfig(
        format=Format.E4M3,
        granularity=ScalingGranularity.TENSORWISE,
    )


def check_correctness(m: int, n: int, k: int, threshold_db: float) -> dict[str, float]:
    a, b, grad = _inputs(m, n, k)

    ref = a @ b.T
    ref.backward(grad)
    ref_out = ref.detach()
    ref_da = a.grad.detach()
    ref_db = b.grad.detach()

    a.grad = None
    b.grad = None
    out = turbo.ops.gemm_fp8(a, b, trans_b=True, config=_fp8_config())
    out.backward(grad)

    result = {
        "forward_snr_db": _snr(ref_out, out.detach()),
        "dgrad_snr_db": _snr(ref_da, a.grad.detach()),
        "wgrad_snr_db": _snr(ref_db, b.grad.detach()),
    }
    failed = {name: value for name, value in result.items() if value < threshold_db}
    if failed:
        raise AssertionError(f"FP8 correctness threshold {threshold_db} dB failed: {failed}")
    return result


def _elapsed_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _time_forward(fn) -> tuple[torch.Tensor, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    end.synchronize()
    return out, float(start.elapsed_time(end))


def _time_backward(out: torch.Tensor, grad: torch.Tensor) -> float:
    return _elapsed_ms(lambda: out.backward(grad))


def benchmark(m: int, n: int, k: int, warmup: int, iterations: int, fp8: bool) -> Timing:
    a, b, grad = _inputs(m, n, k)

    def forward():
        if fp8:
            return turbo.ops.gemm_fp8(a, b, trans_b=True, config=_fp8_config())
        return turbo.ops.gemm(a, b, trans_b=True)

    def one_iteration():
        a.grad = None
        b.grad = None
        forward().backward(grad)

    for _ in range(warmup):
        one_iteration()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    forward_ms = 0.0
    backward_ms = 0.0
    iteration_ms = 0.0
    for _ in range(iterations):
        # Forward timing includes dynamic quantization for the FP8 case.
        out, elapsed = _time_forward(forward)
        forward_ms += elapsed
        a.grad = None
        b.grad = None
        backward_ms += _time_backward(out, grad)
        iteration_ms += _elapsed_ms(one_iteration)

    return Timing(
        forward_ms=forward_ms / iterations,
        backward_ms=backward_ms / iterations,
        iteration_ms=iteration_ms / iterations,
        peak_memory_gib=torch.cuda.max_memory_allocated() / (1024**3),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=32768)
    parser.add_argument("--n", type=int, default=128256)
    parser.add_argument("--k", type=int, default=2880)
    parser.add_argument("--correctness-m", type=int, default=256)
    parser.add_argument("--snr-threshold-db", type=float, default=25.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-bf16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A ROCm/CUDA GPU is required")

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    GlobalBackendManager.set_gemm_backend(BackendType.HIPBLASLT)
    GlobalBackendManager.set_auto_tune(False)

    report = {
        "device": torch.cuda.get_device_name(0),
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "backend": "hipblaslt",
        "fp8_recipe": "tensorwise_e4m3",
    }
    try:
        if not args.skip_correctness:
            report["correctness"] = check_correctness(
                args.correctness_m,
                args.n,
                args.k,
                args.snr_threshold_db,
            )
        if not args.skip_bf16:
            report["bf16"] = asdict(
                benchmark(args.m, args.n, args.k, args.warmup, args.iterations, fp8=False)
            )
        report["fp8"] = asdict(benchmark(args.m, args.n, args.k, args.warmup, args.iterations, fp8=True))
        if "bf16" in report:
            report["speedup"] = {
                name: report["bf16"][name] / report["fp8"][name]
                for name in ("forward_ms", "backward_ms", "iteration_ms")
            }
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        GlobalBackendManager.reset()


if __name__ == "__main__":
    main()
