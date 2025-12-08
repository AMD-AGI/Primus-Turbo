###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
import traceback
from tabulate import tabulate

from torch.nn.functional import ScalingType, scaled_grouped_mm, SwizzleType

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)
from primus_turbo.pytorch.ops import grouped_gemm_fp8
from tests.pytorch.ref.gemm_ref import (
    generate_grouped_gemm_group_lens,
    grouped_gemm_ref,
)
from tests.pytorch.test_utils import compute_snr

M_SIZE_LIST = [512, 1024, 2048, 4096, 8192, 16384]
EP_SIZE_LIST = [32, 16, 8]
EPS = 1e-12
INCLUDE_TORCH_QUANT_TIME = os.environ.get("INCLUDE_TORCH_QUANT_TIME", "0").lower() in ("1", "true", "yes")


def _get_fp8_dtype(format: Format) -> torch.dtype:
    return torch.float8_e4m3fnuz

def _quantize_rowwise_fp8(
    tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Row-wise quantization helper that mirrors the setup from
    test_scaled_grouped_gemm_2d_3d in PyTorch.
    """
    if tensor.numel() == 0:
        scale_shape = tensor.shape[:-1] if tensor.dim() > 0 else (1,)
        scale_inv = torch.ones(scale_shape, dtype=torch.float32, device=tensor.device)
        return tensor.to(fp8_dtype), scale_inv

    contiguous = tensor.contiguous()
    last_dim = contiguous.shape[-1]
    flat = contiguous.view(-1, last_dim).to(torch.float32)

    max_val = torch.finfo(fp8_dtype).max
    amax = torch.amax(flat.abs(), dim=-1, keepdim=True)
    denom = torch.clamp(amax, min=EPS)
    scale = max_val / denom
    scaled = torch.clamp(flat * scale, min=-max_val, max=max_val)
    fp8_flat = scaled.to(fp8_dtype)
    scale_inv = (1.0 / scale).squeeze(-1).to(torch.float32)

    fp8 = fp8_flat.view_as(contiguous).contiguous()
    scale_inv = scale_inv.view(*contiguous.shape[:-1]).contiguous()
    return fp8, scale_inv


def _prepare_torch_scaled_grouped_mm_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    trans_b: bool,
    format: Format,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not trans_b:
        raise ValueError("torch._scaled_grouped_mm reference currently expects trans_b=True")

    if group_lens.dtype != torch.int32:
        group_lens_int = group_lens.to(torch.int32)
    else:
        group_lens_int = group_lens

    if group_lens_int.device != a.device:
        group_lens_int = group_lens_int.to(a.device)

    fp8_dtype = _get_fp8_dtype(format)
    a_fp8, a_scale_inv = _quantize_rowwise_fp8(a, fp8_dtype)
    b_fp8, b_scale_inv = _quantize_rowwise_fp8(b, fp8_dtype)

    a_scale = a_scale_inv.view(-1).contiguous()
    b_scale = b_scale_inv.contiguous()
    b_fp8_t = b_fp8.transpose(-2, -1)

    offs = torch.cumsum(group_lens_int, dim=0).to(torch.int32)
    return a_fp8, b_fp8_t, a_scale, b_scale, offs


def bench_torch_scaled_grouped_mm_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    trans_b: bool,
    format: Format,
    out_ref: torch.Tensor,
    fwd_total_flops: int,
    warmup_iters: int = 20,
    timed_iters: int = 100,
    include_quant_time: bool = True,
) -> tuple[float, float]:
    def _invoke(a_lp, b_lp_t, scale_a, scale_b, offs):
        return scaled_grouped_mm(
            a_lp,
            b_lp_t,
            scale_a=scale_a,
            scale_recipe_a=ScalingType.RowWise,
            scale_b=scale_b,
            scale_recipe_b=ScalingType.RowWise,
            swizzle_a=SwizzleType.NO_SWIZZLE,
            swizzle_b=SwizzleType.NO_SWIZZLE,
            offs=offs,
            output_dtype=a.dtype,
            bias=None,
            use_fast_accum=True,
        )

    prepared_inputs = None
    if not include_quant_time:
        prepared_inputs = _prepare_torch_scaled_grouped_mm_inputs(
            a, b, group_lens, trans_b, format
        )

    def _materialize_inputs():
        if include_quant_time:
            return _prepare_torch_scaled_grouped_mm_inputs(a, b, group_lens, trans_b, format)
        else:
            assert prepared_inputs is not None
            return prepared_inputs

    def _run_once():
        a_lp, b_lp_t, scale_a, scale_b, offs = _materialize_inputs()
        return _invoke(a_lp, b_lp_t, scale_a, scale_b, offs)

    _run_once()
    torch.cuda.synchronize()

    for _ in range(warmup_iters):
        _run_once()
    torch.cuda.synchronize()

    fwd_timer = benchmark.Timer(
        stmt="fn()",
        globals={"fn": _run_once},
    )
    measurement = fwd_timer.timeit(timed_iters)
    torch.cuda.synchronize()

    time_ms = measurement.mean * 1e3
    tflops = fwd_total_flops / (time_ms * 1e-3) / 1e12
    return time_ms, tflops


def _generate_moe_test_cases(
    name_prefix: str,
    n_routed_experts: int,
    moe_intermediate_size: int,
    hidden_size: int,
):
    test_cases = []
    shapes_dict = {
        f"{name_prefix}-GateUP": (2 * moe_intermediate_size, hidden_size),
        f"{name_prefix}-Down": (hidden_size, moe_intermediate_size),
    }

    for ep in EP_SIZE_LIST:
        B = n_routed_experts // ep
        for M in M_SIZE_LIST:
            for name, (N, K) in shapes_dict.items():
                for dtype in [torch.bfloat16]:
                    test_cases.append(
                        {
                            "Case": name,
                            "B": B,
                            "M": M,
                            "N": N,
                            "K": K,
                            "dtype": dtype,
                        }
                    )
    return test_cases


def generate_deepseekv3_test_cases():
    return _generate_moe_test_cases(
        "DSV3", n_routed_experts=256, moe_intermediate_size=2048, hidden_size=7168
    )


def generate_deepseekv2_test_cases():
    return _generate_moe_test_cases(
        "DSV2", n_routed_experts=160, moe_intermediate_size=1536, hidden_size=5120
    )


def generate_deepseekv2_lite_test_cases():
    return _generate_moe_test_cases(
        "DSV2-Lite", n_routed_experts=64, moe_intermediate_size=1408, hidden_size=2048
    )


def bench_grouped_gemm_fp8(
    B,
    M,
    N,
    K,
    ori_dtype,
    format,
    granularity,
    trans_b,
    balance,
    include_torch_quant_time: bool = True,
):
    device = "cuda"
    # Prepare inputs
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance).to(device)
    b_shape = (B, N, K) if trans_b else (B, K, N)
    a = torch.randn((B * M, K), dtype=ori_dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=ori_dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    config = Float8QuantConfig(format=format, granularity=granularity)

    # Reference forward pass
    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=trans_b)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out, retain_graph=True)

    # Forward pass for implementation
    fwd_func = lambda: grouped_gemm_fp8(a, b, group_lens, trans_b=trans_b, config=config)
    bwd_func = lambda: out.backward(grad_out, retain_graph=True)
    out = fwd_func()
    bwd_func()

    # Compute SNRs
    out_snr = compute_snr(out_ref, out)
    a_grad_snr = compute_snr(a_ref.grad, a.grad)
    b_grad_snr = compute_snr(b_ref.grad, b.grad)

    if out_snr <= 20:
        print(f"out_snr too low: {out_snr}")
    if a_grad_snr <= 20:
        print(f"x_grad_snr too low: {a_grad_snr}")
    if b_grad_snr <= 20:
        print(f"w_grad_snr too low: {b_grad_snr}")

    assert out_snr > 20, "out_snr too low"
    assert a_grad_snr > 20, "x_grad_snr too low"
    assert b_grad_snr > 20, "w_grad_snr too low"
    # Calculate FLOPs
    fwd_total_flops = 2 * B * M * N * K
    bwd_total_flops = 2 * fwd_total_flops

    torch_ref_time_ms = None
    torch_ref_tflops = None
    try:
        (
            torch_ref_time_ms,
            torch_ref_tflops,
        ) = bench_torch_scaled_grouped_mm_reference(
            a.detach(),
            b.detach(),
            group_lens,
            trans_b,
            format,
            out_ref.detach(),
            fwd_total_flops,
            include_quant_time=include_torch_quant_time,
        )
        print(
            f"torch.nn.functional.scaled_grouped_mm Mean time: {torch_ref_time_ms:.3f} ms | "
            f"TFLOPS: {torch_ref_tflops:.2f}"
        )
    except Exception as ref_err:
        print(f"torch.nn.functional.scaled_grouped_mm reference failed: {ref_err}")
        traceback.print_exc()

    # Warmup
    warmup = 20
    for _ in range(warmup):
        fwd_func()
        bwd_func()
    torch.cuda.synchronize()

    # Benchmark
    fwd_timer = benchmark.Timer(
        stmt="fn()",
        globals={"fn": fwd_func},
    )
    bwd_timer = benchmark.Timer(
        stmt="fn()",
        globals={"fn": bwd_func},
    )
    fwd_measurement = fwd_timer.timeit(100)
    bwd_measurement = bwd_timer.timeit(100)

    fwd_mean_time_ms = fwd_measurement.mean * 1e3
    bwd_mean_time_ms = bwd_measurement.mean * 1e3
    fwd_tflops = fwd_total_flops / (fwd_mean_time_ms * 1e-3) / 1e12
    bwd_tflops = bwd_total_flops / (bwd_mean_time_ms * 1e-3) / 1e12
    print(f"Forward  Mean time: {fwd_mean_time_ms:.3f} ms | TFLOPS: {fwd_tflops:.2f}")
    print(f"Backward Mean time: {bwd_mean_time_ms:.3f} ms | TFLOPS: {bwd_tflops:.2f}")
    return (
        fwd_mean_time_ms,
        fwd_tflops,
        bwd_mean_time_ms,
        bwd_tflops,
        torch_ref_time_ms,
        torch_ref_tflops,
    )


if __name__ == "__main__":
    dsv2_lite_test_cases = generate_deepseekv2_lite_test_cases()
    dsv2_test_cases = generate_deepseekv2_test_cases()
    dsv3_test_cases = generate_deepseekv3_test_cases()
    test_cases = dsv2_lite_test_cases + dsv2_test_cases + dsv3_test_cases

    # DataFrame to store results
    results = pd.DataFrame(
        columns=[
            "TestID",
            "Case",
            "B",
            "M",
            "N",
            "K",
            "format",
            "granularity",
            "Forward Time (ms)",
            "Forward TFLOPS",
            "Backward Time (ms)",
            "Backward TFLOPS",
            "Torch scaled_grouped_mm Time (ms)",
            "Torch scaled_grouped_mm TFLOPS",
        ]
    )
    test_id = 0
    format = Format.E4M3
    granularity = ScalingGranularity.TENSORWISE
    for case in test_cases:
        B = case["B"]
        M = case["M"]
        N = case["N"]
        K = case["K"]
        dtype = case["dtype"]

        print(f"\n{'='*50}")
        print(f"Testing Case: {case}")
        print(f"{'='*50}")

        trans_b = True
        balance = True
        test_id += 1
        try:
            # Run benchmark
            (
                fwd_time_ms,
                fwd_tflops,
                bwd_time_ms,
                bwd_tflops,
                torch_ref_time_ms,
                torch_ref_tflops,
            ) = bench_grouped_gemm_fp8(
                B=B,
                M=M,
                N=N,
                K=K,
                ori_dtype=dtype,
                format=format,
                granularity=granularity,
                trans_b=trans_b,
                balance=balance,
                include_torch_quant_time=INCLUDE_TORCH_QUANT_TIME,
            )
            # Add to results table
            new_row = {
                "TestID": test_id,
                "Case": case["Case"],
                "B": B,
                "M": M,
                "N": N,
                "K": K,
                "format": format,
                "granularity": granularity,
                "Forward Time (ms)": f"{fwd_time_ms:.2f}",
                "Forward TFLOPS": f"{fwd_tflops:.2f}",
                "Backward Time (ms)": f"{bwd_time_ms:.2f}",
                "Backward TFLOPS": f"{bwd_tflops:.2f}",
                "Torch scaled_grouped_mm Time (ms)": (
                    f"{torch_ref_time_ms:.2f}" if torch_ref_time_ms is not None else "N/A"
                ),
                "Torch scaled_grouped_mm TFLOPS": (
                    f"{torch_ref_tflops:.2f}" if torch_ref_tflops is not None else "N/A"
                ),
            }
            results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        except Exception as e:
            print(f"Failed to run {case}: {str(e)}")
            new_row = {
                "TestID": test_id,
                "Case": case["Case"],
                "B": B,
                "M": M,
                "N": N,
                "K": K,
                "format": format,
                "granularity": granularity,
                "Forward Time (ms)": "N/A",
                "Forward TFLOPS": "N/A",
                "Backward Time (ms)": "N/A",
                "Backward TFLOPS": "N/A",
                "Torch scaled_grouped_mm Time (ms)": "N/A",
                "Torch scaled_grouped_mm TFLOPS": "N/A",
            }
            results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

    # Print results
    print("\nFinal Results:")
    print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))

    # Save to CSV
    results.to_csv("grouped_gemm_fp8_benchmark_results.csv", index=False)
    print("Results saved to grouped_gemm_fp8_benchmark_results.csv")
