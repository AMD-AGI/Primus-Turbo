###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    float8_e4m3,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
)
from primus_turbo.pytorch.ops import grouped_gemm_fp8
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import (
    grouped_gemm_fp8_tensorwise_variable_k_triton_kernel,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA / ROCm device is required")


B = 32
M = 4096
HIDDEN_SIZE = 2880
GATE_UP_SIZE = 5760
INTERMEDIATE_SIZE = 2880
SNR_THRESHOLD = 25.0
FUSED_MAIN_GRAD_INIT = 10.0


GPT_OSS_CASES = [
    pytest.param(
        "GPT-OSS-Forward-UP",
        "forward",
        GATE_UP_SIZE,
        HIDDEN_SIZE,
        False,
        id="forward_up",
    ),
    pytest.param(
        "GPT-OSS-Forward-DOWN",
        "forward",
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        False,
        id="forward_down",
    ),
    pytest.param(
        "GPT-OSS-Backward-UP",
        "backward",
        HIDDEN_SIZE,
        GATE_UP_SIZE,
        True,
        id="backward_up",
    ),
    pytest.param(
        "GPT-OSS-Backward-DOWN",
        "backward",
        INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        True,
        id="backward_down",
    ),
    pytest.param(
        "GPT-OSS-Backward-UP-Weight",
        "wgrad",
        GATE_UP_SIZE,
        HIDDEN_SIZE,
        False,
        id="backward_up_weight",
    ),
    pytest.param(
        "GPT-OSS-Backward-DOWN-Weight",
        "wgrad",
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        False,
        id="backward_down_weight",
    ),
]


def _group_lens(device: str) -> torch.Tensor:
    return torch.full((B,), M, dtype=torch.int64, device=device)


def _snr_from_powers(signal_power: float, noise_power: float) -> float:
    signal = torch.tensor(signal_power, dtype=torch.float64)
    noise = torch.tensor(noise_power, dtype=torch.float64)
    return float(10 * torch.log10(signal / (noise + 1e-12)).item())


def _assert_forward_snr(
    case_name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    group_lens: torch.Tensor,
    trans_b: bool,
) -> None:
    signal_power = 0.0
    noise_power = 0.0
    start = 0
    for group_idx, size in enumerate(group_lens.cpu().tolist()):
        rhs = b[group_idx].t() if trans_b else b[group_idx]
        ref = a[start : start + size] @ rhs
        actual = out[start : start + size]
        ref_f64 = ref.to(torch.float64)
        diff_f64 = (ref - actual).to(torch.float64)
        signal_power += float(ref_f64.norm().pow(2).item())
        noise_power += float(diff_f64.norm().pow(2).item())
        start += size

    snr = _snr_from_powers(signal_power, noise_power)
    print(f"{case_name}: SNR={snr:.2f} dB")
    assert snr > SNR_THRESHOLD


def _assert_fused_wgrad_snr(
    case_name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    group_lens: torch.Tensor,
) -> None:
    signal_power = 0.0
    noise_power = 0.0
    start = 0
    for group_idx, size in enumerate(group_lens.cpu().tolist()):
        ref = a[start : start + size].t() @ b[start : start + size]
        ref = ref + FUSED_MAIN_GRAD_INIT
        actual = out[group_idx]
        ref_f64 = ref.to(torch.float64)
        diff_f64 = (ref - actual).to(torch.float64)
        signal_power += float(ref_f64.norm().pow(2).item())
        noise_power += float(diff_f64.norm().pow(2).item())
        start += size

    snr = _snr_from_powers(signal_power, noise_power)
    print(f"{case_name}: fused wgrad SNR={snr:.2f} dB")
    assert snr > SNR_THRESHOLD


def _run_forward_or_dgrad_case(case_name: str, n: int, k: int, trans_b: bool) -> None:
    device = "cuda:0"
    group_lens = _group_lens(device)
    config = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
    b_shape = (B, n, k) if trans_b else (B, k, n)

    a = torch.randn((B * M, k), dtype=torch.bfloat16, device=device)
    b = torch.randn(b_shape, dtype=torch.bfloat16, device=device)
    out = grouped_gemm_fp8(a, b, group_lens, trans_b=trans_b, config=config)
    torch.cuda.synchronize()
    _assert_forward_snr(case_name, a, b, out, group_lens, trans_b)


def _run_fused_wgrad_case(case_name: str, n: int, k: int) -> None:
    device = "cuda:0"
    group_lens = _group_lens(device)
    group_offs = grouped_gemm_compute_offs(group_lens)

    a = torch.randn((B * M, k), dtype=torch.bfloat16, device=device)
    b = torch.randn((B * M, n), dtype=torch.bfloat16, device=device)
    a_fp8, a_scale_inv = quantize_fp8(a, float8_e4m3, ScalingGranularity.TENSORWISE)
    b_fp8, b_scale_inv = quantize_fp8(b, float8_e4m3, ScalingGranularity.TENSORWISE)

    main_grad = torch.full((B, k, n), FUSED_MAIN_GRAD_INIT, dtype=torch.float32, device=device)
    out = grouped_gemm_fp8_tensorwise_variable_k_triton_kernel(
        a_fp8,
        b_fp8,
        a_scale_inv,
        b_scale_inv,
        group_offs,
        out=main_grad,
        beta=1.0,
    )
    torch.cuda.synchronize()
    assert out is main_grad
    _assert_fused_wgrad_snr(case_name, a, b, out, group_lens)


@pytest.mark.parametrize("case_name,op_type,n,k,trans_b", GPT_OSS_CASES)
def test_grouped_gemm_fp8_tensorwise_gpt_oss_op_precision(case_name, op_type, n, k, trans_b):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.TRITON)
    try:
        if op_type == "wgrad":
            _run_fused_wgrad_case(case_name, n, k)
        else:
            _run_forward_or_dgrad_case(case_name, n, k, trans_b)
    finally:
        GlobalBackendManager.reset()
        torch.cuda.empty_cache()
