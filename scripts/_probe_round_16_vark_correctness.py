"""R16 correctness probe — verify the binary-search → divide swap in
``grouped_var_k_kernel_fp8`` produces bit-identical wgrad output for
the gpt_oss-Down-B4 family (G=4, tiles_per_group=121, slots=192) and
the gpt_oss-GateUP-B32 family (G=32, tiles_per_group=242).

Compare against an FP32 reference (group-by-group bmm). Pass iff
SNR > 40 dB (well above the 25 dB metric gate; numerical equivalence
of integer scheduling is exact, only FP rounding contributes drift).
"""
from __future__ import annotations

import os
import sys

import torch

# Use local Primus-Turbo source, NOT installed venv copy.
sys.path.insert(0, "/workspace/code/Primus-Turbo")

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
os.environ.setdefault("PRIMUS_TURBO_GROUPED_GEMM_BACKEND", "HIPKITTEN")

from primus_turbo.pytorch.core.low_precision import ScalingGranularity  # noqa: E402
from primus_turbo.pytorch.kernels.hipkitten import loader as hk_loader  # noqa: E402
from primus_turbo.pytorch.ops.quantization import quantize_fp8  # noqa: E402

_FP8 = hk_loader.load_fp8()


def _quant_tw(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_fp8(x, torch.float8_e4m3fnuz, ScalingGranularity.TENSORWISE)


def fp32_ref_wgrad(grad_BF16: torch.Tensor, a_BF16: torch.Tensor, group_lens: torch.Tensor) -> torch.Tensor:
    """grad: [G*M, N], a: [G*M, K] (uniform M). Returns dB = a^T @ grad per group, stacked [G, K, N]."""
    out_groups = []
    offset = 0
    for g_len in group_lens.tolist():
        g_a = a_BF16[offset : offset + g_len]
        g_grad = grad_BF16[offset : offset + g_len]
        out_groups.append(g_a.t().to(torch.float32) @ g_grad.to(torch.float32))
        offset += g_len
    return torch.stack(out_groups, dim=0)


def hk_vark_wgrad(grad_BF16: torch.Tensor, a_BF16: torch.Tensor, group_lens: torch.Tensor, *, num_slots: int, group_m: int, num_xcds: int) -> torch.Tensor:
    """Call HK ``grouped_variable_k_crr_dscale`` directly with quantized inputs."""
    a_col, a_s = _quant_tw(a_BF16)
    g_col, g_s = _quant_tw(grad_BF16)

    G = group_lens.numel()
    K = a_BF16.shape[1]
    N = grad_BF16.shape[1]

    out = torch.empty((G, K, N), dtype=torch.bfloat16, device=a_BF16.device)
    group_offs = torch.zeros(G + 1, dtype=torch.int64, device=a_BF16.device)
    group_offs[1:] = torch.cumsum(group_lens, dim=0)

    _FP8.grouped_variable_k_crr_dscale(
        a_col, g_col,
        a_s, g_s,
        group_offs,
        out,
        group_m=group_m,
        num_xcds=num_xcds,
        num_slots=num_slots,
    )
    return out


def snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    diff = (got.to(torch.float32) - ref.to(torch.float32))
    rms_ref = ref.to(torch.float32).pow(2).mean().sqrt().item()
    rms_diff = diff.pow(2).mean().sqrt().item()
    if rms_diff == 0.0:
        return 999.0
    return 20.0 * torch.log10(torch.tensor(rms_ref / rms_diff)).item()


def check(name: str, B: int, M: int, N: int, K: int, num_slots: int, group_m: int, num_xcds: int):
    torch.manual_seed(0xC0FFEE)
    G = B
    M_total = G * M
    a = torch.randn(M_total, K, device="cuda", dtype=torch.bfloat16) * 0.5
    grad = torch.randn(M_total, N, device="cuda", dtype=torch.bfloat16) * 0.5
    group_lens = torch.full((G,), M, dtype=torch.int64, device="cuda")

    ref = fp32_ref_wgrad(grad, a, group_lens)
    got = hk_vark_wgrad(grad, a, group_lens, num_slots=num_slots, group_m=group_m, num_xcds=num_xcds)

    snr = snr_db(ref, got)
    max_abs = (got.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
    has_nan = bool(torch.isnan(got).any().item())
    has_inf = bool(torch.isinf(got).any().item())
    print(f"  {name:<28s} B={B} M={M} N={N} K={K} slots={num_slots} gm={group_m} xcds={num_xcds}: SNR={snr:.1f} dB max_abs={max_abs:.2e} nan={has_nan} inf={has_inf}")
    return snr, has_nan, has_inf


def main():
    print("R16 correctness probe — var-K wgrad after binary-search → divide swap")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print()

    cases = [
        # (name, B, M, N, K, slots, gm, xcds) - matches dispatcher rules
        ("gpt_oss-Down-B4-M2048",   4, 2048, 2880, 2880, 192, 1, 2),
        ("gpt_oss-Down-B4-M4096",   4, 4096, 2880, 2880, 192, 1, 2),
        ("gpt_oss-GateUP-B4-M2048", 4, 2048, 5760, 2880, 256, 1, 2),
        ("gpt_oss-Down-B32-M2048",  32, 2048, 2880, 2880, 256, 8, 4),
        ("gpt_oss-GateUP-B32-M2048", 32, 2048, 5760, 2880, 256, 4, 4),
    ]

    all_pass = True
    for case in cases:
        snr, has_nan, has_inf = check(*case)
        if snr < 25.0 or has_nan or has_inf:
            all_pass = False
            print(f"    FAIL")

    print()
    if all_pass:
        print("PASS — all shapes SNR > 25 dB, no NaN/Inf")
        sys.exit(0)
    else:
        print("FAIL — at least one shape regressed")
        sys.exit(1)


if __name__ == "__main__":
    main()
