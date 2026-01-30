"""Test correctness of the fused cast+transpose kernel."""

import torch

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)


def compute_snr(signal, noise):
    """Compute SNR in dB."""
    signal_power = (signal**2).mean()
    noise_power = (noise**2).mean()
    if noise_power == 0:
        return float("inf")
    return 10 * torch.log10(signal_power / noise_power).item()


def main():
    M, N, K = 8192, 8192, 8192
    dtype = torch.bfloat16
    device = "cuda"
    trans_b = True

    config = Float8QuantConfig(
        format=Format.E4M3,
        granularity=ScalingGranularity.TENSORWISE,
    )

    print("=" * 70)
    print("Correctness Test for FP8 GEMM with Fused Cast+Transpose")
    print("=" * 70)

    # Create inputs
    torch.manual_seed(42)
    a = torch.randn((M, K), dtype=dtype, device=device, requires_grad=True)
    b_shape = (N, K) if trans_b else (K, N)
    b = torch.randn(b_shape, dtype=dtype, device=device, requires_grad=True)

    # Forward pass
    out = turbo.ops.gemm_fp8(a, b, trans_b=trans_b, config=config)

    # Reference forward (BF16)
    if trans_b:
        out_ref = torch.matmul(a, b.t())
    else:
        out_ref = torch.matmul(a, b)

    # Check forward correctness
    fwd_snr = compute_snr(out_ref, out.float() - out_ref)
    print(f"\nForward SNR: {fwd_snr:.1f} dB (threshold: 25 dB)")

    # Backward pass
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    # Reference backward
    a_ref = a.clone().detach().requires_grad_(True)
    b_ref = b.clone().detach().requires_grad_(True)
    if trans_b:
        out_ref = torch.matmul(a_ref, b_ref.t())
    else:
        out_ref = torch.matmul(a_ref, b_ref)
    out_ref.backward(grad_out)

    # Check backward correctness
    da_snr = compute_snr(a_ref.grad, a.grad.float() - a_ref.grad)
    db_snr = compute_snr(b_ref.grad, b.grad.float() - b_ref.grad)
    print(f"d_a SNR:     {da_snr:.1f} dB (threshold: 25 dB)")
    print(f"d_b SNR:     {db_snr:.1f} dB (threshold: 25 dB)")

    # Summary
    min_snr = min(fwd_snr, da_snr, db_snr)
    threshold = 25
    if min_snr >= threshold:
        print(f"\n✓ PASS: All SNRs >= {threshold} dB")
    else:
        print(f"\n✗ FAIL: Min SNR ({min_snr:.1f} dB) < {threshold} dB")

    print("=" * 70)


if __name__ == "__main__":
    main()
