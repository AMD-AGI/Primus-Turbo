###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import (
    Float4QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
    check_mxfp4_support,
)
from primus_turbo.pytorch.ops.grouped_gemm_fp4 import grouped_gemm_fp4
from tests.pytorch.ref.gemm_ref import (
    generate_grouped_gemm_group_lens,
    grouped_gemm_ref,
)
from tests.pytorch.test_utils import compute_snr

torch.manual_seed(42)

# N, K must be multiples of 128 (Triton MXFP4 BLOCK_SIZE_K) — they are the
# fwd/dgrad contraction dims. M is grouped along rows (any size; the wgrad
# wrapper zero-pads each group's M up to 128).
B_VALUES = [1, 2, 4, 8]
M_VALUES = [256, 512, 1024]
NK_VALUES = [(2048, 1536), (1408, 2048), (3072, 5120), (7168, 2048)]
DTYPE_VALUES = [torch.bfloat16, torch.float16]
BALANCE_VALUES = [True, False]


def _run(B, M, N, K, dtype, balance, use_gradient_sr=False):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    supported, reason = check_mxfp4_support()
    if not supported:
        pytest.skip(reason)

    device = "cuda:0"
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance).to(device)
    print(f"\nB={B}, M={M}, N={N}, K={K}, dtype={dtype}, balance={balance}")

    a = torch.randn((B * M, K), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((B, N, K), dtype=dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    torch.cuda.synchronize()

    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=True)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    torch.cuda.synchronize()

    config = Float4QuantConfig(
        granularity=ScalingGranularity.MX_BLOCKWISE,
        format=Format.E2M1_X2,
        block_size=32,
        scale_dtype=ScaleDtype.E8M0,
        use_gradient_sr=use_gradient_sr,
    )
    out = grouped_gemm_fp4(a, b, group_lens, trans_b=True, config=config)
    out.backward(grad_out)
    torch.cuda.synchronize()

    assert out.shape == out_ref.shape
    assert a.grad.shape == a_ref.grad.shape
    assert b.grad.shape == b_ref.grad.shape

    snr_threshold = 8.0  # E2M1 (1-bit mantissa) is intrinsically lossy
    out_snr = compute_snr(out_ref, out)
    a_grad_snr = compute_snr(a_ref.grad, a.grad)
    b_grad_snr = compute_snr(b_ref.grad, b.grad)
    print(f"Out-SNR={out_snr:.2f} dB  AGrad-SNR={a_grad_snr:.2f} dB  BGrad-SNR={b_grad_snr:.2f} dB")
    assert out_snr > snr_threshold, f"out_snr={out_snr:.2f} too low"
    assert a_grad_snr > snr_threshold, f"a_grad_snr={a_grad_snr:.2f} too low"
    assert b_grad_snr > snr_threshold, f"b_grad_snr={b_grad_snr:.2f} too low"


@pytest.mark.parametrize("B", B_VALUES)
@pytest.mark.parametrize("M", M_VALUES)
@pytest.mark.parametrize("NK", NK_VALUES)
@pytest.mark.parametrize("dtype", DTYPE_VALUES)
@pytest.mark.parametrize("balance", BALANCE_VALUES)
def test_grouped_gemm_fp4_mx_blockwise(B, M, NK, dtype, balance):
    """MXFP4 grouped GEMM fwd + dgrad + wgrad on the Triton backend."""
    N, K = NK
    _run(B, M, N, K, dtype, balance)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("M", [512])
@pytest.mark.parametrize("NK", [(2048, 1536)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_grouped_gemm_fp4_gradient_sr(B, M, NK, dtype):
    """Smoke test the stochastic-rounding gradient path."""
    N, K = NK
    _run(B, M, N, K, dtype, balance=False, use_gradient_sr=True)
