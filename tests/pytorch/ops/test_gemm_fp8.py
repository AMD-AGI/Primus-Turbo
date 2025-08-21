###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.float8 import Float8QuantConfig, MXQuantConfig
from primus_turbo.pytorch.ops import gemm_fp8_blockwise, gemm_fp8_tensorwise
from tests.test_utils import compute_snr

torch.manual_seed(42)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("block_size", [128, 256])
@pytest.mark.parametrize("B", [[], [1], [2], [4, 2]])
@pytest.mark.parametrize("M", [257, 4096])
@pytest.mark.parametrize("NK", [(255, 129), (2048, 7168)])
def test_gemm_fp8_blockwise_func(dtype, block_size, B, M, NK):
    if B != []:
        if not (dtype == torch.bfloat16 and block_size == 128 and M == 257 and NK == (255, 129)):
            pytest.skip("Only test one combination for B != []")

    N, K = NK
    device = "cuda:0"

    print(f"\nB={B}, M={M}, N={N}, K={K}, dtype={dtype}, block_size={block_size}")

    x = torch.randn((*B, M, K), dtype=dtype, device=device, requires_grad=True)
    w = torch.randn((N, K), dtype=dtype, device=device, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    w_ref = w.detach().clone().requires_grad_()

    # Ref
    out_ref = x_ref @ w_ref.T
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    x_grad_ref = x_ref.grad
    w_grad_ref = w_ref.grad

    # Config + FWD + BWD
    config = MXQuantConfig(block_size=block_size)
    out = gemm_fp8_blockwise(x, w, trans_a=False, trans_b=True, out_dtype=dtype, config=config)
    out.backward(grad_out)
    x_grad = x.grad
    w_grad = w.grad

    # Check Shape
    print("out shape: ", out.shape, out_ref.shape)
    print("x_grad shape: ", x_grad.shape, x_grad_ref.shape)
    print("w_grad shape: ", w_grad.shape, w_grad_ref.shape)
    assert out.shape == out_ref.shape
    assert x_grad.shape == x_grad_ref.shape
    assert w_grad.shape == w_grad_ref.shape

    # Check Results
    # print("fwd")
    # print(out)
    # print(out_ref)
    out_snr = compute_snr(out_ref, out)
    print(f"Out-SNR: {out_snr:.2f} dB")
    assert out_snr > 20, "out_snr too low"

    # print("dgrad")
    # print(x_grad)
    # print(x_grad_ref)
    xgrad_snr = compute_snr(x_grad_ref, x_grad)
    print(f"XGrad-SNR: {xgrad_snr:.2f} dB")
    assert xgrad_snr > 20, "xgrad_snr too low"

    # print("wgrad")
    # print(w_grad)
    # print(w_grad_ref)
    wgrad_snr = compute_snr(w_grad_ref, w_grad)
    print(f"WGrad-SNR: {wgrad_snr:.2f} dB")
    assert wgrad_snr > 20, "wgrad_snr too low"


@pytest.mark.parametrize("m", [256, 512, 1024, 2048])
@pytest.mark.parametrize("n", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("k", [255, 512, 1024, 2048])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_fp8_tensorwise(m, n, k, dtype):
    device = "cuda:0"

    x = torch.randn((m, k), dtype=dtype, device=device, requires_grad=True)
    w = torch.randn((k, n), dtype=dtype, device=device, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    w_ref = w.detach().clone().requires_grad_()

    # Ref
    out_ref = x_ref @ w_ref
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    x_grad_ref = x_ref.grad
    w_grad_ref = w_ref.grad

    # Config + FWD + BWD
    config = Float8QuantConfig()
    out = gemm_fp8_tensorwise(x, w, False, False, dtype, config)
    out.backward(grad_out)
    x_grad = x.grad
    w_grad = w.grad

    # Check Shape
    print("out shape: ", out.shape, out_ref.shape)
    print("x_grad shape: ", x_grad.shape, x_grad_ref.shape)
    print("w_grad shape: ", w_grad.shape, w_grad_ref.shape)
    assert out.shape == out_ref.shape
    assert x_grad.shape == x_grad_ref.shape
    assert w_grad.shape == w_grad_ref.shape

    # Check Results
    out_snr = compute_snr(out_ref, out)
    print(f"Out-SNR: {out_snr:.2f} dB")
    assert out_snr > 20, "out_snr too low"

    xgrad_snr = compute_snr(x_grad_ref, x_grad)
    print(f"XGrad-SNR: {xgrad_snr:.2f} dB")
    assert xgrad_snr > 20, "xgrad_snr too low"

    wgrad_snr = compute_snr(w_grad_ref, w_grad)
    print(f"WGrad-SNR: {wgrad_snr:.2f} dB")
    assert wgrad_snr > 20, "wgrad_snr too low"
