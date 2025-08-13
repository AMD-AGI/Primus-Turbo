###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch
import torch.nn as nn

from primus_turbo.pytorch.core.float8 import Format, MXQuantConfig
from primus_turbo.pytorch.modules import MXLinear
from tests.test_utils import compute_snr, get_tolerances


# TODO: ori_dtype torch.float32
@pytest.mark.parametrize("ori_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dtype", [Format.E4M3])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("M", [4096])
@pytest.mark.parametrize("NK", [(4096, 4096)])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("enable_torch_compile", [True, False])
def test_mxlinear(ori_dtype, dtype, block_size, M, NK, bias, enable_torch_compile):
    N, K = NK
    device = "cuda:0"

    print(
        f"\nM={M}, N={N}, K={K}, ori_dtype={ori_dtype}, dtype={dtype}, block_size={block_size}, bias={bias}, Compile={enable_torch_compile}"
    )

    x1 = torch.randn((M, K), dtype=ori_dtype, device=device, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()

    # Ref
    model = nn.Linear(K, N, bias=bias).to(device, dtype=ori_dtype)
    out_ref = model(x1)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)

    x_grad_ref = x1.grad.detach().clone()
    w_grad_ref = model.weight.grad.detach().clone()
    if bias:
        bias_grad_ref = model.bias.grad.detach().clone()

    # Clean Grad
    model.weight.grad.zero_()
    if bias:
        model.bias.grad.zero_()
    x2.grad = None

    # MX
    config = MXQuantConfig(block_size=block_size)
    MXLinear.from_float(model, config)
    assert isinstance(model, MXLinear)
    if enable_torch_compile:
        model = torch.compile(model, fullgraph=True, mode="max-autotune")

    out = model(x2)
    out.backward(grad_out)
    x_grad = x2.grad.detach().clone()
    w_grad = model.weight.grad.detach().clone()
    if bias:
        bias_grad = model.bias.grad.detach().clone()

    # Check
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

    if bias:
        torch.testing.assert_close(bias_grad, bias_grad_ref, **get_tolerances(ori_dtype))
