###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch
import torch.nn as nn

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)
from primus_turbo.pytorch.modules import Float8Linear
from tests.pytorch.test_utils import compute_snr, get_tolerances

float8_quant_configs = [
    Float8QuantConfig(granularity=ScalingGranularity.TENSORWISE),
    Float8QuantConfig(granularity=ScalingGranularity.ROWWISE),
    Float8QuantConfig(granularity=ScalingGranularity.BLOCKWISE, block_size=128),
]


@pytest.mark.parametrize("config", float8_quant_configs)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("format", [Format.E4M3, Format.E5M2])
@pytest.mark.parametrize("M", [128, 4096])
@pytest.mark.parametrize("N", [256, 4096])
@pytest.mark.parametrize("K", [512, 2048, 8192])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("enable_torch_compile", [True, False])
def test_float8linear(config, dtype, format, M, N, K, bias, enable_torch_compile):
    device = "cuda:0"

    config.format = format

    print(
        f"\nM={M}, N={N}, K={K}, dtype={dtype}, bias={bias}, config={config}, torch_compile={enable_torch_compile}"
    )

    x1 = torch.randn((M, K), dtype=dtype, device=device, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()

    # Ref
    model = nn.Linear(K, N, bias=bias).to(device, dtype=dtype)
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

    model = Float8Linear.from_float(model, config)
    assert isinstance(model, Float8Linear)
    if enable_torch_compile:
        torch._dynamo.reset()
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
        torch.testing.assert_close(bias_grad, bias_grad_ref, **get_tolerances(dtype))
