###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import copy

import pytest
import torch
import torch.nn as nn

from primus_turbo.pytorch.core.delayed_scaling import DelayedScaling
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    ScalingStrategy,
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
@pytest.mark.parametrize("format", [Format.E4M3, Format.E5M2, Format.HYBRID])
@pytest.mark.parametrize("M", [128, 4096])
@pytest.mark.parametrize("N", [256, 4096])
@pytest.mark.parametrize("K", [512, 2048, 8192])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("enable_torch_compile", [True, False])
def test_float8linear(config, dtype, format, M, N, K, bias, enable_torch_compile):
    if format == Format.HYBRID and config.granularity != ScalingGranularity.TENSORWISE:
        pytest.skip("Format.HYBRID is only supported for TENSORWISE granularity")

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
        model = torch.compile(model, fullgraph=False, mode="max-autotune")

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


@pytest.mark.parametrize("format", [Format.E4M3, Format.HYBRID])
def test_float8linear_tensorwise_fullgraph(format):
    """Tensorwise Float8Linears compile into one graph, and their outputs and input and weight
    gradients are bit-identical to eager. The bias gradient is a reduction Inductor may order
    differently, so it is compared with a tolerance."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    config = Float8QuantConfig(granularity=ScalingGranularity.TENSORWISE, format=format)
    eager = nn.Sequential(
        Float8Linear(1024, 3072, config=config), Float8Linear(3072, 1024, config=config)
    ).to("cuda", dtype=dtype)
    compiled = copy.deepcopy(eager)
    torch._dynamo.reset()
    compiled_forward = torch.compile(compiled, fullgraph=True)

    x = torch.randn(4096, 1024, dtype=dtype, device="cuda")
    grad_out = torch.randn(4096, 1024, dtype=dtype, device="cuda")
    results = []
    for model, forward in ((eager, eager), (compiled, compiled_forward)):
        x_ = x.clone().requires_grad_(True)
        out = forward(x_)
        out.backward(grad_out)
        results.append((out.detach(), x_.grad, [p.grad for p in model.parameters()]))

    (out_e, xg_e, grads_e), (out_c, xg_c, grads_c) = results
    assert torch.equal(out_c, out_e)
    assert torch.equal(xg_c, xg_e)
    names = [name for name, _ in eager.named_parameters()]
    for name, g_c, g_e in zip(names, grads_c, grads_e):
        if name.endswith("bias"):
            torch.testing.assert_close(g_c, g_e, **get_tolerances(dtype))
        else:
            assert torch.equal(g_c, g_e), f"{name} gradient differs"


def _delayed_mlp(recipe, format=Format.E4M3, bias=False, device="cuda"):
    config = Float8QuantConfig(format=format, strategy=ScalingStrategy.DELAYED)
    return nn.Sequential(
        Float8Linear(1024, 3072, bias=bias, config=config, delayed_scaling=recipe, device=device),
        Float8Linear(3072, 1024, bias=bias, config=config, delayed_scaling=recipe, device=device),
    )


@pytest.mark.parametrize("format", [Format.E4M3, Format.HYBRID])
def test_float8linear_delayed_training_fullgraph(format):
    """Four optimizer steps with recipe.step() after each: the compiled model's outputs, weights
    and scales are bit-identical to eager's. Bias-free, so no gradient is a reduction Inductor
    may order differently."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    batches = [
        (
            torch.randn(2048, 1024, dtype=dtype, device="cuda") * (i + 1),
            torch.randn(2048, 1024, dtype=dtype, device="cuda"),
        )
        for i in range(4)
    ]
    init = _delayed_mlp(DelayedScaling(), format).to(dtype=dtype).state_dict()
    runs = []
    for compiled in (False, True):
        torch._dynamo.reset()
        recipe = DelayedScaling()
        model = _delayed_mlp(recipe, format).to(dtype=dtype)
        model.load_state_dict(init)
        forward = torch.compile(model, fullgraph=True) if compiled else model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        outputs = []
        for x, grad_out in batches:
            out = forward(x)
            out.backward(grad_out)
            optimizer.step()
            optimizer.zero_grad()
            recipe.step()
            outputs.append(out.detach())
        scales = [torch.stack(m.scaling_state.scale) for m in model]
        runs.append((outputs, [p.detach().clone() for p in model.parameters()], scales))
    for got, expected in zip(runs[1], runs[0]):
        for x, y in zip(got, expected):
            assert torch.equal(x, y)


def test_float8linear_delayed_matches_dynamic_with_fresh_amax():
    """With bias, step 0 (warm-up) and step 1 on the same input match a dynamic Float8Linear."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    recipe = DelayedScaling()
    delayed = Float8Linear(
        1024, 2048, config=Float8QuantConfig(strategy=ScalingStrategy.DELAYED), delayed_scaling=recipe
    ).to("cuda", dtype=dtype)
    dynamic = Float8Linear(1024, 2048).to("cuda", dtype=dtype)
    dynamic.load_state_dict(delayed.state_dict())
    x = torch.randn(4096, 1024, dtype=dtype, device="cuda")
    grad_out = torch.randn(4096, 2048, dtype=dtype, device="cuda")

    def run(model):
        x_ = x.clone().requires_grad_(True)
        out = model(x_)
        out.backward(grad_out)
        result = (out.detach(), x_.grad, model.weight.grad, model.bias.grad)
        model.zero_grad()
        return result

    expected = run(dynamic)
    for _ in range(2):
        for got, ref in zip(run(delayed), expected):
            assert torch.equal(got, ref)
        recipe.step()


def test_float8linear_delayed_eval_is_dynamic():
    """Eval-mode forwards cast dynamically and leave the state alone."""
    torch.manual_seed(0)
    recipe = DelayedScaling()
    model = _delayed_mlp(recipe).to(dtype=torch.bfloat16)
    reference = copy.deepcopy(model)
    for m in reference:
        if isinstance(m, Float8Linear):
            m.config, m.scaling_state = m._eval_config, None
    x = torch.randn(512, 1024, dtype=torch.bfloat16, device="cuda")
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x), reference(x))
    for state in recipe.states():
        assert not state.ready and not torch.stack(state.amax).any()


def test_float8linear_delayed_from_float_and_meta_init():
    recipe = DelayedScaling()
    config = Float8QuantConfig(strategy=ScalingStrategy.DELAYED)
    linear = nn.Linear(256, 512, device="cuda")
    converted = Float8Linear.from_float(linear, config, delayed_scaling=recipe)
    assert converted is linear and converted.scaling_state.device == linear.weight.device
    copied = Float8Linear.from_float(
        nn.Linear(256, 512, device="cuda"), config, inplace=False, delayed_scaling=recipe
    )
    assert len(recipe.states()) == 2 and copied.scaling_state is not converted.scaling_state

    meta = _delayed_mlp(recipe, device="meta")
    meta.to_empty(device="cuda")
    assert all(s.device.type == "cuda" for s in recipe.states())

    with pytest.raises(ValueError, match="delayed_scaling"):
        Float8Linear(256, 512, config=config)
    with pytest.raises(ValueError, match="delayed_scaling"):
        Float8Linear(256, 512, delayed_scaling=recipe)
