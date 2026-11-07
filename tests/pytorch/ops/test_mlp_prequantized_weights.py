###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Check cache consistency and compare prequantized MLPs with FP32 arithmetic."""

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import (
    Float4QuantConfig,
    Float8QuantConfig,
    ScaleDtype,
    ScalingGranularity,
    ScalingRecipe,
    float4_e2m1fn_x2,
    float8_e4m3,
)
from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor, QuantizedTensorPair
from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.ops.grouped_mlp_fp4 import grouped_mlp_fp4
from primus_turbo.pytorch.ops.grouped_mlp_fp8 import grouped_mlp_fp8
from primus_turbo.pytorch.ops.mlp_fp4 import mlp_fp4
from tests.pytorch.ops.gradient_contract_helpers import caller_thread_backward  # noqa: F401
from tests.pytorch.test_utils import compute_snr

pytestmark = [
    pytest.mark.usefixtures("caller_thread_backward"),
    pytest.mark.skipif(not is_gfx950(), reason="This regression suite requires gfx950"),
]


@pytest.mark.parametrize("kind", ["dense_fp4", "grouped_fp4", "grouped_mx8", "grouped_fp8"])
@pytest.mark.parametrize("cache_transpose", [False, True])
@pytest.mark.parametrize("quantize_activation", [False, True])
def test_mlp_prequantized_inputs(kind, cache_transpose, quantize_activation):
    torch.manual_seed(803)
    grouped = kind.startswith("grouped")
    fp4 = kind.endswith("fp4")
    tensorwise = kind == "grouped_fp8"
    config = (
        Float4QuantConfig()
        if fp4
        else (
            Float8QuantConfig()
            if tensorwise
            else Float8QuantConfig(
                granularity=ScalingGranularity.MX_BLOCKWISE, block_size=32, scale_dtype=ScaleDtype.E8M0
            )
        )
    )
    dtype = float4_e2m1fn_x2 if fp4 else float8_e4m3
    m, k, i, g = 512, 896 if kind == "grouped_fp4" else 1024, 512 if grouped else 1024, 2
    x = (torch.randn(m, k, device="cuda") * 0.1).bfloat16()
    w1 = (torch.randn((g, 2 * i, k) if grouped else (2 * i, k), device="cuda") * 0.02).bfloat16()
    w2 = (torch.randn((g, k, i) if grouped else (k, i), device="cuda") * 0.02).bfloat16()
    lens = torch.tensor([239, 273], device="cuda", dtype=torch.int64)
    probs = torch.rand(m, device="cuda") + 0.25
    inputs = [x, w1, w2]
    quantized = []
    for index, t in enumerate(inputs):
        if index == 0 and not quantize_activation:
            quantized.append(t.detach().requires_grad_())
            continue
        recipe = ScalingRecipe(use_2d_block=index > 0 and not tensorwise)
        extra = {"group_lens": lens} if index == 0 and grouped else {}
        if tensorwise:
            extra.update(pad_align_last=128, pad_align_penultimate=128 if index == 2 else 0)
        q = QuantizedTensor.quantize(
            t,
            dtype,
            config.granularity,
            axis=-1,
            block_size=config.block_size,
            scaling_recipe=recipe,
            **extra,
        )
        q.requires_grad_()
        if cache_transpose and not tensorwise:
            col_recipe = ScalingRecipe(use_2d_block=index > 0, use_rht=index == 0 and fp4)
            # Form the transpose from the same dequantized cached data as the
            # uncached path, making cache-vs-rebuild equality meaningful.
            qt = QuantizedTensor.quantize(
                q.dequantize(),
                dtype,
                config.granularity,
                axis=-2,
                block_size=config.block_size,
                scaling_recipe=col_recipe,
                **extra,
            )
            quantized.append(QuantizedTensorPair(q, qt))
        else:
            quantized.append(q)

    def call(args):
        if grouped:
            fn = grouped_mlp_fp4 if fp4 else grouped_mlp_fp8
            return fn(
                *args, lens, probs=probs, trans_w1=True, trans_w2=True, config=config, activation="silu"
            )
        return mlp_fp4(*args, config=config)

    oracle_inputs = [v.detach().float().requires_grad_() for v in inputs]

    def dense_reference(inp, weight1, weight2):
        gate, up = (inp @ weight1.T).chunk(2, dim=-1)
        return (torch.nn.functional.silu(gate) * up) @ weight2.T

    if grouped:
        oracle = (
            torch.cat(
                [
                    dense_reference(inp, weight1, weight2)
                    for inp, weight1, weight2 in zip(
                        oracle_inputs[0].split([239, 273]), oracle_inputs[1], oracle_inputs[2]
                    )
                ]
            )
            * probs[:, None]
        )
    else:
        oracle = dense_reference(*oracle_inputs)
    ref_inputs = [v.detach().requires_grad_() for v in inputs]
    reference = call(ref_inputs)
    out = call(quantized)
    grad = torch.randn_like(out)
    ref_grads = torch.autograd.grad(reference, ref_inputs, grad)
    leaves = [v.data if isinstance(v, QuantizedTensorPair) else v for v in quantized]
    grads = torch.autograd.grad(out, leaves, grad)
    oracle_grads = torch.autograd.grad(oracle, oracle_inputs, grad.float())
    floor = 6 if fp4 else 20
    for actual, expected in zip([out, *grads], [oracle, *oracle_grads]):
        assert compute_snr(expected.float(), actual.float()) > floor
    for actual, expected in zip([out, *grads], [reference, *ref_grads]):
        assert actual.shape == expected.shape
        assert torch.isfinite(actual).all()
        assert compute_snr(expected.float(), actual.float()) > floor
