###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Public default configuration, invalid formats, and single-expert gradients."""

import importlib

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import Float4QuantConfig, Float8QuantConfig
from primus_turbo.pytorch.ops import grouped_gemm
from tests.pytorch.ops.gradient_contract_helpers import caller_thread_backward  # noqa: F401

pytestmark = pytest.mark.usefixtures("caller_thread_backward")


@pytest.mark.parametrize(
    "name",
    [
        "gemm_fp4",
        "gemm_fp8",
        "grouped_gemm_fp4",
        "grouped_gemm_fp8",
        "mlp_fp4",
        "grouped_mlp_fp4",
        "grouped_mlp_fp8",
    ],
)
def test_public_rejects_invalid_granularity(name):
    mod = importlib.import_module("primus_turbo.pytorch.ops." + name)
    config = Float4QuantConfig() if "fp4" in name else Float8QuantConfig()
    config.granularity = "invalid"
    x = torch.zeros(128, 128)
    weight = torch.zeros((2, 128, 128) if "grouped" in name else (128, 128))
    args = [x, weight]
    kw = {"config": config}
    if "mlp" in name:
        args.append(weight)
    if "grouped" in name:
        args.append(torch.tensor([64, 64]))
        kw["group_offs"] = torch.tensor([0, 64, 128])
    if "grouped_mlp" in name:
        kw["probs"] = torch.ones(128)
        kw["activation"] = "silu"
    with pytest.raises(ValueError, match="Unsupported FP[48] ScalingGranularity"):
        getattr(mod, name)(*args, **kw)


def test_fp4_format_rejects_non_fp4():
    from primus_turbo.pytorch.ops.gemm_fp4 import FP4GemmMXFunction

    with pytest.raises(ValueError, match="Unsupported FP4 format"):
        FP4GemmMXFunction.get_fp4_dtype("invalid")


@pytest.mark.parametrize("limit", [2.0, "3.5", 0, -1])
def test_grouped_fp8_clamp_contract(limit):
    from primus_turbo.pytorch.ops.grouped_mlp_fp8 import _check_activation

    if float(limit) <= 0:
        with pytest.raises(AssertionError, match="must be positive"):
            _check_activation("silu", limit)
    else:
        assert _check_activation("silu", limit) == float(limit)


@pytest.mark.parametrize("accum", [False, True])
@pytest.mark.parametrize("trans_b", [False, True])
def test_single_expert_backward_matches_dense(accum, trans_b):
    torch.manual_seed(912)
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
    b = torch.randn(
        (1, 192, 128) if trans_b else (1, 128, 192), device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    ar, br = a.detach().float().requires_grad_(), b.detach().float().requires_grad_()
    original = torch.randn_like(b, dtype=torch.float32)
    if accum:
        b.main_grad = original.clone()
        b.grad_added_to_main_grad = False
    out = grouped_gemm(
        a,
        b,
        torch.tensor([64], device="cuda"),
        trans_b=trans_b,
        fuse_bgrad_accum_pattern="megatron" if accum else None,
    )
    ref = ar @ (br[0].T if trans_b else br[0])
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad.float())
    torch.testing.assert_close(out.float(), ref, atol=0.15, rtol=0.02)
    torch.testing.assert_close(a.grad.float(), ar.grad, atol=0.15, rtol=0.02)
    torch.testing.assert_close(
        (b.main_grad - original) if accum else b.grad.float(), br.grad, atol=0.15, rtol=0.02
    )
    if accum:
        assert b.grad_added_to_main_grad
