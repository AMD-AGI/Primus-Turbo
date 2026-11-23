###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Block-FP8 Linear wgrad written straight into ``main_grad``: first write (beta 0) vs accumulate (beta 1).

The weight gradient GEMM either overwrites ``main_grad`` (the training framework's first write of the step) or adds
to it in its float32 epilogue. Byte checks: overwrite ignores the old contents and matches accumulate-onto-zero to bf16 precision; accumulate equals the
overwrite result added to the old main_grad in float32; forward output and dgrad do not depend on the policy.
"""

import pytest
import torch

import primus_turbo.pytorch as pt
from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, Format, ScalingGranularity
from primus_turbo.pytorch.ops import utils as ops_utils
from tests.pytorch.ops.gfx1250_helper import bits_equal, rel_l2, requires_flydsl

pytestmark = requires_flydsl
CFG = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.BLOCKWISE, block_size=128)


def _step(kernel, monkeypatch, x, w, gy, mg0, first_write):
    monkeypatch.setenv("PRIMUS_TURBO_FP8_BLOCK_UE8M0", "1")
    monkeypatch.setenv("PRIMUS_TURBO_FP8_BLOCK_KERNEL", kernel)
    monkeypatch.setenv("PRIMUS_TURBO_DSV4UP_PREFER_G90", "")
    monkeypatch.delenv("PRIMUS_TURBO_FP8_WCACHE", raising=False)
    w.main_grad = mg0.clone()
    w.grad = None
    w.grad_added_to_main_grad = False
    ops_utils.set_main_grad_first_write_policy((lambda p: True) if first_write else None)
    try:
        xx = x.detach().requires_grad_(True)
        y = pt.ops.gemm_fp8(
            xx,
            w,
            trans_a=False,
            trans_b=True,
            out_dtype=None,
            config=CFG,
            fuse_bgrad_accum_pattern="megatron",
        )
        y.backward(gy)
        torch.cuda.synchronize()
    finally:
        ops_utils.set_main_grad_first_write_policy(None)
    return y.detach().clone(), xx.grad.clone(), w.main_grad.clone()


@pytest.mark.parametrize("kernel", ["g90", "dsv4up"])
def test_wgrad_first_write_vs_accumulate(monkeypatch, kernel):
    if kernel == "dsv4up":
        from tests.pytorch.ops.gfx1250_helper import gluon_available

        if not gluon_available():
            pytest.skip("Gluon unavailable")
    torch.manual_seed(0)
    t, n, k = 1024, 1024, 1024
    x = (torch.randn(t, k, device="cuda") * 0.5).bfloat16()
    w = (torch.randn(n, k, device="cuda") * 0.02).bfloat16().requires_grad_(True)
    gy = (torch.randn(t, n, device="cuda") * 0.1).bfloat16()
    mg0 = torch.randn(n, k, device="cuda") * 0.01
    zeros = torch.zeros_like(mg0)

    y_a, dx_a, mg_overwrite_dirty = _step(kernel, monkeypatch, x, w, gy, mg0, first_write=True)
    y_b, dx_b, mg_acc_zero = _step(kernel, monkeypatch, x, w, gy, zeros, first_write=False)
    _, _, mg_acc = _step(kernel, monkeypatch, x, w, gy, mg0, first_write=False)

    # overwrite ignores the old contents (same bits as onto a zero buffer) and holds the GEMM result
    _, _, mg_overwrite_zero = _step(kernel, monkeypatch, x, w, gy, zeros, first_write=True)
    assert bits_equal(mg_overwrite_dirty, mg_overwrite_zero)
    assert rel_l2(mg_overwrite_dirty, mg_acc_zero) < 5e-3
    # accumulate = old main_grad + the GEMM result, one float32 add
    assert bits_equal(mg_acc, mg0 + mg_acc_zero)
    # the policy changes only the wgrad write
    assert bits_equal(y_a, y_b) and bits_equal(dx_a, dx_b)
