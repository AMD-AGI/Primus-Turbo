###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Native gfx950 ragged and sparse block-diagonal attention adapter contracts."""

import pytest
import torch

from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_varlen_flydsl_backward_impl,
    flash_attn_varlen_flydsl_forward_impl,
)
from tests.pytorch.ops.gradient_contract_helpers import attention_reference
from tests.pytorch.test_utils import compute_snr

pytestmark = pytest.mark.skipif(not is_gfx950(), reason="This regression suite requires gfx950")


@pytest.mark.parametrize("shape", ["ragged", "sparse", "uniform_sink"])
def test_native_ragged_attention_and_fake_kernels(shape):
    torch.manual_seed(807)
    qlens, klens = (
        ([64, 32], [96, 64])
        if shape == "ragged"
        else (([64] + [0] * 8, [64] * 9) if shape == "sparse" else ([64, 64], [64, 64]))
    )
    cq = torch.tensor([0, *torch.tensor(qlens).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    ck = torch.tensor([0, *torch.tensor(klens).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    xs = [
        torch.randn(n, h, 64, device="cuda", dtype=torch.bfloat16)
        for n, h in [(sum(qlens), 4), (sum(klens), 2), (sum(klens), 2)]
    ]
    refs = [x.float().requires_grad_() for x in xs]
    sink = torch.randn(4, device="cuda") if shape == "uniform_sink" else None
    sr = sink.detach().requires_grad_() if sink is not None else None
    out, lse = flash_attn_varlen_flydsl_forward_impl(
        *xs, cq, ck, max(qlens), max(klens), return_lse=True, sink=sink
    )
    parts = []
    q0 = k0 = 0
    for nq, nk in zip(qlens, klens):
        if nq:
            parts.append(
                attention_reference(
                    refs[0][q0 : q0 + nq][None],
                    refs[1][k0 : k0 + nk][None],
                    refs[2][k0 : k0 + nk][None],
                    sink=sr,
                )[0]
            )
        q0 += nq
        k0 += nk
    reference = torch.cat(parts)
    grad = torch.randn_like(out)
    actual = flash_attn_varlen_flydsl_backward_impl(
        grad, *xs, out, lse, cq, ck, max(qlens), max(klens), sink=sink, deterministic=True
    )
    expected = torch.autograd.grad(reference, refs + ([sr] if sr is not None else []), grad.float())
    for a, b in zip([out, *actual], [reference, *expected]):
        torch.testing.assert_close(a.float(), b, atol=0.025, rtol=0.025)
        assert compute_snr(b, a.float()) > 35
    fwd = torch.ops.primus_turbo.flash_attn_varlen_flydsl_forward.default
    torch.library.opcheck(
        fwd, (*xs, cq, ck, max(qlens), max(klens), -1, True, sink), test_utils=("test_faketensor",)
    )
    bwd = torch.ops.primus_turbo.flash_attn_varlen_flydsl_backward.default
    torch.library.opcheck(
        bwd,
        (grad, *xs, out, lse, cq, ck, max(qlens), max(klens), 0.125, -1, sink, True),
        test_utils=("test_faketensor",),
    )
