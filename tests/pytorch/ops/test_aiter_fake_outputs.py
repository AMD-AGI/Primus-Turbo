###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Compiler metadata agrees with AITER's actual dense and packed outputs."""

import pytest
import torch

from primus_turbo.pytorch.kernels.attention.attention_aiter_impl import (
    attention_aiter_forward_impl,
    attention_aiter_varlen_forward_impl,
)
from tests.pytorch.ops.gradient_contract_helpers import attention_reference


@pytest.mark.parametrize("varlen", [False, True])
def test_aiter_real_and_fake_outputs(varlen):
    torch.manual_seed(824)
    xs = [torch.randn(2, 128, h, 64, device="cuda", dtype=torch.bfloat16) for h in (4, 2, 2)]
    if varlen:
        cu = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int32)
        args = (
            *[x.flatten(0, 1) for x in xs],
            cu,
            cu,
            128,
            128,
            0.0,
            64**-0.5,
            True,
            -1,
            -1,
            None,
            None,
            True,
            False,
        )
        op = attention_aiter_varlen_forward_impl
    else:
        args = (*xs, 0.0, 64**-0.5, True, -1, -1, None, None, True, False)
        op = attention_aiter_forward_impl
    out, lse, _, _ = op(*args)
    ref = attention_reference(*xs)
    torch.testing.assert_close(out.float().reshape(ref.shape), ref, atol=0.02, rtol=0.02)
    assert torch.isfinite(lse).all()
    torch.library.opcheck(op, args, test_utils=("test_faketensor",))
