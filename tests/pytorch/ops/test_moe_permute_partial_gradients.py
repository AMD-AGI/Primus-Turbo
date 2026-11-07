###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unused permute outputs must contribute zero gradients, for both implementations."""

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.ops.moe.moe_permute import moe_permute, moe_unpermute
from tests.pytorch.ops.gradient_contract_helpers import caller_thread_backward  # noqa: F401

pytestmark = pytest.mark.usefixtures("caller_thread_backward")


@pytest.mark.parametrize("backend", [BackendType.TURBO, BackendType.TRITON])
@pytest.mark.parametrize("used", ["tokens", "probs", "both"])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_permute_partial_outputs(backend, used, roundtrip):
    torch.manual_seed(717)
    tokens = torch.randn(17, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    probs = torch.rand(17, 4, device="cuda", requires_grad=True)
    routing = torch.rand(17, 4, device="cuda").argsort(dim=1)[:, :2]
    mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(1, routing, True)
    src_expert, src_token = mask.T.nonzero(as_tuple=True)
    result = moe_permute(
        tokens,
        routing_map=mask,
        num_local_experts=4,
        probs=probs,
        probs_layout="routing_map",
        backend=backend,
    )
    perm, row_map, counts, overflow, _, perm_probs = result
    assert overflow.item() == 0
    torch.testing.assert_close(counts, mask.sum(0))
    torch.testing.assert_close(perm, tokens[src_token])
    torch.testing.assert_close(perm_probs, probs[src_token, src_expert])
    tr, pr = tokens.detach().float().requires_grad_(), probs.detach().clone().requires_grad_()
    rt, rp = tr[src_token], pr[src_token, src_expert]
    if roundtrip:
        perm, perm_probs = moe_unpermute(
            perm,
            row_map,
            restore_shape=tokens.shape,
            num_local_experts=4,
            backend=backend,
            permuted_probs=perm_probs,
        )
        rt = torch.zeros_like(tr).index_add(0, src_token, rt)
        rp = torch.zeros_like(pr).index_put((src_token, src_expert), rp)
        torch.testing.assert_close(perm.float(), rt, atol=0.02, rtol=0.01)
        torch.testing.assert_close(perm_probs, rp)
    outs, refs, grads = [], [], []
    for label, actual, expected in (("tokens", perm, rt), ("probs", perm_probs, rp)):
        if used in (label, "both"):
            outs.append(actual)
            refs.append(expected)
            grads.append(torch.randn_like(actual))
    torch.autograd.backward(outs, grads)
    torch.autograd.backward(refs, [g.float() for g in grads])
    for actual, expected in ((tokens, tr), (probs, pr)):
        expected_grad = expected.grad if expected.grad is not None else torch.zeros_like(expected)
        actual_grad = actual.grad if actual.grad is not None else torch.zeros_like(actual)
        torch.testing.assert_close(actual_grad.float(), expected_grad, atol=0.02, rtol=0.01)
