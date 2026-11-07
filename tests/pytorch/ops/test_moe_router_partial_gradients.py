###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Combined auxiliary/routing losses and optional router defaults."""

import pytest
import torch

from primus_turbo.pytorch.ops.moe import fused_group_topk_routing_with_aux_score
from tests.pytorch.ops.gradient_contract_helpers import caller_thread_backward  # noqa: F401
from tests.pytorch.ref.moe_ref import group_topk_routing_with_aux_score_ref

pytestmark = pytest.mark.usefixtures("caller_thread_backward")


@pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
@pytest.mark.parametrize("loss_outputs", ["scores", "probs", "both"])
@pytest.mark.parametrize("groups,selected,scale", [(None, None, None), (4, 2, 1.7)])
def test_router_partial_losses(score_function, loss_outputs, groups, selected, scale):
    torch.manual_seed(716)
    logits = torch.randn(17, 32, device="cuda", requires_grad=True)
    ref_logits = logits.detach().clone().requires_grad_()
    out = fused_group_topk_routing_with_aux_score(logits, 4, groups, selected, score_function, scale)
    ref = group_topk_routing_with_aux_score_ref(
        ref_logits, 4, groups or 1, selected or 1, score_function, scale or 1.0
    )
    for a, b in zip(out, ref):
        torch.testing.assert_close(a, b)
    indices = {"scores": (0,), "probs": (1,), "both": (0, 1)}[loss_outputs]
    grads = [torch.randn(32, 17, device="cuda").T for _ in indices]
    torch.autograd.backward([out[i] for i in indices], grads)
    torch.autograd.backward([ref[i] for i in indices], grads)
    torch.testing.assert_close(logits.grad, ref_logits.grad)
