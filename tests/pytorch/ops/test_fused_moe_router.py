import pytest
import torch

import primus_turbo.pytorch as pt
from tests.pytorch.ref.moe_ref import (
    MoERouterConfig,
    group_topk_routing_with_aux_score_ref,
)

test_cases = [
    MoERouterConfig(seqlen=8, experts=64, groups=1, selected_groups=1, topk=6),
    MoERouterConfig(seqlen=8, experts=64, groups=1, selected_groups=1, topk=8),
    MoERouterConfig(seqlen=8, experts=256, groups=1, selected_groups=1, topk=16),
    MoERouterConfig(seqlen=8, experts=256, groups=1, selected_groups=1, topk=20),
    MoERouterConfig(seqlen=8, experts=64, groups=4, selected_groups=2, topk=8),
    MoERouterConfig(seqlen=8, experts=256, groups=8, selected_groups=4, topk=16),
]


@pytest.mark.parametrize("batch", [4])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
@pytest.mark.parametrize("scaling_factor", [1.2])
def test_moe_router(batch, config, score_function, scaling_factor):
    # warning: use bfloat16 or lower precision for testing may issue some difference due to the unstable sort
    #          , which supposing having litte effect on final convergence.
    #          float32 may have the same issue at very low probability.
    logits_1 = torch.randn(
        batch * config.seqlen, config.experts, dtype=torch.float, device="cuda", requires_grad=True
    )
    logits_2 = logits_1.clone().detach().requires_grad_(True)

    output_scores, output_topk_logits, output_topk_indices = pt.ops.fused_group_topk_routing_with_aux_score(
        logits_1, config.topk, config.groups, config.selected_groups, score_function, scaling_factor
    )

    ref_scores, ref_topk_logits, refs_topk_indices = group_topk_routing_with_aux_score_ref(
        logits_2, config.topk, config.groups, config.selected_groups, score_function, scaling_factor
    )
    # forward
    torch.testing.assert_close(output_scores, ref_scores)
    assert torch.equal(output_topk_indices, refs_topk_indices)
    torch.testing.assert_close(output_topk_logits, ref_topk_logits)

    # backward
    g_score = torch.randn(batch * config.seqlen, config.experts, dtype=torch.float, device="cuda")
    g_probs = torch.randn(batch * config.seqlen, config.topk, dtype=torch.float, device="cuda")

    output_scores.backward(g_score, retain_graph=True)
    output_topk_logits.backward(g_probs)

    ref_scores.backward(g_score, retain_graph=True)
    ref_topk_logits.backward(g_probs)

    torch.testing.assert_close(logits_1.grad, logits_2.grad)
