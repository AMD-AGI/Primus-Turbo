import pytest
import torch

import primus_turbo.pytorch as pt
from tests.pytorch.ref.moe_ref import MoERouterConfig

test_cases = [
    MoERouterConfig(seqlen=8, experts=64, groups=1, selected_groups=1, topk=8),
    MoERouterConfig(seqlen=8, experts=256, groups=1, selected_groups=1, topk=16),
    MoERouterConfig(seqlen=8, experts=64, groups=4, selected_groups=2, topk=8),
    MoERouterConfig(seqlen=8, experts=256, groups=8, selected_groups=4, topk=16),
]


def group_limited_topk(
    scores: torch.Tensor,
    topk: int,
    num_tokens: int,
    num_experts: int,
    num_groups: int,
    selected_groups: int,
):
    """Perform top-k routing on a subset of expert groups."""
    # Organize the experts into groups
    # Select groups based on sum of top-(topk/group_topk) routing scores within each group
    group_scores = (
        scores.view(num_tokens, num_groups, -1).topk(topk // selected_groups, dim=-1)[0].sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=selected_groups, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)

    # Mask the experts based on selection groups
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, num_experts // num_groups)
        .reshape(num_tokens, -1)
    )

    masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
    probs, top_indices = torch.topk(masked_scores, k=topk, dim=-1)

    return probs, top_indices


def compute_routing_scores_for_aux_loss(logits: torch.Tensor, score_function: str, topk: int) -> torch.Tensor:
    """Compute routing scores based on the score function.

    Args:
        logits (torch.Tensor): The logits tensor after gating, shape: [num_tokens, num_experts].

    Returns:
        torch.Tensor: The normalized routing scores.
    """
    if score_function == "softmax":
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits)

        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")
    return scores


def torch_group_topk_routing_with_aux_score(
    logits: torch.Tensor,
    topk: int,
    num_groups: int,
    selected_groups: int,
    score_function: str,
    scaling_factor: float,
):
    """implemented torch_group_topk and cal aux_score by torch"""
    num_tokens, num_experts = logits.shape

    def compute_topk(scores, topk, num_groups=None, selected_groups=None):
        if selected_groups > 1:
            return group_limited_topk(
                scores=scores,
                topk=topk,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=num_groups,
                selected_groups=selected_groups,
            )
        else:
            return torch.topk(scores, k=topk, dim=1)

    use_pre_softmax = True  # todo add post softmax
    expert_bias = None  # todo add expert bias
    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            probs, top_indices = compute_topk(scores, topk, num_groups, selected_groups)
        else:
            scores, top_indices = compute_topk(logits, topk, num_groups, selected_groups)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float()).type_as(logits)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing, topk, num_groups, selected_groups)
            scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
        else:
            scores, top_indices = compute_topk(scores, topk, num_groups, selected_groups)

        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor

    scores = compute_routing_scores_for_aux_loss(logits, score_function, topk)

    return (
        scores,
        probs,
        top_indices,
    )


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

    ref_scores, ref_topk_logits, refs_topk_indices = torch_group_topk_routing_with_aux_score(
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
