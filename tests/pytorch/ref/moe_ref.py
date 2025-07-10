import torch


class MoERouterConfig:
    def __init__(self, seqlen: int, experts: int, groups: int, selected_groups: int, topk: int):
        self.seqlen = seqlen
        self.experts = experts
        self.groups = groups
        self.selected_groups = selected_groups
        self.topk = topk


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


def group_topk_routing_with_aux_score_ref(
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
