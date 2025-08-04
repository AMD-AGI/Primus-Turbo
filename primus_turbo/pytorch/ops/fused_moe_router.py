import torch

from primus_turbo.pytorch.kernels.moe.fused_moe_router_impl import (
    fused_moe_router_bkwd,
    fused_moe_router_fwd,
)


class FusedGroupTopkRoutingWithAuxScoreFunction(torch.autograd.Function):
    """Fused Scaling GroupTopk and Auxiliary Score Function"""

    @staticmethod
    def forward(
        ctx,
        logits,
        topk: int,
        groups: int,
        selected_groups: int,
        score_function: str = "sigmoid",
        scaling_factor=1.0,
    ):
        s, e = logits.shape
        # only support power of 2 now
        if groups is None:
            groups = 1
            selected_groups = 1
        assert (e & (e - 1)) == 0
        assert (groups & (groups - 1)) == 0
        assert (selected_groups & (selected_groups - 1)) == 0
        if groups > 1:
            assert (topk & (topk - 1)) == 0
        assert selected_groups <= groups
        if scaling_factor is None:
            scaling_factor = 1.0

        output_scores, output_topk_logits, output_topk_indices, raw_topk_logits = fused_moe_router_fwd(
            logits, s, e, groups, topk, selected_groups, score_function, scaling_factor
        )

        ctx.save_for_backward(logits, output_scores, output_topk_indices, output_topk_logits, raw_topk_logits)
        ctx.logit_shape = logits.shape
        ctx.scaling_factor = scaling_factor
        ctx.score_function = score_function
        return output_scores, output_topk_logits, output_topk_indices

    @staticmethod
    def backward(ctx, g_score, g_probs, g_idxs):
        logits, out_scores, topk_indices, topk_logits, raw_topk_logits = ctx.saved_tensors
        logits_grad = torch.zeros(ctx.logit_shape, dtype=topk_logits.dtype, device="cuda")

        if g_score is not None and g_probs is not None:
            g_probs_out, g_score_out = fused_moe_router_bkwd(
                g_probs,
                g_score,
                logits,
                topk_logits,
                raw_topk_logits,
                out_scores,
                ctx.score_function,
                ctx.scaling_factor,
            )

            logits_grad.scatter_(1, topk_indices, g_probs_out)
            if ctx.score_function == "softmax":
                sum_t = torch.sum(logits_grad * out_scores, dim=-1).unsqueeze(-1)
                logits_grad = out_scores * (logits_grad - sum_t)

            logits_grad = logits_grad + g_score_out
            return logits_grad, None, None, None, None, None

        if g_probs is not None:
            if ctx.score_function == "softmax":
                g_probs = ctx.scaling_factor * g_probs
                logits_grad.scatter_(1, topk_indices, g_probs)
                sum_t = torch.sum(logits_grad * out_scores, dim=-1).unsqueeze(-1)
                logits_grad = out_scores * (logits_grad - sum_t)
            else:
                # score / sum(score) grad
                g_probs = ctx.scaling_factor * g_probs
                unscaled_topk_logits = topk_logits / ctx.scaling_factor
                sum_t = (-1) * (g_probs * unscaled_topk_logits * unscaled_topk_logits / raw_topk_logits)
                sum_t = torch.sum(sum_t, dim=-1).unsqueeze(-1)
                g_logits = g_probs * unscaled_topk_logits / raw_topk_logits + sum_t

                # sigmoid grad
                g_logits = g_logits * raw_topk_logits * (1 - raw_topk_logits)

                # scatter_by_idx
                logits_grad.scatter_(1, topk_indices, g_logits)

        if g_score is not None:
            # cal grads of g_score
            if ctx.score_function == "softmax":
                sum_t = torch.sum(g_score * out_scores, dim=-1).unsqueeze(-1)
                grad_x = out_scores * (g_score - sum_t)
            else:
                # score / sum(score) grad (todo-maybe: save the sigmoid logits)
                sigmoid_logits = torch.sigmoid(logits)
                sum_t = (-1) * (g_score * out_scores * out_scores / sigmoid_logits)
                sum_t = torch.sum(sum_t, dim=-1).unsqueeze(-1)
                g_score = g_score * out_scores / sigmoid_logits + sum_t
                # sigmoid grad
                grad_x = g_score * sigmoid_logits * (1 - sigmoid_logits)

            logits_grad = logits_grad + grad_x

        return logits_grad, None, None, None, None, None


def fused_group_topk_routing_with_aux_score(
    logits,
    topk: int,
    groups: int = 1,
    selected_groups: int = 1,
    score_function: str = "sigmoid",
    scaling_factor=1.0,
):
    """
    Fused grouped topk routing with calculating score for moe aux loss
    """
    return FusedGroupTopkRoutingWithAuxScoreFunction.apply(
        logits, topk, groups, selected_groups, score_function, scaling_factor
    )
