###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

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

        output_scores, output_topk_indices, raw_topk_logits, output_probs, output_routing_map = (
            fused_moe_router_fwd(logits, s, e, groups, topk, selected_groups, score_function, scaling_factor)
        )

        ctx.save_for_backward(
            logits, output_scores, output_topk_indices, output_probs, raw_topk_logits, output_routing_map
        )
        ctx.logit_shape = logits.shape
        ctx.scaling_factor = scaling_factor
        ctx.score_function = score_function
        return output_scores, output_probs, output_routing_map.bool()

    @staticmethod
    def backward(ctx, g_score, g_probs, g_routing_map):
        logits, output_scores, output_topk_indices, output_probs, raw_topk_logits, output_routing_map = (
            ctx.saved_tensors
        )
        if g_score is not None and g_probs is not None:
            logits_grad = fused_moe_router_bkwd(
                g_probs,
                g_score,
                logits,
                output_probs,
                output_topk_indices,
                raw_topk_logits,
                output_scores,
                output_routing_map,
                ctx.score_function,
                ctx.scaling_factor,
            )
            return logits_grad, None, None, None, None, None

        g_probs = output_routing_map.to(torch.bfloat16) * g_probs
        logits_grad = torch.zeros(ctx.logit_shape, dtype=logits.dtype, device="cuda")
        if g_probs is not None:
            raw_topk_logits_t = torch.ones_like(g_probs)
            raw_topk_logits_t.scatter_(1, output_topk_indices, raw_topk_logits)
            if ctx.score_function == "softmax":
                g_probs = ctx.scaling_factor * g_probs
                sum_t = torch.sum(g_probs * output_scores, dim=-1).unsqueeze(-1)
                logits_grad = output_scores * (g_probs - sum_t)
            else:
                # score / sum(score) grad
                g_probs = ctx.scaling_factor * g_probs
                unscaled_topk_logits = output_probs / ctx.scaling_factor
                sum_t = (-1) * (g_probs * unscaled_topk_logits * unscaled_topk_logits / raw_topk_logits_t)
                sum_t = torch.sum(sum_t, dim=-1).unsqueeze(-1)
                g_logits = g_probs * unscaled_topk_logits / raw_topk_logits_t + sum_t

                # sigmoid
                logits_grad = g_logits * raw_topk_logits_t * (1 - raw_topk_logits_t)

        if g_score is not None:
            # cal grads of g_score
            if ctx.score_function == "softmax":
                sum_t = torch.sum(g_score * output_scores, dim=-1).unsqueeze(-1)
                grad_x = output_scores * (g_score - sum_t)
            else:
                # score / sum(score) grad (todo-maybe: save the sigmoid logits)
                sigmoid_logits = torch.sigmoid(logits)
                sum_t = (-1) * (g_score * output_scores * output_scores / sigmoid_logits)
                sum_t = torch.sum(sum_t, dim=-1).unsqueeze(-1)
                g_score = g_score * output_scores / sigmoid_logits + sum_t
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
