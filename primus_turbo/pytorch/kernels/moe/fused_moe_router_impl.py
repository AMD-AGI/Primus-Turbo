from typing import Tuple

import torch
import triton
from torch.library import triton_op, wrap_triton

from primus_turbo.triton.moe.fused_router_kernel import (
    fused_scaling_group_sum_routing_backward_kernel,
    fused_scaling_group_sum_routing_kernel,
)


def fused_moe_router_fwd(
    logits: torch.Tensor,
    s: int,
    e: int,
    groups: int,
    topk: int,
    selected_groups: int,
    score_function: str,
    scaling_factor: float,
):
    return torch.ops.primus_turbo.fused_moe_router_fwd_triton.default(
        logits, s, e, groups, topk, selected_groups, score_function, scaling_factor
    )


def fused_moe_router_bkwd(
    g_probs: torch.Tensor,
    g_scores: torch.Tensor,
    logits: torch.Tensor,
    output_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    raw_topk_logits: torch.Tensor,
    out_scores: torch.Tensor,
    routing_map: torch.Tensor,
    score_function: str,
    scaling_factor: float,
):
    return torch.ops.primus_turbo.fused_moe_router_bkwd_triton.default(
        g_probs,
        g_scores,
        logits,
        output_probs,
        topk_indices,
        raw_topk_logits,
        out_scores,
        routing_map,
        score_function,
        scaling_factor,
    )


@triton_op("primus_turbo::fused_moe_router_fwd_triton", mutates_args={})
def fused_moe_router_fwd_triton(
    logits: torch.Tensor,
    s: int,
    e: int,
    groups: int,
    topk: int,
    selected_groups: int,
    score_function: str,
    scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # todo add warmup
    num_stages = 2
    num_programs = s

    BLOCK_SIZE = e
    K_ALIGNED = triton.next_power_of_2(topk)

    # output_topk_logits = torch.empty((s, topk), device="cuda", dtype=logits.dtype)
    raw_topk_logits = torch.empty((s, topk), device="cuda", dtype=logits.dtype)
    output_topk_indices = torch.ones((s, topk), device="cuda", dtype=torch.int64)
    output_scores = torch.empty((s, e), device="cuda", dtype=logits.dtype)

    output_probs = torch.zeros((s, e), device="cuda", dtype=logits.dtype)
    output_routing_map = torch.zeros((s, e), device="cuda", dtype=torch.int32)

    wrap_triton(fused_scaling_group_sum_routing_kernel)[(num_programs,)](
        logits,
        output_scores,
        # output_topk_logits,
        output_topk_indices,
        raw_topk_logits,
        output_probs,
        output_routing_map,
        s,
        e,
        groups,
        topk,
        selected_groups,
        K_ALIGNED,
        BLOCK_SIZE,
        num_stages,
        0 if score_function == "sigmoid" else 1,
        scaling_factor,
    )

    return output_scores, output_topk_indices, raw_topk_logits, output_probs, output_routing_map


@triton_op("primus_turbo::fused_moe_router_bkwd_triton", mutates_args={})
def fused_moe_router_bkwd_triton(
    g_probs: torch.Tensor,
    g_scores: torch.Tensor,
    logits: torch.Tensor,
    output_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    raw_topk_logits: torch.Tensor,
    out_scores: torch.Tensor,
    routing_map: torch.Tensor,
    score_function: str,
    scaling_factor: float,
) -> torch.Tensor:
    s, e = out_scores.shape
    k = topk_indices.shape[1]

    num_stages = 2
    num_programs = s

    BLOCK_SIZE = e
    K_ALIGNED = triton.next_power_of_2(k)

    g_probs = g_probs.contiguous()
    g_scores = g_scores.contiguous()

    output_g_probs = torch.zeros_like(g_probs)
    output_g_scores = torch.empty_like(g_scores)

    wrap_triton(fused_scaling_group_sum_routing_backward_kernel)[(num_programs,)](
        g_probs,
        g_scores,
        logits,
        output_probs,
        topk_indices,
        raw_topk_logits,
        out_scores,
        routing_map,
        output_g_probs,
        output_g_scores,
        s,
        e,
        k,
        K_ALIGNED,
        BLOCK_SIZE,
        num_stages,
        0 if score_function == "sigmoid" else 1,
        scaling_factor,
    )

    output_g_logits = output_g_probs + output_g_scores

    return output_g_logits
