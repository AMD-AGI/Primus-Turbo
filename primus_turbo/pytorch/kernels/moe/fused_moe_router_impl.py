from typing import Tuple

import torch
import triton
from torch.library import triton_op, wrap_triton

from primus_turbo.triton.moe.fused_router_kernel import (
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
) -> torch.Tensor:
    return torch.ops.primus_turbo.fused_moe_router_fwd_triton.default(
        logits, s, e, groups, topk, selected_groups, score_function, scaling_factor
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # todo add warmup
    num_stages = 2
    num_programs = s

    BLOCK_SIZE = e

    topk_aligned = triton.next_power_of_2(topk)
    output_topk_logits = torch.empty((s, topk_aligned), device="cuda", dtype=logits.dtype)
    raw_topk_logits = torch.empty_like(output_topk_logits)
    output_topk_indices = torch.ones((s, topk_aligned), device="cuda", dtype=torch.int64)
    output_scores = torch.empty((s, e), device="cuda", dtype=logits.dtype)

    wrap_triton(fused_scaling_group_sum_routing_kernel)[(num_programs,)](
        logits,
        output_scores,
        output_topk_logits,
        output_topk_indices,
        raw_topk_logits,
        s,
        e,
        groups,
        topk_aligned,
        selected_groups,
        BLOCK_SIZE,
        num_stages,
        0 if score_function == "sigmoid" else 1,
        scaling_factor,
    )

    if topk_aligned != topk:
        output_topk_logits = output_topk_logits[:, :topk]
        output_topk_indices = output_topk_indices[:, :topk]
        raw_topk_logits = raw_topk_logits[:, :topk]

    return output_scores, output_topk_logits, output_topk_indices, raw_topk_logits
