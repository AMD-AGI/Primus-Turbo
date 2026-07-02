###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``MegaMoE``: nn.Module wrapper over the fully fused mega MoE op."""

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from primus_turbo.pytorch.modules.moe.mega_moe_utils import apply_random_logits
from primus_turbo.pytorch.ops.moe.fused_moe_router import (
    fused_group_topk_routing_with_aux_score,
)
from primus_turbo.pytorch.ops.moe.mega_moe_fused import mega_moe_fused

__all__ = ["MegaMoE"]


class MegaMoE(nn.Module):
    """Fused expert-parallel MoE layer; routing-free or self-routing."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        ep_group: torch.distributed.ProcessGroup,
        *,
        top_k: Optional[int] = None,
        num_groups: Optional[int] = None,
        topk_group: Optional[int] = None,
        score_function: str = "sigmoid",
        routed_scaling_factor: float = 1.0,
        force_load_balancing: bool = False,
        add_bias_linear: bool = False,
        enable_expert_bias: bool = False,
        shared_expert_intermediate_size: Optional[int] = None,
        shared_expert_gate: bool = False,
        init_method: Optional[Callable[[torch.Tensor], None]] = None,
        output_layer_init_method: Optional[Callable[[torch.Tensor], None]] = None,
        router_dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = torch.bfloat16,
    ) -> None:
        super().__init__()

        # EP topology: experts sharded evenly across the EP group
        self.ep_group: torch.distributed.ProcessGroup = ep_group
        self.ep_size: int = ep_group.size()
        self.ep_rank: int = ep_group.rank()
        assert self.ep_size > 0, "Expected positive expert parallel size"
        assert num_experts % self.ep_size == 0, "num_experts must be divisible by EP size"
        self.experts_per_rank: int = num_experts // self.ep_size
        # this rank's global expert ids
        offset = self.ep_rank * self.experts_per_rank
        self.local_expert_indices: list[int] = [offset + i for i in range(self.experts_per_rank)]

        self.hidden_size: int = hidden_size
        self.intermediate_size: int = intermediate_size
        self.num_experts: int = num_experts
        self.dtype: torch.dtype = dtype or torch.bfloat16
        assert self.dtype == torch.bfloat16, "MegaMoE fused kernel only supports bfloat16"
        self.init_method: Optional[Callable[[torch.Tensor], None]] = init_method
        self.output_layer_init_method: Optional[Callable[[torch.Tensor], None]] = output_layer_init_method

        # optional internal router; None top_k => routing done upstream
        self.top_k: Optional[int] = top_k
        self.has_router: bool = top_k is not None
        self.num_groups: Optional[int] = num_groups
        self.topk_group: Optional[int] = topk_group
        assert score_function in ("sigmoid", "softmax"), f"unsupported score_function {score_function}"
        self.score_function: str = score_function
        self.routed_scaling_factor: float = routed_scaling_factor
        self.force_load_balancing: bool = force_load_balancing
        self.router_dtype: torch.dtype = router_dtype
        if self.has_router:
            assert top_k <= num_experts, "top_k cannot exceed num_experts"
            # group-limited routing needs power-of-2 groups
            if num_groups is not None and num_groups > 1:
                assert num_groups & (num_groups - 1) == 0, "num_groups must be a power of 2"
                assert num_experts % num_groups == 0, "num_experts must be divisible by num_groups"
                assert topk_group is not None and 0 < topk_group <= num_groups, "invalid topk_group"
                assert top_k % topk_group == 0, "top_k must be divisible by topk_group"

        factory_kwargs = {"device": device, "dtype": self.dtype}

        # routing gate weight [E, H] + optional logit bias [E], fp32 for stability
        if self.has_router:
            self.gate_weight = nn.Parameter(
                torch.empty((num_experts, hidden_size), device=device, dtype=torch.float32)
            )
            self.gate_bias = (
                nn.Parameter(torch.empty((num_experts,), device=device, dtype=torch.float32))
                if add_bias_linear
                else None
            )
            # aux-free balancing not supported: the fused router does not inject
            # expert_bias into the selection score, so it would be a silent no-op
            if enable_expert_bias:
                raise NotImplementedError(
                    "MegaMoE does not support enable_expert_bias (aux-loss-free balancing): "
                    "the fused router ignores expert_bias for expert selection. "
                    "Set moe_router_enable_expert_bias=False or use the non-fused router."
                )
            self.expert_bias = None
            self.local_tokens_per_expert = None
        else:
            self.gate_weight = None
            self.gate_bias = None
            self.expert_bias = None
            self.local_tokens_per_expert = None

        # per-rank expert shard: w1 [g, 2I, H] (gate+up), w2 [g, H, I]
        self.w1 = nn.Parameter(
            torch.empty((self.experts_per_rank, 2 * intermediate_size, hidden_size), **factory_kwargs)
        )
        self.w2 = nn.Parameter(
            torch.empty((self.experts_per_rank, hidden_size, intermediate_size), **factory_kwargs)
        )
        # expert-sharded weights: exclude from dense DP all-reduce (EP>1).
        # Megatron reads param.allreduce for DDP bucketing/optimizer grouping;
        # untagged w1/w2 would be reduced across EP as if replicated.
        expert_parallel = self.ep_size > 1
        for param in (self.w1, self.w2):
            param.allreduce = not expert_parallel

        # optional shared expert (replicated, run on every token)
        self.shared_expert_intermediate_size: Optional[int] = shared_expert_intermediate_size
        self.shared_expert_gate: bool = shared_expert_gate
        if shared_expert_intermediate_size is not None:
            s = shared_expert_intermediate_size
            self.shared_w1 = nn.Parameter(torch.empty((2 * s, hidden_size), **factory_kwargs))
            self.shared_w2 = nn.Parameter(torch.empty((hidden_size, s), **factory_kwargs))
            self.shared_gate_weight = (
                nn.Parameter(torch.empty((1, hidden_size), **factory_kwargs)) if shared_expert_gate else None
            )
        else:
            self.shared_w1 = self.shared_w2 = self.shared_gate_weight = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Init gate + expert weights via init_method, else fan_in normal."""
        init1 = self.init_method
        init2 = self.output_layer_init_method or self.init_method
        with torch.no_grad():
            if self.gate_weight is not None:
                (init1 or (lambda w: nn.init.kaiming_uniform_(w, a=math.sqrt(5))))(self.gate_weight)
            # init w1/w2 independently so output_layer_init_method is honored
            # even when init_method is None
            if init1 is not None:
                init1(self.w1)
            else:
                self.w1.normal_(mean=0.0, std=2.0 / math.sqrt(self.hidden_size))
            if init2 is not None:
                init2(self.w2)
            else:
                self.w2.normal_(mean=0.0, std=2.0 / math.sqrt(self.intermediate_size))
            if self.gate_bias is not None:
                self.gate_bias.zero_()
            if self.expert_bias is not None:
                self.expert_bias.zero_()
            if self.local_tokens_per_expert is not None:
                self.local_tokens_per_expert.zero_()
            if self.shared_w1 is not None:
                (init1 or (lambda w: w.normal_(std=2.0 / math.sqrt(self.hidden_size))))(self.shared_w1)
                (init2 or (lambda w: w.normal_(std=2.0 / math.sqrt(self.shared_expert_intermediate_size))))(
                    self.shared_w2
                )
                if self.shared_gate_weight is not None:
                    (init1 or (lambda w: nn.init.kaiming_uniform_(w, a=math.sqrt(5))))(
                        self.shared_gate_weight
                    )

    def route(
        self,
        hidden_states: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route tokens -> (topk_idx [T,K] i32, topk_weights [T,K] f32)."""
        assert self.has_router, "MegaMoE built routing-free (top_k=None); no internal router"
        x = hidden_states.to(self.router_dtype)
        assert valid_mask is None or valid_mask.shape[0] == x.shape[0], "valid_mask/token count mismatch"
        gate_bias = self.gate_bias.to(self.router_dtype) if self.gate_bias is not None else None
        # gate logits [T, E]
        logits = F.linear(x, self.gate_weight.to(self.router_dtype), gate_bias)
        if self.force_load_balancing:
            # benchmark: random logits via Megatron RandomSTE
            logits = apply_random_logits(logits)
        # fused score fn + group-limited top-k; probs/routing_map dense [T,E]
        _, probs, routing_map = fused_group_topk_routing_with_aux_score(
            logits,
            self.top_k,
            self.num_groups or 1,
            self.topk_group or 1,
            self.score_function,
            self.routed_scaling_factor,
        )
        # dense -> sparse: recover selected experts + weights
        topk_weights, topk_idx = probs.topk(self.top_k, dim=-1)
        # zero out padded tokens so they contribute nothing through combine
        if valid_mask is not None:
            topk_weights = topk_weights * valid_mask.to(topk_weights.dtype).unsqueeze(-1)
        # accumulate per-expert counts for aux-free balancing (skip random logits)
        if (
            self.local_tokens_per_expert is not None
            and torch.is_grad_enabled()
            and not self.force_load_balancing
        ):
            with torch.no_grad():
                rmap = routing_map if valid_mask is None else routing_map & valid_mask.unsqueeze(-1)
                self.local_tokens_per_expert += rmap.sum(dim=0).to(self.local_tokens_per_expert.dtype)
        return topk_idx.to(torch.int32), topk_weights.to(torch.float32)

    def expert_compute(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Fused dispatch -> L1 -> SwiGLU -> L2 -> combine -> reduce; y [T, hidden]."""
        return mega_moe_fused(self.ep_group, x, topk_idx, topk_weights, self.w1, self.w2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        *,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fused expert compute (+ optional shared expert); y keeps input leading shape."""
        in_shape = hidden_states.shape
        in_dtype = hidden_states.dtype
        flat = hidden_states.reshape(-1, self.hidden_size)
        if topk_idx is None or topk_weights is None:
            # Megatron parity: padding_mask [seq, bsz] True = PADDING; invert to valid_mask [T]
            valid_mask = None
            if padding_mask is not None:
                valid_mask = (~padding_mask.reshape(-1).bool()).to(flat.device)
            # route on original precision
            topk_idx, topk_weights = self.route(flat, valid_mask)
        else:
            # external topk path does not consume padding_mask; caller must mask upstream
            assert padding_mask is None, (
                "padding_mask is ignored when topk_idx/topk_weights are provided; "
                "zero out padded tokens' weights upstream"
            )
            topk_idx = topk_idx.reshape(-1, topk_idx.shape[-1]).to(torch.int32)
            topk_weights = topk_weights.reshape(-1, topk_weights.shape[-1]).to(torch.float32)
        x = flat.to(self.dtype)
        y = self.expert_compute(x, topk_idx, topk_weights)
        if self.shared_w1 is not None:
            y = y + self._shared_expert(x)
        return y.reshape(in_shape).to(in_dtype)

    def _shared_expert(self, x: torch.Tensor) -> torch.Tensor:
        """Replicated SwiGLU shared expert run on every token (optional gate)."""
        # shared_w1 row-packed [gate; up]: silu(first half) * second half
        gate, up = F.linear(x, self.shared_w1).chunk(2, dim=-1)
        out = F.linear(F.silu(gate) * up, self.shared_w2)
        if self.shared_gate_weight is not None:
            # fp32 sigmoid for numeric stability (Megatron parity)
            out = torch.sigmoid(F.linear(x, self.shared_gate_weight).float()).to(out.dtype) * out
        return out

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}, "
            f"num_experts={self.num_experts}, top_k={self.top_k}, "
            f"experts_per_rank={self.experts_per_rank}, ep_size={self.ep_size}"
        )
