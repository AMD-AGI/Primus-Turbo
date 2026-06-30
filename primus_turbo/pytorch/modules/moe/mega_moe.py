###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``MegaMoE``: nn.Module wrapper over the fully fused mega MoE op (route + expert_compute)."""

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
    """Fused expert-parallel mixture-of-experts layer (num_experts is global)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        ep_group: torch.distributed.group,
        *,
        num_groups: Optional[int] = None,
        topk_group: Optional[int] = None,
        score_function: str = "sigmoid",
        routed_scaling_factor: float = 1.0,
        force_load_balancing: bool = False,
        add_bias_linear: bool = False,
        enable_expert_bias: bool = False,
        shared_expert_intermediate_size: Optional[int] = None,
        shared_expert_gate: bool = False,
        init_method: Optional[Callable] = None,
        output_layer_init_method: Optional[Callable] = None,
        router_dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = torch.bfloat16,
    ) -> None:
        super().__init__()

        # EP topology: experts sharded evenly across the EP group
        self.ep_group = ep_group
        self.ep_size = ep_group.size()
        self.ep_rank = ep_group.rank()
        assert self.ep_size > 0, "Expected non-negative expert parallel size"
        assert num_experts % self.ep_size == 0, "num_experts must divide EP size"
        assert top_k <= num_experts, "top_k cannot exceed num_experts"
        self.experts_per_rank = num_experts // self.ep_size
        # this rank's global expert ids (Megatron local_expert_indices parity)
        offset = self.ep_rank * self.experts_per_rank
        self.local_expert_indices = [offset + i for i in range(self.experts_per_rank)]

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.dtype = dtype or torch.bfloat16
        assert self.dtype == torch.bfloat16, "MegaMoE fused kernel only supports bfloat16"
        self.layer_number = None

        # routing config
        self.num_groups = num_groups
        self.topk_group = topk_group
        assert score_function in ("sigmoid", "softmax"), f"unsupported score_function {score_function}"
        self.score_function = score_function
        self.routed_scaling_factor = routed_scaling_factor
        self.force_load_balancing = force_load_balancing
        self.router_dtype = router_dtype
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method
        # group-limited routing constraints (fused router op needs power-of-2 groups)
        if num_groups is not None and num_groups > 1:
            assert num_groups & (num_groups - 1) == 0, "num_groups must be a power of 2"
            assert num_experts % num_groups == 0, "num_experts must divide num_groups"
            assert topk_group is not None and 0 < topk_group <= num_groups, "invalid topk_group"
            assert top_k % topk_group == 0, "top_k must be divisible by topk_group"

        factory_kwargs = {"device": device, "dtype": self.dtype}

        # routing gate weight [E, H]; fp32 for routing stability
        self.gate_weight = nn.Parameter(
            torch.empty((num_experts, hidden_size), device=device, dtype=torch.float32)
        )
        # optional router gating bias [E] added to logits
        self.gate_bias = (
            nn.Parameter(torch.empty((num_experts,), device=device, dtype=torch.float32))
            if add_bias_linear
            else None
        )
        # TODO(zhuang12): aux-free balancing state: counter only (fused router op does not bias selection, mirrors Primus)
        # local_tokens_per_expert accumulates selected counts (trainer updates expert_bias from it)
        if enable_expert_bias:
            self.register_buffer(
                "expert_bias", torch.zeros((num_experts,), device=device, dtype=torch.float32)
            )
            self.register_buffer(
                "local_tokens_per_expert",
                torch.zeros((num_experts,), device=device, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.expert_bias = None
            self.local_tokens_per_expert = None

        # per-rank expert shard: w1 [g, 2I, H] (gate+up), w2 [g, H, I]
        self.w1 = nn.Parameter(
            torch.empty((self.experts_per_rank, 2 * intermediate_size, hidden_size), **factory_kwargs)
        )
        self.w2 = nn.Parameter(
            torch.empty((self.experts_per_rank, hidden_size, intermediate_size), **factory_kwargs)
        )

        # optional shared expert (replicated, run on every token, added to output)
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.shared_expert_gate = shared_expert_gate
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
        """Init gate + expert weights via init_method (w2 via output_layer_init_method), else fan_in normal."""
        init1 = self.init_method
        init2 = self.output_layer_init_method or self.init_method
        with torch.no_grad():
            if init1 is not None:
                init1(self.gate_weight)
                init1(self.w1)
                (init2 or init1)(self.w2)
            else:
                nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))
                self.w1.normal_(mean=0.0, std=2.0 / math.sqrt(self.hidden_size))
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

    # Per-stage encapsulated APIs
    def route(
        self,
        hidden_states: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route tokens -> (topk_idx [T,K] i32, topk_weights [T,K] f32) via the turbo fused router.

        valid_mask [T] bool (True = real token): padded tokens get zero weight and are
        excluded from the load-balance counter (Megatron padding_mask parity).
        """
        x = hidden_states.to(self.router_dtype)
        assert valid_mask is None or valid_mask.shape[0] == x.shape[0], "valid_mask/token count mismatch"
        gate_bias = self.gate_bias.to(self.router_dtype) if self.gate_bias is not None else None
        logits = F.linear(x, self.gate_weight.to(self.router_dtype), gate_bias)  # [T, E]
        if self.force_load_balancing:
            logits = apply_random_logits(logits)  # benchmark: random logits via Megatron RandomSTE
        # fused score fn + group-limited top-k + scaling; returns dense [T,E] probs (op has no expert_bias arg)
        _, probs, _ = fused_group_topk_routing_with_aux_score(
            logits,
            self.top_k,
            self.num_groups or 1,
            self.topk_group or 1,
            self.score_function,
            self.routed_scaling_factor,
        )
        # dense -> sparse: top-k recovers the selected experts + their weights (op hides its indices)
        topk_weights, topk_idx = probs.topk(self.top_k, dim=-1)
        # zero out padded tokens so they contribute nothing through combine
        if valid_mask is not None:
            topk_weights = topk_weights * valid_mask.to(topk_weights.dtype).unsqueeze(-1)
        # accumulate per-expert selection counts for aux-free balancing (trainer updates the bias)
        # skip under force_load_balancing: random selections would pollute the counter
        if (
            self.local_tokens_per_expert is not None
            and self.training
            and torch.is_grad_enabled()
            and not self.force_load_balancing
        ):
            with torch.no_grad():
                counted_idx = topk_idx if valid_mask is None else topk_idx[valid_mask]
                self.local_tokens_per_expert += torch.bincount(
                    counted_idx.flatten(), minlength=self.num_experts
                ).to(self.local_tokens_per_expert.dtype)
        return topk_idx.to(torch.int32), topk_weights.to(torch.float32)

    def expert_compute(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Fused dispatch -> L1 GEMM -> SwiGLU -> L2 GEMM -> combine -> reduce; returns y [T, hidden]."""
        return mega_moe_fused(
            self.ep_group, x, topk_idx, topk_weights, self.w1, self.w2, num_experts=self.num_experts
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """MoE forward: route -> fused expert compute (+ shared expert).

        Matches Megatron MoELayer: returns ``(output, mlp_bias)``; mlp_bias is always None.
        """
        in_shape = hidden_states.shape
        in_dtype = hidden_states.dtype
        flat = hidden_states.reshape(-1, self.hidden_size)
        # Megatron parity: padding_mask True = PADDING; invert to valid_mask [T] (True = real token)
        valid_mask = None
        if padding_mask is not None:
            valid_mask = (~padding_mask.transpose(0, 1).reshape(-1).bool()).to(flat.device)
        topk_idx, topk_weights = self.route(flat, valid_mask)  # route on original precision (Megatron parity)
        x = flat.to(self.dtype)
        y = self.expert_compute(x, topk_idx, topk_weights)
        if self.shared_w1 is not None:
            y = y + self._shared_expert(x)
        return y.reshape(in_shape).to(in_dtype), None

    def set_layer_number(self, layer_number: int) -> None:
        """Megatron parity: record the transformer layer index."""
        self.layer_number = layer_number

    # Routing helpers
    def _shared_expert(self, x: torch.Tensor) -> torch.Tensor:
        """Replicated SwiGLU shared expert run on every token (optional sigmoid gate)."""
        # shared_w1 row-packed [gate; up] to match Megatron swiglu: silu(first half) * second half
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
