###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``MegaMoE``: nn.Module wrapper over the fully fused mega MoE expert compute (routing-free)."""

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from primus_turbo.pytorch.ops.moe.mega_moe_fused import mega_moe_fused

__all__ = ["MegaMoE"]


class MegaMoE(nn.Module):
    """Fused expert-parallel MoE expert compute (routing-free; num_experts is global).

    Routing is computed externally (e.g. Megatron's native ``TopKRouter``) and passed in
    as ``topk_idx``/``topk_weights``; this module only runs the fully fused
    dispatch -> L1 GEMM -> SwiGLU -> L2 GEMM -> combine (+ optional shared expert).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        ep_group: torch.distributed.group,
        *,
        shared_expert_intermediate_size: Optional[int] = None,
        shared_expert_gate: bool = False,
        init_method: Optional[Callable] = None,
        output_layer_init_method: Optional[Callable] = None,
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
        self.experts_per_rank = num_experts // self.ep_size
        # this rank's global expert ids (Megatron local_expert_indices parity)
        offset = self.ep_rank * self.experts_per_rank
        self.local_expert_indices = [offset + i for i in range(self.experts_per_rank)]

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.dtype = dtype or torch.bfloat16
        assert self.dtype == torch.bfloat16, "MegaMoE fused kernel only supports bfloat16"
        self.layer_number = None
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        factory_kwargs = {"device": device, "dtype": self.dtype}

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
        """Init expert weights via init_method (w2 via output_layer_init_method), else fan_in normal."""
        init1 = self.init_method
        init2 = self.output_layer_init_method or self.init_method
        with torch.no_grad():
            if init1 is not None:
                init1(self.w1)
                (init2 or init1)(self.w2)
            else:
                self.w1.normal_(mean=0.0, std=2.0 / math.sqrt(self.hidden_size))
                self.w2.normal_(mean=0.0, std=2.0 / math.sqrt(self.intermediate_size))
            if self.shared_w1 is not None:
                (init1 or (lambda w: w.normal_(std=2.0 / math.sqrt(self.hidden_size))))(self.shared_w1)
                (init2 or (lambda w: w.normal_(std=2.0 / math.sqrt(self.shared_expert_intermediate_size))))(
                    self.shared_w2
                )
                if self.shared_gate_weight is not None:
                    (init1 or (lambda w: nn.init.kaiming_uniform_(w, a=math.sqrt(5))))(
                        self.shared_gate_weight
                    )

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
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Fused expert compute for pre-routed tokens (+ optional shared expert).

        ``topk_idx`` [T, K] int32 and ``topk_weights`` [T, K] float32 are the routing
        decision produced upstream; returns y with the same leading shape as ``hidden_states``.
        """
        in_shape = hidden_states.shape
        in_dtype = hidden_states.dtype
        x = hidden_states.reshape(-1, self.hidden_size).to(self.dtype)
        topk_idx = topk_idx.reshape(-1, topk_idx.shape[-1]).to(torch.int32)
        topk_weights = topk_weights.reshape(-1, topk_weights.shape[-1]).to(torch.float32)
        y = self.expert_compute(x, topk_idx, topk_weights)
        if self.shared_w1 is not None:
            y = y + self._shared_expert(x)
        return y.reshape(in_shape).to(in_dtype)

    def set_layer_number(self, layer_number: int) -> None:
        """Megatron parity: record the transformer layer index."""
        self.layer_number = layer_number

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
            f"num_experts={self.num_experts}, "
            f"experts_per_rank={self.experts_per_rank}, ep_size={self.ep_size}"
        )
