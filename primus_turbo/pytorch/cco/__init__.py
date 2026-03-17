import torch

import primus_turbo.pytorch as turbo

from typing import Tuple

lib = torch.library.Library("cco", "DEF")  # noqa: TOR901
lib.define(
    "fused_dispatch_groupedgemm("
    "Tensor A, Tensor B, Tensor token_indices, Tensor token_probs, int num_experts, str group_name, *, bool return_A = True) -> (Tensor?, Tensor?)",
    tags=[torch._C.Tag.needs_fixed_stride_order],
)

@torch.library.impl(lib, "fused_dispatch_groupedgemm", "CUDA")
def _fused_dispatch_groupedgemm(A: torch.Tensor,
                           B: torch.Tensor,
                           input_routing_map: torch.Tensor,
                           token_probs: torch.Tensor,
                           num_experts: int,
                           group_name: str,                      
                           *,
                           return_A: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
     
    group = torch.distributed.group.WORLD
    global_routing_map = torch.empty(
        input_routing_map.size(0) * group.size(),
        input_routing_map.size(1),
        device=input_routing_map.device,
        dtype=input_routing_map.dtype,
    )
    torch.distributed.all_gather_into_tensor(global_routing_map, input_routing_map, group)
    (sparse_to_dense_map, rdma_to_attn_map,
                         attn_to_rdma_map, num_of_tokens_for_experts,
                         local_expert_routing_map) = turbo._C.get_dispatch_layout(
        global_routing_map=global_routing_map,
        node_rank=0,       # TODO: get from distributed config
        local_rank=group.rank(),# TODO: get from distributed config
        num_of_tokens_per_rank=A.size(0),
    )
    return sparse_to_dense_map, num_of_tokens_for_experts