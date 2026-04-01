from typing import Optional, Tuple, Union

import torch
import torch.distributed._symmetric_memory as _symm_mem
import torch.distributed.distributed_c10d as c10d

import primus_turbo.pytorch._C as cpp_extensions


def _fused_dispatch(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                    group_name: str,
                    topk_idx: Optional[torch.Tensor] = None,
                    topk_weights: Optional[torch.Tensor] = None,
                    num_experts: Optional[int] = None,
                    expert_alignment: int = 1,
                    *,
                    num_sms: int = None,
                    handle=None,
                    ):
    x, x_scales = x if isinstance(x, tuple) else (x, None)
    assert x.dim() == 2

    group = c10d._resolve_process_group(group_name)
    num_ranks = group.size()

    num_tokens, hidden_size = x.shape
    num_max_tokens = num_tokens * num_ranks
    workspace_size = num_max_tokens * hidden_size * \
        num_ranks * x.element_size() + num_ranks * num_ranks * 4

    symm = _symm_mem.get_symm_mem_workspace(group_name, workspace_size)
    rank = symm.rank
    num_ranks = symm.world_size
    workspace = symm.get_buffer(rank, [workspace_size], torch.uint8)

    rank_prefix_matrix = symm.get_buffer(
        rank, [num_ranks * num_ranks], torch.int32, storage_offset=0)

    offset_bytes = rank_prefix_matrix.nbytes

    recv_x = symm.get_buffer(rank, [num_max_tokens, hidden_size], x.dtype,
                             storage_offset=offset_bytes // x.element_size())

    if topk_idx is not None and topk_weights is not None:
        num_topk = topk_idx.size(1)
        offset_bytes += recv_x.nbytes
        recv_topk_idx = symm.get_buffer(rank, [num_max_tokens, num_topk], torch.int64,
                                        storage_offset=offset_bytes // topk_idx.element_size())
        offset_bytes += recv_topk_idx.nbytes
        recv_topk_weights = symm.get_buffer(rank, [num_max_tokens, num_topk], torch.float32,
                                            storage_offset=offset_bytes // topk_weights.element_size())
    else:
        recv_topk_idx = None
        recv_topk_weights = None

    return *cpp_extensions.fused_dispatch(
        x,
        x_scales,
        topk_idx,
        topk_weights,
        workspace,
        group_name,
        num_experts,
        expert_alignment,
        num_sms,
    ), recv_x, recv_topk_idx, recv_topk_weights


# @torch.library.impl(lib, "fused_dispatch_groupedgemm", "CUDA")
# def _fused_dispatch_groupedgemm(
#     x: torch.Tensor,
#     group_name: str,
#     x_scales: Optional[torch.Tensor] = None,
#     topk_idx: Optional[torch.Tensor] = None,
#     topk_weights: Optional[torch.Tensor] = None,
#     num_experts: Optional[int] = -1,
#     *,
#     num_sms: int,
#     return_x: bool,
# ) -> Tuple[
#     torch.Tensor,
#     Optional[torch.Tensor],
#     Optional[torch.Tensor],
#     Optional[torch.Tensor],
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
#     Optional[torch.Tensor],
#     Optional[torch.Tensor],
#     Optional[torch.Tensor],
# ]:
#     workspace_size = int(4096 * 7168 * 8 * 4)
#     symm = _symm_mem.get_symm_mem_workspace(group_name, workspace_size)
#     rank = symm.rank
#     num_ranks = symm.world_size
#     workspace = symm.get_buffer(rank, [workspace_size], torch.uint8)

#     num_tokens = x.size(0)

#     return fused_dispatch_groupedgemm_cpp(
#         x, x_scales, topk_idx, topk_weights, num_experts, workspace, group_name, num_sms
#     )
