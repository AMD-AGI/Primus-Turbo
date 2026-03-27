from typing import Optional, Tuple

import torch
import torch.distributed._symmetric_memory as _symm_mem

import primus_turbo.pytorch._C

lib = torch.library.Library("cco", "DEF")  # noqa: TOR901
lib.define(
    "fused_dispatch_groupedgemm("
    "Tensor x, str group_name, Tensor? x_scales, Tensor? topk_idx, Tensor? topk_weights, int num_experts=-1, *, int num_sms, bool return_x)"
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
    tags=[torch._C.Tag.needs_fixed_stride_order],
)


@torch.library.impl(lib, "fused_dispatch_groupedgemm", "CUDA")
def _fused_dispatch_groupedgemm(
    x: torch.Tensor,
    group_name: str,
    x_scales: Optional[torch.Tensor] = None,
    topk_idx: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    num_experts: Optional[int] = -1,
    *,
    num_sms: int,
    return_x: bool,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    workspace_size = int(4096 * 7168 * 8 * 4)
    symm = _symm_mem.get_symm_mem_workspace(group_name, workspace_size)
    rank = symm.rank
    workspace = symm.get_buffer(rank, [workspace_size], torch.uint8)
    return torch.ops.primus_turbo_cpp_extension.fused_dispatch_groupedgemm(
        x, x_scales, topk_idx, topk_weights, num_experts, workspace, group_name, num_sms
    )
