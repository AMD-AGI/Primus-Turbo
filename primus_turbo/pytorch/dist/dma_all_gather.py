from typing import Optional

import torch
import torch.distributed as dist

import primus_turbo.pytorch._C.dist as turbo_dist


def dma_all_gather_into_tensor(
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: Optional[dist.ProcessGroup] = None):
    print("dma_all_gather_into_tensor")
    turbo_dist.dma_all_gather_into_tensor_nobuffer(
        0,
        output_tensor,
        input_tensor,
        dist.get_rank(group),
        0,
        0,
    )
    pass
