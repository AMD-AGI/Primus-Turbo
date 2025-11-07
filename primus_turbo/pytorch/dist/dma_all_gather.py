import torch
import torch.distributed as dist

import primus_turbo.pytorch._C.dist as turbo_dist


def dma_all_gather_into_tensor(
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group=None,
        async_op=False):

    group = group or dist.group.WORLD
    work = turbo_dist.dma_all_gather_into_tensor(
        output_tensor,
        input_tensor,
        group,
    )
    if async_op:
        return work
    if work is not None:
        work.wait()
