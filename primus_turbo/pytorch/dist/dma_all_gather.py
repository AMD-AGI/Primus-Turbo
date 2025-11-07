import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

import primus_turbo.pytorch._C.dist as turbo_dist


def dma_all_gather_into_tensor(
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group=None,
        async_op=False):

    group = group or dist.group.WORLD
    group_tag = c10d._get_group_tag(group)
    work = turbo_dist.dma_all_gather_into_tensor(
        output_tensor,
        input_tensor,
        group,
        group_tag,
    )
    if async_op:
        return work
    if work is not None:
        work.wait()
