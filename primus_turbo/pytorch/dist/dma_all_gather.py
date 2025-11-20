###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

import primus_turbo.pytorch._C.dist as turbo_dist

origin_all_gather_into_tensor = dist.all_gather_into_tensor


def _fallback_to_torch(output_tensor: torch.Tensor, input_tensor: torch.Tensor, group, async_op):
    if async_op:
        return True

    if not (output_tensor.is_cuda and input_tensor.is_cuda):
        return True

    if output_tensor.device != input_tensor.device:
        return True

    if not (output_tensor.is_contiguous() and input_tensor.is_contiguous()):
        return True

    # check all ranks are in a single node
    # dict of (global_rank: local_rank)
    rank_dict = c10d._world.pg_group_ranks[group]
    maxdiff = max(rank_dict) - min(rank_dict)
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", f"{torch.cuda.device_count()}"))
    if maxdiff >= local_world_size:
        return True

    return False


def dma_all_gather_into_tensor(
    output_tensor: torch.Tensor, input_tensor: torch.Tensor, group=None, async_op=False
):
    group = group or dist.group.WORLD

    if _fallback_to_torch(output_tensor, input_tensor, group, async_op):
        return origin_all_gather_into_tensor(output_tensor, input_tensor, group, async_op)

    # TODO (limou)
    # if multiple training jobs are running concurrently on the same node, group_tag may be duplicated.
    # currently, only the single-training-job scenario is considered.
    group_tag = c10d._get_group_tag(group)
    work = turbo_dist.dma_all_gather_into_tensor(
        output_tensor,
        input_tensor,
        group,
        group_tag,
    )
    if not async_op and work is not None:
        work.wait()
    else:
        return work
