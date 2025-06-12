from typing import List, Optional, Tuple

import torch
import torch.distributed.distributed_c10d as c10d

from ..kernels.async_tp.ag_gemm_impl import _pipeline_fused_all_gather_matmul_impl

__all__ = ["fused_all_gather_matmul"]


def fused_all_gather_matmul(
    A_shard: torch.Tensor,
    Bs: List[torch.Tensor],
    layouts: List[str],
    gather_dim: int,
    group_name: str,
    gemm_streams: List[torch.cuda.Stream],
    comm_streams: List[torch.cuda.Stream],
    copy_streams: List[torch.cuda.Stream],
    *,
    comm_method: str = "pipeline",
    num_splits: int = 2,
    skip_copy_local_A: bool = False,  # only needed for te
    enable_sdma: bool = False,
    return_A: bool = True,
    A_out: Optional[torch.Tensor] = None,
    outputs: Optional[List[torch.Tensor]] = None,
    out_dtypes: Optional[List[torch.dtype]] = None,
) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
    # check input
    if A_shard.dim() < 2:
        raise ValueError("A_shard must be a matrix")
    for B in Bs:
        if B.dim() != 2:
            raise ValueError("B must be a matrix")
    if len(layouts) != len(Bs):
        raise ValueError("len(layouts) must be the same as len(Bs)")

    if gather_dim < 0 or gather_dim >= A_shard.dim():
        raise ValueError("Invalid gather_dim")

    if comm_method not in ["auto", "ring_exchange", "pipeline"]:
        raise ValueError(f"Invalid comm_method: {comm_method}")

    group = c10d._resolve_process_group(group_name)

    if return_A and A_out is not None:
        if A_out.dtype != A_shard.dtype:
            raise ValueError(
                f"Invalid dtype: A_out ({A_out.dtype}) difference with A_shard ({A_shard.dtype})!"
            )

        if A_out.numel() != A_shard.numel() * group.size():
            raise ValueError(f"A_out size must equal group size * A_shard size.")

    A_shard_flat = A_shard.movedim(gather_dim, 0)
    leading_dims = [group.size()] + list(A_shard_flat.shape[:-1])
    A_shard_flat = A_shard_flat.flatten(0, -2)
    A_shard_flat = A_shard_flat if layouts[0][0] == "N" else A_shard_flat.T
    for i, (layout, B) in enumerate(zip(layouts, Bs)):
        Bs[i] = B if layout[1] == "N" else B.T

    def unflatten(t: torch.Tensor) -> torch.Tensor:
        return t.view(*leading_dims, -1).flatten(0, 1).movedim(0, gather_dim)

    if return_A and A_out is None:
        A_out = A_shard_flat.new_empty(
            A_shard_flat.shape[0] * group.size(),
            A_shard_flat.shape[1],
        )

    out_dtypes = out_dtypes or [B.dtype for B in Bs]

    if outputs is None:
        outputs = [
            A_shard.new_empty(A_shard_flat.shape[0] * group.size(), B.shape[1], dtype=out_dtype or B.dtype)
            for B, out_dtype in zip(Bs, out_dtypes)
        ]

    if comm_method in ["ring_exchange", "auto"]:
        raise NotImplementedError()
    else:
        with torch.profiler.record_function("pipeline_fused_all_gather_matmul"):
            _pipeline_fused_all_gather_matmul_impl(
                A_shard_flat,
                Bs,
                torch.ops.aten.mm,
                group_name,
                comm_streams,
                copy_streams,
                gemm_streams,
                num_splits=num_splits,
                enable_sdma=enable_sdma,
                skip_copy_local_A=skip_copy_local_A,
                return_A=return_A,
                A_out=A_out,
                outputs=outputs,
            )

    A = unflatten(A_out) if return_A else None
    return A, [unflatten(output) for output in outputs]
