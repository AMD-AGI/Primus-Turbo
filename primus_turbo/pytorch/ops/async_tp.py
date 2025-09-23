###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import operator
from functools import reduce
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed.distributed_c10d as c10d

from primus_turbo.pytorch.kernels.async_tp import ag_gemm_impl, gemm_rs_impl

__all__ = [
    "fused_all_gather_matmul",
    "fused_matmul_reduce_scatter",
    "fused_all_gather_scaled_matmul",
    "fused_scaled_matmul_reduce_scatter",
]


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
    skip_copy_local_ag_out: bool = False,  # only needed for te
    enable_sdma: bool = False,
    return_A: bool = True,
    A_out: Optional[torch.Tensor] = None,
    outputs: Optional[List[torch.Tensor]] = None,
    out_dtypes: Optional[List[torch.dtype]] = None,
) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
    """
    Perform the following logic with micro-pipelined computation and
    communication:
        A = all_gather_tensor(A_shard, gather_dim, group_name)
        C_0 = torch.matmul(A, B_0)
        C_1 = torch.matmul(A, B_1)
        C_2 = torch.matmul(A, B_2)

    Optimal stride order for A_shard - if A_shard.movedim(gather_dim, 0) is
    contiguous, no extra copy is required for input layout transformation.
    Otherwise A_shard needs to be copied once.

    Parameters:
        A_shard (torch.Tensor): local sharded input tensor of gemm.
        Bs (List[torch.Tensor]): a list of local sharded weight tensor of gemm.
        layouts (List[str]): the layout of A B tensor 'NN' or 'NT' or 'TN'
        gather_dim (int): A_shard's gather dim
        group_name (str): tp group's name
        gemm_streams (List[torch.cuda.Stream]): multi streams for gemm
        comm_streams (List[torch.cuda.Stream]): multi streams for all gather
        copy_streams (List[torch.cuda.Stream]): multi streams for copy

    Keyword Arguments:
        comm_method (str, optional): specify internal algorithm of the implementation, 'ring_exchange' or 'pipeline' or 'auto'. Defaults to "pipeline".
        num_splits (int, optional): number of chunk splits by pipeline method. Defaults to 2.
        skip_copy_local_ag_out (bool, optional): skip copy A_shard to A_out, only used for transformer engine. Defaults to False.
        return_A (bool, optional): if return_A is True, return the result of all-gathered A_shard. Defaults to True.
        A_out (Optional[torch.Tensor], optional): the output of all-gathered A_shard, Defaults to None.
        outputs (Optional[List[torch.Tensor]], optional): the output tensors of matmul, Defaults to None.
        out_dtypes (Optional[List[torch.dtype]], optional): the output dtype of matmul. Defaults to None.

    Returns:
        Tuple[Optional[torch.Tensor], List[torch.Tensor]]: all-gathered A_shard and output tensors of matmul.

    Example:
        >>> A_shard = torch.randn(2, 3)
        >>> B = torch.randn(3, 3)
        >>> tp_group = torch.distributed.new_group(...)
        >>> gemm_streams = [torch.cuda.current_stream()]
        >>> comm_streams = [torch.cuda.Stream() for i in range(tp_group.size() - 1)]
        >>> copy_streams = [torch.cuda.Stream()]
        >>> A_out, outputs = primus_turbo.pytorch.ops.fused_all_gather_matmul(A_shard, [B], ['NN'], 0, tp_group.group_name, gemm_streams, comm_streams, copy_streams)
    """

    with torch.profiler.record_function(f"{comm_method}_fused_all_gather_matmul"):
        return ag_gemm_impl._fused_all_gather_matmul_impl(
            torch.ops.aten.mm.out,
            A_shard,
            Bs,
            layouts,
            None,
            [{} for _ in Bs],
            [None for _ in Bs] if out_dtypes is None else out_dtypes,
            gather_dim,
            group_name,
            return_A=return_A,
            comm_method=comm_method,
            num_splits=num_splits,
            enable_sdma=enable_sdma,
            comm_stream_pool=comm_streams,
            copy_stream_pool=copy_streams,
            gemm_stream_pool=gemm_streams,
            skip_copy_local_ag_out=skip_copy_local_ag_out,
            A_out=A_out,
            mm_out=outputs,
        )


def fused_all_gather_scaled_matmul(
    A_shard: torch.Tensor,
    Bs: list[torch.Tensor],
    layouts: list[str],
    A_scale: torch.Tensor,
    kwargs_list: list[dict[str, Any]],
    gather_dim: int,
    group_name: str,
    out_dtypes: list[Optional[torch.dtype]],
    gemm_streams: List[torch.cuda.Stream],
    comm_streams: List[torch.cuda.Stream],
    copy_streams: List[torch.cuda.Stream],
    *,
    comm_method: str = "pipeline",
    num_splits=2,
    skip_copy_local_ag_out: bool = False,  # only needed for te
    enable_sdma=False,
    A_out: Optional[torch.Tensor] = None,
    mm_out: Optional[List[torch.Tensor]] = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:

    with torch.profiler.record_function(f"{comm_method}_fused_all_gather_scaled_matmul"):
        A, res = ag_gemm_impl._fused_all_gather_matmul_impl(
            torch.ops.aten._scaled_mm.out,
            A_shard,
            Bs,
            layouts,
            A_scale,
            kwargs_list,
            out_dtypes,
            gather_dim,
            group_name,
            comm_method=comm_method,
            num_splits=num_splits,
            enable_sdma=enable_sdma,
            comm_stream_pool=comm_streams,
            copy_stream_pool=copy_streams,
            gemm_stream_pool=gemm_streams,
            return_A=True,
            skip_copy_local_ag_out=skip_copy_local_ag_out,
            A_out=A_out,
            mm_out=mm_out,
        )
        assert A is not None
        return A, res


def fused_matmul_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    layout: str,
    reduce_op: str,
    scatter_dim: int,
    group_name: str,
    gemm_streams: List[torch.cuda.Stream],
    comm_streams: List[torch.cuda.Stream],
    *,
    comm_method: str = "pipeline",
    num_splits: int = 2,
    enable_sdma: bool = False,
    output: Optional[torch.Tensor] = None,
    rs_out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Perform the following logic with micro-pipelined computation and
    communication:
        C_total = torch.matmul(A, B)
        C = reduce_scatter_tensor(C, "avg", scatter_dim, group)

    Optimal stride order for A - if A.movedim(scatter_dim, 0) is
    contiguous, no extra copy is required for input layout transformation.
    Otherwise A needs to be copied once.

    Parameters:
        A (torch.Tensor): input tensor of gemm.
        B (torch.Tensor): weight tensor of gemm.
        layout (str): the layout of A B tensor 'NN' or 'NT'
        reduce_op(str): reduce method 'avg' or 'sum'
        scatter_dim (int): C's scatter dim
        group_name (str): tp group's name

    Keyword Arguments:
        output(torch.Tensor, optional): the output tensor of matmul and scatter. Defaults to None.
        rs_out(torch.Tensor, optional): the output tensor of reduce. Defaults to None.
        output_dtype(torch.dtype, optional): the output dtype of matmul and reduce. Defaults to None.

    Return:
        torch.Tensor: the output tensor of mutmul_reduce_scatter.

    Example:
        >>> A = torch.randn(2, 3)
        >>> B = torch.randn(3, 3)
        >>> tp_group = torch.distributed.new_group(...)
        >>> comm_method = "pipeline"
        >>> gemm_streams = []
        >>> comm_streams = []
        >>> if comm_method == "pipeline":
        >>>     gemm_streams = [torch.cuda.current_stream()]
        >>>     comm_streams = [torch.cuda.Stream() for i in range(tp_group.size())]
        >>> rs_output = primus_turbo.pytorch.ops.fused_matmul_reduce_scatter(A, B, 'NN', 'sum', 0, tp_group.group_name, gemm_streams, comm_streams)
    """
    # check input
    if A.dim() < 2:
        raise ValueError("A must be a matrix")
    if scatter_dim < 0 or scatter_dim >= A.dim():
        raise ValueError("Invalid scatter_dim")
    if B.dim() != 2:
        raise ValueError("B must be a matrix")
    if reduce_op not in ["sum", "avg"]:
        raise ValueError("reduce_op must be sum or avg")
    if layout[0] == "T":
        raise ValueError(f"layout must be NN or NT, but {layout}")
    if out_dtype is None:
        out_dtype = A.dtype

    group = c10d._resolve_process_group(group_name)
    if comm_method == "pipeline":
        B = B.T if layout[1] == "T" else B
        N = B.shape[1]
    else:
        B = B.T if layout[1] == "N" else B
        N = B.shape[0]

    x = A.movedim(scatter_dim, 0)
    leading_dims = list(x.shape[:-1])
    leading_dims[0] //= group.size()
    x = x.flatten(0, -2)
    M, K = x.shape

    if comm_method == "tile" and ((M // group.size() < 256) or (M % 256) or (N % 256) or (K % 256)):
        raise ValueError(
            f"M, N, and K must be divisible by 256, and M divided by group size must not be less than 256 when comm_method use tile."
        )

    if rs_out is not None:
        if rs_out.dtype != A.dtype:
            raise ValueError(f"Invalid dtype: rs_out ({rs_out.dtype}) is different with A ({A.dtype})!")
        if rs_out.numel() != reduce(operator.mul, leading_dims, 1) * N:
            raise ValueError(
                f"Invalid shape: rs_out ({rs_out.shape}) is not unexpected as ({leading_dims}, {N})!"
            )
    else:
        rs_out = torch.empty((M // group.size(), N), dtype=out_dtype, device=A.device)

    if output is not None:
        if output.dtype != A.dtype:
            raise ValueError(f"Invalid dtype: output ({output.dtype}) is different with A ({A.dtype})!")

        if output.numel() != rs_out.numel() * group.size():
            raise ValueError(f"output size must equal group size * rs_out size.")
        output = output.view(-1, N)

    with torch.profiler.record_function(f"{comm_method}_fused_matmul_scatter_out"):
        if comm_method == "tile":
            rs_output = gemm_rs_impl._tiled_fused_matmul_scatter_out_impl(
                input=x,
                weight=B,
                group_name=group_name,
                reduce_op=reduce_op,
                output=output,
                rs_output=rs_out,
                out_dtype=out_dtype,
                stream=torch.cuda.current_stream(),
            )
        else:
            rs_output = gemm_rs_impl._pipeline_scaled_matmul_scatter_out_impl(
                mm_out_op=torch.ops.aten.mm.out,
                A=x,
                B=B,
                A_scale=None,
                kwargs={},
                group_name=group_name,
                reduce_op=reduce_op,
                num_splits=num_splits,
                enable_sdma=enable_sdma,
                gemm_stream_pool=gemm_streams,
                comm_stream_pool=comm_streams,
                output=output,
                rs_output=rs_out,
                out_dtype=out_dtype,
            )
        return rs_output.view(*leading_dims, -1).movedim(0, scatter_dim).contiguous()


def fused_scaled_matmul_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    layout: str,
    A_scale: torch.Tensor,
    kwargs: dict[str, Any],
    reduce_op: str,
    orig_scatter_dim: int,
    scatter_dim_after_maybe_reshape: int,
    group_name: str,
    output_shape: list[int],
    out_dtype: Optional[torch.dtype],
    gemm_streams: List[torch.cuda.Stream],
    comm_streams: List[torch.cuda.Stream],
    *,
    comm_method: str = "pipeline",
    num_splits: int = 2,
    enable_sdma: bool = False,
    output: Optional[torch.Tensor] = None,
    rs_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if A.dim() < 2:
        raise ValueError("A_shard must be a matrix")
    if scatter_dim_after_maybe_reshape < 0 or scatter_dim_after_maybe_reshape >= A.dim():
        raise ValueError("Invalid scatter dim for 2D tensor input to scaled_mm")
    if orig_scatter_dim < 0 or orig_scatter_dim >= len(output_shape):
        raise ValueError("Invalid scatter dim for 3D+ output tensor")
    if B.dim() != 2:
        raise ValueError("B must be a matrix")

    if reduce_op not in ["sum", "avg"]:
        raise ValueError("reduce_op must be sum or avg")
    if layout != "NN":
        raise ValueError(f"layout must be NN, otherwise scale should be processed, but {layout}")
    if comm_method != "pipeline":
        raise ValueError(f"{comm_method} not supported yet, please use pipeline")

    group = c10d._resolve_process_group(group_name)

    # Move scatter to first dim, then shard the tensor along the first dim, so the chunk producer
    # can perform matmuls along the first dim.
    A_with_scatter_dim_0 = A.movedim(scatter_dim_after_maybe_reshape, 0)
    leading_dims = list(A_with_scatter_dim_0.shape[:-1])
    leading_dims[0] //= group.size()

    # To handle case where A is 3D+, reshape to 2D to prepare for mm which requires 2D inputs.
    A_2D_with_scatter_dim_0 = A_with_scatter_dim_0.flatten(0, -2)

    # Now that 'A' is sharded along the first dim, we need to update its scale(s) accordingly.
    # How we do this depends on if we are using tensorwise scaling, rowwise scaling, or no scaling.
    tensorwise_scaling = A_scale is not None and A_scale.numel() == 1
    rowwise_scaling = A_scale is not None and A_scale.numel() > 1

    if not tensorwise_scaling and not rowwise_scaling:
        raise ValueError("A_scale cannot be none for scaled_mm")

    if rowwise_scaling:
        if A_scale.shape[:-1] != A.shape[:-1]:
            raise ValueError(
                "For row-wise scaling, the leading dims of A_scale "
                "must match the leading dims of A "
                f"(A shape: {A.shape}, A_scale shape: {A_scale.shape})"
            )
        A_scale = A_scale.movedim(scatter_dim_after_maybe_reshape, 0).contiguous().flatten(0, -2)

    def check_shape(shape_0, shape_1):
        if len(shape_0) != len(shape_1):
            return False
        for i in range(len(shape_0)):
            if shape_0[i] != shape_1[i]:
                return False
        return True

    output_shape[orig_scatter_dim] //= group.size()
    if rs_out is not None:
        if rs_out.dtype != out_dtype:
            raise ValueError(
                f"Invalid dtype: rs_output ({rs_out.dtype}) is different with out_dtype ({out_dtype})!"
            )
        if not check_shape(output_shape, list(rs_out.shape)):
            raise ValueError(f"Invalid shape: rs_out ({rs_out.shape}) is not unexpected as ({output_shape})!")
    else:
        rs_out = torch.empty(
            (A_2D_with_scatter_dim_0.shape[0] // group.size(), B.shape[-1]), dtype=out_dtype, device=A.device
        )

    if output is not None:
        if output.dtype != out_dtype:
            raise ValueError(
                f"Invalid dtype: output ({output.dtype}) is different with out_dtype ({out_dtype})!"
            )

        if output.numel() != rs_out.numel() * group.size():
            raise ValueError(f"output size must equal group size * rs_out size.")
        output = output.view(-1, B.shape[-1])

    with torch.profiler.record_function(f"{comm_method}_fused_scaled_matmul_scatter_out"):
        rs_output = gemm_rs_impl._pipeline_scaled_matmul_scatter_out_impl(
            mm_out_op=torch.ops.aten._scaled_mm.out,
            A=A_2D_with_scatter_dim_0,
            B=B,
            A_scale=A_scale,
            kwargs=kwargs,
            group_name=group_name,
            reduce_op=reduce_op,
            num_splits=num_splits,
            enable_sdma=enable_sdma,
            gemm_stream_pool=gemm_streams,
            comm_stream_pool=comm_streams,
            output=output,
            rs_output=rs_out,
            out_dtype=out_dtype,
        )
        return (
            rs_output.view(*leading_dims, -1)
            .movedim(0, scatter_dim_after_maybe_reshape)
            .view(output_shape)
            .contiguous()
        )
