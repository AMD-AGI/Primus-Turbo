###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch


def _empty_tensor(device):
    return torch.Tensor().to(device)


def gemm_impl(
    A: torch.Tensor,
    transA: bool,
    B: torch.Tensor,
    transB: bool,
    out_dtype: torch.dtype,
    transC: bool,
    backend="hipblaslt",
) -> torch.Tensor:
    assert backend in ("hipblaslt")

    args = (
        A,
        _empty_tensor(device=A.device),
        B,
        _empty_tensor(device=B.device),
        out_dtype,
        transA,
        transB,
        transC,
    )

    if backend == "hipblaslt":
        # TODO(ruibzhan): support more backends.
        out = torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm(*args)

    return out
