###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch


def _empty_tensor(device):
    return torch.Tensor().to(device)


def gemm_impl(
    a: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    trans_c: bool,
    backend="hipblaslt",
) -> torch.Tensor:
    assert backend in ("hipblaslt")

    args = (
        a,
        _empty_tensor(device=a.device),
        b,
        _empty_tensor(device=b.device),
        out_dtype,
        trans_a,
        trans_b,
        trans_c,
    )

    if backend == "hipblaslt":
        # TODO(ruibzhan): support more backends.
        out = torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm(*args)

    return out
