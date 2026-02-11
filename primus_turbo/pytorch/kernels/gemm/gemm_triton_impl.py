###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
from torch.library import triton_op

from primus_turbo.triton.gemm.gemm_kernel import gemm_triton_kernel

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def gemm_triton_impl(a: torch.Tensor, b: torch.Tensor, layout: str = "NN") -> torch.Tensor:
    return torch.ops.primus_turbo.gemm_triton.default(a, b, layout)


@triton_op("primus_turbo::gemm_triton", mutates_args={})
def gemm_triton(a: torch.Tensor, b: torch.Tensor, layout: str = "NN") -> torch.Tensor:
    assert layout in ["NN", "NT", "TN"], f"Unsupported layout: {layout}"
    assert (
        a.dtype in _SUPPORTED_DTYPES and b.dtype in _SUPPORTED_DTYPES
    ), f"Only bf16/fp16 supported, got a={a.dtype}, b={b.dtype}"

    trans_a = layout[0] == "T"
    trans_b = layout[1] == "T"

    return gemm_triton_kernel(a, b, trans_a, trans_b, out_dtype=a.dtype)


@gemm_triton.register_fake
def gemm_triton_meta(a: torch.Tensor, b: torch.Tensor, layout: str = "NN") -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2, f"Expect 2D tensors, got {a.shape}, {b.shape}"
    if layout == "NN":
        m, k1 = a.shape
        k2, n = b.shape
    elif layout == "NT":
        m, k1 = a.shape
        n, k2 = b.shape
    elif layout == "TN":
        k1, m = a.shape
        k2, n = b.shape
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    assert k1 == k2, f"Incompatible matmul dims: k1={k1}, k2={k2}"
    out_dtype = torch.result_type(a, b)
    return torch.empty((m, n), device=a.device, dtype=out_dtype)
