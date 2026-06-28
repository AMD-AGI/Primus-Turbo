###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import math

import torch
import torch.nn as nn


def grouped_gemm_ref(a, b, seg_lens, trans_b=True):
    # NOTE: the per-group matmul is accumulated in fp32 (operands upcast, result
    # cast back to ``a.dtype``) for two reasons:
    #   1. Correctness: an fp32-accumulated golden is the right reference for a
    #      low-precision (fp4/fp8) product comparison.
    #   2. Robustness: on ROCm/hipBLASLt a *bf16* NT GEMM (row-major A @ a
    #      transposed, column-major B) selects a kernel that reads out of bounds
    #      for some odd group M (e.g. a 2078-row slice with N=2816, K=2048),
    #      hard-faulting the GPU ("Memory access fault ... Reason: Unknown") in
    #      both the fwd matmul and the autograd bwd (which re-introduces a
    #      transposed bf16 operand). Running in fp32 dodges the bad bf16 kernel.
    #      The product kernel under test runs on Triton, so only this reference
    #      hit the library bug.
    seg_lens = seg_lens.cpu().numpy()
    out = []
    start = 0
    for i, size in enumerate(seg_lens):
        rhs = b[i, :, :].t() if trans_b else b[i, :, :]
        out.append((a[start : start + size, :].float() @ rhs.float()).to(a.dtype))
        start += size
    return torch.cat(out)


def grouped_gemm_variable_k_ref(a, b, seg_lens, trans_a=True, trans_b=False):
    assert trans_a == True and trans_b == False, "Only trans_a=True and trans_b=False are supported."
    seg_lens = seg_lens.cpu().numpy()
    B = len(seg_lens)
    M = a.shape[1]
    N = b.shape[1]
    out = torch.zeros((B, M, N), dtype=a.dtype, device="cuda", requires_grad=False)
    start = 0
    for i, size in enumerate(seg_lens):
        a_tmp = a[start : start + size, :].t()
        b_tmp = b[start : start + size, :]
        out_tmp = a_tmp @ b_tmp
        out[i] = out_tmp
        start += size
    return out


class GroupedLinearRef(torch.nn.Module):
    def __init__(
        self,
        batch: int,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.in_features = in_features  # K
        self.out_features = out_features  # N
        self.batch = batch
        self.dtype = dtype
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((batch, self.out_features, self.in_features), **factory_kwargs)
        )  # [B,N,K]
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(
        self,
        x: torch.Tensor,  # [B * M, K],
        seg_lens: torch.Tensor,  # [B,] int64
    ) -> torch.Tensor:
        out = grouped_gemm_ref(x, self.weight, seg_lens)
        return out


def generate_grouped_gemm_group_lens(b, m, balance: bool):
    if balance:
        return torch.full((b,), m, dtype=torch.int64)
    else:
        dist = 0.2 + 0.8 * torch.rand(b)
        dist /= dist.sum()
        group_lens = (dist * b * m).to(torch.int64)
        error = b * m - group_lens.sum()
        group_lens[-1] += error
        return group_lens
