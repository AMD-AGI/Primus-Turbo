###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.kernels.grouped_gemm import grouped_gemm_impl as impl


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("trans_b", [False, True])
def test_meta_grouped_gemm(dtype, trans_b):
    a = torch.empty(8, 32, device="meta", dtype=dtype)
    b = torch.empty((3, 48, 32) if trans_b else (3, 32, 48), device="meta", dtype=dtype)
    lens = torch.empty(3, device="meta", dtype=torch.int64)
    offs = torch.empty(4, device="meta", dtype=torch.int64)
    out = impl.grouped_gemm_impl(a, b, lens, offs, False, trans_b, None, 0)
    assert out.shape == (8, 48) and out.dtype == dtype
    assert out.device.type == "meta"


@pytest.mark.parametrize("trans_c, shape", [(False, (3, 32, 48)), (True, (3, 48, 32))])
def test_meta_variable_k_output_transpose(trans_c, shape):
    a = torch.empty(8, 32, device="meta", dtype=torch.bfloat16)
    b = torch.empty(8, 48, device="meta", dtype=torch.bfloat16)
    lens = torch.empty(3, device="meta", dtype=torch.int64)
    offs = torch.empty(4, device="meta", dtype=torch.int64)
    out = impl.grouped_gemm_variable_k_impl(a, b, lens, offs, True, False, trans_c, None, 0)
    assert out.shape == shape and out.dtype == a.dtype


@pytest.mark.parametrize("trans_a, trans_b", [(False, False), (False, True), (True, True)])
def test_meta_variable_k_rejects_unsupported_layout(trans_a, trans_b):
    a = torch.empty(8, 32, device="meta", dtype=torch.bfloat16)
    b = torch.empty(8, 48, device="meta", dtype=torch.bfloat16)
    lens = torch.empty(3, device="meta", dtype=torch.int64)
    with pytest.raises(AssertionError, match="Only trans_a=True and trans_b=False"):
        impl.grouped_gemm_variable_k_impl(a, b, lens, lens, trans_a, trans_b, False, None, 0)
