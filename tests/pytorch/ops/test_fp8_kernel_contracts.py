###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FP8 kernel metadata, output orientation and accumulation against FP32."""

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import Format, ScalingGranularity, float8_e4m3
from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_accum_impl, gemm_fp8_impl
from tests.pytorch.ops.gradient_contract_helpers import caller_thread_backward  # noqa: F401
from tests.pytorch.ops.test_grouped_gemm_fp8 import _run_grouped_gemm_fp8_test

pytestmark = [
    pytest.mark.usefixtures("caller_thread_backward"),
    pytest.mark.skipif(not is_gfx950(), reason="This regression suite requires gfx950"),
]


@pytest.mark.parametrize("trans_c", [False, True])
@pytest.mark.parametrize("backend", [BackendType.HIPBLASLT, BackendType.FLYDSL, None])
def test_dense_fp8_output_and_accumulation(trans_c, backend):
    GlobalBackendManager.set_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(backend is None)
    torch.manual_seed(818)
    a = torch.randn(128, 256, device="cuda").to(float8_e4m3)
    b = torch.randn(192, 256, device="cuda").to(float8_e4m3)
    sa = torch.tensor([0.25], device="cuda")
    sb = torch.tensor([0.5], device="cuda")
    ref = (a.float() * sa) @ (b.float() * sb).T
    if trans_c:
        ref = ref.T.contiguous()
    args = (a, sa, False, b, sb, True, torch.bfloat16, trans_c, ScalingGranularity.TENSORWISE.value)
    default = BackendType.HIPBLASLT.value
    out = gemm_fp8_impl(*args, default)
    torch.testing.assert_close(out.float(), ref, rtol=0.008, atol=0.02)
    torch.library.opcheck(gemm_fp8_impl, (*args, default), test_utils=("test_faketensor",))
    accum = torch.randn_like(out)
    initial = accum.clone()
    accum_args = (*args, accum, default)
    for step in range(1, 3):
        gemm_fp8_accum_impl(*accum_args)
        initial = (initial.float() + ref).to(accum.dtype)
        torch.testing.assert_close(accum, initial, rtol=0.008, atol=0.02)
    torch.library.opcheck(gemm_fp8_accum_impl, accum_args, test_utils=("test_faketensor",))


@pytest.mark.parametrize("trans_b", [False, True])
def test_triton_rowwise_grouped_gradients(trans_b):
    _run_grouped_gemm_fp8_test(
        3,
        128,
        128,
        256,
        torch.bfloat16,
        Format.E4M3,
        ScalingGranularity.ROWWISE,
        trans_b,
        backend=BackendType.TRITON,
    )
