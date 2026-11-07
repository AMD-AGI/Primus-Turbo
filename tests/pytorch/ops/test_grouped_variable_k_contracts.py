###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Variable-K schedules, output orientation and in-place output metadata."""

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3
from primus_turbo.pytorch.core.utils import build_ck, is_gfx950
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_fp8_variable_k_accum_impl,
    grouped_gemm_fp8_variable_k_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
    grouped_gemm_variable_k_accum_impl,
    grouped_gemm_variable_k_impl,
)
from tests.pytorch.ops.gradient_contract_helpers import caller_thread_backward  # noqa: F401

pytestmark = pytest.mark.usefixtures("caller_thread_backward")


@pytest.mark.parametrize("trans_c", [False, True])
@pytest.mark.parametrize(
    "backend,schedule",
    [
        pytest.param(
            BackendType.CK,
            "static",
            marks=pytest.mark.skipif(not build_ck(), reason="CK backend is not built"),
        ),
        pytest.param(
            BackendType.CK,
            "work_steal",
            marks=pytest.mark.skipif(
                not build_ck() or not is_gfx950(), reason="CK work-steal requires gfx950 and a CK build"
            ),
        ),
        (BackendType.HIPBLASLT, "static"),
    ],
)
@pytest.mark.parametrize("fp8", [False, True])
def test_variable_k_output_and_accumulation(fp8, backend, schedule, trans_c):
    GlobalBackendManager.set_grouped_gemm_backend(backend)
    torch.manual_seed(823)
    lens = torch.tensor([128, 256], device="cuda", dtype=torch.int64)
    offs = torch.tensor([0, 128, 384], device="cuda", dtype=torch.int64)
    dtype = float8_e4m3 if fp8 else torch.bfloat16
    a = torch.randn(384, 128, device="cuda").to(dtype)
    b = torch.randn(384, 256, device="cuda").to(dtype)
    sa = torch.tensor([0.25], device="cuda")
    sb = torch.tensor([0.5], device="cuda")
    ref = torch.stack([x.float().T @ y.float() for x, y in zip(a.split([128, 256]), b.split([128, 256]))])
    if fp8:
        ref = ref * sa * sb
    if trans_c:
        ref = ref.transpose(-1, -2).contiguous()
    if fp8:
        args = (
            a,
            b,
            sa,
            sb,
            lens,
            offs,
            True,
            False,
            trans_c,
            torch.bfloat16,
            ScalingGranularity.TENSORWISE.value,
            None,
            backend.value,
        )
        op, accum_op = grouped_gemm_fp8_variable_k_impl, grouped_gemm_fp8_variable_k_accum_impl
        kw = {}
    else:
        args = (a, b, lens, offs, True, False, trans_c, None, backend.value)
        op, accum_op = grouped_gemm_variable_k_impl, grouped_gemm_variable_k_accum_impl
        kw = {"schedule": schedule}
    out = op(*args, **kw)
    torch.testing.assert_close(out.float(), ref, rtol=0.008, atol=0.04)
    if backend is BackendType.HIPBLASLT:
        accum = torch.randn_like(out)
        expected = (accum.float() + ref).to(out.dtype)
        accum_op(*args, out=accum)
        torch.testing.assert_close(accum, expected, rtol=0.008, atol=0.04)
        torch.library.opcheck(accum_op, args, {"out": accum}, test_utils=("test_faketensor",))
