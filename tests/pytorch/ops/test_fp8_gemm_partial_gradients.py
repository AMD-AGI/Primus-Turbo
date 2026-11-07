###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FP8 frozen-weight/frozen-activation gradients across scaling granularities."""

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, Format, ScaleDtype, ScalingGranularity
from primus_turbo.pytorch.ops import gemm_fp8, grouped_gemm_fp8
from tests.pytorch.ops.gradient_contract_helpers import caller_thread_backward  # noqa: F401
from tests.pytorch.test_utils import compute_snr

pytestmark = pytest.mark.usefixtures("caller_thread_backward")

CONFIGS = [
    (ScalingGranularity.TENSORWISE, BackendType.CK),
    (ScalingGranularity.ROWWISE, BackendType.CK),
    (ScalingGranularity.BLOCKWISE, BackendType.TRITON),
    (ScalingGranularity.MX_BLOCKWISE, BackendType.HIPBLASLT),
]


@pytest.mark.parametrize("granularity,backend", CONFIGS)
@pytest.mark.parametrize("grad_inputs", ["a", "b", "ab"])
@pytest.mark.parametrize("grouped", [False, True])
def test_fp8_partial_gradients(granularity, backend, grad_inputs, grouped):
    torch.manual_seed(719)
    # Grouped MXFP8 is implemented by Triton; hipBLASLt covers dense MXFP8.
    if grouped and granularity is ScalingGranularity.MX_BLOCKWISE:
        backend = BackendType.TRITON
    (GlobalBackendManager.set_grouped_gemm_backend if grouped else GlobalBackendManager.set_gemm_backend)(
        backend
    )
    kwargs = {"format": Format.E4M3, "granularity": granularity}
    if granularity is ScalingGranularity.BLOCKWISE:
        kwargs["block_size"] = 128
    elif granularity is ScalingGranularity.MX_BLOCKWISE:
        kwargs.update(block_size=32, scale_dtype=ScaleDtype.E8M0)
    config = Float8QuantConfig(**kwargs)
    lengths = [128, 256]
    a = torch.randn(384, 256, device="cuda", dtype=torch.bfloat16).requires_grad_("a" in grad_inputs)
    b = torch.randn(
        (2, 128, 256) if grouped else (128, 256), device="cuda", dtype=torch.bfloat16
    ).requires_grad_("b" in grad_inputs)
    ar, br = [x.detach().float().requires_grad_(x.requires_grad) for x in (a, b)]
    if grouped:
        lens = torch.tensor(lengths, device="cuda", dtype=torch.int64)
        out = grouped_gemm_fp8(a, b, lens, trans_b=True, config=config)
        ref = torch.cat([x @ w.T for x, w in zip(ar.split(lengths), br)])
    else:
        out = gemm_fp8(a, b, trans_a=False, trans_b=True, config=config)
        ref = ar @ br.T
    grad = torch.randn(128, 384, device="cuda", dtype=torch.bfloat16).T
    out.backward(grad)
    ref.backward(grad.float())
    pairs = [(out, ref)]
    for actual, expected in ((a, ar), (b, br)):
        if actual.requires_grad:
            pairs.append((actual.grad, expected.grad))
        else:
            assert actual.grad is None
    # Match the existing FP8 suite's E4M3 SNR floor; no tolerance relaxation.
    for actual, expected in pairs:
        assert actual.shape == expected.shape
        assert torch.isfinite(actual).all()
        assert compute_snr(expected.float(), actual.float()) > 25
