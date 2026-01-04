###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
    Float4QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
)
from primus_turbo.pytorch.ops.gemm_fp4 import gemm_fp4
from tests.pytorch.test_utils import compute_snr

torch.manual_seed(42)


@pytest.mark.parametrize("m", [256, 512, 1024])
@pytest.mark.parametrize("n", [256, 352, 1024, 2048])
@pytest.mark.parametrize("k", [128, 160, 512, 1024])
@pytest.mark.parametrize("layout", ["NT"])
@pytest.mark.parametrize(
    "format",
    [
        Format.E2M1_X2,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        torch.float16,
    ],
)
@pytest.mark.parametrize("granularity", [ScalingGranularity.MX_BLOCKWISE])
@pytest.mark.parametrize("backend", [None, BackendType.HIPBLASLT])
@pytest.mark.parametrize("auto_tune", [False, True])
def test_gemm_fp4_mx_blockwise(m, n, k, layout, format, dtype, granularity, backend, auto_tune):
    # Skip redundant test: auto_tune is ignored when backend is explicitly specified
    if backend is not None and auto_tune:
        pytest.skip("auto_tune is ignored when backend is explicitly specified")

    # NOTE: user need to ensure m, n and k are multiples of 16.
    assert m % 16 == 0 and n % 16 == 0 and k % 16 == 0, "Assume m, n and k are multiples of 16."

    from primus_turbo.pytorch.core.low_precision import check_mxfp4_support

    # Skip unit test on gfx942.
    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    # Set backend and auto_tune config
    GlobalBackendManager.set_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(auto_tune)

    print(
        f"\nM={m}, N={n}, K={k}, layout={layout}, dtype={dtype}, format={format}, "
        f"backend={backend}, auto_tune={auto_tune}"
    )

    device = "cuda:0"

    trans_act = layout[0] == "T"
    trans_w = layout[1] == "T"

    act_shape = (m, k) if not trans_act else (k, m)
    w_shape = (k, n) if not trans_w else (n, k)

    act = torch.randn(act_shape, dtype=dtype, device=device, requires_grad=True)
    w = torch.randn(w_shape, dtype=dtype, device=device, requires_grad=True)

    act_ref = act.detach().clone().requires_grad_()
    w_ref = w.detach().clone().requires_grad_()
    torch.cuda.synchronize()

    # Ref
    act_mat = act_ref.T if trans_act else act_ref
    w_mat = w_ref.T if trans_w else w_ref
    out_ref = act_mat @ w_mat
    out_ref.backward(torch.ones_like(out_ref))
    torch.cuda.synchronize()

    # Config + FWD + BWD
    # NOTE: scaling recipe reference: https://arxiv.org/pdf/2509.25149
    config = Float4QuantConfig(
        granularity=granularity, format=format, block_size=32, scale_dtype=ScaleDtype.E8M0
    )
    c = gemm_fp4(act, w, trans_act, trans_w, dtype, config)
    c.backward(torch.ones_like(c))

    # Check Shape
    assert c.shape == out_ref.shape
    assert act.grad.shape == act_ref.grad.shape
    assert w.grad.shape == w_ref.grad.shape

    snr_threshold = 10
    # Check Results
    out_snr = compute_snr(out_ref, c)
    print(f"Out-SNR: {out_snr:.2f} dB")
    assert out_snr > snr_threshold, "out_snr too low"

    act_grad_snr = compute_snr(act_ref.grad, act.grad)
    print(f"Activation Grad-SNR: {act_grad_snr:.2f} dB")
    assert act_grad_snr > snr_threshold, "act_grad_snr too low"

    w_grad_snr = compute_snr(w_ref.grad, w.grad)
    print(f"Weight Grad-SNR: {w_grad_snr:.2f} dB")
    assert w_grad_snr > snr_threshold, "w_grad_snr too low"

    # Reset config and caches
    GlobalBackendManager.reset()
