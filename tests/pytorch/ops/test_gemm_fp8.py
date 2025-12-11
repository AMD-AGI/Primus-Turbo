###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
)
from primus_turbo.pytorch.ops import gemm_fp8
from tests.pytorch.test_utils import compute_snr

torch.manual_seed(42)


@pytest.mark.parametrize("m", [255, 507, 1032, 2056])
@pytest.mark.parametrize("n", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("k", [256, 512, 576, 1024, 2048])
@pytest.mark.parametrize("layout", ["NN", "NT"])
@pytest.mark.parametrize("format", [Format.E4M3, Format.E5M2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("granularity", [ScalingGranularity.TENSORWISE, ScalingGranularity.ROWWISE])
@pytest.mark.parametrize("backend", [None, BackendType.CK, BackendType.HIPBLASLT])
@pytest.mark.parametrize("auto_tune", [False, True])
def test_gemm_fp8_ck(m, n, k, layout, format, dtype, granularity, backend, auto_tune):
    # Skip redundant test: auto_tune is ignored when backend is explicitly specified
    if backend is not None and auto_tune:
        pytest.skip("auto_tune is ignored when backend is explicitly specified")

    # Set backend and auto_tune config
    GlobalBackendManager.set_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(auto_tune)

    print(
        f"\nM={m}, N={n}, K={k}, layout={layout}, dtype={dtype}, format={format}, backend={backend}, auto_tune={auto_tune}"
    )

    device = "cuda:0"

    trans_a = layout[0] == "T"
    trans_b = layout[1] == "T"

    a_shape = (m, k) if not trans_a else (k, m)
    b_shape = (k, n) if not trans_b else (n, k)

    a = torch.randn(a_shape, dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    torch.cuda.synchronize()

    # Ref
    a_mat = a_ref.T if trans_a else a_ref
    b_mat = b_ref.T if trans_b else b_ref
    c_ref = a_mat @ b_mat
    grad_c = torch.randn_like(c_ref)
    c_ref.backward(grad_c)
    torch.cuda.synchronize()

    # Config + FWD + BWD
    config = Float8QuantConfig(granularity=granularity, format=format)
    print(config)
    c = gemm_fp8(a, b, trans_a, trans_b, dtype, config)
    c.backward(grad_c)

    # Check Shape
    # print("out shape: ", out.shape, out_ref.shape)
    # print("x_grad shape: ", x_grad.shape, x_grad_ref.shape)
    # print("w_grad shape: ", w_grad.shape, w_grad_ref.shape)
    assert c.shape == c_ref.shape
    assert a.grad.shape == a_ref.grad.shape
    assert b.grad.shape == b_ref.grad.shape

    snr_threshold = 25 if format == Format.E4M3 else 20
    # Check Results
    c_snr = compute_snr(c_ref, c)
    print(f"C-SNR: {c_snr:.2f} dB")
    assert c_snr > snr_threshold, "c_snr too low"

    a_grad_snr = compute_snr(a_ref.grad, a.grad)
    print(f"AGrad-SNR: {a_grad_snr:.2f} dB")
    assert a_grad_snr > snr_threshold, "a_grad_snr too low"

    b_grad_snr = compute_snr(b_ref.grad, b.grad)
    print(f"BGrad-SNR: {b_grad_snr:.2f} dB")
    assert b_grad_snr > snr_threshold, "b_grad_snr too low"

    # Reset config and caches
    GlobalBackendManager.reset()


# n and k must be a multiple of 128
# block_size must be 128
@pytest.mark.parametrize("m", [255, 257, 512, 1024])
@pytest.mark.parametrize("n", [256, 512, 1024, 4096])
@pytest.mark.parametrize("k", [256, 1024, 4096])
@pytest.mark.parametrize("layout", ["NT", "NN"])
@pytest.mark.parametrize("format", [Format.E4M3, Format.E5M2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("granularity", [ScalingGranularity.BLOCKWISE])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("backend", [None, BackendType.CK, BackendType.HIPBLASLT])
@pytest.mark.parametrize("auto_tune", [False, True])
def test_gemm_fp8_blockwise(m, n, k, layout, format, dtype, granularity, block_size, backend, auto_tune):
    # Skip redundant test: auto_tune is ignored when backend is explicitly specified
    if backend is not None and auto_tune:
        pytest.skip("auto_tune is ignored when backend is explicitly specified")

    # Set backend and auto_tune config
    GlobalBackendManager.set_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(auto_tune)

    print(
        f"\nM={m}, N={n}, K={k}, layout={layout}, dtype={dtype}, format={format}, "
        f"granularity={granularity}, block_size={block_size}, backend={backend}, auto_tune={auto_tune}"
    )

    device = "cuda:0"

    trans_a = layout[0] == "T"
    trans_b = layout[1] == "T"

    a_shape = (m, k) if not trans_a else (k, m)
    b_shape = (k, n) if not trans_b else (n, k)

    a = torch.randn(a_shape, dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    torch.cuda.synchronize()

    # Ref
    a_mat = a_ref.T if trans_a else a_ref
    b_mat = b_ref.T if trans_b else b_ref
    c_ref = a_mat @ b_mat
    grad_c = torch.randn_like(c_ref)
    c_ref.backward(grad_c)
    torch.cuda.synchronize()

    # Config + FWD + BWD
    config = Float8QuantConfig(granularity=granularity, format=format, block_size=block_size)
    c = gemm_fp8(a, b, trans_a, trans_b, dtype, config)
    c.backward(grad_c)

    assert c.shape == c_ref.shape
    assert a.grad.shape == a_ref.grad.shape
    assert b.grad.shape == b_ref.grad.shape

    snr_threshold = 25 if format == Format.E4M3 else 20

    # Check Results
    c_snr = compute_snr(c_ref, c)
    print(f"C-SNR: {c_snr:.2f} dB")
    assert c_snr > snr_threshold, "c_snr too low"

    a_grad_snr = compute_snr(a_ref.grad, a.grad)
    print(f"AGrad-SNR: {a_grad_snr:.2f} dB")
    assert a_grad_snr > snr_threshold, "a_grad_snr too low"

    b_grad_snr = compute_snr(b_ref.grad, b.grad)
    print(f"BGrad-SNR: {b_grad_snr:.2f} dB")
    assert b_grad_snr > snr_threshold, "b_grad_snr too low"

    # Reset config and caches
    GlobalBackendManager.reset()


@pytest.mark.parametrize("m", [256, 512, 1024])
@pytest.mark.parametrize("n", [256, 352, 1024, 2048])
@pytest.mark.parametrize("k", [128, 160, 512, 1024])
@pytest.mark.parametrize("layout", ["NT"])
@pytest.mark.parametrize("format", [Format.E4M3, Format.E5M2, Format.HYBRID])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("granularity", [ScalingGranularity.MX_BLOCKWISE])
@pytest.mark.parametrize("backend", [None, BackendType.CK, BackendType.HIPBLASLT])
@pytest.mark.parametrize("auto_tune", [False, True])
def test_gemm_mxfp8(m, n, k, layout, format, dtype, granularity, backend, auto_tune):
    # Skip redundant test: auto_tune is ignored when backend is explicitly specified
    if backend is not None and auto_tune:
        pytest.skip("auto_tune is ignored when backend is explicitly specified")

    # NOTE: user need to ensure m, n and k are multiples of 16.
    assert m % 16 == 0 and n % 16 == 0 and k % 16 == 0, "Assume m, n and k are multiples of 16."

    from primus_turbo.pytorch.core.low_precision import check_mxfp8_support

    # Skip unit test on gfx942.
    mxfp8_supported, reason = check_mxfp8_support()
    if not mxfp8_supported:
        pytest.skip(reason)

    # Set backend and auto_tune config
    GlobalBackendManager.set_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(auto_tune)

    print(
        f"\nM={m}, N={n}, K={k}, layout={layout}, dtype={dtype}, format={format}, "
        f"backend={backend}, auto_tune={auto_tune}"
    )

    device = "cuda:0"

    trans_a = layout[0] == "T"
    trans_b = layout[1] == "T"

    a_shape = (m, k) if not trans_a else (k, m)
    b_shape = (k, n) if not trans_b else (n, k)

    a = torch.randn(a_shape, dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    torch.cuda.synchronize()

    # Ref
    a_mat = a_ref.T if trans_a else a_ref
    b_mat = b_ref.T if trans_b else b_ref
    c_ref = a_mat @ b_mat
    grad_c = torch.randn_like(c_ref)
    c_ref.backward(grad_c)
    torch.cuda.synchronize()

    # Config + FWD + BWD
    config = Float8QuantConfig(
        granularity=granularity, format=format, block_size=32, scale_dtype=ScaleDtype.E8M0
    )
    print(config)
    c = gemm_fp8(a, b, trans_a, trans_b, dtype, config)
    c.backward(grad_c)

    # Check Shape
    assert c.shape == c_ref.shape
    assert a.grad.shape == a_ref.grad.shape
    assert b.grad.shape == b_ref.grad.shape

    snr_threshold = 25 if format == Format.E4M3 else 20
    # Check Results
    c_snr = compute_snr(c_ref, c)
    print(f"C-SNR: {c_snr:.2f} dB")
    assert c_snr > snr_threshold, "c_snr too low"

    a_grad_snr = compute_snr(a_ref.grad, a.grad)
    print(f"AGrad-SNR: {a_grad_snr:.2f} dB")
    assert a_grad_snr > snr_threshold, "a_grad_snr too low"

    b_grad_snr = compute_snr(b_ref.grad, b.grad)
    print(f"BGrad-SNR: {b_grad_snr:.2f} dB")
    assert b_grad_snr > snr_threshold, "b_grad_snr too low"

    # Reset config and caches
    GlobalBackendManager.reset()
