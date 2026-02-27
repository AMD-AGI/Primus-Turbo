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
    check_mxfp8_support,
)
from primus_turbo.pytorch.ops import gemm_fp8
from tests.pytorch.test_utils import compute_snr

torch.manual_seed(42)


def _run_gemm_fp8_test(
    m: int,
    n: int,
    k: int,
    layout: str,
    format: Format,
    dtype: torch.dtype,
    granularity: ScalingGranularity,
    backend: BackendType | None,
    auto_tune: bool,
    block_size: int | None = None,
):
    """Common test logic for gemm_fp8 with different scaling granularities."""
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
    if block_size is not None:
        scale_dtype = ScaleDtype.E8M0 if granularity == ScalingGranularity.MX_BLOCKWISE else ScaleDtype.FP32
        config = Float8QuantConfig(
            granularity=granularity, format=format, block_size=block_size, scale_dtype=scale_dtype
        )
    else:
        config = Float8QuantConfig(granularity=granularity, format=format)
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


def _run_gemm_fp8_deterministic_test(
    m: int,
    n: int,
    k: int,
    layout: str,
    format: Format,
    dtype: torch.dtype,
    granularity: ScalingGranularity,
    backend: BackendType,
    repeats: int = 10,
    block_size: int | None = None,
):
    """Determinism + correctness check for gemm_fp8 on a small set of configs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Keep deterministic test focused: no autotune (reduces variability).
    GlobalBackendManager.set_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(False)

    try:
        device = "cuda:0"
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        trans_a = layout[0] == "T"
        trans_b = layout[1] == "T"

        print(
            f"\n[deterministic] M={m}, N={n}, K={k}, layout={layout}, dtype={dtype}, format={format}, "
            f"granularity={granularity}, block_size={block_size}, backend={backend}"
        )

        a_shape = (m, k) if not trans_a else (k, m)
        b_shape = (k, n) if not trans_b else (n, k)

        a0 = torch.randn(a_shape, dtype=dtype, device=device)
        b0 = torch.randn(b_shape, dtype=dtype, device=device)

        # Reference (correctness)
        a_ref = a0.detach().clone().requires_grad_()
        b_ref = b0.detach().clone().requires_grad_()
        a_mat = a_ref.T if trans_a else a_ref
        b_mat = b_ref.T if trans_b else b_ref
        c_ref = a_mat @ b_mat
        grad_c = torch.randn_like(c_ref)
        c_ref.backward(grad_c)
        torch.cuda.synchronize()

        if block_size is not None:
            scale_dtype = (
                ScaleDtype.E8M0 if granularity == ScalingGranularity.MX_BLOCKWISE else ScaleDtype.FP32
            )
            config = Float8QuantConfig(
                granularity=granularity, format=format, block_size=block_size, scale_dtype=scale_dtype
            )
        else:
            config = Float8QuantConfig(granularity=granularity, format=format)

        def _run_once():
            a = a0.detach().clone().requires_grad_()
            b = b0.detach().clone().requires_grad_()
            c = gemm_fp8(a, b, trans_a, trans_b, dtype, config)
            c.backward(grad_c)
            return c.detach(), a.grad.detach(), b.grad.detach()

        outs = []
        for _ in range(repeats):
            outs.append(_run_once())
            torch.cuda.synchronize()

        c0, da0, db0 = outs[0]
        for i in range(1, repeats):
            ci, dai, dbi = outs[i]
            torch.testing.assert_close(c0, ci, rtol=0, atol=0)
            torch.testing.assert_close(da0, dai, rtol=0, atol=0)
            torch.testing.assert_close(db0, dbi, rtol=0, atol=0)

        # Correctness (SNR) - FP8 uses looser thresholds
        snr_threshold = 25 if format == Format.E4M3 else 20
        c_snr = compute_snr(c_ref, c0)
        a_grad_snr = compute_snr(a_ref.grad, da0)
        b_grad_snr = compute_snr(b_ref.grad, db0)
        print(
            f"deterministic fp8: C-SNR={c_snr:.2f} dB, AGrad-SNR={a_grad_snr:.2f} dB, BGrad-SNR={b_grad_snr:.2f} dB"
        )
        assert c_snr > snr_threshold, "c_snr too low"
        assert a_grad_snr > snr_threshold, "a_grad_snr too low"
        assert b_grad_snr > snr_threshold, "b_grad_snr too low"
    finally:
        GlobalBackendManager.reset()


@pytest.mark.parametrize("m", [255, 507, 1032, 2056])
@pytest.mark.parametrize("n", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("k", [256, 512, 576, 1024, 2048])
@pytest.mark.parametrize("layout", ["NN", "NT"])
@pytest.mark.parametrize("format", [Format.E4M3, Format.E5M2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", [None, BackendType.TRITON, BackendType.CK, BackendType.HIPBLASLT])
@pytest.mark.parametrize("auto_tune", [False, True])
def test_gemm_fp8_tensorwise(m, n, k, layout, format, dtype, backend, auto_tune):
    _run_gemm_fp8_test(
        m=m,
        n=n,
        k=k,
        layout=layout,
        format=format,
        dtype=dtype,
        granularity=ScalingGranularity.TENSORWISE,
        backend=backend,
        auto_tune=auto_tune,
    )


@pytest.mark.parametrize("m", [255, 507, 1032, 2056])
@pytest.mark.parametrize("n", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("k", [256, 512, 576, 1024, 2048])
@pytest.mark.parametrize("layout", ["NN", "NT"])
@pytest.mark.parametrize("format", [Format.E4M3, Format.E5M2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", [None, BackendType.TRITON, BackendType.CK])
@pytest.mark.parametrize("auto_tune", [False, True])
def test_gemm_fp8_rowwise(m, n, k, layout, format, dtype, backend, auto_tune):
    _run_gemm_fp8_test(
        m=m,
        n=n,
        k=k,
        layout=layout,
        format=format,
        dtype=dtype,
        granularity=ScalingGranularity.ROWWISE,
        backend=backend,
        auto_tune=auto_tune,
    )


@pytest.mark.parametrize("m", [255, 257, 512, 1024])
@pytest.mark.parametrize("n", [256, 512, 1024, 4096])
@pytest.mark.parametrize("k", [256, 1024, 4096])
@pytest.mark.parametrize("layout", ["NT", "NN"])
@pytest.mark.parametrize("format", [Format.E4M3, Format.E5M2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("backend", [BackendType.TRITON, BackendType.CK])
@pytest.mark.parametrize("auto_tune", [False, True])
def test_gemm_fp8_blockwise(m, n, k, layout, format, dtype, block_size, backend, auto_tune):
    _run_gemm_fp8_test(
        m=m,
        n=n,
        k=k,
        layout=layout,
        format=format,
        dtype=dtype,
        granularity=ScalingGranularity.BLOCKWISE,
        backend=backend,
        auto_tune=auto_tune,
        block_size=block_size,
    )


@pytest.mark.parametrize("m", [256, 512, 1024])
@pytest.mark.parametrize("n", [256, 352, 1024, 2048])
@pytest.mark.parametrize("k", [128, 160, 512, 1024])
@pytest.mark.parametrize("layout", ["NT"])
@pytest.mark.parametrize("format", [Format.E4M3, Format.E5M2, Format.HYBRID])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", [None, BackendType.HIPBLASLT])
@pytest.mark.parametrize("auto_tune", [False, True])
def test_gemm_fp8_mx_blockwise(m, n, k, layout, format, dtype, backend, auto_tune):
    # NOTE: m, n and k must be multiples of 16 for MX_BLOCKWISE.
    assert m % 16 == 0 and n % 16 == 0 and k % 16 == 0, "m, n and k must be multiples of 16"

    # Skip unit test on gfx942.
    mxfp8_supported, reason = check_mxfp8_support()
    if not mxfp8_supported:
        pytest.skip(reason)

    _run_gemm_fp8_test(
        m=m,
        n=n,
        k=k,
        layout=layout,
        format=format,
        dtype=dtype,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        backend=backend,
        auto_tune=auto_tune,
        block_size=32,
    )


@pytest.mark.parametrize("m", [255, 507, 1032, 2056])
@pytest.mark.parametrize("n", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("k", [256, 512, 576, 1024, 2048])
@pytest.mark.parametrize("layout", ["NN", "NT"])
@pytest.mark.parametrize("format", [Format.E4M3, Format.E5M2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", [BackendType.CK, BackendType.HIPBLASLT])
@pytest.mark.deterministic
def test_gemm_fp8_tensorwise_deterministic(m, n, k, layout, format, dtype, backend):
    _run_gemm_fp8_deterministic_test(
        m=m,
        n=n,
        k=k,
        layout=layout,
        format=format,
        dtype=dtype,
        granularity=ScalingGranularity.TENSORWISE,
        backend=backend,
        block_size=None,
    )


@pytest.mark.parametrize("m", [255, 507, 1032, 2056])
@pytest.mark.parametrize("n", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("k", [256, 512, 576, 1024, 2048])
@pytest.mark.parametrize("layout", ["NN", "NT"])
@pytest.mark.parametrize("format", [Format.E4M3, Format.E5M2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", [BackendType.CK])
@pytest.mark.deterministic
def test_gemm_fp8_rowwise_deterministic(m, n, k, layout, format, dtype, backend):
    _run_gemm_fp8_deterministic_test(
        m=m,
        n=n,
        k=k,
        layout=layout,
        format=format,
        dtype=dtype,
        granularity=ScalingGranularity.ROWWISE,
        backend=backend,
        block_size=None,
    )


@pytest.mark.parametrize("m", [255, 257, 512, 1024])
@pytest.mark.parametrize("n", [256, 512, 1024, 4096])
@pytest.mark.parametrize("k", [256, 1024, 4096])
@pytest.mark.parametrize("layout", ["NT", "NN"])
@pytest.mark.parametrize("format", [Format.E4M3, Format.E5M2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", [BackendType.CK])
@pytest.mark.deterministic
def test_gemm_fp8_blockwise_deterministic(m, n, k, layout, format, dtype, backend):
    _run_gemm_fp8_deterministic_test(
        m=m,
        n=n,
        k=k,
        layout=layout,
        format=format,
        dtype=dtype,
        granularity=ScalingGranularity.BLOCKWISE,
        backend=backend,
        block_size=128,
    )
