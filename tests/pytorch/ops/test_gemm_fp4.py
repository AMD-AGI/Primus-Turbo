###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch
import torch.nn.functional as F

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
    Float4QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
    ScalingRecipe,
    float4_e2m1fn_x2,
)
from primus_turbo.pytorch.core.quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorPair,
)
from primus_turbo.pytorch.ops.gemm_fp4 import FP4GemmMXFunction, gemm_fp4
from tests.pytorch.test_utils import compute_snr

torch.manual_seed(42)


def test_gemm_fp4_dense_quant_tail_regression():
    """Eligible dense shapes must not feed unwritten FlyDSL quant rows to GEMM."""
    from primus_turbo.pytorch.core.low_precision import check_mxfp4_support

    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if (props.major, props.minor) != (9, 5):
        pytest.skip("FlyDSL MXFP4 GEMM requires gfx950")

    torch.manual_seed(20260908)
    a = torch.randn((128, 256), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
    reference = a @ b.T
    config = Float4QuantConfig(
        granularity=ScalingGranularity.MX_BLOCKWISE,
        format=Format.E2M1_X2,
        block_size=32,
        scale_dtype=ScaleDtype.E8M0,
        use_preshuffle=False,
    )

    GlobalBackendManager.set_gemm_backend(BackendType.FLYDSL)
    GlobalBackendManager.set_auto_tune(False)
    try:
        actual = gemm_fp4(a, b, trans_a=False, trans_b=True, out_dtype=torch.bfloat16, config=config)
        torch.cuda.synchronize()
    finally:
        GlobalBackendManager.reset()

    assert torch.isfinite(actual).all()
    assert compute_snr(reference, actual) > 10


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
@pytest.mark.parametrize("backend", [BackendType.AITER, BackendType.FLYDSL])
@pytest.mark.parametrize("auto_tune", [False, True])
@pytest.mark.parametrize("preshuffle", [False, True])
def test_gemm_fp4_mx_blockwise(m, n, k, layout, format, dtype, granularity, backend, auto_tune, preshuffle):
    if backend != BackendType.AITER and preshuffle:
        pytest.skip("Preshuffle is only supported for AITER backend")

    if backend == BackendType.AITER and dtype != torch.bfloat16:
        pytest.skip("AITER backend only supports bfloat16 dtype")

    if backend == BackendType.FLYDSL:
        if not (m % 64 == 0 and n % 64 == 0 and k % 64 == 0):
            pytest.skip("FlyDSL MXFP4 backend requires M/N/K all multiples of 64")

    # Skip redundant test: auto_tune is ignored when backend is explicitly specified
    if backend is not None and auto_tune:
        pytest.skip("auto_tune is ignored when backend is explicitly specified")

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
        f"backend={backend}, auto_tune={auto_tune}, preshuffle={preshuffle}"
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
    c_ref.backward(torch.ones_like(c_ref))
    torch.cuda.synchronize()

    # Config + FWD + BWD
    # NOTE: scaling recipe reference: https://arxiv.org/pdf/2509.25149
    config = Float4QuantConfig(
        granularity=granularity,
        format=format,
        block_size=32,
        scale_dtype=ScaleDtype.E8M0,
        use_preshuffle=preshuffle,
    )
    print(config)
    c = gemm_fp4(a, b, trans_a, trans_b, dtype, config)
    c.backward(torch.ones_like(c))

    # Check Shape
    assert c.shape == c_ref.shape
    assert a.grad.shape == a_ref.grad.shape
    assert b.grad.shape == b_ref.grad.shape

    snr_threshold = 10
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


def _run_gemm_fp4_mx_quantized_tensor_test(
    m: int,
    n: int,
    k: int,
    layout: str,
    format: Format,
    dtype: torch.dtype,
    backend: BackendType | None,
    preshuffle: bool,
):
    """Shared helper: externally quantize both ``a`` and ``b`` into
    :class:`QuantizedTensor`, pass them into :func:`gemm_fp4`, and validate
    forward/backward SNR vs a high-precision reference.
    """
    from primus_turbo.pytorch.core.low_precision import check_mxfp4_support

    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    assert m % 16 == 0 and n % 16 == 0 and k % 16 == 0, "Assume m, n and k are multiples of 16."

    GlobalBackendManager.set_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(False)

    device = "cuda:0"
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    trans_a = layout[0] == "T"
    trans_b = layout[1] == "T"
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)

    a = torch.randn(a_shape, dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    torch.cuda.synchronize()

    # Reference (high precision)
    a_mat = a_ref.T if trans_a else a_ref
    b_mat = b_ref.T if trans_b else b_ref
    c_ref = a_mat @ b_mat
    grad_c = torch.ones_like(c_ref)
    c_ref.backward(grad_c)
    torch.cuda.synchronize()

    config = Float4QuantConfig(
        granularity=ScalingGranularity.MX_BLOCKWISE,
        format=format,
        block_size=32,
        scale_dtype=ScaleDtype.E8M0,
        use_preshuffle=preshuffle,
    )

    fp4_dtype = FP4GemmMXFunction.get_fp4_dtype(format)

    # Externally construct QuantizedTensor with the SAME scaling recipes that
    # gemm_fp4's autograd Function uses internally, so the forward result
    # should match the non-QT path bit-for-bit. shuffle_scale / shuffle_out
    qt_a = QuantizedTensor.quantize(
        a,
        fp4_dtype,
        config.granularity,
        block_size=config.block_size,
        axis=1,
        scaling_recipe=ScalingRecipe(
            use_2d_block=False,
            use_sr=False,
            use_rht=False,
            shuffle_scale=config.use_preshuffle,
            shuffle_out=False,
        ),
    )

    qt_b = QuantizedTensor.quantize(
        b,
        fp4_dtype,
        config.granularity,
        block_size=config.block_size,
        axis=1,
        scaling_recipe=ScalingRecipe(
            use_2d_block=True,
            use_sr=False,
            use_rht=False,
            shuffle_scale=config.use_preshuffle,
            shuffle_out=config.use_preshuffle,
        ),
    )

    c = gemm_fp4(
        QuantizedTensorPair(data=qt_a, data_t=None),
        QuantizedTensorPair(data=qt_b, data_t=None),
        trans_a,
        trans_b,
        dtype,
        config,
    )
    c.backward(torch.ones_like(c))
    torch.cuda.synchronize()

    assert c.shape == c_ref.shape
    assert qt_a.grad is not None and qt_a.grad.shape == a.shape
    assert qt_b.grad is not None and qt_b.grad.shape == b.shape

    snr_threshold = 10
    c_snr = compute_snr(c_ref, c)
    a_grad_snr = compute_snr(a_ref.grad, qt_a.grad)
    b_grad_snr = compute_snr(b_ref.grad, qt_b.grad)
    print(
        f"\n[QT-MXFP4] M={m}, N={n}, K={k}, layout={layout}, format={format}, "
        f"dtype={dtype}, backend={backend}: "
        f"C-SNR={c_snr:.2f} dB, AGrad-SNR={a_grad_snr:.2f} dB, BGrad-SNR={b_grad_snr:.2f} dB"
    )
    assert c_snr > snr_threshold, f"c_snr={c_snr:.2f} too low"
    assert a_grad_snr > snr_threshold, f"a_grad_snr={a_grad_snr:.2f} too low"
    assert b_grad_snr > snr_threshold, f"b_grad_snr={b_grad_snr:.2f} too low"

    GlobalBackendManager.reset()


@pytest.mark.parametrize("m", [256, 1024])
@pytest.mark.parametrize("n", [256, 1024])
@pytest.mark.parametrize("k", [128, 512])
@pytest.mark.parametrize("layout", ["NT"])
@pytest.mark.parametrize("format", [Format.E2M1_X2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", [None, BackendType.HIPBLASLT, BackendType.FLYDSL])
@pytest.mark.parametrize("preshuffle", [False, True])
def test_gemm_fp4_mx_blockwise_quantized_tensor(m, n, k, layout, format, dtype, backend, preshuffle):
    """MX_BLOCKWISE gemm_fp4 with pre-quantized QuantizedTensor inputs.

    HipBLASLt / default-dispatch / FlyDSL coverage. AITER QT coverage is in
    :func:`test_gemm_fp4_mx_blockwise_quantized_tensor_aiter_preshuffled`
    below because AITER lacks tuned GEMM configs for these small shapes
    (default config produces near-zero SNR).
    """
    if backend != BackendType.AITER and preshuffle:
        pytest.skip("Preshuffle is only supported for AITER backend")
    if backend == BackendType.AITER and dtype != torch.bfloat16:
        pytest.skip("AITER backend only supports bfloat16 dtype")
    if backend == BackendType.FLYDSL and not (m % 64 == 0 and n % 64 == 0 and k % 64 == 0):
        pytest.skip("FlyDSL MXFP4 backend requires M/N/K all multiples of 64")

    _run_gemm_fp4_mx_quantized_tensor_test(
        m=m,
        n=n,
        k=k,
        layout=layout,
        format=format,
        dtype=dtype,
        backend=backend,
        preshuffle=preshuffle,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", [None, BackendType.FLYDSL])
def test_gemm_fp4_mx_fused_grad_accum(dtype, backend):
    """``fuse_bgrad_accum_pattern`` must leave ``main_grad`` holding previous + wgrad.

    FlyDSL is the only FP4 backend with the accumulate epilogue and its store is 16-bit,
    so ``main_grad`` is allocated in the weight's own dtype rather than Megatron's fp32.
    """
    from primus_turbo.pytorch.core.low_precision import check_mxfp4_support

    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = "cuda:0"
    m, n, k = 256, 512, 256  # FlyDSL MXFP4 needs M/N/K all multiples of 256

    GlobalBackendManager.set_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(False)

    config = Float4QuantConfig(
        granularity=ScalingGranularity.MX_BLOCKWISE,
        format=Format.E2M1_X2,
        block_size=32,
        scale_dtype=ScaleDtype.E8M0,
        use_gradient_sr=False,  # the two runs must quantize grad_out identically
    )

    a = torch.randn((m, k), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((n, k), dtype=dtype, device=device, requires_grad=True)
    grad_out = torch.randn((m, n), dtype=dtype, device=device)
    a_fused = a.detach().clone().requires_grad_(True)
    b_fused = b.detach().clone().requires_grad_(True)
    torch.cuda.synchronize()

    # Baseline: ordinary autograd, b.grad holds the weight gradient.
    out = gemm_fp4(a, b, trans_b=True, out_dtype=dtype, config=config)
    out.backward(grad_out)
    torch.cuda.synchronize()

    # Fused: the wgrad is accumulated into a pre-seeded main_grad buffer.
    previous = torch.randn(b_fused.shape, dtype=dtype, device=device)
    b_fused.main_grad = previous.clone()
    b_fused.grad_added_to_main_grad = False

    out_fused = gemm_fp4(
        a_fused,
        b_fused,
        trans_b=True,
        out_dtype=dtype,
        config=config,
        fuse_bgrad_accum_pattern="megatron",
    )
    out_fused.backward(grad_out)
    torch.cuda.synchronize()

    torch.testing.assert_close(out_fused, out)
    assert b_fused.grad_added_to_main_grad is True, "weight must be flagged during forward"
    assert b_fused.grad.shape == b_fused.shape, "dummy wgrad must keep the weight's shape"
    assert b_fused.grad.dtype == b_fused.dtype, "dummy wgrad must keep the weight's dtype"

    snr_threshold = 10

    a_grad_snr = compute_snr(a.grad, a_fused.grad)
    print(f"AGrad-SNR: {a_grad_snr:.2f} dB")
    assert a_grad_snr > snr_threshold, "a_grad_snr too low"

    accumulated = b_fused.main_grad.float() - previous.float()
    b_grad_snr = compute_snr(b.grad.float(), accumulated)
    print(f"BGrad-SNR: {b_grad_snr:.2f} dB")
    assert b_grad_snr > snr_threshold, "b_grad_snr too low"

    GlobalBackendManager.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_use_gradient_sr_false():
    """Gradient quantization with use_gradient_sr=False should be deterministic (identical)."""
    from primus_turbo.pytorch.core.low_precision import check_mxfp4_support

    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    device = "cuda:0"
    m, k, n = 256, 512, 256
    dtype = torch.bfloat16

    config = Float4QuantConfig(use_gradient_sr=False)

    a = torch.randn(m, k, dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(n, k, dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn(m, n, dtype=dtype, device=device)

    out1 = gemm_fp4(a, b, trans_b=True, config=config)
    out1.backward(grad_output)
    a_grad1 = a.grad.clone()
    b_grad1 = b.grad.clone()
    a.grad = None
    b.grad = None

    out2 = gemm_fp4(a, b, trans_b=True, config=config)
    out2.backward(grad_output)
    a_grad2 = a.grad.clone()
    b_grad2 = b.grad.clone()

    assert torch.equal(a_grad1, a_grad2), "A gradients should be identical without stochastic rounding"
    assert torch.equal(b_grad1, b_grad2), "B gradients should be identical without stochastic rounding"


@pytest.mark.parametrize("m", [256])
@pytest.mark.parametrize("n", [256])
@pytest.mark.parametrize("k", [128, 512])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_fp4_mx_blockwise_torch_compile_backward(m, n, k, dtype):
    """Regression: FP4 GEMM forward+backward must trace and run under torch.compile.

    Covers the Inductor-traced backward path that the ``FP4GemmMXFunction`` backward
    compile-compatibility fix targets (contiguous ``grad_out`` + ``shuffle_scale=False``
    on the gradient scaling recipe). Verifies the compiled path traces end-to-end and
    produces finite, correctly shaped gradients matching a high-precision reference (SNR).
    """
    from primus_turbo.pytorch.core.low_precision import check_mxfp4_support

    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    assert m % 16 == 0 and n % 16 == 0 and k % 16 == 0, "Assume m, n and k are multiples of 16."

    device = "cuda:0"
    torch.manual_seed(42)

    # NT layout (a: [m, k], b: [n, k]); matches the production FP4 GEMM usage.
    trans_a, trans_b = False, True

    # Pin a concrete backend so the test is deterministic and independent of the
    # PRIMUS_TURBO_GEMM_BACKEND environment variable; the compile-backward fix is
    # backend-independent (it concerns grad_out quantization, not the GEMM impl).
    GlobalBackendManager.set_gemm_backend(BackendType.HIPBLASLT)
    GlobalBackendManager.set_auto_tune(False)

    config = Float4QuantConfig(
        granularity=ScalingGranularity.MX_BLOCKWISE,
        format=Format.E2M1_X2,
        block_size=32,
        scale_dtype=ScaleDtype.E8M0,
    )

    a = torch.randn((m, k), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((n, k), dtype=dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    torch.cuda.synchronize()

    # High-precision reference.
    c_ref = a_ref @ b_ref.T
    c_ref.backward(torch.ones_like(c_ref))
    torch.cuda.synchronize()

    def fp4_gemm(x, w):
        return gemm_fp4(x, w, trans_a, trans_b, dtype, config)

    compiled = torch.compile(fp4_gemm)
    c = compiled(a, b)
    c.backward(torch.ones_like(c))
    torch.cuda.synchronize()

    assert c.shape == c_ref.shape
    assert a.grad is not None and a.grad.shape == a_ref.grad.shape
    assert b.grad is not None and b.grad.shape == b_ref.grad.shape
    assert torch.isfinite(c).all(), "compiled forward produced non-finite values"
    assert torch.isfinite(a.grad).all(), "compiled a.grad is non-finite"
    assert torch.isfinite(b.grad).all(), "compiled b.grad is non-finite"

    snr_threshold = 10
    assert compute_snr(c_ref, c) > snr_threshold, "compiled c_snr too low"
    assert compute_snr(a_ref.grad, a.grad) > snr_threshold, "compiled a_grad_snr too low"
    assert compute_snr(b_ref.grad, b.grad) > snr_threshold, "compiled b_grad_snr too low"

    GlobalBackendManager.reset()


# ----------------------------------------------------------------------------
# Determinism suite (run with --deterministic-only): bit-exact across repeats.
# ----------------------------------------------------------------------------
_DET_GEMM_FP4_MNK = [(256, 256, 256), (512, 512, 256), (1024, 1024, 512)]


@pytest.mark.parametrize("mnk", _DET_GEMM_FP4_MNK)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "backend",
    [BackendType.AITER, BackendType.HIPBLASLT, BackendType.FLYDSL],
    ids=["AITER", "HIPBLASLT", "FLYDSL"],
)
@pytest.mark.deterministic
def test_gemm_fp4_deterministic(mnk, dtype, backend):
    """Dense MXFP4 GEMM fwd + bwd are bit-exact across 10 repeats (SR off), and
    match a high-precision reference (SNR)."""
    from primus_turbo.pytorch.core.low_precision import check_mxfp4_support

    supported, reason = check_mxfp4_support()
    if not supported:
        pytest.skip(reason)

    m, n, k = mnk
    if backend == BackendType.AITER and dtype != torch.bfloat16:
        pytest.skip("AITER backend only supports bfloat16 dtype")
    if backend == BackendType.FLYDSL and not (m % 64 == 0 and n % 64 == 0 and k % 64 == 0):
        pytest.skip("FlyDSL MXFP4 backend requires M/N/K all multiples of 64")

    GlobalBackendManager.set_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(False)

    device = "cuda:0"
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # NT layout (a: [m, k], b: [n, k]) -- the production FP4 GEMM usage.
    trans_a, trans_b = False, True
    config = Float4QuantConfig(
        granularity=ScalingGranularity.MX_BLOCKWISE,
        format=Format.E2M1_X2,
        block_size=32,
        scale_dtype=ScaleDtype.E8M0,
    )
    print(f"\n[deterministic] M={m}, N={n}, K={k}, dtype={dtype}, backend={backend}")

    a0 = torch.randn((m, k), dtype=dtype, device=device)
    b0 = torch.randn((n, k), dtype=dtype, device=device)
    a0 = a0 / a0.abs().max()
    b0 = b0 / b0.abs().max()

    # Reference (high precision)
    a_ref = a0.detach().clone().requires_grad_()
    b_ref = b0.detach().clone().requires_grad_()
    c_ref = a_ref @ b_ref.T
    grad_c = torch.randn_like(c_ref)
    c_ref.backward(grad_c)
    torch.cuda.synchronize()

    def _run_once():
        # Clean memory each iter so the caching allocator can't alias a buffer
        # still being written by a pending op from a prior case.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        a = a0.detach().clone().requires_grad_()
        b = b0.detach().clone().requires_grad_()
        c = gemm_fp4(a, b, trans_a, trans_b, dtype, config)
        c.backward(grad_c)
        return c.detach(), a.grad.detach(), b.grad.detach()

    try:
        repeats = 10
        outs = []
        for _ in range(repeats):
            outs.append(_run_once())
            torch.cuda.synchronize()

        c0, da0, db0 = outs[0]
        # Determinism (bitwise identical across runs)
        for i in range(1, repeats):
            ci, dai, dbi = outs[i]
            torch.testing.assert_close(c0, ci, rtol=0, atol=0)
            torch.testing.assert_close(da0, dai, rtol=0, atol=0)
            torch.testing.assert_close(db0, dbi, rtol=0, atol=0)

        # Correctness (close to reference)
        snr_threshold = 10
        assert compute_snr(c_ref.detach(), c0) > snr_threshold, "c_snr too low"
        assert compute_snr(a_ref.grad.detach(), da0) > snr_threshold, "a_grad_snr too low"
        assert compute_snr(b_ref.grad.detach(), db0) > snr_threshold, "b_grad_snr too low"
    finally:
        GlobalBackendManager.reset()


# ---------------------------------------------------------------------------
# AITER preshuffle fast-path coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m,n,k",
    [
        # Flux 12B shapes (have AITER tuned GEMM configs); ensures real
        # coverage on shapes the production fast path actually hits.
        (16384, 3072, 3072),
        (16384, 3072, 12288),
        (16384, 12288, 3072),
        # Small shapes (fall back to AITER's default config; useful for
        # CI without a Flux-scale GPU).
        (256, 256, 128),
        (1024, 1024, 512),
    ],
)
def test_gemm_fp4_impl_aiter_preshuffle_parity(m, n, k):
    """``gemm_fp4_impl(preshuffled=True)`` must produce bitwise-identical
    output to ``gemm_fp4_impl(preshuffled=False)`` when both reach the same
    ``aiter.gemm_a4w4(..., bpreshuffle=True)`` call, just with the
    shuffles applied at different points (inside execute vs upfront).
    Locks the fast-path correctness against future regressions in AITER's
    ``shuffle_scale`` / ``shuffle_weight`` ops.
    """
    from primus_turbo.pytorch.core.low_precision import check_mxfp4_support
    from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import gemm_fp4_impl

    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    import aiter

    if aiter.get_GEMM_config(m, n, k) is None:
        pytest.skip("AITER does not advertise a GEMM config for this shape")

    GlobalBackendManager.set_gemm_backend(BackendType.AITER)
    GlobalBackendManager.set_auto_tune(False)

    try:
        device = "cuda:0"
        torch.manual_seed(42)
        dtype = torch.bfloat16

        a_hp = torch.randn((m, k), dtype=dtype, device=device)
        b_hp = torch.randn((n, k), dtype=dtype, device=device)

        fp4_dtype = torch.float4_e2m1fn_x2
        granularity = ScalingGranularity.MX_BLOCKWISE
        block_size = 32

        # Vanilla (unshuffled) quantize; AITER's execute will do the
        # 3 shuffles internally.
        qa = QuantizedTensor.quantize(
            a_hp,
            fp4_dtype,
            granularity,
            block_size=block_size,
            axis=1,
            scaling_recipe=ScalingRecipe(
                use_2d_block=False,
                use_sr=False,
                use_rht=False,
                shuffle_scale=False,
                shuffle_out=False,
            ),
        )
        qb = QuantizedTensor.quantize(
            b_hp,
            fp4_dtype,
            granularity,
            block_size=block_size,
            axis=1,
            scaling_recipe=ScalingRecipe(
                use_2d_block=True,
                use_sr=False,
                use_rht=False,
                shuffle_scale=False,
                shuffle_out=False,
            ),
        )

        out_no_pre = gemm_fp4_impl(
            qa.qdata,
            qa.scale_inv,
            False,
            qb.qdata,
            qb.scale_inv,
            True,
            dtype,
            False,
            granularity=granularity.value,
            default_backend=BackendType.AITER.value,
            preshuffled=False,
        )

        # Pre-shuffle once outside the call; AITER's execute now skips
        # the 3 shuffle kernels and goes straight to gemm_a4w4.
        shuffled_a_scale = torch.ops.primus_turbo_cpp_extension.shuffle_scale(qa.scale_inv, [16, 16])
        shuffled_b_scale = torch.ops.primus_turbo_cpp_extension.shuffle_scale(qb.scale_inv, [16, 16])
        shuffled_b_data = torch.ops.primus_turbo_cpp_extension.shuffle_weight(qb.qdata, [16, 16])

        out_pre = gemm_fp4_impl(
            qa.qdata,
            shuffled_a_scale,
            False,
            shuffled_b_data,
            shuffled_b_scale,
            True,
            dtype,
            False,
            granularity=granularity.value,
            default_backend=BackendType.AITER.value,
            preshuffled=True,
        )

        torch.cuda.synchronize()
        assert out_no_pre.shape == out_pre.shape
        # Both paths converge on the same aiter.gemm_a4w4(bpreshuffle=True)
        # consuming the same byte-identical shuffled inputs -> outputs match.
        assert torch.equal(out_no_pre, out_pre), (
            "preshuffled=True must match preshuffled=False bit-for-bit on the AITER backend"
        )
    finally:
        GlobalBackendManager.reset()


# (M, K, I): the smallest that clear dense_glu_epi_quant_supported (M % 256, I % 64,
# K // 256 >= 4) and the FlyDSL GEMM's 64-multiple requirement.
_MLP_MKI = (512, 1024, 1024)
# Llama-3.1-8B's own MLP, one case: it is far more work than the small one.
_MLP_MKI_LLAMA = (8192, 4096, 14336)

_MLP_GATES = {"silu": F.silu, "gelu": lambda t: F.gelu(t, approximate="tanh")}
# Sized against these leaves rather than borrowed: a limit small enough to saturate
# most of l1 leaves dx with no signal, and the test then measures nothing.
_MLP_CLAMP_LIMIT = 1.0
_MLP_SNR_THRESHOLD = 6.0
# The clamp's dead band costs the gradients a little.
_MLP_CLAMP_SNR_THRESHOLD = 7.0


def _mlp_fp4_leaves(dtype, seed=42, mki=None):
    m, k, i = mki or _MLP_MKI
    torch.manual_seed(seed)
    device = "cuda:0"
    x = torch.randn((m, k), dtype=dtype, device=device, requires_grad=True)
    w1 = (torch.randn((2 * i, k), dtype=dtype, device=device) * 0.02).requires_grad_(True)
    w2 = (torch.randn((k, i), dtype=dtype, device=device) * 0.02).requires_grad_(True)
    grad_out = torch.randn((m, k), dtype=dtype, device=device) * 0.1
    return x, w1, w2, grad_out


def _mlp_fp4_run(dtype, prequantize_x=False, activation="silu", clamp_limit=None, mki=None):
    from primus_turbo.pytorch.core.quantized_tensor import (
        QuantizedTensor,
        QuantizedTensorPair,
    )
    from primus_turbo.pytorch.ops import mlp_fp4

    x, w1, w2, grad_out = _mlp_fp4_leaves(dtype, mki=mki)
    x_in = x
    if prequantize_x:
        # data is the row-wise operand, data_t the col-wise (RHT) wgrad one.
        from primus_turbo.pytorch.ops.quantization import quantize_fp4_with_trans

        row_recipe, col_recipe = ScalingRecipe(), ScalingRecipe(use_rht=True)
        row, row_scale, col, col_scale = quantize_fp4_with_trans(
            x.detach(),
            float4_e2m1fn_x2,
            ScalingGranularity.MX_BLOCKWISE,
            block_size=32,
            scaling_recipe=row_recipe,
            scaling_recipe_for_trans=col_recipe,
        )

        def _wrap(data, scale_inv, shape, recipe, axis):
            return QuantizedTensor(
                data,
                scale_inv,
                shape=shape,
                orig_dtype=x.dtype,
                dest_dtype=float4_e2m1fn_x2,
                granularity=ScalingGranularity.MX_BLOCKWISE,
                block_size=32,
                scaling_recipe=recipe,
                quantized_axis=axis,
            )

        m, k, _ = mki or _MLP_MKI
        x_in = QuantizedTensorPair(
            _wrap(row, row_scale, torch.Size((m, k)), row_recipe, -1),
            _wrap(col, col_scale, torch.Size((k, m)), col_recipe, -2),
        )
    out = mlp_fp4(x_in, w1, w2, activation=activation, clamp_limit=clamp_limit)
    out.backward(grad_out)
    return out.detach(), (None if prequantize_x else x.grad), w1.grad, w2.grad


def _mlp_fp4_reference(dtype, activation="silu", clamp_limit=None, mki=None):
    i = (mki or _MLP_MKI)[2]
    x, w1, w2, grad_out = _mlp_fp4_leaves(dtype, mki=mki)
    l1 = x.float() @ w1.float().t()
    gate, up = l1[:, :i], l1[:, i:]
    if clamp_limit is not None:
        # The gate saturates from above only; the linear half from both sides.
        gate = gate.clamp(max=clamp_limit)
        up = up.clamp(min=-clamp_limit, max=clamp_limit)
    act = _MLP_GATES[activation](gate) * up
    out = act @ w2.float().t()
    out.backward(grad_out.float())
    return out.detach(), x.grad, w1.grad, w2.grad


_MLP_FP4_TENSORS = ("out", "dx", "dw1", "dw2")


@pytest.mark.parametrize(
    "mki, activation, clamp_limit",
    [
        (_MLP_MKI, "silu", None),
        (_MLP_MKI, "gelu", None),
        (_MLP_MKI, "silu", _MLP_CLAMP_LIMIT),
        (_MLP_MKI, "gelu", _MLP_CLAMP_LIMIT),
        pytest.param(_MLP_MKI_LLAMA, "silu", None, id="llama3.1-8b"),
    ],
)
def test_mlp_fp4_mx_blockwise(mki, activation, clamp_limit):
    """``mlp_fp4`` end to end against an eager fp32 reference.

    The floor is low for the same reason the grouped MLP's is: MXFP4 carries ~2 mantissa
    bits and this stacks four quantizations plus the wgrad operands' RHT, so it catches a
    wrong answer rather than the quantization.
    """
    from primus_turbo.pytorch.core.low_precision import check_mxfp4_support

    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    floor = _MLP_SNR_THRESHOLD if clamp_limit is None else _MLP_CLAMP_SNR_THRESHOLD
    ref = _mlp_fp4_reference(torch.bfloat16, activation, clamp_limit, mki)
    got = _mlp_fp4_run(torch.bfloat16, activation=activation, clamp_limit=clamp_limit, mki=mki)
    for name, r, g in zip(_MLP_FP4_TENSORS, ref, got):
        snr = compute_snr(r.float(), g.float())
        print(f"{name}-SNR: {snr:.2f} dB")
        assert snr > floor, f"{name} snr too low"


def test_mlp_fp4_accepts_a_prequantized_x():
    """A caller that already has x quantized must get the same numbers.

    This is the only way to hand the op a pre-quantized activation, so it is also
    what the fused RMSNorm entry point feeds it.
    """
    from primus_turbo.pytorch.core.low_precision import check_mxfp4_support

    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    plain = _mlp_fp4_run(torch.bfloat16)
    pre = _mlp_fp4_run(torch.bfloat16, prequantize_x=True)
    for name, a, b in zip(_MLP_FP4_TENSORS, plain, pre):
        if a is None or b is None:  # dx: a QuantizedTensor x is not a leaf
            continue
        snr = compute_snr(a.float(), b.float())
        print(f"{name}-SNR: {snr:.2f} dB")
        assert snr > 100, f"{name} must be reproduced, got {snr:.2f} dB"


def test_rmsnorm_residual_fp4_feeds_mlp_fp4():
    """The fused norm's pair feeds the MLP directly and still carries the gradient."""
    from primus_turbo.pytorch.core.low_precision import check_mxfp4_support
    from primus_turbo.pytorch.ops import mlp_fp4
    from primus_turbo.pytorch.ops.normalization import rmsnorm_residual_fp4

    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    m, k, i = _MLP_MKI
    torch.manual_seed(42)
    device = "cuda:0"
    x = torch.randn((m, k), dtype=torch.bfloat16, device=device, requires_grad=True)
    residual = torch.randn((m, k), dtype=torch.bfloat16, device=device, requires_grad=True)
    gamma = torch.randn((k,), dtype=torch.bfloat16, device=device, requires_grad=True)
    w1 = (torch.randn((2 * i, k), dtype=torch.bfloat16, device=device) * 0.02).requires_grad_(True)
    w2 = (torch.randn((k, i), dtype=torch.bfloat16, device=device) * 0.02).requires_grad_(True)

    y, x_plus_r, y_fp4 = rmsnorm_residual_fp4(x, residual, gamma)
    out = mlp_fp4(y_fp4, w1, w2)
    out.backward(torch.randn_like(out) * 0.1)

    assert out.shape == (m, k)
    assert torch.isfinite(out).all()
    assert torch.isfinite(y).all() and torch.isfinite(x_plus_r).all()
    # The pair is a straight-through estimator for y, so the dgrad has to land here.
    for name, t in (("x", x), ("residual", residual), ("gamma", gamma)):
        assert t.grad is not None, f"{name} got no gradient through the quantized pair"
        assert torch.isfinite(t.grad).all(), f"{name} gradient is not finite"
