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
    ScalingRecipe,
)
from primus_turbo.pytorch.core.quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorPair,
)
from primus_turbo.pytorch.ops.gemm_fp4 import FP4GemmMXFunction, gemm_fp4
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
@pytest.mark.parametrize("backend", [None, BackendType.HIPBLASLT, BackendType.AITER])
@pytest.mark.parametrize("auto_tune", [False, True])
def test_gemm_fp4_mx_blockwise(m, n, k, layout, format, dtype, granularity, backend, auto_tune):
    if backend == BackendType.AITER:
        if dtype != torch.bfloat16:
            pytest.skip("AITER backend only supports bfloat16 dtype")
        import aiter

        aiter_gemm_config = aiter.get_GEMM_config(m, n, k)
        if aiter_gemm_config is None:
            pytest.skip("AITER does not support this gemm configuration. Have potential numerical issue.")

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

    # End-to-end wiring check: with AITER pinned + autotune off, the
    # preshuffle fast path must be active so FP4GemmMXFunction emits
    # pre-shuffled tensors and gemm_fp4_impl(preshuffled=True) runs.
    from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import enable_preshuffle

    if backend == BackendType.AITER and not auto_tune:
        assert (
            enable_preshuffle() is True
        ), "AITER + autotune-off should enable the preshuffle fast path"
    else:
        assert enable_preshuffle() is False, (
            f"Preshuffle must be off for backend={backend}, auto_tune={auto_tune}"
        )

    print(
        f"\nM={m}, N={n}, K={k}, layout={layout}, dtype={dtype}, format={format}, "
        f"backend={backend}, auto_tune={auto_tune}, preshuffle={enable_preshuffle()}"
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
        granularity=granularity, format=format, block_size=32, scale_dtype=ScaleDtype.E8M0
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

    # Read the live preshuffle policy AFTER set_gemm_backend, so the
    # externally-constructed QuantizedTensor recipes match what
    # FP4GemmMXFunction will derive internally. Under AITER this becomes
    # True and shuffle_scale / shuffle_out flip on.
    from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import enable_preshuffle

    preshuffle = enable_preshuffle()

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
    )

    fp4_dtype = FP4GemmMXFunction.get_fp4_dtype(format)

    # Externally construct QuantizedTensor with the SAME scaling recipes that
    # gemm_fp4's autograd Function uses internally, so the forward result
    # should match the non-QT path bit-for-bit. shuffle_scale / shuffle_out
    # MUST match enable_preshuffle() or check_quantized_tensor's strict
    # equality assert fires under AITER.
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
            shuffle_scale=preshuffle,
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
            shuffle_scale=preshuffle,
            shuffle_out=preshuffle,
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
@pytest.mark.parametrize("backend", [None, BackendType.HIPBLASLT])
def test_gemm_fp4_mx_blockwise_quantized_tensor(m, n, k, layout, format, dtype, backend):
    """MX_BLOCKWISE gemm_fp4 with pre-quantized QuantizedTensor inputs.

    HipBLASLt / default-dispatch coverage. AITER QT coverage is in
    :func:`test_gemm_fp4_mx_blockwise_quantized_tensor_aiter_preshuffled`
    below because AITER lacks tuned GEMM configs for these small shapes
    (default config produces near-zero SNR).
    """
    _run_gemm_fp4_mx_quantized_tensor_test(
        m=m,
        n=n,
        k=k,
        layout=layout,
        format=format,
        dtype=dtype,
        backend=backend,
    )



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_use_gradient_sr_true():
    """use_gradient_sr=True should select the SR kernel path, producing gradients
    that differ from the deterministic RTN path."""
    from primus_turbo.pytorch.core.low_precision import check_mxfp4_support

    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    device = "cuda:0"
    m, k, n = 256, 512, 256
    dtype = torch.bfloat16

    config_sr = Float4QuantConfig(use_gradient_sr=True)
    config_rtn = Float4QuantConfig(use_gradient_sr=False)

    a = torch.randn(m, k, dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(n, k, dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn(m, n, dtype=dtype, device=device)

    out_sr = gemm_fp4(a, b, trans_b=True, config=config_sr)
    out_sr.backward(grad_output)
    a_grad_sr = a.grad.clone()
    b_grad_sr = b.grad.clone()
    a.grad = None
    b.grad = None

    out_rtn = gemm_fp4(a, b, trans_b=True, config=config_rtn)
    out_rtn.backward(grad_output)
    a_grad_rtn = a.grad.clone()
    b_grad_rtn = b.grad.clone()

    # SR-rounded gradients should differ from RTN-rounded ones.
    assert not torch.equal(a_grad_sr, a_grad_rtn) or not torch.equal(
        b_grad_sr, b_grad_rtn
    ), "SR gradients should differ from RTN gradients when use_gradient_sr=True selects SR kernel"


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

@pytest.mark.parametrize(
    "m,n,k",
    [
        # Flux 12B shapes with AITER tuned GEMM configs.
        (16384, 3072, 3072),
        (16384, 3072, 12288),
        (16384, 12288, 3072),
    ],
)
def test_gemm_fp4_mx_blockwise_quantized_tensor_aiter_preshuffled(m, n, k):
    """AITER + pre-quantized QuantizedTensorPair contract.

    The externally constructed rowwise + colwise tensors must carry
    matching ``shuffle_scale`` / ``shuffle_out`` flags
    ``enable_preshuffle()`` implies, or
    :func:`check_quantized_tensor`'s strict equality assert fires
    under :class:`BackendType.AITER`. The colwise tensor (``data_t``)
    must be supplied (cannot be derived via ``dequantize()`` on a
    preshuffled-scale forward tensor).
    """
    from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import enable_preshuffle

    from primus_turbo.pytorch.core.low_precision import check_mxfp4_support

    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    GlobalBackendManager.set_gemm_backend(BackendType.AITER)
    GlobalBackendManager.set_auto_tune(False)
    assert enable_preshuffle() is True

    try:
        device = "cuda:0"
        dtype = torch.bfloat16
        fp4_dtype = FP4GemmMXFunction.get_fp4_dtype(Format.E2M1_X2)
        torch.manual_seed(42)

        a = torch.randn((m, k), dtype=dtype, device=device, requires_grad=True)
        b = torch.randn((n, k), dtype=dtype, device=device, requires_grad=True)
        a_ref = a.detach().clone().requires_grad_()
        b_ref = b.detach().clone().requires_grad_()
        c_ref = a_ref @ b_ref.T
        c_ref.backward(torch.ones_like(c_ref))
        torch.cuda.synchronize()

        config = Float4QuantConfig(
            granularity=ScalingGranularity.MX_BLOCKWISE,
            format=Format.E2M1_X2,
            block_size=32,
            scale_dtype=ScaleDtype.E8M0,
        )

        # Recipes match FP4GemmMXFunction internal construction.
        qt_a = QuantizedTensor.quantize(
            a, fp4_dtype, config.granularity, block_size=32, axis=1,
            scaling_recipe=ScalingRecipe(
                use_2d_block=False, use_sr=False, use_rht=False,
                shuffle_scale=True, shuffle_out=False,
            ),
        )
        qt_a_t = QuantizedTensor.quantize(
            a, fp4_dtype, config.granularity, block_size=32, axis=0,
            scaling_recipe=ScalingRecipe(
                use_2d_block=False, use_sr=False, use_rht=True,
                shuffle_scale=True, shuffle_out=True,
            ),
        )
        qt_b = QuantizedTensor.quantize(
            b, fp4_dtype, config.granularity, block_size=32, axis=1,
            scaling_recipe=ScalingRecipe(
                use_2d_block=True, use_sr=False, use_rht=False,
                shuffle_scale=True, shuffle_out=True,
            ),
        )
        qt_b_t = QuantizedTensor.quantize(
            b, fp4_dtype, config.granularity, block_size=32, axis=0,
            scaling_recipe=ScalingRecipe(
                use_2d_block=True, use_sr=False, use_rht=True,
                shuffle_scale=True, shuffle_out=True,
            ),
        )

        c = gemm_fp4(
            QuantizedTensorPair(data=qt_a, data_t=qt_a_t),
            QuantizedTensorPair(data=qt_b, data_t=qt_b_t),
            False, True, dtype, config,
        )
        c.backward(torch.ones_like(c))
        torch.cuda.synchronize()

        snr_threshold = 10
        c_snr = compute_snr(c_ref, c)
        a_grad_snr = compute_snr(a_ref.grad, qt_a.grad)
        b_grad_snr = compute_snr(b_ref.grad, qt_b.grad)
        print(
            f"\n[QT-MXFP4-AITER-pre] M={m} N={n} K={k}: "
            f"C-SNR={c_snr:.2f} dB, AGrad-SNR={a_grad_snr:.2f} dB, "
            f"BGrad-SNR={b_grad_snr:.2f} dB"
        )
        assert c_snr > snr_threshold, f"c_snr={c_snr:.2f} too low"
        assert a_grad_snr > snr_threshold, f"a_grad_snr={a_grad_snr:.2f} too low"
        assert b_grad_snr > snr_threshold, f"b_grad_snr={b_grad_snr:.2f} too low"
    finally:
        GlobalBackendManager.reset()


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
            a_hp, fp4_dtype, granularity, block_size=block_size, axis=1,
            scaling_recipe=ScalingRecipe(
                use_2d_block=False, use_sr=False, use_rht=False,
                shuffle_scale=False, shuffle_out=False,
            ),
        )
        qb = QuantizedTensor.quantize(
            b_hp, fp4_dtype, granularity, block_size=block_size, axis=1,
            scaling_recipe=ScalingRecipe(
                use_2d_block=True, use_sr=False, use_rht=False,
                shuffle_scale=False, shuffle_out=False,
            ),
        )

        out_no_pre = gemm_fp4_impl(
            qa.qdata, qa.scale_inv, False,
            qb.qdata, qb.scale_inv, True,
            dtype, False,
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
            qa.qdata, shuffled_a_scale, False,
            shuffled_b_data, shuffled_b_scale, True,
            dtype, False,
            granularity=granularity.value,
            default_backend=BackendType.AITER.value,
            preshuffled=True,
        )

        torch.cuda.synchronize()
        assert out_no_pre.shape == out_pre.shape
        # Both paths converge on the same aiter.gemm_a4w4(bpreshuffle=True)
        # consuming the same byte-identical shuffled inputs -> outputs match.
        assert torch.equal(out_no_pre, out_pre), (
            "preshuffled=True must match preshuffled=False bit-for-bit "
            "on the AITER backend"
        )
    finally:
        GlobalBackendManager.reset()


def test_gemm_fp4_impl_hipblaslt_preshuffle_opt_out():
    """User pins HipBLASLt + asks for ``preshuffled=True`` is a logically
    incoherent request (HipBLASLt does not understand the AITER tile
    layout; see ``csrc/kernels/gemm/hipblaslt_gemm.cu``).

    Dispatch Path 1 must raise ``ValueError`` (instead of silently feeding
    preshuffled bytes into hipBLASLt, which would produce garbage). Locks
    the safety contract for the HipBLASLt opt-out.
    """
    from primus_turbo.pytorch.core.low_precision import check_mxfp4_support
    from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import gemm_fp4_impl

    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    GlobalBackendManager.set_gemm_backend(BackendType.HIPBLASLT)
    GlobalBackendManager.set_auto_tune(False)

    try:
        device = "cuda:0"
        m, n, k = 256, 256, 128
        dtype = torch.bfloat16
        fp4_dtype = torch.float4_e2m1fn_x2
        granularity = ScalingGranularity.MX_BLOCKWISE

        a_hp = torch.randn((m, k), dtype=dtype, device=device)
        b_hp = torch.randn((n, k), dtype=dtype, device=device)
        qa = QuantizedTensor.quantize(
            a_hp, fp4_dtype, granularity, block_size=32, axis=1,
            scaling_recipe=ScalingRecipe(
                use_2d_block=False, use_sr=False, use_rht=False,
                shuffle_scale=False, shuffle_out=False,
            ),
        )
        qb = QuantizedTensor.quantize(
            b_hp, fp4_dtype, granularity, block_size=32, axis=1,
            scaling_recipe=ScalingRecipe(
                use_2d_block=True, use_sr=False, use_rht=False,
                shuffle_scale=False, shuffle_out=False,
            ),
        )

        with pytest.raises(ValueError, match="cannot handle"):
            gemm_fp4_impl(
                qa.qdata, qa.scale_inv, False,
                qb.qdata, qb.scale_inv, True,
                dtype, False,
                granularity=granularity.value,
                default_backend=BackendType.HIPBLASLT.value,
                preshuffled=True,
            )
    finally:
        GlobalBackendManager.reset()
