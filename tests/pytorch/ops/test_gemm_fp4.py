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
from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor
from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import enable_preshuffle
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
    # should match the non-QT path bit-for-bit.
    qt_a = QuantizedTensor(
        a,
        fp4_dtype,
        config.granularity,
        block_size=config.block_size,
        keep_trans_cache=True,
        scaling_recipe=ScalingRecipe(
            use_2d_block=False,
            use_sr=False,
            use_rht=False,
            shuffle_scale=enable_preshuffle(),
        ),
        scaling_recipe_for_trans=ScalingRecipe(
            use_2d_block=False,
            use_sr=False,
            use_rht=True,
            shuffle_scale=enable_preshuffle(),
            shuffle_out=enable_preshuffle(),
        ),
    )

    qt_b = QuantizedTensor(
        b,
        fp4_dtype,
        config.granularity,
        block_size=config.block_size,
        keep_trans_cache=True,
        scaling_recipe=ScalingRecipe(
            use_2d_block=False,
            use_sr=False,
            use_rht=False,
            shuffle_scale=enable_preshuffle(),
        ),
        scaling_recipe_for_trans=ScalingRecipe(
            use_2d_block=False,
            use_sr=False,
            use_rht=True,
            shuffle_scale=enable_preshuffle(),
            shuffle_out=enable_preshuffle(),
        ),
    )

    c = gemm_fp4(qt_a, qt_b, trans_a, trans_b, dtype, config)
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
    """MX_BLOCKWISE gemm_fp4 with pre-quantized QuantizedTensor inputs."""
    _run_gemm_fp4_mx_quantized_tensor_test(
        m=m,
        n=n,
        k=k,
        layout=layout,
        format=format,
        dtype=dtype,
        backend=backend,
    )
