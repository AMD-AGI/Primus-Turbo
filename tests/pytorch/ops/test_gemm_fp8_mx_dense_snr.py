###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Dense MXFP8 (E4M3, 1x32 E8M0) forward / dgrad / wgrad against BF16, through the backend registry.

A correct MXFP8 GEMM gives ~26-29 dB SNR against BF16 on Gaussian inputs. On gfx1250 the hipBLASLt
MX_BLOCKWISE GEMM gives ~5 dB without raising, so this suite also pins down that hipBLASLt is
never chosen for MXFP8 there and that an MX case no backend can serve raises instead of
returning numbers.
"""

import pytest
import torch

from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    dispatch_report,
    reset_dispatch_report,
)
from primus_turbo.pytorch.core.low_precision import (
    MXFP8_BLOCK_SIZE,
    Float8QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
    float8_e4m3,
)
from primus_turbo.pytorch.core.utils import is_gfx950, is_gfx1250
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import (
    GEMMFP8FlyDSLBackend,
    GEMMFP8HipBLASLtBackend,
    gemm_fp8_impl,
)
from primus_turbo.pytorch.ops import gemm_fp8
from tests.pytorch.test_utils import compute_snr

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not (is_gfx1250() or is_gfx950()),
    reason="dense MXFP8 needs gfx1250 or gfx950",
)

SNR_MIN_DB = 20.0

# (M, N, K): M = tokens, N = out features, K = in features.
DENSE_MX_SHAPES = [
    (1024, 4096, 4096),  # Llama-3.1-8B attention out-proj (reduced M)
    (1024, 14336, 4096),  # Llama-3.1-8B MLP up (reduced M)
    (1024, 4096, 14336),  # Llama-3.1-8B MLP down (reduced M)
    (1024, 5120, 2880),  # gpt-oss QKV: K = hidden 2880, not a multiple of 128
    (1024, 2880, 4096),  # gpt-oss attention out-proj: N = 2880
    (2048, 2880, 2880),  # K = N = 2880
]


def _mx8_config() -> Float8QuantConfig:
    return Float8QuantConfig(
        granularity=ScalingGranularity.MX_BLOCKWISE,
        format=Format.E4M3,
        block_size=MXFP8_BLOCK_SIZE,
        scale_dtype=ScaleDtype.E8M0,
    )


@pytest.fixture(autouse=True)
def _default_backends():
    GlobalBackendManager.reset()
    yield
    GlobalBackendManager.reset()


def _fwd_bwd_vs_bf16(m, n, k, config):
    """gemm_fp8 fwd + bwd against BF16 on identical inputs: ({name: SNR dB}, GEMMFP8 backends used)."""
    torch.manual_seed(1234)
    device = "cuda"
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=device, dtype=torch.bfloat16) / k**0.5
    g = torch.randn(m, n, device=device, dtype=torch.bfloat16)

    a_ref, b_ref = a.clone().requires_grad_(), b.clone().requires_grad_()
    out_ref = a_ref @ b_ref.t()
    out_ref.backward(g)

    a_q, b_q = a.clone().requires_grad_(), b.clone().requires_grad_()
    reset_dispatch_report()
    out_q = gemm_fp8(a_q, b_q, trans_a=False, trans_b=True, out_dtype=torch.bfloat16, config=config)
    out_q.backward(g)

    snrs = {}
    for name, ref, got in (
        ("fwd", out_ref, out_q),
        ("dgrad", a_ref.grad, a_q.grad),
        ("wgrad", b_ref.grad, b_q.grad),
    ):
        assert got.shape == ref.shape, f"{name}: shape {tuple(got.shape)} != {tuple(ref.shape)}"
        assert torch.isfinite(got).all(), f"{name}: non-finite values"
        snrs[name] = compute_snr(ref, got)
    used = {str(row["backend"]) for row in dispatch_report() if "GEMMFP8" in row["dispatcher"]}
    return snrs, used


@pytest.mark.parametrize("m, n, k", DENSE_MX_SHAPES)
def test_dense_mxfp8_fwd_dgrad_wgrad_snr(m, n, k):
    snrs, used = _fwd_bwd_vs_bf16(m, n, k, _mx8_config())
    for name, snr in snrs.items():
        assert snr >= SNR_MIN_DB, f"{name} ({m}x{n}x{k}): SNR {snr:.1f} dB < {SNR_MIN_DB} dB"
    if is_gfx1250():
        assert not any("HIPBLASLT" in u for u in used), f"hipBLASLt ran an MXFP8 GEMM on gfx1250: {used}"
        assert any("FLYDSL" in u for u in used), f"expected the FlyDSL gfx1250 MX backend, got {used}"


@pytest.mark.skipif(not is_gfx1250(), reason="gfx1250-only gate")
@pytest.mark.parametrize("m, n, k", DENSE_MX_SHAPES[:1])
def test_fp8_tensorwise_unchanged_on_gfx1250(m, n, k):
    # The gfx1250 FlyDSL kernel serves MX_BLOCKWISE only; FP8 tensorwise keeps its backend.
    a = torch.zeros(m, k, device="cuda", dtype=torch.uint8).view(float8_e4m3)
    b = torch.zeros(n, k, device="cuda", dtype=torch.uint8).view(float8_e4m3)
    one = torch.ones((), device="cuda", dtype=torch.float32)
    assert not GEMMFP8FlyDSLBackend.can_handle(
        a, one, False, b, one, True, torch.bfloat16, False, ScalingGranularity.TENSORWISE
    )
    snrs, used = _fwd_bwd_vs_bf16(m, n, k, Float8QuantConfig())
    for name, snr in snrs.items():
        assert snr >= SNR_MIN_DB, f"tensorwise {name} ({m}x{n}x{k}): SNR {snr:.1f} dB < {SNR_MIN_DB} dB"
    assert used and not any("FLYDSL" in u for u in used), f"tensorwise FP8 dispatched to {used}"


def _mx_operands(m, n, k, trans_a=False):
    a_shape = (k, m) if trans_a else (m, k)
    a = torch.zeros(a_shape, device="cuda", dtype=torch.uint8).view(float8_e4m3)
    b = torch.zeros(n, k, device="cuda", dtype=torch.uint8).view(float8_e4m3)
    sa = torch.full((m, k // 32), 127, device="cuda", dtype=torch.uint8)
    sb = torch.full((n, k // 32), 127, device="cuda", dtype=torch.uint8)
    return a, sa, b, sb


@pytest.mark.skipif(not is_gfx1250(), reason="gfx1250-only gate")
def test_hipblaslt_never_takes_mxfp8_on_gfx1250():
    a, sa, b, sb = _mx_operands(256, 256, 256)
    kwargs = dict(
        a=a,
        a_scale_inv=sa,
        trans_a=False,
        b=b,
        b_scale_inv=sb,
        trans_b=True,
        out_dtype=torch.bfloat16,
        trans_c=False,
        granularity=ScalingGranularity.MX_BLOCKWISE,
    )
    assert not GEMMFP8HipBLASLtBackend.can_handle(**kwargs)
    out = torch.zeros(256, 256, device="cuda", dtype=torch.float32)
    assert not GEMMFP8HipBLASLtBackend.can_handle(**kwargs, inplace_add_to_out=True, out=out)

    # Pinning hipBLASLt for MXFP8 must fail loudly, not run the wrong kernel.
    GlobalBackendManager.set_gemm_backend(BackendType.HIPBLASLT)
    with pytest.raises(ValueError, match="cannot handle"):
        gemm_fp8_impl(
            a,
            sa,
            False,
            b,
            sb,
            True,
            torch.bfloat16,
            False,
            ScalingGranularity.MX_BLOCKWISE.value,
            BackendType.FLYDSL.value,
        )


@pytest.mark.skipif(not is_gfx1250(), reason="gfx1250-only gate")
def test_config_default_hipblaslt_does_not_route_mxfp8_to_hipblaslt_on_gfx1250():
    """A Float8QuantConfig whose default_backend is hipBLASLt still gets FlyDSL for MXFP8 on gfx1250 (the
    hipBLASLt backend refuses MX there), with the same accuracy as the built-in default."""
    m, n, k = DENSE_MX_SHAPES[0]
    config = Float8QuantConfig(
        granularity=ScalingGranularity.MX_BLOCKWISE,
        format=Format.E4M3,
        block_size=MXFP8_BLOCK_SIZE,
        scale_dtype=ScaleDtype.E8M0,
        default_backend=BackendType.HIPBLASLT,
    )
    snrs, used = _fwd_bwd_vs_bf16(m, n, k, config)
    assert used and not any("HIPBLASLT" in u for u in used), f"hipBLASLt ran an MXFP8 GEMM on gfx1250: {used}"
    assert any("FLYDSL" in u for u in used), used
    for name, snr in snrs.items():
        assert snr >= SNR_MIN_DB, f"{name}: {snr:.1f} dB"


@pytest.mark.skipif(not is_gfx1250(), reason="gfx1250-only gate")
def test_unsupported_mxfp8_case_raises_on_gfx1250():
    # TN layout: the gfx1250 MX kernel is NT only and hipBLASLt is refused, so nothing may run.
    a, sa, b, sb = _mx_operands(256, 256, 256, trans_a=True)
    with pytest.raises(ValueError, match="No compatible backend"):
        gemm_fp8_impl(
            a,
            sa,
            True,
            b,
            sb,
            True,
            torch.bfloat16,
            False,
            ScalingGranularity.MX_BLOCKWISE.value,
            BackendType.FLYDSL.value,
        )
