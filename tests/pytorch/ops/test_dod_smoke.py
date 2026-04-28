###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""DoD smoke regression set (per-checkpoint gate).

A small but representative subset of the four canonical GEMM / FP8 / grouped
test files (~200K parametrized cases each) that finishes in tens of minutes
instead of hours. The file is curated for the auto_optimize DoD checkpoint:
each iteration of the optimizer runs this set as a pre-merge gate.

Backend coverage policy (matches the task brief):
  * Dense GEMM:    HIPBLASLT + HIPKITTEN only (~300 cases)
  * Grouped GEMM:  HIPKITTEN + TRITON only   (~300 cases)

Other backends (CK / TURBO) are exercised only via their dedicated unit tests
(test_gemm_fp8.py, test_grouped_gemm_fp8.py) and are NOT part of the DoD gate.

Important HIPKITTEN-specific constraints honored in this file:
  * Dense BF16/FP8 fwd+bwd both go through the SAME dispatcher with the SAME
    backend choice. The backward pass implies two extra GEMM shapes:
      bwd_a uses logical (M, K, N), bwd_b uses logical (K, N, M).
    Probing the analysis caches at HEAD shows only TWO shapes have all six
    keys in the cache: (4096, 4096, 4096) and (8192, 8192, 8192). Anything
    else used in fwd+bwd raises "HIPKITTEN cannot handle". We therefore split
    the HIPKITTEN dense suite into a forward-only block (8 shapes x 3 layouts)
    and a fwd+bwd block (2 safe shapes x 3 layouts).
  * Grouped BF16 backward also dispatches through HIPKITTEN; only one
    forward whitelist entry, (4096, 4096, 7168) RCR, has a matching
    backward whitelist entry. The other forward shapes are tested
    forward-only.
  * Triton grouped backend SUPPORTED_DTYPES = (fp16, bf16) only — fp32 is
    excluded from the grouped Triton dense suite (it's still covered for
    DENSE GEMM via HIPBLASLT in section 1/2).
  * Numerical comparisons for HIPKITTEN BF16 grouped use SNR (>=35 dB)
    rather than torch.testing.assert_close, matching test_grouped_gemm.py.
    BF16 with K=7168 routinely has a few ten-thousandths of elements with
    |diff| > 1e-2 even when the kernel is correct.

Coverage matrix (totals are pytest-collected counts; some skip on gfx942):

  Section                                              Cases   Notes
  -------------------------------------------------------------------
  DENSE GEMM (target ~300, HIPBLASLT + HIPKITTEN only)
    (1)  HIPBLASLT bf16/fp16/fp32                        72    8 shapes x 3 layouts x 3 dtypes
    (2)  HIPBLASLT bf16/fp16/fp32 deterministic          18    2 shapes x 3 layouts x 3 dtypes
    (3)  HIPBLASLT fp8 tensorwise                        84    7 shapes x 2 layouts x 3 fmts x 2 dtypes
    (4)  HIPBLASLT fp8 tensorwise deterministic          24    2 shapes x 2 layouts x 3 fmts x 2 dtypes
    (5)  HIPBLASLT fp8 mx_blockwise                      36    6 shapes x 3 fmts x 2 dtypes (gfx950)
    (6)  HIPBLASLT fp8 mx_blockwise deterministic        12    2 shapes x 3 fmts x 2 dtypes (gfx950)
    (7a) HIPKITTEN bf16 forward-only                     24    8 cache shapes x 3 layouts (no_grad)
    (7b) HIPKITTEN bf16 fwd+bwd safe                      6    2 safe shapes x 3 layouts
    (8a) HIPKITTEN fp8 tensorwise forward-only           16    8 cache shapes x 2 layouts (FP8 NN/NT only)
    (8b) HIPKITTEN fp8 tensorwise fwd+bwd safe            4    2 safe shapes x 2 layouts
    (9)  HIPKITTEN bf16 deterministic                     6    2 shapes x 3 layouts (repeats=4)
    (10) HIPKITTEN fp8 deterministic                      4    2 shapes x 2 layouts (repeats=4)
    (11) HIPKITTEN bf16 reject paths                      2    unsupported shape + uncached bwd
    (12) HIPKITTEN fp8 reject path                        1    rowwise rejected (tensorwise-only)
    Subtotal                                            309

  GROUPED GEMM (target ~300, HIPKITTEN + TRITON only)
    (13) TRITON bf16/fp16 dense grouped                  64    8 shapes x 2 dtypes x 2 trans_b x 2 balance
    (14) TRITON bf16/fp16 dense grouped deterministic     8    2 shapes x 2 dtypes x 2 trans_b
    (15) TRITON fp8 tensorwise                           64    4 shapes x 2 fmts x 2 dtypes x 2 trans_b x 2 balance
    (16) TRITON fp8 rowwise                              64    4 shapes x 2 fmts x 2 dtypes x 2 trans_b x 2 balance
    (17) TRITON fp8 blockwise                            24    3 shapes x 2 fmts x 2 dtypes x 2 balance (NT only)
    (18) TRITON fp8 tensorwise deterministic             16    2 shapes x 2 fmts x 2 dtypes x 2 trans_b
    (19) TRITON fp8 rowwise deterministic                16    2 shapes x 2 fmts x 2 dtypes x 2 trans_b
    (20) TRITON fp8 blockwise deterministic               8    2 shapes x 2 fmts x 2 dtypes (NT only)
    (21a) HIPKITTEN bf16 forward-only (whitelist)        12    4 (shape,layout) configs x 3 B values
    (21b) HIPKITTEN bf16 fwd+bwd (only safe shape)        3    (4096,4096,7168) RCR x B=[1,2,4]
    (22a) HIPKITTEN fp8 tensorwise forward-only          16    8 cache shapes x 2 layouts
    (22b) HIPKITTEN fp8 tensorwise fwd+bwd                2    2 safe shapes (4096,_,_) RCR
    (23) TRITON bf16 zero-length-group regression         1    MoE bug
    (24) TRITON fp8 blockwise zero-length-group           1    MoE bwd bug
    (25) HIPKITTEN bf16 grouped reject                    1    unsupported shape
    (26) HIPKITTEN fp8 grouped reject                     1    rowwise rejected (tensorwise-only)
    Subtotal                                            301

  TOTAL                                                 610
"""

import pytest
import torch

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    check_mxfp8_support,
)
from primus_turbo.pytorch.kernels import hipkitten as _hipkitten
from primus_turbo.pytorch.ops import grouped_gemm, grouped_gemm_fp8

# Reuse the well-tested driver helpers from the canonical FP8 test files.
# Importing private helpers is intentional: the smoke set is a strict subset
# of the same logic, so any helper change is exercised here too.
from tests.pytorch.ops.test_gemm_fp8 import (
    _run_gemm_fp8_deterministic_test,
    _run_gemm_fp8_test,
)
from tests.pytorch.ops.test_grouped_gemm_fp8 import (
    _run_grouped_gemm_fp8_deterministic_test,
    _run_grouped_gemm_fp8_test,
)
from tests.pytorch.ref.gemm_ref import (
    generate_grouped_gemm_group_lens,
    grouped_gemm_ref,
)
from tests.pytorch.test_utils import compute_snr, get_tolerances


# ---------------------------------------------------------------- constants ---
# HipKittens BF16 / FP8 dense FORWARD shape pool. Every entry has rcr/rrr/crr
# autotuned in the cache, so all three test layouts (NN/NT/TN) work for the
# forward call. These are tested forward-only.
_HIPKITTEN_DENSE_FWD_SHAPES = [
    (4096, 4096, 4096),
    (4096, 4096, 11008),
    (4096, 6144, 4096),
    (4096, 8192, 8192),
    (4096, 12288, 4096),
    (4096, 22016, 4096),
    (8192, 4096, 4096),
    (8192, 8192, 8192),
]

# HipKittens BF16 / FP8 dense FWD+BWD-safe shape pool. The dense autograd path
# dispatches two extra GEMMs whose logical (m,n,k) are (M,K,N) and (K,N,M);
# only shapes for which BOTH (M,K,N) and (K,N,M) ALSO have entries in the
# autotune cache survive a fwd+bwd test. Probed against
#   HipKittens/analysis/bf16_gemm/mi350x/bench_bf16_no_jit_final.json
#   HipKittens/analysis/fp8_gemm/mi350x/.autotune_cache.json
# both reduce to the two square shapes below.
_HIPKITTEN_DENSE_SAFE_SHAPES = [
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]

# Single safe shape for the deterministic suite (4-repeat fwd+bwd is heavy).
_HIPKITTEN_DENSE_DET_SHAPES = [(4096, 4096, 4096), (8192, 8192, 8192)]

# HipKittens BF16 grouped forward whitelist (see _grouped_bf16_supported in
# primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_impl.py). The CRR
# entries are reserved for the variable-K backward path and are not exposed
# in the forward grouped_gemm() entry point.
_HIPKITTEN_GROUPED_BF16_FORWARD = [
    # (M, N, K, trans_b)  --  M is per-group rows; trans_b=True => RCR layout.
    (4096, 4096, 7168, True),  # rcr (DeepSeek GateUP)
    (4096, 2048, 7168, False),  # rrr (DeepSeek Down dA)
    (4096, 4096, 7168, False),  # rrr (DeepSeek dA RRR variant)
    (4096, 7168, 4096, False),  # rrr (DeepSeek GateUP dA)
]

# HipKittens FP8 grouped forward (cache shared with FP8 dense). Pick 8 shapes
# from the FP8 dense pool (the kernel pads to 256/256/128 internally, so any
# shape in the FP8 dense cache is fine). Forward path uses RRR / RCR.
_HIPKITTEN_GROUPED_FP8_SHAPES = [
    (4096, 4096, 4096),
    (4096, 4096, 11008),
    (4096, 4096, 14336),
    (4096, 6144, 4096),
    (4096, 8192, 8192),
    (4096, 10240, 8192),
    (4096, 12288, 4096),
    (4096, 22016, 4096),
]

# HipKittens FP8 grouped fwd+bwd-safe shapes: backward dB goes through the
# variable-K HipKitten backend whose can_handle whitelist (crr_4096_7168_4096,
# crr_7168_2048_4096) constrains us to forward shapes RCR (4096,4096,7168)
# and RCR (4096,7168,2048).
_HIPKITTEN_GROUPED_FP8_FWDBWD_SHAPES = [
    (4096, 4096, 7168),
    (4096, 7168, 2048),
]

# Generic small shapes for HIPBLASLT-only paths. Two FP8-divisibility classes:
# all 16-aligned for tensorwise, all 32-aligned for MX_BLOCKWISE (block_size=32).
_HBL_DENSE_SHAPES = [
    (256, 256, 256),
    (256, 512, 384),
    (512, 256, 512),
    (512, 512, 512),
    (768, 1024, 256),
    (1024, 256, 512),
    (1024, 512, 384),
    (768, 768, 512),
]
_HBL_DET_SHAPES = [(256, 256, 256), (512, 512, 512)]

_HBL_FP8_TENSORWISE_SHAPES = [
    (256, 256, 256),
    (256, 512, 384),
    (512, 256, 512),
    (512, 512, 512),
    (768, 1024, 256),
    (1024, 512, 512),
    (768, 768, 384),
]
_HBL_FP8_TENSORWISE_DET_SHAPES = [(256, 256, 256), (512, 512, 512)]

_HBL_FP8_MX_SHAPES = [
    (384, 384, 384),
    (384, 512, 384),
    (512, 384, 384),
    (512, 512, 384),
    (384, 384, 512),
    (512, 384, 512),
]
_HBL_FP8_MX_DET_SHAPES = [(384, 384, 384), (512, 512, 384)]

_DENSE_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# Triton grouped backend SUPPORTED_DTYPES = (fp16, bf16) only — no fp32.
_GROUPED_DENSE_DTYPES = [torch.float16, torch.bfloat16]
_FP8_OUT_DTYPES = [torch.bfloat16, torch.float16]
_FP8_FORMATS = [Format.E4M3, Format.E5M2, Format.HYBRID]
_LAYOUTS_3 = ["NN", "NT", "TN"]
_LAYOUTS_2 = ["NN", "NT"]


# ----------------------------------------------------------------- helpers ---
def _skip_if_no_hipkitten() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        _hipkitten.load_bf16()
    except ImportError as exc:
        pytest.skip(str(exc))


def _skip_if_no_hipkitten_fp8() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        _hipkitten.load_fp8()
    except ImportError as exc:
        pytest.skip(str(exc))


def _run_dense_gemm(
    m: int,
    n: int,
    k: int,
    layout: str,
    dtype: torch.dtype,
    backend: BackendType | None,
) -> None:
    """fwd + bwd correctness vs torch matmul, with explicit backend pinning."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    GlobalBackendManager.set_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(False)
    try:
        trans_a = layout[0] == "T"
        trans_b = layout[1] == "T"
        device = "cuda"
        torch.manual_seed(42)

        a_shape = (m, k) if not trans_a else (k, m)
        b_shape = (k, n) if not trans_b else (n, k)
        a = torch.randn(a_shape, dtype=dtype, device=device)
        b = torch.randn(b_shape, dtype=dtype, device=device)
        a = (a / a.abs().max()).requires_grad_()
        b = (b / b.abs().max()).requires_grad_()
        a_ref = a.detach().clone().requires_grad_()
        b_ref = b.detach().clone().requires_grad_()

        c_ref = (a_ref.T if trans_a else a_ref) @ (b_ref.T if trans_b else b_ref)
        c = turbo.ops.gemm(a, b, trans_a, trans_b, dtype)
        torch.testing.assert_close(c, c_ref, **get_tolerances(dtype))

        grad = torch.randn_like(c)
        c_ref.backward(grad)
        c.backward(grad)
        torch.testing.assert_close(a.grad, a_ref.grad, **get_tolerances(dtype))
        torch.testing.assert_close(b.grad, b_ref.grad, **get_tolerances(dtype))
    finally:
        GlobalBackendManager.reset()


def _run_dense_gemm_deterministic(
    m: int,
    n: int,
    k: int,
    layout: str,
    dtype: torch.dtype,
    backend: BackendType,
    repeats: int = 4,
) -> None:
    """Bit-exact determinism + correctness vs torch matmul (lightweight)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    GlobalBackendManager.set_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(False)
    try:
        trans_a = layout[0] == "T"
        trans_b = layout[1] == "T"
        device = "cuda"
        torch.manual_seed(42)

        a_shape = (m, k) if not trans_a else (k, m)
        b_shape = (k, n) if not trans_b else (n, k)
        a0 = torch.randn(a_shape, dtype=dtype, device=device)
        b0 = torch.randn(b_shape, dtype=dtype, device=device)
        a0 = a0 / a0.abs().max()
        b0 = b0 / b0.abs().max()

        a_ref = a0.detach().clone().requires_grad_()
        b_ref = b0.detach().clone().requires_grad_()
        c_ref = (a_ref.T if trans_a else a_ref) @ (b_ref.T if trans_b else b_ref)
        grad = torch.randn_like(c_ref)
        c_ref.backward(grad)
        torch.cuda.synchronize()

        def _once():
            a = a0.detach().clone().requires_grad_()
            b = b0.detach().clone().requires_grad_()
            c = turbo.ops.gemm(a, b, trans_a, trans_b, dtype)
            c.backward(grad)
            return c.detach(), a.grad.detach(), b.grad.detach()

        outs = [_once() for _ in range(repeats)]
        for i in range(1, repeats):
            torch.testing.assert_close(outs[0][0], outs[i][0], rtol=0, atol=0)
            torch.testing.assert_close(outs[0][1], outs[i][1], rtol=0, atol=0)
            torch.testing.assert_close(outs[0][2], outs[i][2], rtol=0, atol=0)

        torch.testing.assert_close(outs[0][0], c_ref, **get_tolerances(dtype))
        torch.testing.assert_close(outs[0][1], a_ref.grad, **get_tolerances(dtype))
        torch.testing.assert_close(outs[0][2], b_ref.grad, **get_tolerances(dtype))
    finally:
        GlobalBackendManager.reset()


def _run_grouped_dense(
    B: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    trans_b: bool,
    balance: bool,
    backend: BackendType | None,
) -> None:
    """Minimal fwd+bwd correctness for grouped_gemm with explicit backend."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    GlobalBackendManager.set_grouped_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(False)
    try:
        device = "cuda"
        torch.manual_seed(42)
        group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance).to(device)
        b_shape = (B, N, K) if trans_b else (B, K, N)
        a = torch.randn((B * M, K), dtype=torch.float32, device=device).to(dtype).requires_grad_(True)
        b = torch.randn(b_shape, dtype=torch.float32, device=device).to(dtype).requires_grad_(True)
        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)

        out = grouped_gemm(a, b, group_lens, trans_b=trans_b)
        out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens.clone(), trans_b)
        torch.testing.assert_close(out, out_ref, **get_tolerances(dtype))

        grad = torch.randn_like(out_ref)
        out_ref.backward(grad)
        out.backward(grad)
        snr_threshold = 45 if dtype == torch.bfloat16 else 50
        assert compute_snr(out_ref, out) > snr_threshold
        assert compute_snr(a_ref.grad, a.grad) > snr_threshold
        assert compute_snr(b_ref.grad, b.grad) > snr_threshold
    finally:
        GlobalBackendManager.reset()


def _run_grouped_dense_deterministic(
    B: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    trans_b: bool,
    backend: BackendType,
    repeats: int = 4,
) -> None:
    """Bit-exact determinism for grouped_gemm + correctness vs ref."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    GlobalBackendManager.set_grouped_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(False)
    try:
        device = "cuda"
        torch.manual_seed(42)
        group_lens = generate_grouped_gemm_group_lens(B, M, balance=True).to(device)
        b_shape = (B, N, K) if trans_b else (B, K, N)
        a0 = torch.randn((B * M, K), dtype=torch.float32, device=device).to(dtype)
        b0 = torch.randn(b_shape, dtype=torch.float32, device=device).to(dtype)

        a_ref = a0.detach().clone().requires_grad_(True)
        b_ref = b0.detach().clone().requires_grad_(True)
        out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens.clone(), trans_b)
        grad = torch.randn_like(out_ref)
        out_ref.backward(grad)
        torch.cuda.synchronize()

        def _once():
            a = a0.detach().clone().requires_grad_(True)
            b = b0.detach().clone().requires_grad_(True)
            out = grouped_gemm(a, b, group_lens.clone(), trans_b=trans_b)
            out.backward(grad)
            return out.detach(), a.grad.detach(), b.grad.detach()

        outs = [_once() for _ in range(repeats)]
        for i in range(1, repeats):
            torch.testing.assert_close(outs[0][0], outs[i][0], rtol=0, atol=0)
            torch.testing.assert_close(outs[0][1], outs[i][1], rtol=0, atol=0)
            torch.testing.assert_close(outs[0][2], outs[i][2], rtol=0, atol=0)

        snr_threshold = 45 if dtype == torch.bfloat16 else 50
        assert compute_snr(out_ref, outs[0][0]) > snr_threshold
        assert compute_snr(a_ref.grad, outs[0][1]) > snr_threshold
        assert compute_snr(b_ref.grad, outs[0][2]) > snr_threshold
    finally:
        GlobalBackendManager.reset()


# =============================================================================
# DENSE GEMM (target ~300 cases, HIPBLASLT + HIPKITTEN only)
# =============================================================================

# (1) HIPBLASLT bf16 / fp16 / fp32 — 6 x 3 x 3 = 54
@pytest.mark.parametrize("dtype", _DENSE_DTYPES)
@pytest.mark.parametrize("layout", _LAYOUTS_3)
@pytest.mark.parametrize("m, n, k", _HBL_DENSE_SHAPES)
def test_smoke_gemm_hipblaslt(m, n, k, layout, dtype):
    _run_dense_gemm(m, n, k, layout, dtype, BackendType.HIPBLASLT)


# (2) HIPBLASLT bf16 / fp16 / fp32 deterministic — 2 x 3 x 3 = 18
@pytest.mark.parametrize("dtype", _DENSE_DTYPES)
@pytest.mark.parametrize("layout", _LAYOUTS_3)
@pytest.mark.parametrize("m, n, k", _HBL_DET_SHAPES)
def test_smoke_gemm_hipblaslt_deterministic(m, n, k, layout, dtype):
    _run_dense_gemm_deterministic(m, n, k, layout, dtype, BackendType.HIPBLASLT)


# (3) HIPBLASLT FP8 tensorwise — 6 x 2 x 3 x 2 = 72
@pytest.mark.parametrize("dtype", _FP8_OUT_DTYPES)
@pytest.mark.parametrize("format", _FP8_FORMATS)
@pytest.mark.parametrize("layout", _LAYOUTS_2)
@pytest.mark.parametrize("m, n, k", _HBL_FP8_TENSORWISE_SHAPES)
def test_smoke_gemm_fp8_tensorwise_hipblaslt(m, n, k, layout, format, dtype):
    _run_gemm_fp8_test(
        m=m,
        n=n,
        k=k,
        layout=layout,
        format=format,
        dtype=dtype,
        granularity=ScalingGranularity.TENSORWISE,
        backend=BackendType.HIPBLASLT,
        auto_tune=False,
    )


# (4) HIPBLASLT FP8 tensorwise deterministic — 2 x 2 x 3 x 2 = 24
@pytest.mark.parametrize("dtype", _FP8_OUT_DTYPES)
@pytest.mark.parametrize("format", _FP8_FORMATS)
@pytest.mark.parametrize("layout", _LAYOUTS_2)
@pytest.mark.parametrize("m, n, k", _HBL_FP8_TENSORWISE_DET_SHAPES)
def test_smoke_gemm_fp8_tensorwise_hipblaslt_deterministic(m, n, k, layout, format, dtype):
    _run_gemm_fp8_deterministic_test(
        m=m,
        n=n,
        k=k,
        layout=layout,
        format=format,
        dtype=dtype,
        granularity=ScalingGranularity.TENSORWISE,
        backend=BackendType.HIPBLASLT,
        repeats=4,
    )


# (5) HIPBLASLT FP8 mx_blockwise — 4 x 3 x 2 = 24 (skipped on non-gfx950)
@pytest.mark.parametrize("dtype", _FP8_OUT_DTYPES)
@pytest.mark.parametrize("format", _FP8_FORMATS)
@pytest.mark.parametrize("m, n, k", _HBL_FP8_MX_SHAPES)
def test_smoke_gemm_fp8_mx_blockwise_hipblaslt(m, n, k, format, dtype):
    mxfp8_supported, reason = check_mxfp8_support()
    if not mxfp8_supported:
        pytest.skip(reason)
    _run_gemm_fp8_test(
        m=m,
        n=n,
        k=k,
        layout="NT",
        format=format,
        dtype=dtype,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        backend=BackendType.HIPBLASLT,
        auto_tune=False,
        block_size=32,
    )


# (6) HIPBLASLT FP8 mx_blockwise deterministic — 2 x 3 x 2 = 12
@pytest.mark.parametrize("dtype", _FP8_OUT_DTYPES)
@pytest.mark.parametrize("format", _FP8_FORMATS)
@pytest.mark.parametrize("m, n, k", _HBL_FP8_MX_DET_SHAPES)
def test_smoke_gemm_fp8_mx_blockwise_hipblaslt_deterministic(m, n, k, format, dtype):
    mxfp8_supported, reason = check_mxfp8_support()
    if not mxfp8_supported:
        pytest.skip(reason)
    _run_gemm_fp8_deterministic_test(
        m=m,
        n=n,
        k=k,
        layout="NT",
        format=format,
        dtype=dtype,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        backend=BackendType.HIPBLASLT,
        repeats=4,
        block_size=32,
    )


# (7a) HIPKITTEN bf16 forward-only cache-hit — 8 x 3 = 24
# All shapes here have rcr/rrr/crr autotuned, but the implied bwd shapes
# (M,K,N) and (K,N,M) are NOT in the cache for most of them, so we only
# exercise the forward path; bwd correctness is covered in 7b.
@pytest.mark.parametrize("layout", _LAYOUTS_3)
@pytest.mark.parametrize("m, n, k", _HIPKITTEN_DENSE_FWD_SHAPES)
def test_smoke_gemm_hipkitten_bf16_forward(m, n, k, layout):
    _skip_if_no_hipkitten()
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    GlobalBackendManager.set_gemm_backend(BackendType.HIPKITTEN)
    GlobalBackendManager.set_auto_tune(False)
    try:
        trans_a = layout[0] == "T"
        trans_b = layout[1] == "T"
        device = "cuda"
        torch.manual_seed(42)
        a_shape = (m, k) if not trans_a else (k, m)
        b_shape = (k, n) if not trans_b else (n, k)
        a = torch.randn(a_shape, dtype=torch.bfloat16, device=device)
        b = torch.randn(b_shape, dtype=torch.bfloat16, device=device)
        a = a / a.abs().max()
        b = b / b.abs().max()
        with torch.no_grad():
            c = turbo.ops.gemm(a, b, trans_a, trans_b, torch.bfloat16)
            c_ref = (a.T if trans_a else a) @ (b.T if trans_b else b)
        assert compute_snr(c_ref, c) > 35
    finally:
        GlobalBackendManager.reset()


# (7b) HIPKITTEN bf16 fwd+bwd safe — 2 x 3 = 6
@pytest.mark.parametrize("layout", _LAYOUTS_3)
@pytest.mark.parametrize("m, n, k", _HIPKITTEN_DENSE_SAFE_SHAPES)
def test_smoke_gemm_hipkitten_bf16_fwdbwd(m, n, k, layout):
    _skip_if_no_hipkitten()
    _run_dense_gemm(m, n, k, layout, torch.bfloat16, BackendType.HIPKITTEN)


# (8a) HIPKITTEN fp8 tensorwise forward-only — 8 x 2 = 16
# FP8 GEMM only supports trans_a=False (NN / NT layouts).
@pytest.mark.parametrize("layout", _LAYOUTS_2)
@pytest.mark.parametrize("m, n, k", _HIPKITTEN_DENSE_FWD_SHAPES)
def test_smoke_gemm_hipkitten_fp8_tensorwise_forward(m, n, k, layout):
    _skip_if_no_hipkitten_fp8()
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    GlobalBackendManager.set_gemm_backend(BackendType.HIPKITTEN)
    GlobalBackendManager.set_auto_tune(False)
    try:
        trans_a = layout[0] == "T"
        trans_b = layout[1] == "T"
        device = "cuda"
        torch.manual_seed(42)
        a_shape = (m, k) if not trans_a else (k, m)
        b_shape = (k, n) if not trans_b else (n, k)
        a = torch.randn(a_shape, dtype=torch.bfloat16, device=device, requires_grad=False)
        b = torch.randn(b_shape, dtype=torch.bfloat16, device=device, requires_grad=False)
        a = a / a.abs().max()
        b = b / b.abs().max()
        config = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
        with torch.no_grad():
            c = turbo.ops.gemm_fp8(a, b, trans_a, trans_b, torch.bfloat16, config)
            c_ref = (a.T if trans_a else a) @ (b.T if trans_b else b)
        assert compute_snr(c_ref.to(torch.bfloat16), c) > 25
    finally:
        GlobalBackendManager.reset()


# (8b) HIPKITTEN fp8 tensorwise fwd+bwd safe — 2 x 2 = 4
@pytest.mark.parametrize("layout", _LAYOUTS_2)
@pytest.mark.parametrize("m, n, k", _HIPKITTEN_DENSE_SAFE_SHAPES)
def test_smoke_gemm_hipkitten_fp8_tensorwise_fwdbwd(m, n, k, layout):
    _skip_if_no_hipkitten_fp8()
    _run_gemm_fp8_test(
        m=m,
        n=n,
        k=k,
        layout=layout,
        format=Format.E4M3,
        dtype=torch.bfloat16,
        granularity=ScalingGranularity.TENSORWISE,
        backend=BackendType.HIPKITTEN,
        auto_tune=False,
    )


# (9) HIPKITTEN bf16 deterministic — 2 x 3 = 6
@pytest.mark.parametrize("layout", _LAYOUTS_3)
@pytest.mark.parametrize("m, n, k", _HIPKITTEN_DENSE_DET_SHAPES)
def test_smoke_gemm_hipkitten_bf16_deterministic(m, n, k, layout):
    _skip_if_no_hipkitten()
    _run_dense_gemm_deterministic(m, n, k, layout, torch.bfloat16, BackendType.HIPKITTEN, repeats=4)


# (10) HIPKITTEN fp8 deterministic — 2 x 2 = 4
@pytest.mark.parametrize("layout", _LAYOUTS_2)
@pytest.mark.parametrize("m, n, k", _HIPKITTEN_DENSE_DET_SHAPES)
def test_smoke_gemm_hipkitten_fp8_tensorwise_deterministic(m, n, k, layout):
    _skip_if_no_hipkitten_fp8()
    _run_gemm_fp8_deterministic_test(
        m=m,
        n=n,
        k=k,
        layout=layout,
        format=Format.E4M3,
        dtype=torch.bfloat16,
        granularity=ScalingGranularity.TENSORWISE,
        backend=BackendType.HIPKITTEN,
        repeats=4,
    )


# (11) HIPKITTEN bf16 reject paths — 2
def test_smoke_gemm_hipkitten_rejects_unsupported_shape():
    """Shape that is 256-tile aligned but not in the cache must error out."""
    _skip_if_no_hipkitten()
    GlobalBackendManager.set_gemm_backend(BackendType.HIPKITTEN)
    GlobalBackendManager.set_auto_tune(False)
    try:
        a = torch.randn((256, 256), dtype=torch.bfloat16, device="cuda")
        b = torch.randn((384, 256), dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match="HIPKITTEN cannot handle"):
            turbo.ops.gemm(a, b, trans_b=True)
    finally:
        GlobalBackendManager.reset()


def test_smoke_gemm_hipkitten_rejects_uncached_backward_shape():
    """Forward shape is in cache but the implied backward shape is not."""
    _skip_if_no_hipkitten()
    GlobalBackendManager.set_gemm_backend(BackendType.HIPKITTEN)
    GlobalBackendManager.set_auto_tune(False)
    try:
        a = torch.randn((4096, 12288), dtype=torch.bfloat16, device="cuda")
        b = torch.randn((12288, 4096), dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match="HIPKITTEN cannot handle"):
            turbo.ops.gemm(a, b)
    finally:
        GlobalBackendManager.reset()


# (12) HIPKITTEN fp8 reject path (rowwise) — 1
def test_smoke_gemm_hipkitten_fp8_rejects_rowwise():
    """HIPKITTEN FP8 only supports TENSORWISE; ROWWISE must be rejected."""
    _skip_if_no_hipkitten_fp8()
    GlobalBackendManager.set_gemm_backend(BackendType.HIPKITTEN)
    GlobalBackendManager.set_auto_tune(False)
    try:
        a = torch.randn((4096, 4096), dtype=torch.bfloat16, device="cuda", requires_grad=True)
        b = torch.randn((4096, 4096), dtype=torch.bfloat16, device="cuda", requires_grad=True)
        config = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.ROWWISE)
        with pytest.raises(ValueError, match="HIPKITTEN cannot handle"):
            turbo.ops.gemm_fp8(a, b, False, True, torch.bfloat16, config)
    finally:
        GlobalBackendManager.reset()


# =============================================================================
# GROUPED GEMM (target ~300 cases, HIPKITTEN + TRITON only)
# =============================================================================

_GROUPED_TRITON_DENSE_SHAPES = [
    # (B, M, N, K)
    (1, 256, 256, 256),
    (2, 256, 512, 384),
    (2, 512, 256, 512),
    (4, 256, 1024, 256),
    (4, 512, 512, 384),
    (8, 256, 256, 256),
    (4, 768, 256, 384),
    (2, 1024, 256, 512),
]
_GROUPED_TRITON_DET_SHAPES = [
    (2, 256, 512, 384),
    (4, 512, 256, 512),
]
_GROUPED_TRITON_FP8_SHAPES = [
    (1, 256, 256, 256),
    (2, 256, 512, 384),
    (2, 512, 512, 384),
    (4, 256, 1024, 256),
]
_GROUPED_TRITON_FP8_DET_SHAPES = [
    (2, 256, 512, 384),
    (4, 256, 1024, 256),
]
_GROUPED_TRITON_FP8_BLOCKWISE_SHAPES = [
    (1, 256, 256, 256),
    (2, 256, 512, 384),
    (4, 256, 1024, 256),
]
_GROUPED_TRITON_FP8_BLOCKWISE_DET_SHAPES = [
    (2, 256, 512, 384),
    (4, 256, 1024, 256),
]

# TRITON FP8 supports only the COMMON dtype set (no hybrid).
_GROUPED_TRITON_FP8_FORMATS = [Format.E4M3, Format.E5M2]


# (13) TRITON dense bf16/fp16 — 8 x 2 x 2 x 2 = 64
# fp32 is intentionally excluded: GroupedGEMMTritonBackend's SUPPORTED_DTYPES
# is (fp16, bf16); attempting fp32 hits the "TRITON cannot handle" path.
@pytest.mark.parametrize("balance", [True, False])
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("dtype", _GROUPED_DENSE_DTYPES)
@pytest.mark.parametrize("B, M, N, K", _GROUPED_TRITON_DENSE_SHAPES)
def test_smoke_grouped_gemm_triton(B, M, N, K, dtype, trans_b, balance):
    _run_grouped_dense(B, M, N, K, dtype, trans_b, balance, BackendType.TRITON)


# (14) TRITON dense bf16/fp16 deterministic — 2 x 2 x 2 = 8
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("B, M, N, K", _GROUPED_TRITON_DET_SHAPES)
def test_smoke_grouped_gemm_triton_deterministic(B, M, N, K, dtype, trans_b):
    _run_grouped_dense_deterministic(B, M, N, K, dtype, trans_b, BackendType.TRITON, repeats=4)


# (15) TRITON FP8 tensorwise — 4 x 4 x 2 x 2 = 64
@pytest.mark.parametrize("balance", [True, False])
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("dtype", _FP8_OUT_DTYPES)
@pytest.mark.parametrize("format", _GROUPED_TRITON_FP8_FORMATS)
@pytest.mark.parametrize("B, M, N, K", _GROUPED_TRITON_FP8_SHAPES)
def test_smoke_grouped_gemm_fp8_tensorwise_triton(B, M, N, K, format, dtype, trans_b, balance):
    _run_grouped_gemm_fp8_test(
        B=B,
        M=M,
        N=N,
        K=K,
        ori_dtype=dtype,
        format=format,
        granularity=ScalingGranularity.TENSORWISE,
        trans_b=trans_b,
        balance=balance,
        backend=BackendType.TRITON,
        auto_tune=False,
    )


# (16) TRITON FP8 rowwise — 4 x 4 x 2 x 2 = 64
@pytest.mark.parametrize("balance", [True, False])
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("dtype", _FP8_OUT_DTYPES)
@pytest.mark.parametrize("format", _GROUPED_TRITON_FP8_FORMATS)
@pytest.mark.parametrize("B, M, N, K", _GROUPED_TRITON_FP8_SHAPES)
def test_smoke_grouped_gemm_fp8_rowwise_triton(B, M, N, K, format, dtype, trans_b, balance):
    _run_grouped_gemm_fp8_test(
        B=B,
        M=M,
        N=N,
        K=K,
        ori_dtype=dtype,
        format=format,
        granularity=ScalingGranularity.ROWWISE,
        trans_b=trans_b,
        balance=balance,
        backend=BackendType.TRITON,
        auto_tune=False,
    )


# (17) TRITON FP8 blockwise (NT only) — 3 x 4 x 2 = 24
@pytest.mark.parametrize("balance", [True, False])
@pytest.mark.parametrize("dtype", _FP8_OUT_DTYPES)
@pytest.mark.parametrize("format", _GROUPED_TRITON_FP8_FORMATS)
@pytest.mark.parametrize("B, M, N, K", _GROUPED_TRITON_FP8_BLOCKWISE_SHAPES)
def test_smoke_grouped_gemm_fp8_blockwise_triton(B, M, N, K, format, dtype, balance):
    _run_grouped_gemm_fp8_test(
        B=B,
        M=M,
        N=N,
        K=K,
        ori_dtype=dtype,
        format=format,
        granularity=ScalingGranularity.BLOCKWISE,
        trans_b=True,
        balance=balance,
        backend=BackendType.TRITON,
        auto_tune=False,
        block_size=128,
    )


# (18) TRITON FP8 tensorwise deterministic — 2 x 4 x 2 = 16
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("dtype", _FP8_OUT_DTYPES)
@pytest.mark.parametrize("format", _GROUPED_TRITON_FP8_FORMATS)
@pytest.mark.parametrize("B, M, N, K", _GROUPED_TRITON_FP8_DET_SHAPES)
def test_smoke_grouped_gemm_fp8_tensorwise_triton_deterministic(B, M, N, K, format, dtype, trans_b):
    _run_grouped_gemm_fp8_deterministic_test(
        B=B,
        M=M,
        N=N,
        K=K,
        ori_dtype=dtype,
        format=format,
        granularity=ScalingGranularity.TENSORWISE,
        trans_b=trans_b,
        balance=False,
        backend=BackendType.TRITON,
        repeats=4,
    )


# (19) TRITON FP8 rowwise deterministic — 2 x 4 x 2 = 16
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("dtype", _FP8_OUT_DTYPES)
@pytest.mark.parametrize("format", _GROUPED_TRITON_FP8_FORMATS)
@pytest.mark.parametrize("B, M, N, K", _GROUPED_TRITON_FP8_DET_SHAPES)
def test_smoke_grouped_gemm_fp8_rowwise_triton_deterministic(B, M, N, K, format, dtype, trans_b):
    _run_grouped_gemm_fp8_deterministic_test(
        B=B,
        M=M,
        N=N,
        K=K,
        ori_dtype=dtype,
        format=format,
        granularity=ScalingGranularity.ROWWISE,
        trans_b=trans_b,
        balance=False,
        backend=BackendType.TRITON,
        repeats=4,
    )


# (20) TRITON FP8 blockwise deterministic (NT) — 2 x 4 = 8
@pytest.mark.parametrize("dtype", _FP8_OUT_DTYPES)
@pytest.mark.parametrize("format", _GROUPED_TRITON_FP8_FORMATS)
@pytest.mark.parametrize("B, M, N, K", _GROUPED_TRITON_FP8_BLOCKWISE_DET_SHAPES)
def test_smoke_grouped_gemm_fp8_blockwise_triton_deterministic(B, M, N, K, format, dtype):
    _run_grouped_gemm_fp8_deterministic_test(
        B=B,
        M=M,
        N=N,
        K=K,
        ori_dtype=dtype,
        format=format,
        granularity=ScalingGranularity.BLOCKWISE,
        trans_b=True,
        balance=False,
        backend=BackendType.TRITON,
        repeats=4,
        block_size=128,
    )


# (21a) HIPKITTEN BF16 grouped FORWARD-ONLY (whitelist) — 4 x 3 = 12
# Forward whitelist coverage; backward is intentionally NOT exercised here
# because the grouped HIPKITTEN BF16 backward path only has cache entries
# for one of these (M,N,K) tuples, and a cache miss raises "HIPKITTEN cannot
# handle" at the dispatcher rather than falling back. SNR (>= 35 dB) is used
# instead of assert_close to mirror test_grouped_gemm.py: BF16 with K=7168
# routinely has a few ten-thousandths of elements with |diff| > 1e-2.
@pytest.mark.parametrize("B", [2, 4, 8])
@pytest.mark.parametrize("M, N, K, trans_b", _HIPKITTEN_GROUPED_BF16_FORWARD)
def test_smoke_grouped_gemm_hipkitten_bf16_forward(B, M, N, K, trans_b):
    _skip_if_no_hipkitten()
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.HIPKITTEN)
    GlobalBackendManager.set_auto_tune(False)
    try:
        device = "cuda"
        torch.manual_seed(42)
        group_lens = torch.full((B,), M, dtype=torch.int64, device=device)
        a = torch.randn((B * M, K), dtype=torch.bfloat16, device=device)
        b_shape = (B, N, K) if trans_b else (B, K, N)
        b = torch.randn(b_shape, dtype=torch.bfloat16, device=device)

        with torch.no_grad():
            out = grouped_gemm(a, b, group_lens, trans_b=trans_b)
            out_ref = grouped_gemm_ref(a.clone(), b.clone(), group_lens.clone(), trans_b=trans_b)
        assert compute_snr(out_ref, out) > 35
    finally:
        GlobalBackendManager.reset()


# (21b) HIPKITTEN BF16 grouped FWD+BWD on the single safe shape — 1 x 3 = 3
# Only RCR (4096,4096,7168) has both a forward and a backward HIPKITTEN
# whitelist entry, so this is the only place we can verify the autograd path
# through HIPKITTEN BF16 grouped without a cache miss.
@pytest.mark.parametrize("B", [1, 2, 4])
def test_smoke_grouped_gemm_hipkitten_bf16_fwdbwd(B):
    _skip_if_no_hipkitten()
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.HIPKITTEN)
    GlobalBackendManager.set_auto_tune(False)
    try:
        M, N, K, trans_b = 4096, 4096, 7168, True
        device = "cuda"
        torch.manual_seed(42)
        group_lens = torch.full((B,), M, dtype=torch.int64, device=device)
        a = torch.randn((B * M, K), dtype=torch.bfloat16, device=device, requires_grad=True)
        b_shape = (B, N, K) if trans_b else (B, K, N)
        b = torch.randn(b_shape, dtype=torch.bfloat16, device=device, requires_grad=True)
        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)

        out = grouped_gemm(a, b, group_lens, trans_b=trans_b)
        out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens.clone(), trans_b=trans_b)
        grad = torch.randn_like(out_ref)
        out_ref.backward(grad)
        out.backward(grad)

        assert compute_snr(out_ref, out) > 35
        assert compute_snr(a_ref.grad, a.grad) > 35
        assert compute_snr(b_ref.grad, b.grad) > 35
    finally:
        GlobalBackendManager.reset()


# (22a) HIPKITTEN FP8 tensorwise grouped FORWARD-ONLY — 8 x 2 = 16
# The HIPKITTEN FP8 grouped backend pads to (256, 256, 128) tiles internally,
# so any cache shape works for forward. Backward goes through the variable-K
# CRR backend whose tile/group_m comes from a separate (small) cache; we keep
# fwd+bwd to a dedicated section below to avoid sweeping over kernel configs
# we did not pre-tune.
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("M, N, K", _HIPKITTEN_GROUPED_FP8_SHAPES)
def test_smoke_grouped_gemm_hipkitten_fp8_tensorwise_forward(M, N, K, trans_b):
    _skip_if_no_hipkitten_fp8()
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.HIPKITTEN)
    GlobalBackendManager.set_auto_tune(False)
    try:
        B = 2
        device = "cuda"
        torch.manual_seed(42)
        group_lens = torch.full((B,), M, dtype=torch.int64, device=device)
        a = torch.randn((B * M, K), dtype=torch.bfloat16, device=device)
        b_shape = (B, N, K) if trans_b else (B, K, N)
        b = torch.randn(b_shape, dtype=torch.bfloat16, device=device)
        config = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)

        with torch.no_grad():
            out = grouped_gemm_fp8(a, b, group_lens, trans_b=trans_b, config=config)
            out_ref = grouped_gemm_ref(a.clone(), b.clone(), group_lens.clone(), trans_b=trans_b)
        assert compute_snr(out_ref, out) > 25
    finally:
        GlobalBackendManager.reset()


# (22b) HIPKITTEN FP8 tensorwise grouped FWD+BWD — 2 cases
# Backward dB goes through the variable-K HIPKITTEN backend whose can_handle
# only checks dtypes/dims, so any RCR (4096,_,_) shape from the FP8 grouped
# fwd+bwd whitelist works at runtime.
@pytest.mark.parametrize("M, N, K", _HIPKITTEN_GROUPED_FP8_FWDBWD_SHAPES)
def test_smoke_grouped_gemm_hipkitten_fp8_tensorwise_fwdbwd(M, N, K):
    _skip_if_no_hipkitten_fp8()
    _run_grouped_gemm_fp8_test(
        B=2,
        M=M,
        N=N,
        K=K,
        ori_dtype=torch.bfloat16,
        format=Format.E4M3,
        granularity=ScalingGranularity.TENSORWISE,
        trans_b=True,
        balance=True,
        backend=BackendType.HIPKITTEN,
        auto_tune=False,
    )


# (23) TRITON BF16 zero-length-group regression — 1
def test_smoke_grouped_gemm_zero_group_triton():
    """MoE bug: backward must zero-fill weight grads for empty experts."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.TRITON)
    GlobalBackendManager.set_auto_tune(False)
    try:
        device = "cuda"
        B, M, N, K = 8, 1024, 2048, 1024
        torch.manual_seed(42)
        nz = B - 2
        base = (B * M) // nz
        rem = (B * M) % nz
        group_lens_cpu = torch.zeros(B, dtype=torch.int64)
        group_lens_cpu[:nz] = base
        group_lens_cpu[:rem] += 1
        group_lens = group_lens_cpu.to(device)
        zero_idx = (group_lens_cpu == 0).nonzero(as_tuple=False).flatten().tolist()
        assert zero_idx, "test setup: expected at least one zero-length group"

        a = torch.randn((B * M, K), dtype=torch.bfloat16, device=device, requires_grad=True)
        b = torch.randn((B, N, K), dtype=torch.bfloat16, device=device, requires_grad=True)
        out = grouped_gemm(a, b, group_lens, trans_b=True)
        grad = torch.randn_like(out)
        out.backward(grad)
        for i in zero_idx:
            torch.testing.assert_close(
                b.grad[i],
                torch.zeros_like(b.grad[i]),
                rtol=0.0,
                atol=0.0,
                msg=f"b.grad[{i}] must be zero when group_lens[{i}] == 0",
            )
    finally:
        GlobalBackendManager.reset()


# (24) TRITON FP8 BLOCKWISE zero-length-group regression — 1
def test_smoke_grouped_gemm_fp8_blockwise_zero_group_triton():
    """MoE bwd crash regression for FP8 blockwise grouped GEMM (Triton)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.TRITON)
    GlobalBackendManager.set_auto_tune(False)
    try:
        device = "cuda"
        ori_dtype = torch.bfloat16
        E, K, N = 8, 2048, 2048
        group_lens_list = [2048, 2048, 0, 0, 0, 0, 0, 0]
        group_lens = torch.tensor(group_lens_list, dtype=torch.int64, device=device)
        total_m = sum(group_lens_list)

        torch.manual_seed(42)
        a = torch.randn((total_m, K), dtype=ori_dtype, device=device, requires_grad=True)
        b = torch.randn((E, N, K), dtype=ori_dtype, device=device, requires_grad=True)
        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)

        out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=True)
        grad = torch.randn_like(out_ref)
        out_ref.backward(grad)

        config = Float8QuantConfig(
            format=Format.E4M3, granularity=ScalingGranularity.BLOCKWISE, block_size=128
        )
        out = grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=config)
        out.backward(grad)

        assert out.shape == out_ref.shape
        assert a.grad.shape == a_ref.grad.shape
        assert b.grad.shape == b_ref.grad.shape

        snr_threshold = 25
        assert compute_snr(out_ref, out) > snr_threshold, "out_snr too low"
        assert compute_snr(a_ref.grad, a.grad) > snr_threshold, "a_grad_snr too low"
        assert compute_snr(b_ref.grad, b.grad) > snr_threshold, "b_grad_snr too low"
    finally:
        GlobalBackendManager.reset()


# (25) HIPKITTEN BF16 grouped reject — 1
def test_smoke_grouped_gemm_hipkitten_rejects_unsupported_shape():
    """HIPKITTEN grouped only accepts the small BF16 whitelist; everything else
    must surface as 'HIPKITTEN cannot handle'."""
    _skip_if_no_hipkitten()
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.HIPKITTEN)
    GlobalBackendManager.set_auto_tune(False)
    try:
        B, M, N, K = 2, 512, 4096, 7168
        device = "cuda"
        group_lens = torch.full((B,), M, dtype=torch.int64, device=device)
        a = torch.randn((B * M, K), dtype=torch.bfloat16, device=device)
        b = torch.randn((B, N, K), dtype=torch.bfloat16, device=device)
        with pytest.raises(ValueError, match="HIPKITTEN cannot handle"):
            grouped_gemm(a, b, group_lens, trans_b=True)
    finally:
        GlobalBackendManager.reset()


# (26) HIPKITTEN FP8 grouped reject (rowwise) — 1
def test_smoke_grouped_gemm_hipkitten_fp8_rejects_rowwise():
    """HIPKITTEN grouped FP8 backend only supports TENSORWISE."""
    _skip_if_no_hipkitten_fp8()
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.HIPKITTEN)
    GlobalBackendManager.set_auto_tune(False)
    try:
        B, M, N, K = 2, 4096, 4096, 4096
        device = "cuda"
        group_lens = torch.full((B,), M, dtype=torch.int64, device=device)
        a = torch.randn((B * M, K), dtype=torch.bfloat16, device=device, requires_grad=True)
        b = torch.randn((B, N, K), dtype=torch.bfloat16, device=device, requires_grad=True)
        config = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.ROWWISE)
        with pytest.raises(ValueError, match="HIPKITTEN cannot handle"):
            grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=config)
    finally:
        GlobalBackendManager.reset()
