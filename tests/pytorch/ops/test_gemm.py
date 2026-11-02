###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from tests.pytorch.ops.determinism_helper import (  # noqa: F401  (_det_seed: fixture)
    ACCUM_DENSE_SHAPE,
    _det_seed,
    check_determinism,
    check_fused_grad_accum,
    dense_ref,
    fp32_leaves,
    make_generator,
    randn,
)
from tests.pytorch.ops.gemm_shapes_helper import GEMM_SHAPES
from tests.pytorch.test_utils import get_tolerances


@pytest.mark.parametrize("m, n, k", GEMM_SHAPES)
@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gemm(m, n, k, layout, dtype):
    trans_a = layout[0] == "T"
    trans_b = layout[1] == "T"

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    torch.manual_seed(42)

    print(f"\nM={m}, N={n}, K={k}, trans_a={trans_a}, trans_b={trans_b}, dtype={dtype}")

    a_shape = (m, k) if not trans_a else (k, m)
    b_shape = (k, n) if not trans_b else (n, k)

    a = torch.randn(a_shape, dtype=dtype, device=device)
    b = torch.randn(b_shape, dtype=dtype, device=device)
    a = a / a.abs().max()
    b = b / b.abs().max()
    a.requires_grad_()
    b.requires_grad_()
    a.grad = None
    b.grad = None
    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    torch.cuda.synchronize()

    # Reference output
    a_mat = a_ref.T if trans_a else a_ref
    b_mat = b_ref.T if trans_b else b_ref
    c_ref = a_mat @ b_mat

    # Turbo
    c = turbo.ops.gemm(a, b, trans_a, trans_b, dtype)

    # print("a:", a.shape)
    # print("b:", b.shape)
    # print("c: ", c, c.shape)
    # print("c_ref: ", c_ref, c_ref.shape)

    # Check fwd
    torch.testing.assert_close(c, c_ref, **get_tolerances(dtype))

    # Backward
    grad_c = torch.randn_like(c)
    c_ref.backward(grad_c)
    c.backward(grad_c)
    torch.testing.assert_close(a.grad, a_ref.grad, **get_tolerances(dtype))
    torch.testing.assert_close(b.grad, b_ref.grad, **get_tolerances(dtype))


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("main_grad_dtype", [torch.float32, None])
@pytest.mark.parametrize("backend", [None, BackendType.TRITON, BackendType.HIPBLASLT])
def test_gemm_fused_grad_accum(layout, dtype, main_grad_dtype, backend):
    """``fuse_bgrad_accum_pattern`` must leave ``main_grad`` holding previous + wgrad.

    The fused path hands the weight's ``main_grad`` to the wgrad GEMM as a beta=1
    accumulate target instead of returning a gradient for autograd to add on top, so
    the check is that the buffer moved by exactly the wgrad the ordinary path produces.
    ``main_grad_dtype=None`` keeps the accumulator in the weight's own dtype; Megatron
    uses FP32.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    m, n, k = 256, 512, 256
    accum_dtype = main_grad_dtype if main_grad_dtype is not None else dtype

    # A pinned backend exercises that backend's beta=1 accumulate epilogue; ``None``
    # leaves the dispatcher to resolve one, which is how training actually runs.
    GlobalBackendManager.set_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(False)

    device = "cuda"
    torch.manual_seed(42)

    trans_a = layout[0] == "T"
    trans_b = layout[1] == "T"
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)

    a = torch.randn(a_shape, dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=dtype, device=device, requires_grad=True)
    grad_out = torch.randn((m, n), dtype=dtype, device=device)
    a_fused = a.detach().clone().requires_grad_(True)
    b_fused = b.detach().clone().requires_grad_(True)
    torch.cuda.synchronize()

    # Baseline: ordinary autograd, b.grad holds the weight gradient.
    out = turbo.ops.gemm(a, b, trans_a, trans_b, dtype)
    out.backward(grad_out)

    # Fused: the wgrad is accumulated into a pre-seeded main_grad buffer.
    previous = torch.randn(b_fused.shape, dtype=accum_dtype, device=device)
    b_fused.main_grad = previous.clone()
    b_fused.grad_added_to_main_grad = False

    out_fused = turbo.ops.gemm(a_fused, b_fused, trans_a, trans_b, dtype, fuse_bgrad_accum_pattern="megatron")
    out_fused.backward(grad_out)
    torch.cuda.synchronize()

    # The forward is untouched, and the weight is flagged so the training framework's
    # own accumulation step stands down. Following Megatron, backward still hands back
    # a dummy tensor rather than None (it keeps the DDP hooks on the main thread), so
    # what matters is the flag plus the dummy matching the weight's shape and dtype --
    # a dtype mismatch would make autograd allocate and cast a full-size copy.
    torch.testing.assert_close(out_fused, out)
    assert b_fused.grad_added_to_main_grad is True, "weight must be flagged during forward"
    assert b_fused.grad.shape == b_fused.shape, "dummy wgrad must keep the weight's shape"
    assert b_fused.grad.dtype == b_fused.dtype, "dummy wgrad must keep the weight's dtype"

    torch.testing.assert_close(a.grad, a_fused.grad, **get_tolerances(dtype))
    # Compare the accumulated buffer rather than the wgrad recovered by subtracting
    # `previous`: that subtraction cancels the leading digits and blows the rounding of
    # a low-precision accumulator up past any sensible tolerance. Tolerances follow the
    # GEMM's dtype, since the baseline wgrad was rounded to it before autograd stored it.
    expected = (previous.float() + b.grad.float()).to(accum_dtype)
    torch.testing.assert_close(b_fused.main_grad, expected, **get_tolerances(dtype))

    GlobalBackendManager.reset()


# ---------------------------------------------------------------------------
# Determinism (--deterministic-only): bit-exact across repeats on determinism_helper's harness.
# ---------------------------------------------------------------------------
# Triton has no float32 kernel.
_DET_BACKEND_DTYPES = [
    (backend, dtype)
    for backend in (None, BackendType.TRITON, BackendType.HIPBLASLT)
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    if not (backend is BackendType.TRITON and dtype == torch.float32)
]


@pytest.mark.parametrize("m, n, k", [(1, 4096, 2048), (128, 129, 127), (1024, 1024, 512), (2048, 4096, 2048)])
@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("backend, dtype", _DET_BACKEND_DTYPES)
@pytest.mark.deterministic
def test_det_gemm(request, det_seed, m, n, k, layout, backend, dtype):
    if backend is BackendType.TRITON and min(m, n, k) < 64:
        pytest.skip(
            "Triton persistent kernel uses BLOCK_K=64 / BLOCK_M=256 / BLOCK_N=256; "
            "small dimensions cause illegal memory access in pytest environment"
        )
    GlobalBackendManager.set_gemm_backend(backend)
    trans_a, trans_b = layout[0] == "T", layout[1] == "T"
    gen = make_generator(det_seed)
    a0 = randn((k, m) if trans_a else (m, k), dtype, gen, normalize=True)
    b0 = randn((n, k) if trans_b else (k, n), dtype, gen, normalize=True)
    g = randn((m, n), dtype, gen)

    def run_once(mid):
        a = a0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        c = turbo.ops.gemm(a, b, trans_a, trans_b, dtype)
        mid()
        c.backward(g)
        return {"out": c, "dgrad": a.grad, "wgrad": b.grad}

    check_determinism(request.node.nodeid, run_once, dense_ref(a0, b0, g, trans_a, trans_b), snr_floor=40)


@pytest.mark.parametrize(
    "m, n, k, layout",
    [
        (256, 256, 256, "NN"),
        (256, 512, 128, "TN"),
        (512, 256, 128, "NT"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.deterministic
def test_det_gemm_subview(request, det_seed, m, n, k, layout, dtype):
    """Regression: misaligned storage_offset with aligned strides.

    Simulates TP/MoE weight slicing where a subview has aligned strides but
    a non-16-element-aligned base address.  Without the data_ptr() guard in
    the kernel launcher, tl.multiple_of would generate misaligned vector
    loads, producing garbage data and potential NaN in training.
    """
    GlobalBackendManager.set_gemm_backend(BackendType.TRITON)
    trans_a, trans_b = layout[0] == "T", layout[1] == "T"
    gen = make_generator(det_seed)
    misalign = 3

    # The buffer is normalized before the view is taken: an out-of-place op on the view would
    # materialize a dense, aligned copy.
    def subview(rows, cols):
        stride = (cols + 15) // 16 * 16
        buf = randn((misalign + rows * stride,), dtype, gen, normalize=True)
        return torch.as_strided(buf, (rows, cols), (stride, 1), storage_offset=misalign)

    a0 = subview(k, m) if trans_a else subview(m, k)
    b0 = subview(n, k) if trans_b else subview(k, n)
    assert a0.data_ptr() % (16 * a0.element_size()) != 0, (
        "test setup: a0 base pointer must NOT be 16-element-aligned"
    )
    assert a0.stride(0) % 16 == 0, "test setup: a0 stride must be 16-aligned"

    # The kernel gets the subviews themselves, never a copy. Forward only.
    def run_once(mid):
        return {"out": turbo.ops.gemm(a0, b0, trans_a, trans_b, dtype)}

    a, b = fp32_leaves(a0, b0)
    ref = (a.t() if trans_a else a) @ (b.t() if trans_b else b)
    check_determinism(request.node.nodeid, run_once, {"out": ref.detach()}, snr_floor=40, repeats=5)


@pytest.mark.parametrize("microbatches", [1, 2], ids=["mb1", "mb2"])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(BackendType.HIPBLASLT, id="gemm-bf16-HIPBLASLT"),
        pytest.param(BackendType.TRITON, id="gemm-bf16-TRITON"),
    ],
)
@pytest.mark.deterministic
def test_det_fused_grad_accum(request, det_seed, backend, microbatches):
    GlobalBackendManager.set_gemm_backend(backend)
    check_fused_grad_accum(
        request.node.nodeid,
        det_seed,
        lambda a, b, _: turbo.ops.gemm(
            a, b, False, True, torch.bfloat16, fuse_bgrad_accum_pattern="megatron"
        ),
        ACCUM_DENSE_SHAPE,
        torch.float32,
        40,
        microbatches,
    )
