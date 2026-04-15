###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from tests.pytorch.test_utils import get_tolerances


@pytest.mark.parametrize("m", [1, 16, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("n", [1, 16, 129, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("k", [1, 16, 127, 255, 512, 1024, 2048])
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


@pytest.mark.parametrize("m", [1, 16, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("n", [1, 16, 129, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("k", [1, 16, 127, 255, 512, 1024, 2048])
@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backend", [None, BackendType.TRITON, BackendType.HIPBLASLT])
@pytest.mark.deterministic
def test_gemm_deterministic(m, n, k, layout, dtype, backend):
    trans_a = layout[0] == "T"
    trans_b = layout[1] == "T"

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if backend is BackendType.TRITON and dtype == torch.float32:
        pytest.skip("Triton backend does not support float32")

    if backend is BackendType.TRITON and min(m, n, k) < 64:
        pytest.skip(
            "Triton persistent kernel uses BLOCK_K=64 / BLOCK_M=256 / BLOCK_N=256; "
            "small dimensions cause illegal memory access in pytest environment"
        )

    GlobalBackendManager.set_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(False)

    device = "cuda"
    torch.manual_seed(42)

    print(
        f"\n[deterministic] M={m}, N={n}, K={k}, trans_a={trans_a}, trans_b={trans_b}, "
        f"dtype={dtype}, backend={backend}"
    )

    a_shape = (m, k) if not trans_a else (k, m)
    b_shape = (k, n) if not trans_b else (n, k)

    a0 = torch.randn(a_shape, dtype=dtype, device=device)
    b0 = torch.randn(b_shape, dtype=dtype, device=device)
    a0 = a0 / a0.abs().max()
    b0 = b0 / b0.abs().max()

    # Reference output (correctness)
    a_ref = a0.detach().clone().requires_grad_()
    b_ref = b0.detach().clone().requires_grad_()
    a_mat = a_ref.T if trans_a else a_ref
    b_mat = b_ref.T if trans_b else b_ref
    c_ref = a_mat @ b_mat
    grad_c = torch.randn_like(c_ref)
    c_ref.backward(grad_c)
    torch.cuda.synchronize()

    def _run_once():
        a = a0.detach().clone().requires_grad_()
        b = b0.detach().clone().requires_grad_()
        c = turbo.ops.gemm(a, b, trans_a, trans_b, dtype)
        c.backward(grad_c)
        return c.detach(), a.grad.detach(), b.grad.detach()

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
    torch.testing.assert_close(c0, c_ref.detach(), **get_tolerances(dtype))
    torch.testing.assert_close(da0, a_ref.grad.detach(), **get_tolerances(dtype))
    torch.testing.assert_close(db0, b_ref.grad.detach(), **get_tolerances(dtype))

    GlobalBackendManager.reset()


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
def test_gemm_deterministic_subview(m, n, k, layout, dtype):
    """Regression: misaligned storage_offset with aligned strides.

    Simulates TP/MoE weight slicing where a subview has aligned strides but
    a non-16-element-aligned base address.  Without the data_ptr() guard in
    the kernel launcher, tl.multiple_of would generate misaligned vector
    loads, producing garbage data and potential NaN in training.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    GlobalBackendManager.set_gemm_backend(BackendType.TRITON)
    GlobalBackendManager.set_auto_tune(False)

    trans_a = layout[0] == "T"
    trans_b = layout[1] == "T"
    device = "cuda"
    torch.manual_seed(42)

    elem_bytes = torch.tensor([], dtype=dtype).element_size()

    # Build subviews with 16-aligned strides but misaligned base pointers.
    # Use as_strided to place data at a storage_offset that is NOT a
    # multiple of 16, while keeping row stride 16-aligned.
    # Normalize the underlying buffer BEFORE creating the view so that the
    # view's data_ptr stays misaligned (out-of-place ops would materialize
    # a new dense tensor and lose the misalignment).
    misalign = 3
    cols_a = k if not trans_a else m
    rows_a = m if not trans_a else k
    stride_a = (cols_a + 15) // 16 * 16
    buf_a = torch.randn(misalign + rows_a * stride_a, dtype=dtype, device=device)
    buf_a /= buf_a.abs().max()
    a0 = torch.as_strided(buf_a, (rows_a, cols_a), (stride_a, 1), storage_offset=misalign)

    cols_b = n if not trans_b else k
    rows_b = k if not trans_b else n
    stride_b = (cols_b + 15) // 16 * 16
    buf_b = torch.randn(misalign + rows_b * stride_b, dtype=dtype, device=device)
    buf_b /= buf_b.abs().max()
    b0 = torch.as_strided(buf_b, (rows_b, cols_b), (stride_b, 1), storage_offset=misalign)

    assert (
        a0.data_ptr() % (16 * elem_bytes) != 0
    ), "test setup: a0 base pointer must NOT be 16-element-aligned"
    assert a0.stride(0) % 16 == 0, "test setup: a0 stride must be 16-aligned"

    # Reference output (correctness baseline uses contiguous copies)
    a_ref = a0.clone()
    b_ref = b0.clone()
    a_mat = a_ref.T if trans_a else a_ref
    b_mat = b_ref.T if trans_b else b_ref
    c_ref = a_mat @ b_mat

    # Pass the SUBVIEW directly — no clone() — so the kernel sees the
    # misaligned data_ptr and must handle it correctly.
    repeats = 5
    outs = []
    for _ in range(repeats):
        c = turbo.ops.gemm(a0, b0, trans_a, trans_b, dtype)
        outs.append(c.detach())
        torch.cuda.synchronize()

    # Determinism (bitwise identical across runs)
    for i in range(1, repeats):
        torch.testing.assert_close(outs[0], outs[i], rtol=0, atol=0)

    # Correctness (close to PyTorch reference)
    torch.testing.assert_close(outs[0], c_ref, **get_tolerances(dtype))

    GlobalBackendManager.reset()
