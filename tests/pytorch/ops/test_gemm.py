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
