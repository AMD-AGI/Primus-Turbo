import pytest
import torch

from tests.test_utils import (
    cosine_similarity,
    ulp_error,
    max_abs_error,
    mean_squared_error,
    relative_error,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shapes", [(512, 128, 256),
                                    (8192, 8192, 8192),
                                    (1, 2048, 128)])
def test_gemm_numerical(dtype, shapes):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(42)

    m, n, k = shapes
    device = "cuda"

    a = torch.randn(m, k, device=device, dtype=dtype, requires_grad=True)
    a_cpu = a.float().detach().clone().cpu().requires_grad_()

    b = torch.randn(n, k, device=device, dtype=dtype, requires_grad=True)
    b_cpu = b.float().detach().clone().cpu().requires_grad_()

    out = torch.matmul(a, b.T)
    out = out.float().cpu()
    ref = torch.matmul(a_cpu, b_cpu.T)

    ulp = ulp_error(out, ref)

    print(f"\n[GEMM] dtype={dtype}, shape={shapes}, result:")
    print(f"RelError:   {relative_error(ref, out):.3e}")
    print(f"MAE:        {max_abs_error(ref, out):.3e}")
    print(f"MSE:        {mean_squared_error(ref, out):.3e}")
    print(f"CosSim:     {cosine_similarity(ref, out):.6f}")
    print(f"ULP(max):   {ulp.max().item()}, ULP(mean): {ulp.float().mean().item():.2f}")


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fnuz])
@pytest.mark.parametrize("shapes", [(512, 128, 256),
                                    (8192, 8192, 8192),
                                    (1, 2048, 128)])
def test_fp8_gemm_numerical(dtype, shapes):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(42)

    m, n, k = shapes
    device = "cuda"

    a = torch.randn(m, k, device=device, requires_grad=True)
    a_cpu = a.float().detach().clone().cpu().requires_grad_()
    scale_a = torch.tensor(1.0, dtype=torch.float32, device=device)
    a = a.to(dtype)

    b = torch.randn(n, k, device=device, requires_grad=True)
    b_cpu = b.float().detach().clone().cpu().requires_grad_()
    scale_b = torch.tensor(1.0, dtype=torch.float32, device=device)
    b = b.to(dtype)

    out = torch._scaled_mm(a, b.T, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float32)
    out = out.float().cpu()
    ref = torch.matmul(a_cpu, b_cpu.T)

    ulp = ulp_error(out, ref)

    print(f"\n[GEMM] dtype={dtype}, shape={shapes}, result:")
    print(f"RelError:   {relative_error(ref, out):.3e}")
    print(f"MAE:        {max_abs_error(ref, out):.3e}")
    print(f"MSE:        {mean_squared_error(ref, out):.3e}")
    print(f"CosSim:     {cosine_similarity(ref, out):.6f}")
    print(f"ULP(max):   {ulp.max().item()}, ULP(mean): {ulp.float().mean().item():.2f}")
 