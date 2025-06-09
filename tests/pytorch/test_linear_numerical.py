import pytest
import torch

from tests.test_utils import (
    cosine_similarity,
    l2_norm,
    max_abs_error,
    mean_squared_error,
    relative_error,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shapes", [(512, 128, 256)])
def test_linear_numerical(dtype, shapes):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(42)

    seq_len, in_features, out_features = shapes
    device = "cuda"

    torch_linear = torch.nn.Linear(in_features, out_features, device=device, dtype=dtype)
    ref_linear = torch.nn.Linear(in_features, out_features, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        ref_linear.weight.copy_(torch_linear.weight.cpu())

    x1 = torch.randn(seq_len, in_features, device=device, dtype=dtype, requires_grad=True)
    x2 = x1.float().detach().clone().requires_grad_().cpu()

    # === Forward ===
    out = torch_linear(x1)
    ref = ref_linear(x2)

    # === Backward ===
    grad_output = torch.ones_like(out)
    grad_output_ref = torch.ones_like(ref)
    out.backward(grad_output)
    ref.backward(grad_output_ref)

    out = out.float().cpu()
    grad_output = grad_output.float().cpu()

    grad_weight = torch_linear.weight.grad.float().detach().clone().cpu()
    grad_weight_ref = ref_linear.weight.grad

    print(f"\n[Linear] dtype={dtype}, shape={shapes}, output result:")
    print(f"RelError:   {relative_error(ref, out):.3e}")
    print(f"MAE:        {max_abs_error(ref, out):.3e}")
    print(f"MSE:        {mean_squared_error(ref, out):.3e}")
    print(f"CosSim:     {cosine_similarity(ref, out):.6f}")
    print(f"L2 Norm:    {l2_norm(ref, out):.3e}")

    print(f"\n[Linear] dtype={dtype}, shape={shapes}, grad output result:")
    print(f"RelError:   {relative_error(grad_output_ref, grad_output):.3e}")
    print(f"MAE:        {max_abs_error(grad_output_ref, grad_output):.3e}")
    print(f"MSE:        {mean_squared_error(grad_output_ref, grad_output):.3e}")
    print(f"CosSim:     {cosine_similarity(grad_output_ref, grad_output):.6f}")
    print(f"L2 Norm:    {l2_norm(grad_output_ref, grad_output):.3e}")

    print(f"\n[Linear] dtype={dtype}, shape={shapes}, grad weight result:")
    print(f"RelError:   {relative_error(grad_weight_ref, grad_weight):.3e}")
    print(f"MAE:        {max_abs_error(grad_weight_ref, grad_weight):.3e}")
    print(f"MSE:        {mean_squared_error(grad_weight_ref, grad_weight):.3e}")
    print(f"CosSim:     {cosine_similarity(grad_weight_ref, grad_weight):.6f}")
    print(f"L2 Norm:    {l2_norm(grad_weight_ref, grad_weight):.3e}")
