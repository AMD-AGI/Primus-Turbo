import pytest
import torch

from tests.test_utils import (
    cosine_similarity,
    max_abs_error,
    mean_squared_error,
    relative_error,
    ulp_error,
)

FUNC_TABLE = {
    "exp": lambda x: torch.exp(x),
    "log": lambda x: torch.log(x.abs() + 1e-3),
    "sqrt": lambda x: torch.sqrt(x.abs()),
    "sigmoid": lambda x: torch.sigmoid(x),
    "tanh": lambda x: torch.tanh(x),
    "pow": lambda x, y: torch.pow(x.abs() + 1e-3, y.abs()),
}


@pytest.mark.parametrize("func_name", FUNC_TABLE.keys())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1024,), (64, 64)])
def test_special_function_accuracy(func_name, dtype, shape):
    torch.manual_seed(0)
    device = torch.device("cuda")

    x = torch.randn(*shape, device=device).to(dtype)

    if func_name == "pow":
        y = torch.randn(*shape, device=device).to(dtype)
        ref = FUNC_TABLE[func_name](x.cpu(), y.cpu())
        out = FUNC_TABLE[func_name](x, y)
    else:
        ref = FUNC_TABLE[func_name](x.cpu())
        out = FUNC_TABLE[func_name](x)

    out = out.float().cpu()
    ref = ref.float()

    ulp = ulp_error(out, ref)

    print(f"\n[{func_name.upper()}] dtype={dtype}, shape={shape}")
    print(f"RelError:   {relative_error(ref, out):.3e}")
    print(f"MaxAbsErr:  {max_abs_error(ref, out):.3e}")
    print(f"MSE:        {mean_squared_error(ref, out):.3e}")
    print(f"CosSim:     {cosine_similarity(ref, out):.6f}")
    print(f"ULP(max):   {ulp.max().item()}, ULP(mean): {ulp.float().mean().item():.2f}")
