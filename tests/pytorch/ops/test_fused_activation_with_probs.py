import pytest
import torch

from primus_turbo.pytorch.ops import fused_activation_with_probs
from tests.pytorch.ref.activation_with_probs import activation_with_probs_ref
from tests.test_utils import get_tolerances


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("act_type", ("swiglu", "geglu"))
@pytest.mark.parametrize("num_experts", (8,))
@pytest.mark.parametrize("hidden_size", (4096,))
@pytest.mark.parametrize("num_tokens", (4096,))
def test_fused_activation_with_probs(
    num_tokens: int, hidden_size: int, num_experts: int, act_type: str, dtype: torch.dtype
):
    torch.manual_seed(1234)
    even = num_tokens // num_experts

    x = torch.rand((num_tokens, hidden_size), dtype=dtype, device="cuda")
    probs = torch.rand((num_tokens,), dtype=torch.float32, device="cuda")
    tokens_per_expert = torch.randint(low=0, high=even, size=(num_experts,), device="cuda")

    cpu_actual_num_tokens = tokens_per_expert.sum().item()

    ref_out = activation_with_probs_ref(x, probs.unsqueeze(-1), act_type)
    turbo_out = fused_activation_with_probs(x, probs, tokens_per_expert, act_type)

    tol = get_tolerances(dtype)
    assert torch.allclose(ref_out[:cpu_actual_num_tokens], turbo_out[:cpu_actual_num_tokens], **tol)
