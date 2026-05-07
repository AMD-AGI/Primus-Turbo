###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for ``moe_permute`` / ``moe_unpermute`` (HIP MoE kernels).
"""

import pytest
import torch

from primus_turbo.pytorch.ops.moe.permute import moe_permute, moe_unpermute
from tests.pytorch.ref.permuatation_ref import (
    pytorch_permute_mask_map,
    pytorch_unpermute_mask_map,
)
from tests.pytorch.test_utils import get_tolerances


def generate_routing_map(num_tokens: int, num_experts: int, num_topk: int, *, seed: int) -> torch.Tensor:
    """Bool routing_map ``[num_tokens, num_experts]`` with exactly ``num_topk`` ones per row."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    perm = torch.argsort(torch.rand(num_tokens, num_experts, generator=g, device="cuda"), dim=1)
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool, device="cuda")
    routing_map.scatter_(1, perm[:, :num_topk], True)
    return routing_map


def routing_map_to_expert_map(routing_map: torch.Tensor, num_topk: int, kind: str) -> torch.Tensor:
    """Return ``routing_map`` itself for the bool path, or ``topk_idx[num_tokens, num_topk]`` otherwise."""
    if kind == "routing_map":
        return routing_map
    dtype = {"topk_idx_int32": torch.int32, "topk_idx_int64": torch.int64}[kind]
    num_tokens, num_experts = routing_map.shape
    expert_ids = torch.arange(num_experts, device="cuda").unsqueeze(0).expand(num_tokens, -1).contiguous()
    return expert_ids[routing_map].view(num_tokens, num_topk).to(dtype)


# -----------------------------------------------------------------------------
# Forward + backward correctness (pad_multiple = 0)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("num_topk", [1, 2, 4, 8])
@pytest.mark.parametrize("expert_map_kind", ["routing_map", "topk_idx_int32", "topk_idx_int64"])
@pytest.mark.parametrize("num_tokens", [4096])
@pytest.mark.parametrize("num_experts", [16])
@pytest.mark.parametrize("hidden_size", [4096])
def test_moe_permutation(num_topk, expert_map_kind, num_tokens, num_experts, hidden_size):
    routing_map = generate_routing_map(num_tokens, num_experts, num_topk, seed=1234)
    expert_map = routing_map_to_expert_map(routing_map, num_topk, expert_map_kind)
    ndtt = torch.tensor([num_tokens], dtype=torch.int32, device="cuda")

    base = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")
    tokens_ref = base.detach().clone().requires_grad_(True)
    tokens_turbo = base.detach().clone().requires_grad_(True)

    # --- reference -------------------------------------------------------
    ref_perm, sorted_idx = pytorch_permute_mask_map(tokens_ref, routing_map)
    grad_perm = torch.randn_like(ref_perm)
    ref_perm.backward(grad_perm, retain_graph=True)

    ref_unp_in = ref_perm.detach().clone().requires_grad_(True)
    ref_unp_out = pytorch_unpermute_mask_map(ref_unp_in, sorted_idx, tokens_ref.shape, probs=None)
    grad_unp = torch.randn_like(ref_unp_out)
    ref_unp_out.backward(grad_unp, retain_graph=True)

    # --- turbo: permute (forward + backward) -----------------------------
    permuted_tokens, row_id_map, tokens_per_expert, overflow_flag, _, _ = moe_permute(
        tokens_turbo,
        expert_map,
        ndtt,
        num_tokens,
        num_local_experts=num_experts,
        num_topk=0 if expert_map_kind == "routing_map" else num_topk,
    )
    assert int(overflow_flag.item()) == 0
    torch.testing.assert_close(
        tokens_per_expert.cpu(),
        routing_map.sum(dim=0).to(torch.int32).cpu(),
    )
    permuted_tokens.backward(grad_perm, retain_graph=True)

    # --- turbo: unpermute (forward + backward) ---------------------------
    turbo_unp_in = permuted_tokens.detach().clone().requires_grad_(True)
    turbo_unp_out, _ = moe_unpermute(
        turbo_unp_in,
        row_id_map,
        ndtt,
        num_dispatched=num_tokens,
        num_local_experts=num_experts,
    )
    turbo_unp_out.backward(grad_unp, retain_graph=True)

    tol = get_tolerances(permuted_tokens.dtype)
    # PyTorch native reference accumulates backward gradients in bf16, which is less accurate than the turbo implementation.
    bwd_tol = dict(atol=max(tol["atol"], 0.01 * num_topk), rtol=max(tol["rtol"], 0.01 * num_topk))

    # --- compare ---------------------------------------------------------
    torch.testing.assert_close(permuted_tokens, ref_perm, **tol)
    torch.testing.assert_close(tokens_turbo.grad, tokens_ref.grad, **bwd_tol)
    torch.testing.assert_close(turbo_unp_out, ref_unp_out, **tol)
    torch.testing.assert_close(turbo_unp_in.grad, ref_unp_in.grad, **bwd_tol)


# -----------------------------------------------------------------------------
# pad_multiple: tokens_per_expert is rounded up to the requested alignment
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("num_topk", [1, 2, 4])
@pytest.mark.parametrize("pad_multiple", [8, 16, 64])
def test_moe_permute_pad_multiple(num_topk, pad_multiple):
    """``pad_multiple > 0`` ⇒ ``tokens_per_expert`` rounds each expert's
    real count up to a multiple of ``pad_multiple``. Verified against the
    ``pad_multiple = 0`` baseline."""
    num_tokens, num_experts, hidden_size = 1024, 8, 64
    routing_map = generate_routing_map(num_tokens, num_experts, num_topk, seed=1234)
    ndtt = torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
    tokens = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")

    _, _, base_per_expert, _, _, _ = moe_permute(
        tokens,
        routing_map,
        ndtt,
        num_tokens,
        num_local_experts=num_experts,
        pad_multiple=0,
    )
    _, _, padded_per_expert, _, _, _ = moe_permute(
        tokens,
        routing_map,
        ndtt,
        num_tokens,
        num_local_experts=num_experts,
        pad_multiple=pad_multiple,
    )

    expected = ((base_per_expert + pad_multiple - 1) // pad_multiple) * pad_multiple
    torch.testing.assert_close(padded_per_expert, expected)
    assert (padded_per_expert % pad_multiple == 0).all()


# -----------------------------------------------------------------------------
# overflow_flag
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pad_multiple,cap_kind,expected",
    [
        (0, "uncapped", 0),
        (0, ("real", 0), 0),
        (0, ("real", +256), 0),
        (0, "half", 1),
        (0, "tiny", 1),
        (8, ("padded", 0), 0),
        (8, ("padded", +8), 0),
        (8, ("real", 0), 1),
        (8, "half", 1),
        (64, ("padded", 0), 0),
        (64, ("real", 0), 1),
    ],
)
def test_overflow_flag(pad_multiple, cap_kind, expected):
    num_tokens, num_experts, num_topk = 1024, 8, 2
    hidden_size = 64
    routing_map = generate_routing_map(num_tokens, num_experts, num_topk, seed=4321)
    real_per_expert = routing_map.sum(dim=0)
    real = int(real_per_expert.sum().item())
    padded = int(
        (((real_per_expert + pad_multiple - 1) // pad_multiple) * pad_multiple).sum().item()
        if pad_multiple > 0
        else real
    )

    if cap_kind == "uncapped":
        cap = -1
    elif cap_kind == "half":
        cap = real // 2
    elif cap_kind == "tiny":
        cap = 1
    else:
        base, delta = cap_kind
        cap = (real if base == "real" else padded) + delta

    tokens = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")
    _, _, _, overflow_flag, _, _ = moe_permute(
        tokens,
        routing_map,
        torch.tensor([num_tokens], dtype=torch.int32, device="cuda"),
        num_tokens,
        num_local_experts=num_experts,
        pad_multiple=pad_multiple,
        num_permuted_tokens=cap,
    )
    torch.cuda.synchronize()

    assert int(overflow_flag.item()) == expected
