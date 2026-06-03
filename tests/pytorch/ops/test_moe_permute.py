###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for ``moe_permute`` / ``moe_unpermute`` (HIP MoE kernels).
"""

import pytest
import torch

from primus_turbo.pytorch.ops.moe.moe_permute import moe_permute, moe_unpermute
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


def expected_permuted_layout(routing_map: torch.Tensor, pad_multiple: int):
    """Build the (token_id, expert_idx) source map for each row of the
    expert-major permuted output, including padded rows (marked with -1).

    The kernel writes ``[expert 0 real | expert 0 pad zeros | expert 1 real |
    expert 1 pad zeros | ...]``: padding is inserted **per-expert**, NOT
    appended once at the end, so the first ``real_total`` rows are not a
    contiguous slice of the real entries when ``pad_multiple > 0``.
    """
    device = routing_map.device
    num_tokens, num_experts = routing_map.shape
    real_per_expert = routing_map.sum(dim=0).tolist()
    if pad_multiple > 0:
        padded_per_expert = [((r + pad_multiple - 1) // pad_multiple) * pad_multiple for r in real_per_expert]
    else:
        padded_per_expert = list(real_per_expert)
    total = sum(padded_per_expert)

    src_token = torch.full((total,), -1, dtype=torch.long, device=device)
    src_expert = torch.full((total,), -1, dtype=torch.long, device=device)
    offset = 0
    for e in range(num_experts):
        real_e = real_per_expert[e]
        if real_e > 0:
            routed = routing_map[:, e].nonzero(as_tuple=True)[0]
            src_token[offset : offset + real_e] = routed
            src_expert[offset : offset + real_e] = e
        offset += padded_per_expert[e]
    assert offset == total
    return src_token, src_expert, total


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
    permuted_tokens, row_id_map, tokens_per_expert, overflow_flag, ndtt, _, _ = moe_permute(
        tokens_turbo,
        expert_map,
        num_local_experts=num_experts,
        num_topk=0 if expert_map_kind == "routing_map" else num_topk,
        probs_layout="routing_map",
    )
    assert int(overflow_flag.item()) == 0
    assert int(ndtt.item()) == num_tokens
    torch.testing.assert_close(
        tokens_per_expert.cpu(),
        routing_map.sum(dim=0).to(torch.int64).cpu(),
    )
    permuted_tokens.backward(grad_perm, retain_graph=True)

    # --- turbo: unpermute (forward + backward) ---------------------------
    turbo_unp_in = permuted_tokens.detach().clone().requires_grad_(True)
    turbo_unp_out, _ = moe_unpermute(
        turbo_unp_in,
        row_id_map,
        ndtt,
        restore_shape=tokens_turbo.shape,
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
    tokens = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")

    _, _, base_per_expert, _, _, _, _ = moe_permute(
        tokens,
        routing_map,
        num_local_experts=num_experts,
        pad_multiple=0,
        probs_layout="routing_map",
    )
    _, _, padded_per_expert, _, _, _, _ = moe_permute(
        tokens,
        routing_map,
        num_local_experts=num_experts,
        pad_multiple=pad_multiple,
        probs_layout="routing_map",
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
    _, _, _, overflow_flag, _, _, _ = moe_permute(
        tokens,
        routing_map,
        num_local_experts=num_experts,
        pad_multiple=pad_multiple,
        num_permuted_tokens=cap,
        probs_layout="routing_map",
    )
    torch.cuda.synchronize()

    assert int(overflow_flag.item()) == expected


# -----------------------------------------------------------------------------
# Fast path: num_tokens = 0 must short-circuit forward AND backward without
# touching the preprocessing kernel (which hard-asserts max_num_dispatched > 0).
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("with_probs", [False, True])
@pytest.mark.parametrize("pad_multiple", [0, 8])
def test_moe_permute_empty_input(with_probs, pad_multiple):
    """Empty dispatch (``num_tokens == 0``) takes the Python-side fast path."""
    num_experts, hidden_size = 8, 64
    routing_map = torch.zeros((0, num_experts), dtype=torch.bool, device="cuda")
    tokens = torch.randn((0, hidden_size), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    probs = (
        torch.zeros((0, num_experts), dtype=torch.float32, device="cuda", requires_grad=True)
        if with_probs
        else None
    )

    (
        permuted_tokens,
        row_id_map,
        tokens_per_expert,
        overflow_flag,
        ndtt,
        _,
        permuted_probs,
    ) = moe_permute(
        tokens,
        routing_map,
        num_local_experts=num_experts,
        pad_multiple=pad_multiple,
        probs=probs,
        probs_layout="routing_map",
    )

    assert permuted_tokens.shape == (0, hidden_size)
    assert tokens_per_expert.shape == (num_experts,)
    assert int(tokens_per_expert.sum().item()) == 0
    assert int(overflow_flag.item()) == 0
    assert int(ndtt.item()) == 0
    assert row_id_map.shape == (pad_multiple, 2 * num_experts + 1)
    if with_probs:
        assert permuted_probs is not None
        assert permuted_probs.numel() == 0
    else:
        assert permuted_probs is None

    # Backward over an empty permuted_tokens must not launch the kernel.
    grad_perm = torch.zeros_like(permuted_tokens)
    if with_probs:
        grad_pp = torch.zeros_like(permuted_probs)
        torch.autograd.backward([permuted_tokens, permuted_probs], [grad_perm, grad_pp])
    else:
        permuted_tokens.backward(grad_perm)

    assert tokens.grad is not None
    assert tokens.grad.shape == (0, hidden_size)
    if with_probs:
        assert probs.grad is not None
        assert probs.grad.shape == (0, num_experts)

    # moe_unpermute fast path: zero permuted rows back to zero dispatched rows.
    permuted = torch.zeros((0, hidden_size), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    unpermuted, _ = moe_unpermute(
        permuted,
        row_id_map,
        ndtt,
        restore_shape=tokens.shape,
        num_local_experts=num_experts,
    )
    assert unpermuted.shape == (0, hidden_size)
    unpermuted.backward(torch.zeros_like(unpermuted))
    assert permuted.grad is not None and permuted.grad.shape == (0, hidden_size)


# -----------------------------------------------------------------------------
# Probs path: forward writes [permuted, ] probs from [T, E] multihot, and the
# backward routes ``permuted_probs.grad`` back to ``probs`` as well as
# ``permuted_tokens.grad`` back to ``tokens``.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("num_topk", [1, 2, 4])
@pytest.mark.parametrize("pad_multiple", [0, 16])
def test_moe_permute_with_probs_fwd_bwd(num_topk, pad_multiple):
    num_tokens, num_experts, hidden_size = 256, 8, 128
    routing_map = generate_routing_map(num_tokens, num_experts, num_topk, seed=7)

    tokens = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    # Use random multihot probs so the gather is observable; non-routed slots
    # stay zero — that's the contract that ``[T, E]`` probs encode.
    probs_dense = torch.rand((num_tokens, num_experts), dtype=torch.float32, device="cuda")
    probs_dense = probs_dense * routing_map.float()
    probs = probs_dense.detach().clone().requires_grad_(True)

    permuted_tokens, row_id_map, _, overflow_flag, _, _, permuted_probs = moe_permute(
        tokens,
        routing_map,
        num_local_experts=num_experts,
        pad_multiple=pad_multiple,
        probs=probs,
        probs_layout="routing_map",
    )
    assert int(overflow_flag.item()) == 0
    assert permuted_probs is not None
    assert permuted_probs.dtype == torch.float32
    assert permuted_probs.shape[0] == permuted_tokens.shape[0]

    # Reference: gather probs[T, E] into expert-major order with per-expert
    # padding zeros, mirroring how the kernel writes ``permuted_probs`` from
    # ``[token_id, expert_idx]``.
    src_token, src_expert, total = expected_permuted_layout(routing_map, pad_multiple)
    assert permuted_probs.shape[0] == total
    real_mask = src_token >= 0
    expected_pp = torch.zeros(total, dtype=torch.float32, device="cuda")
    expected_pp[real_mask] = probs[src_token[real_mask], src_expert[real_mask]]
    torch.testing.assert_close(permuted_probs.float(), expected_pp, atol=0.0, rtol=0.0)

    # Backward: random gradient on permuted_probs must flow back to probs at
    # the matching ``[token_id, expert_idx]`` slots only; padded rows do not
    # contribute (their src_token == -1).
    grad_perm = torch.randn_like(permuted_tokens)
    grad_pp = torch.randn_like(permuted_probs)
    torch.autograd.backward([permuted_tokens, permuted_probs], [grad_perm, grad_pp])

    expected_probs_grad = torch.zeros_like(probs)
    expected_probs_grad[src_token[real_mask], src_expert[real_mask]] = grad_pp[real_mask]

    torch.testing.assert_close(probs.grad, expected_probs_grad, atol=1e-5, rtol=1e-5)
    assert tokens.grad is not None
    assert tokens.grad.shape == tokens.shape


# -----------------------------------------------------------------------------
# Unpermute probs backward: a forward gradient on ``unpermuted_probs`` must
# flow back to ``permuted_probs.grad`` via the permute kernel.
# -----------------------------------------------------------------------------


def test_moe_unpermute_with_probs_fwd_bwd():
    num_tokens, num_experts, num_topk, hidden_size = 256, 8, 2, 128
    routing_map = generate_routing_map(num_tokens, num_experts, num_topk, seed=11)

    tokens = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")
    probs = torch.rand((num_tokens, num_experts), dtype=torch.float32, device="cuda") * routing_map.float()
    permuted_tokens, row_id_map, _, _, ndtt, _, permuted_probs = moe_permute(
        tokens,
        routing_map,
        num_local_experts=num_experts,
        probs=probs,
        probs_layout="routing_map",
    )

    permuted_probs = permuted_probs.detach().clone().requires_grad_(True)
    permuted_in = permuted_tokens.detach().clone().requires_grad_(True)
    unpermuted_tokens, unpermuted_probs = moe_unpermute(
        permuted_in,
        row_id_map,
        ndtt,
        restore_shape=tokens.shape,
        num_local_experts=num_experts,
        permuted_probs=permuted_probs,
    )

    assert unpermuted_probs is not None
    assert unpermuted_probs.shape == (num_tokens, num_experts)

    grad_unp = torch.randn_like(unpermuted_tokens)
    grad_unp_probs = torch.randn_like(unpermuted_probs)
    torch.autograd.backward([unpermuted_tokens, unpermuted_probs], [grad_unp, grad_unp_probs])

    assert permuted_in.grad is not None and permuted_in.grad.shape == permuted_tokens.shape
    assert permuted_probs.grad is not None and permuted_probs.grad.shape == permuted_probs.shape

    # Forward-then-backward roundtrip: the gradient on permuted_probs at row r
    # comes from grad_unp_probs[token_id, expert_idx] for the (token, expert)
    # pair encoded at row r of the expert-major layout (or 0 for padded rows).
    src_token, src_expert, total = expected_permuted_layout(routing_map, pad_multiple=0)
    real_mask = src_token >= 0
    expected_pp_grad = torch.zeros_like(permuted_probs)
    expected_pp_grad[real_mask] = grad_unp_probs[src_token[real_mask], src_expert[real_mask]]
    torch.testing.assert_close(permuted_probs.grad, expected_pp_grad, atol=1e-5, rtol=1e-5)


# -----------------------------------------------------------------------------
# fp16 dtype: forward + backward, exercising the unpermute fp16 dispatch added
# in the C++ launcher.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("num_topk", [1, 4])
def test_moe_permute_fp16_fwd_bwd(num_topk):
    num_tokens, num_experts, hidden_size = 1024, 8, 256
    routing_map = generate_routing_map(num_tokens, num_experts, num_topk, seed=21)

    base = torch.randn((num_tokens, hidden_size), dtype=torch.float16, device="cuda")
    tokens_ref = base.detach().clone().requires_grad_(True)
    tokens_turbo = base.detach().clone().requires_grad_(True)

    ref_perm, sorted_idx = pytorch_permute_mask_map(tokens_ref, routing_map)
    grad_perm = torch.randn_like(ref_perm)
    ref_perm.backward(grad_perm, retain_graph=True)

    ref_unp_in = ref_perm.detach().clone().requires_grad_(True)
    ref_unp_out = pytorch_unpermute_mask_map(ref_unp_in, sorted_idx, tokens_ref.shape, probs=None)
    grad_unp = torch.randn_like(ref_unp_out)
    ref_unp_out.backward(grad_unp, retain_graph=True)

    permuted_tokens, row_id_map, _, overflow_flag, ndtt, _, _ = moe_permute(
        tokens_turbo,
        routing_map,
        num_local_experts=num_experts,
        probs_layout="routing_map",
    )
    assert int(overflow_flag.item()) == 0
    permuted_tokens.backward(grad_perm, retain_graph=True)

    turbo_unp_in = permuted_tokens.detach().clone().requires_grad_(True)
    turbo_unp_out, _ = moe_unpermute(
        turbo_unp_in,
        row_id_map,
        ndtt,
        restore_shape=tokens_turbo.shape,
        num_local_experts=num_experts,
    )
    assert turbo_unp_out.dtype == torch.float16
    turbo_unp_out.backward(grad_unp, retain_graph=True)

    tol = get_tolerances(torch.float16)
    bwd_tol = dict(atol=max(tol["atol"], 0.01 * num_topk), rtol=max(tol["rtol"], 0.01 * num_topk))

    torch.testing.assert_close(permuted_tokens, ref_perm, **tol)
    torch.testing.assert_close(tokens_turbo.grad, tokens_ref.grad, **bwd_tol)
    torch.testing.assert_close(turbo_unp_out, ref_unp_out, **tol)
    torch.testing.assert_close(turbo_unp_in.grad, ref_unp_in.grad, **bwd_tol)


# -----------------------------------------------------------------------------
# pad_multiple > 0: end-to-end (forward + backward) correctness check.
#   * permute forward must still match the reference for the real rows.
#   * unpermute output shape must equal [num_dispatched, hidden] (i.e. the
#     padding rows are stripped via the new ``pad_multiple`` parameter).
#   * backward gradients on tokens / permuted_tokens must match the
#     pad_multiple = 0 baseline (padding rows do not contribute).
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("num_topk", [1, 2, 4])
@pytest.mark.parametrize("pad_multiple", [8, 64])
def test_moe_permute_pad_multiple_fwd_bwd(num_topk, pad_multiple):
    num_tokens, num_experts, hidden_size = 512, 8, 128
    routing_map = generate_routing_map(num_tokens, num_experts, num_topk, seed=33)

    base = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")

    # Reference: pad_multiple = 0.
    tokens_ref = base.detach().clone().requires_grad_(True)
    perm_ref, row_id_map_ref, _, _, ndtt_ref, _, _ = moe_permute(
        tokens_ref,
        routing_map,
        num_local_experts=num_experts,
        pad_multiple=0,
        probs_layout="routing_map",
    )
    grad_perm_ref = torch.randn_like(perm_ref)
    perm_ref.backward(grad_perm_ref, retain_graph=True)

    unp_in_ref = perm_ref.detach().clone().requires_grad_(True)
    unp_out_ref, _ = moe_unpermute(
        unp_in_ref,
        row_id_map_ref,
        ndtt_ref,
        restore_shape=tokens_ref.shape,
        num_local_experts=num_experts,
    )
    grad_unp_ref = torch.randn_like(unp_out_ref)
    unp_out_ref.backward(grad_unp_ref, retain_graph=True)

    # Padded run: forward output interleaves padding zeros per expert
    # (``[expert0 real | expert0 pad | expert1 real | expert1 pad | ...]``),
    # so the real entries do NOT form a contiguous prefix in general.
    tokens_pad = base.detach().clone().requires_grad_(True)
    perm_pad, row_id_map_pad, tokens_per_expert_pad, _, ndtt_pad, _, _ = moe_permute(
        tokens_pad,
        routing_map,
        num_local_experts=num_experts,
        pad_multiple=pad_multiple,
        probs_layout="routing_map",
    )
    src_token_pad, _src_expert_pad, total_pad = expected_permuted_layout(routing_map, pad_multiple)
    real_mask_pad = src_token_pad >= 0
    real_total = perm_ref.shape[0]
    assert perm_pad.shape[0] == total_pad
    assert int(real_mask_pad.sum().item()) == real_total
    assert int(tokens_per_expert_pad.sum().item()) == perm_pad.shape[0]

    # Real rows of the padded output must equal the reference (same expert-
    # major order) and padded rows must be zeros.
    torch.testing.assert_close(perm_pad[real_mask_pad], perm_ref, **get_tolerances(torch.bfloat16))
    if perm_pad.shape[0] > real_total:
        assert torch.all(perm_pad[~real_mask_pad] == 0)

    # Backward: feed the same gradient on real rows, zeros on padded rows.
    grad_perm_pad = torch.zeros_like(perm_pad)
    grad_perm_pad[real_mask_pad] = grad_perm_ref
    perm_pad.backward(grad_perm_pad, retain_graph=True)

    tol = get_tolerances(torch.bfloat16)
    torch.testing.assert_close(tokens_pad.grad, tokens_ref.grad, **tol)

    # Unpermute strips padding via ``restore_shape``; output collapses to
    # [num_tokens, hidden].
    unp_in_pad = perm_pad.detach().clone().requires_grad_(True)
    unp_out_pad, _ = moe_unpermute(
        unp_in_pad,
        row_id_map_pad,
        ndtt_pad,
        restore_shape=tokens_pad.shape,
        num_local_experts=num_experts,
        pad_multiple=pad_multiple,
    )
    assert unp_out_pad.shape == (num_tokens, hidden_size)
    torch.testing.assert_close(unp_out_pad, unp_out_ref, **tol)

    # Backward through unpermute: gradient on real rows matches the
    # pad_multiple = 0 baseline, padded rows must remain zero.
    unp_out_pad.backward(grad_unp_ref, retain_graph=True)
    assert unp_in_pad.grad is not None
    torch.testing.assert_close(unp_in_pad.grad[real_mask_pad], unp_in_ref.grad, **tol)
    if unp_in_pad.grad.shape[0] > real_total:
        assert torch.all(unp_in_pad.grad[~real_mask_pad] == 0)
