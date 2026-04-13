###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from primus_turbo.jax.core.low_precision import float8_e4m3
from primus_turbo.jax.lax.moe import moe_combine, moe_dispatch
from tests.jax.test_utils import skip_if_lt_x_gpu

key = jax.random.PRNGKey(123)
num_ranks = jax.local_device_count()


# ============================================================================
# Helpers
# ============================================================================


def _generate(num_tokens, hidden, num_topk, num_experts):
    """Generate test data with leading device axis (num_ranks, ...)."""
    rank_ids_bf16 = jnp.arange(num_ranks, dtype=jnp.bfloat16)[:, None, None]
    x = jnp.ones((num_ranks, num_tokens, hidden), dtype=jnp.bfloat16) * rank_ids_bf16

    scores = jnp.abs(jax.random.normal(key, (num_ranks, num_tokens, num_experts), dtype=jnp.float32)) + 1

    rank_ids_f32 = jnp.arange(num_ranks, dtype=jnp.float32)[:, None, None]
    topk_weights = jnp.ones((num_ranks, num_tokens, num_topk), dtype=jnp.float32) * rank_ids_f32

    return x, scores, topk_weights


def calc_diff(x, y):
    x, y = np.asarray(x, dtype=np.float64) + 1, np.asarray(y, dtype=np.float64) + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return float(1 - sim)


def _per_token_cast_to_fp8(x):
    """BF16 [n, h] -> (FP8 E4M3 [n, h], scales [n, h//128])."""
    n, h = x.shape
    assert h % 128 == 0, f"hidden size must be divisible by 128, got {h}"
    fp8_max = jnp.finfo(float8_e4m3).max.astype(jnp.float32)
    x_f32 = x.astype(jnp.float32).reshape(n, -1, 128)
    amax = jnp.abs(x_f32).max(axis=-1)
    scale = jnp.maximum(amax / fp8_max, jnp.finfo(jnp.float32).tiny)
    x_scaled = x_f32 / scale[..., None]
    x_fp8 = x_scaled.reshape(n, h).astype(float8_e4m3)
    return x_fp8, scale


def _per_token_cast_back(x_fp8, x_scales):
    """(FP8 E4M3 [n, h], scales [n, h//128]) -> BF16 [n, h]."""
    n, h = x_fp8.shape
    x_f32 = x_fp8.astype(jnp.float32).reshape(n, -1, 128)
    return (x_f32 * x_scales[..., None]).reshape(n, h).astype(jnp.bfloat16)


def _dispatch_and_combine(x, topk_idx, topk_weights, num_experts, use_fp8):
    """Dispatch (optionally FP8) then combine. Returns (output, handle)."""
    if use_fp8:
        x_fp8, x_scales = _per_token_cast_to_fp8(x)
        recv_x_tuple, recv_topk_idx, recv_topk_weights, handle = moe_dispatch(
            (x_fp8, x_scales), topk_idx, topk_weights, num_experts
        )
        recv_x = _per_token_cast_back(*recv_x_tuple)
    else:
        recv_x, recv_topk_idx, recv_topk_weights, handle = moe_dispatch(
            x, topk_idx, topk_weights, num_experts
        )
    return recv_x, recv_topk_idx, recv_topk_weights, handle


# ============================================================================
# Forward Test
# ============================================================================


@pytest.mark.multigpu
@pytest.mark.parametrize("use_fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("num_tokens", [4096])
@pytest.mark.parametrize("hidden", [7168])
@pytest.mark.parametrize("num_topk", [8])
@pytest.mark.parametrize("num_experts", [256])
@skip_if_lt_x_gpu(8)
def test_moe_dispatch_combine(num_tokens, hidden, num_topk, num_experts, use_fp8):
    """Test MoE dispatch/combine forward pass (BF16 and FP8).

    Verifies:
      1. Dispatch correctness - tokens from rank i carry value i.
      2. recv_topk_idx validity - values in [-1, num_experts_per_rank).
      3. Combine round-trip - combine(dispatch(x)) approx x after normalisation.
    """
    x, scores, topk_weights = _generate(num_tokens, hidden, num_topk, num_experts)

    @jax.pmap
    def _test_fwd(x, scores, topk_weights):
        num_topk = topk_weights.shape[1]
        topk_idx = jax.lax.top_k(scores, num_topk)[1].astype(jnp.int32)
        num_experts = scores.shape[1]

        recv_x, recv_topk_idx, recv_topk_weights, handle = _dispatch_and_combine(
            x, topk_idx, topk_weights, num_experts, use_fp8
        )
        rank_prefix_matrix = handle[0]
        is_token_in_rank = handle[4]

        amax_recv_topk_weights = jnp.broadcast_to(
            jnp.amax(recv_topk_weights, axis=1, keepdims=True), recv_topk_weights.shape
        )
        check_recv_topk_weights = jax.lax.select(
            jnp.equal(recv_topk_idx, -1), amax_recv_topk_weights, recv_topk_weights
        )

        combined_x = moe_combine(recv_x, handle)
        check_combine_x = combined_x.astype(jnp.float32) / jnp.expand_dims(
            is_token_in_rank.sum(axis=1), axis=1
        )

        return recv_x, check_recv_topk_weights, rank_prefix_matrix, recv_topk_idx, check_combine_x

    recv_x, check_recv_topk_weights, rank_prefix_matrix, recv_topk_idx, check_combine_x = _test_fwd(
        x, scores, topk_weights
    )

    recv_x = np.array(recv_x)
    check_recv_topk_weights = np.array(check_recv_topk_weights)
    rank_prefix_matrix = np.array(rank_prefix_matrix)
    recv_topk_idx = np.array(recv_topk_idx)
    check_combine_x = np.array(check_combine_x)

    # --- 1. Dispatch ---
    def check_dispatch(per_rank_x, per_rank_rpm):
        for rank in range(num_ranks):
            cx = per_rank_x[rank]
            rpm = per_rank_rpm[rank]
            recv_size = int(rpm[num_ranks - 1][rank])
            assert np.allclose(np.amin(cx[:recv_size], axis=1), np.amax(cx[:recv_size], axis=1))
            start = 0
            for src in range(num_ranks):
                end = int(rpm[src][rank])
                assert (cx[start:end, :].astype(np.int32) - src).sum() == 0
                start = end

    check_dispatch(recv_x, rank_prefix_matrix)
    check_dispatch(check_recv_topk_weights, rank_prefix_matrix)

    # --- 2. recv_topk_idx ---
    experts_per_rank = num_experts // num_ranks
    for rank in range(num_ranks):
        idx = recv_topk_idx[rank]
        valid = (idx == -1) | ((idx >= 0) & (idx < experts_per_rank))
        assert valid.sum() == idx.size

    # --- 3. Combine round-trip ---
    fwd_tol = 5e-4 if use_fp8 else 5e-6
    for rank in range(num_ranks):
        diff = calc_diff(check_combine_x[rank], x[rank])
        assert diff < fwd_tol, f"Combine round-trip: rank {rank}, diff={diff}"


# ============================================================================
# Backward Test (custom_vjp backward rules via jax.vjp)
# ============================================================================


@pytest.mark.multigpu
@pytest.mark.parametrize("use_fp8", [False], ids=["bf16"])
@pytest.mark.parametrize("num_tokens", [4096])
@pytest.mark.parametrize("hidden", [7168])
@pytest.mark.parametrize("num_topk", [8])
@pytest.mark.parametrize("num_experts", [256])
@skip_if_lt_x_gpu(8)
def test_moe_dispatch_combine_backward(num_tokens, hidden, num_topk, num_experts, use_fp8):
    """Test dispatch/combine backward via jax.vjp exercising custom_vjp rules.

    The round-trip f(x) = combine(dispatch(x)) / K approx identity, so J approx I.
    With cotangent v = f(x) approx x, vjp gives grad_x = J^T @ v approx x.

    For FP8: scales are stop-gradiented so backward only flows through
    FP8 cast (STE) -> dispatch -> cast back -> combine.

    Reference:
      https://github.com/NVIDIA/Megatron-LM/blob/fe5291fa/tests/unit_tests/
      transformer/moe/test_token_dispatcher.py#L116
    """
    x, scores, topk_weights = _generate(num_tokens, hidden, num_topk, num_experts)

    @jax.pmap
    def _compute_fwd_and_grad(x, scores, topk_weights):
        num_topk = topk_weights.shape[1]
        topk_idx = jax.lax.top_k(scores, num_topk)[1].astype(jnp.int32)
        num_experts = scores.shape[1]

        def fwd(x):
            if use_fp8:
                x_fp8, x_scales = _per_token_cast_to_fp8(x)
                x_scales = jax.lax.stop_gradient(x_scales)
                recv_tuple, _, _, handle = moe_dispatch(
                    (x_fp8, x_scales), topk_idx, topk_weights, num_experts
                )
                recv_x = _per_token_cast_back(*recv_tuple)
            else:
                recv_x, _, _, handle = moe_dispatch(x, topk_idx, topk_weights, num_experts)
            combined_x = moe_combine(recv_x, handle)
            is_token_in_rank = handle[4]
            scale = jnp.expand_dims(is_token_in_rank.sum(axis=1), axis=1)
            normalized = combined_x.astype(jnp.float32) / scale
            return normalized.astype(x.dtype)

        output, vjp_fn = jax.vjp(fwd, x)
        (grad_x,) = vjp_fn(output)
        return output, grad_x

    restored, grad_x = _compute_fwd_and_grad(x, scores, topk_weights)

    restored = np.array(restored)
    grad_x = np.array(grad_x)
    x_np = np.array(x)

    fwd_tol = 5e-6
    bwd_tol = 5e-3 if use_fp8 else 5e-6

    for rank in range(num_ranks):
        diff = calc_diff(restored[rank], x_np[rank])
        assert diff < fwd_tol, f"Forward: restored != x at rank {rank}, diff={diff}"

    for rank in range(num_ranks):
        diff = calc_diff(grad_x[rank], x_np[rank])
        assert diff < bwd_tol, f"Backward: grad_x != x at rank {rank}, diff={diff}"
