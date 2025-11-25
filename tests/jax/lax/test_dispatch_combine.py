###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec

from primus_turbo.jax.lax import get_dispatch_config, moe_combine, moe_dispatch
from primus_turbo.jax.primitive.moe.moe_dispatch import moe_dispatch_p

key = jax.random.PRNGKey(123)

num_ranks = jax.device_count()

mesh = jax.make_mesh((num_ranks,), ("x",), axis_types=(jax.sharding.AxisType.Explicit,))


def _generate(num_tokens, hidden, num_topk, num_experts):
    x = jnp.ones((num_ranks, num_tokens, hidden), dtype=jnp.bfloat16)
    scores = jnp.abs(jax.random.normal(key, (num_ranks, num_tokens, num_experts), dtype=jnp.float32)) + 1

    topk_weights = jnp.ones((num_ranks, num_tokens, num_topk), dtype=jnp.float32)

    for i in range(num_ranks):
        x = x.at[i].set(i)
        topk_weights = topk_weights.at[i].set(i)
    return (
        jnp.reshape(x, (num_ranks * num_tokens, hidden)),
        jnp.reshape(scores, (num_ranks * num_tokens, num_experts)),
        jnp.reshape(topk_weights, (num_ranks * num_tokens, num_topk)),
    )


def inplace_unique(x: jax.Array, num_slots: int):
    assert x.ndim == 2

    x_padded = jnp.where(x < 0, num_slots, x)

    def batch_bincount(x_row):
        return jnp.bincount(x_row, length=num_slots + 1)

    bin_count = jax.vmap(batch_bincount)(x_padded)
    bin_count = bin_count[:, :num_slots]

    sorted_indices = jnp.argsort(-bin_count, axis=-1)
    valid_mask = bin_count[jnp.arange(bin_count.shape[0])[:, None], sorted_indices] > 0

    result = jnp.where(
        jnp.arange(x.shape[1])[None, :] < valid_mask.sum(axis=1, keepdims=True),
        sorted_indices[:, : x.shape[1]],
        -1,
    )

    return result


def calc_diff(x: jax.Array, y: jax.Array):
    x, y = x.astype(jnp.float64) + 1, y.astype(jnp.float64) + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


@jax.jit
@jax.shard_map(mesh=mesh, in_specs=PartitionSpec("x"), out_specs=PartitionSpec("x"))
def _test_moe_dispatch_combine(x, scores, topk_weights):
    assert scores.ndim == 2, f"scores must be a 2D array, but got {scores.ndim}"
    assert x.ndim == 2, f"x must be a 2D array, but got {x.ndim}"
    assert topk_weights.ndim == 2, f"topk_weights must be a 2D array, but got {topk_weights.ndim}"

    num_topk = topk_weights.shape[1]
    topk_idx = jax.lax.top_k(scores, num_topk)[1]
    topk_idx = topk_idx.astype(jnp.int64)

    num_tokens, hidden = x.shape
    num_experts = scores.shape[1]

    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.astype(jnp.int64)
    rank_idx = inplace_unique(rank_idx, num_ranks)

    # expert meta
    num_tokens_per_expert = jax.vmap(lambda i: jnp.sum(topk_idx == i))(jnp.arange(num_experts))
    gbl_num_tokens_per_expert = jax.lax.psum(num_tokens_per_expert, "x")

    # dispatch layout
    config = get_dispatch_config()
    (
        _,
        _,
        _,
        _,
        ref_is_token_in_rank,
        ref_num_tokens_per_rank,
        ref_num_tokens_per_expert,
        _,
        _,
        _,
        _,
        _,
    ) = moe_dispatch_p.bind(
        x,
        jnp.array([], dtype=jnp.float32),
        topk_idx,
        topk_weights,
        num_experts=num_experts,
        expert_alignment=1,
        num_worst_tokens=num_tokens * num_ranks,
        num_sms=config.num_sms,
        num_max_nvl_chunked_send_tokens=config.num_max_nvl_chunked_send_tokens,
        num_max_nvl_chunked_recv_tokens=config.num_max_nvl_chunked_recv_tokens,
        num_max_rdma_chunked_send_tokens=config.num_max_rdma_chunked_send_tokens,
        num_max_rdma_chunked_recv_tokens=config.num_max_rdma_chunked_recv_tokens,
    )

    # 2. Rank layout meta
    num_tokens_per_rank = jnp.empty((num_ranks,), dtype=jnp.int32)
    for i in range(num_ranks):
        num_tokens_per_rank = num_tokens_per_rank.at[i].set(jnp.sum((rank_idx == i)))
    gbl_num_tokens_per_rank = jax.lax.psum(num_tokens_per_rank, "x")

    # 3. Test Dispatch
    recv_x, recv_topk_idx, recv_topk_weights, handle = moe_dispatch(
        x, topk_idx=topk_idx, topk_weights=topk_weights, num_experts=num_experts
    )
    rank_prefix_matrix = handle[0]
    recv_topk_weights_copy = jnp.copy(recv_topk_weights)
    amax_recv_topk_weights = jnp.broadcast_to(
        jnp.amax(recv_topk_weights_copy, axis=1, keepdims=True), recv_topk_weights.shape
    )
    check_recv_topk_weights = jax.lax.select(
        jnp.equal(recv_topk_idx, -1), amax_recv_topk_weights, recv_topk_weights_copy
    )

    # 4. Test cached dispatch (must without top-k staffs)
    cached_recv_x, _, _, _ = moe_dispatch(x, handle=handle)

    # 5. Test Combine
    combined_x, combined_topk_weights = moe_combine(
        recv_x, handle=handle, topk_weights=check_recv_topk_weights
    )
    check_combine_x = combined_x.astype(jnp.float32) / jnp.expand_dims(
        ref_is_token_in_rank.sum(axis=1), axis=1
    )
    check_combine_weights = combined_topk_weights / jnp.expand_dims(ref_is_token_in_rank.sum(axis=1), axis=1)

    return (
        [
            (num_tokens_per_expert, ref_num_tokens_per_expert),
            (num_tokens_per_rank, ref_num_tokens_per_rank),
        ],
        [
            (recv_x, rank_prefix_matrix, gbl_num_tokens_per_rank),
            (cached_recv_x, rank_prefix_matrix, gbl_num_tokens_per_rank),
            (check_recv_topk_weights, rank_prefix_matrix, gbl_num_tokens_per_rank),
        ],
        (recv_topk_idx, gbl_num_tokens_per_expert),
        [(check_combine_x, x), (check_combine_weights, topk_weights)],
    )


@pytest.mark.parametrize("num_tokens", [4096])
@pytest.mark.parametrize("hidden", [7168])
@pytest.mark.parametrize("num_topk", [8])
@pytest.mark.parametrize("num_experts", [256])
def test_moe_dispatch_combine(num_tokens, hidden, num_topk, num_experts):
    x, scores, topk_weights = _generate(num_tokens, hidden, num_topk, num_experts)

    dispatch_layout_result, recv_result, recv_topk_idx_result, combine_result = _test_moe_dispatch_combine(
        x, scores, topk_weights
    )

    for base, ref in dispatch_layout_result:
        np.testing.assert_allclose(base, ref)

    def check_dispatch(ref_check_x, ref_rank_prefix_matrix, ref_size_tensor):
        ref_recv_check_x = jnp.reshape(ref_check_x, (num_ranks, num_ranks * num_tokens, -1))
        ref_rank_prefix_matrix = jnp.reshape(ref_rank_prefix_matrix, (num_ranks, num_ranks, num_ranks))
        ref_size_tensor = jnp.reshape(ref_size_tensor, (num_ranks, num_ranks))
        for rank in range(num_ranks):
            check_x = ref_recv_check_x[rank]
            rank_prefix_matrix = ref_rank_prefix_matrix[rank]
            recv_size = ref_size_tensor[rank][rank].item()

            assert jnp.allclose(jnp.amin(check_x[:recv_size], axis=1), jnp.amax(check_x[:recv_size], axis=1))
            check_start = 0
            for i in range(num_ranks):
                check_end = rank_prefix_matrix[i][rank].item()
                assert (check_x[check_start:check_end, :].astype(jnp.int32) - i).sum().item() == 0
                check_start = check_end

    for base, ref, size_tensor in recv_result:
        check_dispatch(base, ref, size_tensor)

    def check_topk_idx(ref_topk_idx, ref_gbl_num_tokens_per_expert):
        ref_recv_topk_idx = jnp.reshape(ref_topk_idx, (num_ranks, num_ranks * num_tokens, num_topk))
        ref_gbl_num_tokens_per_expert = jnp.reshape(ref_gbl_num_tokens_per_expert, (num_ranks, num_experts))

        for rank in range(num_ranks):
            recv_topk_idx = ref_recv_topk_idx[rank]

            assert (
                jnp.equal(recv_topk_idx, -1)
                | ((recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks)))
            ).sum().item() == recv_topk_idx.size

    check_topk_idx(*recv_topk_idx_result)

    def check_combine(ref_combine_x, ref_x, diff_threshold=5e-6):
        ref_combine_x = jnp.reshape(ref_combine_x, (num_ranks, num_tokens, -1))
        ref_x = jnp.reshape(ref_x, (num_ranks, num_tokens, -1))
        for rank in range(num_ranks):
            combine_x = ref_combine_x[rank]
            x = ref_x[rank]
            diff = calc_diff(combine_x, x)
            assert diff < diff_threshold

    for (base, ref), diff_threshold in zip(combine_result, [5e-6, 1e-9]):
        check_combine(base, ref, diff_threshold)


@pytest.mark.parametrize("num_tokens", [4096])
@pytest.mark.parametrize("hidden", [7168])
@pytest.mark.parametrize("num_topk", [8])
@pytest.mark.parametrize("num_experts", [256])
def test_moe_dispatch_combine_backward(num_tokens, hidden, num_topk, num_experts):

    @jax.shard_map(mesh=mesh, in_specs=PartitionSpec("x"), out_specs=PartitionSpec("x"))
    def _test_mode_dispatch_combine_backward(x, scores, topk_weights):
        assert scores.ndim == 2, f"scores must be a 2D array, but got {scores.ndim}"
        assert x.ndim == 2, f"x must be a 2D array, but got {x.ndim}"
        assert topk_weights.ndim == 2, f"topk_weights must be a 2D array, but got {topk_weights.ndim}"

        num_topk = topk_weights.shape[1]
        topk_idx = jax.lax.top_k(scores, num_topk)[1]
        topk_idx = topk_idx.astype(jnp.int64)

        num_experts = scores.shape[1]

        recv_x, _, rect_topk_weights, handle = moe_dispatch(
            x, topk_idx=topk_idx, topk_weights=topk_weights, num_experts=num_experts
        )

        combined_x = moe_combine(recv_x, handle=handle)

        return combined_x, rect_topk_weights

    @jax.jit
    def _test_mode_dispatch_combine_backward_grad_fn(x, scores, topk_weights):
        return jax.vjp(_test_mode_dispatch_combine_backward, x, scores, topk_weights)

    x, scores, topk_weights = _generate(num_tokens, hidden, num_topk, num_experts)

    primals, f_vjp = _test_mode_dispatch_combine_backward_grad_fn(x, scores, topk_weights)
    grad_out = f_vjp(primals)

    jax.debug.print("grad_out: {grad_out}", grad_out=grad_out)
