import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import functools

from primus_turbo.jax.lax.moe_dispatch import moe_dispatch
key = jax.random.key(1000)
P = jax.sharding.PartitionSpec

shards = jax.device_count()


dtype = jnp.bfloat16

mesh = jax.make_mesh(
    (shards,), ("x",), axis_types=(jax.sharding.AxisType.Explicit,)
)


num_tokens = 4096
hidden = 7168
num_topk = 8
num_experts = 256


@jax.jit
@functools.partial(jax.shard_map, mesh=mesh, in_specs=P(None), out_specs=P(None))
def ref_fn(x, topk_idx, topk_weights):
    rank = jax.lax.axis_index('x')
    x = x * rank
    topk_weights = topk_weights * rank
    recv_x, recv_topk_idx, recv_topk_weights, handle = moe_dispatch(x, num_experts,
                                                                    topk_idx=topk_idx, topk_weights=topk_weights)

    return recv_x, recv_topk_idx, recv_topk_weights, handle


x = jnp.ones((num_tokens, hidden), dtype=jnp.bfloat16)
scores = jnp.abs(jax.random.normal(
    key, (num_tokens, num_experts), dtype=jnp.float32)) + 1
topk_idx = jax.lax.top_k(scores, num_topk)[1]
topk_idx = topk_idx.astype(jnp.int64)
topk_weights = jnp.ones((num_tokens, num_topk), dtype=jnp.float32)
recv_x = ref_fn(
    x, topk_idx, topk_weights)

# (
#     rank_prefix_matrix,
#     channel_prefix_matrix,
#     recv_channel_prefix_matrix,
#     recv_src_idx,
#     is_token_in_rank,
#     send_head,
# ) = handle
# print(rank_prefix_matrix, channel_prefix_matrix, is_token_in_rank)
