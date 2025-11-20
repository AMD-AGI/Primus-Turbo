import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax

from primus_turbo.jax.lax.moe_dispatch import moe_dispatch
key = jax.random.key(1000)

Explicit = jax.sharding.AxisType.Explicit
mesh = jax.make_mesh((8, 1), ('i', 'j'), axis_types=(Explicit,) * 2)


@jax.jit
@jax.shard_map(mesh=mesh, out_specs=jax.sharding.PartitionSpec('i', 'j'))
def test_function(x, topk_idx, topk_weights):
    out = moe_dispatch(x, 256, topk_idx=topk_idx, topk_weights=topk_weights,
                 rank=0, num_ranks=8)
    return out

x = np.arange(16).reshape(4, 4)
topk_idx = jnp.ones(x.shape, dtype=jnp.int64)
topk_weights = jnp.ones(x.shape, dtype=jnp.float32)
y = test_function(x, topk_idx, topk_weights)
print(y)
