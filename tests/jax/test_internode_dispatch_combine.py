###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Multi-node (internode) MoE dispatch/combine tests using DeepEP
``per_process`` mode with ep_size > 8.

Requires more than 8 GPUs (typically 2+ nodes of 8 GPUs each).
Each process manages 1 GPU, and rocSHMEM is used for cross-node RDMA.

Run with::

    pytest tests/jax/lax/test_internode_dispatch_combine.py --dist-only
"""

import pytest

from tests.jax.test_utils import JaxMultiProcessTestCase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NUM_TOKENS = 4096
_HIDDEN = 7168
_NUM_TOPK = 8
_EXPERTS_PER_RANK = 32

# ---------------------------------------------------------------------------
# Helpers (top-level for pickling; framework imports are late)
# ---------------------------------------------------------------------------


def _calc_diff(x, y):
    import numpy as np

    x = np.asarray(x, dtype=np.float64) + 1
    y = np.asarray(y, dtype=np.float64) + 1
    denom = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denom
    return float(1 - sim)


def _init_per_process():
    import os

    os.environ["PRIMUS_TURBO_JAX_DEEPEP_MODE"] = "per_process"

    import jax
    import jax.numpy as jnp
    import numpy as np
    import primus_turbo.jax  # noqa: F401

    primus_turbo.jax.initialize()
    return jax, jnp, np


def _generate(rank, world_size):
    import jax
    import jax.numpy as jnp

    num_experts = _EXPERTS_PER_RANK * world_size
    key = jax.random.PRNGKey(rank)

    x = jnp.ones((_NUM_TOKENS, _HIDDEN), dtype=jnp.bfloat16) * rank
    scores = (
        jnp.abs(jax.random.normal(key, (_NUM_TOKENS, num_experts), dtype=jnp.float32))
        + 1
    )
    topk_idx = jax.lax.top_k(scores, _NUM_TOPK)[1].astype(jnp.int32)
    topk_weights = jnp.ones((_NUM_TOKENS, _NUM_TOPK), dtype=jnp.float32) * rank

    return x, topk_idx, topk_weights, num_experts


# ---------------------------------------------------------------------------
# Worker functions
# ---------------------------------------------------------------------------


def _worker_internode_dispatch_combine_fwd_bf16(rank, world_size):
    """BF16 internode forward: dispatch + combine round-trip."""
    import jax.numpy as jnp

    _init_per_process()
    from primus_turbo.jax.lax.moe import moe_combine, moe_dispatch

    x, topk_idx, topk_weights, num_experts = _generate(rank, world_size)

    recv_x, recv_topk_idx, recv_topk_weights, handle = moe_dispatch(
        x, topk_idx, topk_weights, num_experts
    )

    is_token_in_rank = handle[5]  # internode handle index

    combined_x = moe_combine(recv_x, handle)
    assert combined_x.shape == (_NUM_TOKENS, _HIDDEN)

    scale = jnp.expand_dims(is_token_in_rank.sum(axis=1), axis=1)
    normalised = combined_x.astype(jnp.float32) / scale

    diff = _calc_diff(normalised, x)
    assert diff < 5e-6, f"Rank {rank}: internode combine round-trip diff={diff}"


def _worker_internode_dispatch_combine_bwd_bf16(rank, world_size):
    """BF16 internode backward: jax.vjp round-trip gradient check."""
    import jax
    import jax.numpy as jnp

    _init_per_process()
    from primus_turbo.jax.lax.moe import moe_combine, moe_dispatch

    x, topk_idx, topk_weights, num_experts = _generate(rank, world_size)

    def fwd(x):
        recv_x, _, _, handle = moe_dispatch(x, topk_idx, topk_weights, num_experts)
        combined_x = moe_combine(recv_x, handle)
        is_token_in_rank = handle[5]
        scale = jnp.expand_dims(is_token_in_rank.sum(axis=1), axis=1)
        return (combined_x.astype(jnp.float32) / scale).astype(x.dtype)

    output, vjp_fn = jax.vjp(fwd, x)
    (grad_x,) = vjp_fn(output)

    fwd_diff = _calc_diff(output, x)
    assert fwd_diff < 5e-6, f"Rank {rank}: internode forward diff={fwd_diff}"

    bwd_diff = _calc_diff(grad_x, x)
    assert bwd_diff < 5e-6, f"Rank {rank}: internode backward diff={bwd_diff}"


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestInternodeDispatchCombine(JaxMultiProcessTestCase):
    """Multi-node DeepEP MoE dispatch/combine tests (internode, >8 GPUs)."""

    def _skip_if_insufficient_gpus(self):
        import jax

        if jax.local_device_count() <= 8:
            pytest.skip("Need more than 8 GPUs for internode mode")

    @pytest.mark.multigpu
    def test_internode_dispatch_combine_fwd_bf16(self):
        """BF16 forward: internode dispatch + combine round-trip."""
        self._skip_if_insufficient_gpus()
        self.run_multiprocess(_worker_internode_dispatch_combine_fwd_bf16)

    @pytest.mark.multigpu
    def test_internode_dispatch_combine_bwd_bf16(self):
        """BF16 backward: internode jax.vjp round-trip gradient check."""
        self._skip_if_insufficient_gpus()
        self.run_multiprocess(_worker_internode_dispatch_combine_bwd_bf16)
