###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import partial

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import xla

from primus_turbo.jax.deep_ep import runtime as deep_ep_runtime
from primus_turbo.jax.primitive import ABSTRACT_EVAL_TABLE, IMPL_TABLE, LOWERING_TABLE

# ----------------------------------------
# Step-1: Primitive Define
# ----------------------------------------
moe_combine_p = Primitive("moe_combine")
moe_combine_p.multiple_results = True

moe_internode_combine_p = Primitive("moe_internode_combine")
moe_internode_combine_p.multiple_results = True


# ----------------------------------------
# Step-2: Impl
# ----------------------------------------
IMPL_TABLE[moe_combine_p] = partial(xla.apply_primitive, moe_combine_p)
IMPL_TABLE[moe_internode_combine_p] = partial(xla.apply_primitive, moe_internode_combine_p)
# ----------------------------------------
# Step-3: Abstract eval
# ----------------------------------------


def _moe_combine_abstract_eval(
    x: jnp.ndarray,
    topk_weights: jnp.ndarray,
    bias_0: jnp.ndarray,
    bias_1: jnp.ndarray,
    src_idx: jnp.ndarray,
    rank_prefix_matrix: jnp.ndarray,
    channel_prefix_matrix: jnp.ndarray,
    send_head: jnp.ndarray,
    ep_size: int,
    launch_mode: int,
    num_sms: int,
    num_max_nvl_chunked_send_tokens: int,
    num_max_nvl_chunked_recv_tokens: int,
    num_max_rdma_chunked_send_tokens: int,
    num_max_rdma_chunked_recv_tokens: int,
):
    del bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix
    del ep_size, launch_mode, num_sms, num_max_nvl_chunked_send_tokens
    del num_max_nvl_chunked_recv_tokens, num_max_rdma_chunked_send_tokens
    del num_max_rdma_chunked_recv_tokens

    assert x.ndim == 2, f"x must be a 2D array, but got {x.ndim}"
    assert send_head.ndim == 2, f"send_head must be a 2D array, but got {send_head.ndim}"

    num_recv_tokens = send_head.shape[0]
    _, hidden_size = x.shape
    recv_x = ShapedArray((num_recv_tokens, hidden_size), x.dtype)
    shape = (num_recv_tokens, topk_weights.shape[1]) if topk_weights.size > 0 else topk_weights.shape
    recv_topk_weights = ShapedArray(shape, topk_weights.dtype)
    return recv_x, recv_topk_weights


ABSTRACT_EVAL_TABLE[moe_combine_p] = _moe_combine_abstract_eval


def _moe_internode_combine_abstract_eval(
    x: jnp.ndarray,
    topk_weights: jnp.ndarray,
    src_meta: jnp.ndarray,
    is_combined_token_in_rank: jnp.ndarray,
    rdma_channel_prefix_matrix: jnp.ndarray,
    rdma_rank_prefix_sum: jnp.ndarray,
    gbl_channel_prefix_matrix: jnp.ndarray,
    gbl_rank_prefix_sum: jnp.ndarray,
    combined_rdma_head: jnp.ndarray,
    combined_nvl_head: jnp.ndarray,
    ep_size: int,
    launch_mode: int,
    num_sms: int,
    num_max_nvl_chunked_send_tokens: int,
    num_max_nvl_chunked_recv_tokens: int,
    num_max_rdma_chunked_send_tokens: int,
    num_max_rdma_chunked_recv_tokens: int,
):
    del src_meta, rdma_channel_prefix_matrix, rdma_rank_prefix_sum
    del gbl_channel_prefix_matrix, gbl_rank_prefix_sum
    del combined_rdma_head, combined_nvl_head
    del ep_size, launch_mode, num_sms
    del num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens
    del num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens

    assert x.ndim == 2

    num_combined_tokens = is_combined_token_in_rank.shape[0]
    _, hidden_size = x.shape
    combined_x = ShapedArray((num_combined_tokens, hidden_size), x.dtype)
    shape = (num_combined_tokens, topk_weights.shape[1]) if topk_weights.size > 0 else topk_weights.shape
    combined_topk_weights = ShapedArray(shape, topk_weights.dtype)
    return combined_x, combined_topk_weights


ABSTRACT_EVAL_TABLE[moe_internode_combine_p] = _moe_internode_combine_abstract_eval


# ----------------------------------------
# Step-4: JIT Lowering
# ----------------------------------------
def _moe_combine_lowering(ctx, *args, **kwargs):
    target = deep_ep_runtime.get_target_name("moe_combine", launch_mode=kwargs.get("launch_mode"))
    return jax.ffi.ffi_lowering(target, has_side_effect=True)(ctx, *args, **kwargs)


LOWERING_TABLE[moe_combine_p] = _moe_combine_lowering


def _moe_internode_combine_lowering(ctx, *args, **kwargs):
    return jax.ffi.ffi_lowering("moe_internode_combine_per_process", has_side_effect=True)(
        ctx, *args, **kwargs
    )


LOWERING_TABLE[moe_internode_combine_p] = _moe_internode_combine_lowering
# ----------------------------------------
# Step-5: batching
# ----------------------------------------
# moe_combine is a cross-rank IPC primitive (same family as moe_dispatch).  Its
# 2 outputs (recv_x, recv_topk_weights) carry per-rank state that cannot be
# merged across the batch axis.  See primus_turbo.jax.primitive._batching for
# the shared factory and the pipeline-parallel rationale.
from jax.interpreters import batching

from primus_turbo.jax.primitive._batching import make_ipc_batchers

(
    batching.primitive_batchers[moe_combine_p],
    batching.fancy_primitive_batchers[moe_combine_p],
) = make_ipc_batchers(moe_combine_p, num_outputs=2)


__all__ = ["moe_combine_p", "moe_internode_combine_p"]
