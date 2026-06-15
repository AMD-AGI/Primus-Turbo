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

NUM_MAX_NVL_PEERS = deep_ep_runtime.NUM_MAX_NVL_PEERS

# ----------------------------------------
# Step-1: Primitive Define
# ----------------------------------------
moe_dispatch_p = Primitive("moe_dispatch")
moe_cached_dispatch_p = Primitive("moe_cached_dispatch")
moe_cached_dispatch_p.multiple_results = True
moe_dispatch_p.multiple_results = True

moe_internode_dispatch_p = Primitive("moe_internode_dispatch")
moe_internode_dispatch_p.multiple_results = True
moe_internode_cached_dispatch_p = Primitive("moe_internode_cached_dispatch")
moe_internode_cached_dispatch_p.multiple_results = True


# ----------------------------------------
# Step-2: Impl
# ----------------------------------------
IMPL_TABLE[moe_dispatch_p] = partial(xla.apply_primitive, moe_dispatch_p)
IMPL_TABLE[moe_cached_dispatch_p] = partial(xla.apply_primitive, moe_cached_dispatch_p)
IMPL_TABLE[moe_internode_dispatch_p] = partial(xla.apply_primitive, moe_internode_dispatch_p)
IMPL_TABLE[moe_internode_cached_dispatch_p] = partial(xla.apply_primitive, moe_internode_cached_dispatch_p)
# ----------------------------------------
# Step-3: Abstract eval
# ----------------------------------------


def _get_recv_x_scale_shape(x_scales: jnp.ndarray, num_worst_tokens: int) -> ShapedArray:
    if x_scales.size > 0:
        if x_scales.ndim == 1:
            return ShapedArray((num_worst_tokens,), jnp.float32)
        else:
            return ShapedArray((num_worst_tokens, x_scales.shape[1]), jnp.float32)
    else:
        return ShapedArray(x_scales.shape, jnp.float32)


def _get_num_rdma_ranks(ep_size: int) -> int:
    assert (
        ep_size % NUM_MAX_NVL_PEERS == 0
    ), f"internode ep_size must be divisible by NUM_MAX_NVL_PEERS={NUM_MAX_NVL_PEERS}, got {ep_size}"
    return ep_size // NUM_MAX_NVL_PEERS


def _moe_dispatch_abstract_eval(
    x: jnp.ndarray,
    x_scales,
    topk_idx,
    topk_weights,
    num_experts: int,
    expert_alignment: int,
    num_worst_tokens: int,
    ep_size: int,
    launch_mode: int,
    num_sms: int,
    num_max_nvl_chunked_send_tokens: int,
    num_max_nvl_chunked_recv_tokens: int,
    num_max_rdma_chunked_send_tokens: int,
    num_max_rdma_chunked_recv_tokens: int,
):
    del expert_alignment, launch_mode, num_max_nvl_chunked_send_tokens
    del num_max_nvl_chunked_recv_tokens, num_max_rdma_chunked_send_tokens
    del num_max_rdma_chunked_recv_tokens

    assert x.ndim == 2, "x must be a 2D array, but got {}".format(x.ndim)
    assert topk_idx.ndim == 2, "topk_idx must be a 2D array, but got {}".format(topk_idx.ndim)

    num_ranks = ep_size

    num_tokens, hidden_size = x.shape
    num_topk = topk_idx.shape[1]
    num_channels = num_sms // 2
    recv_x_scales = _get_recv_x_scale_shape(x_scales, num_worst_tokens)
    recv_x = ShapedArray((num_worst_tokens, hidden_size), x.dtype)
    recv_topk_idx = ShapedArray((num_worst_tokens, num_topk), jnp.int32)
    recv_topk_weights = ShapedArray((num_worst_tokens, num_topk), jnp.float32)
    is_token_in_rank = ShapedArray((num_tokens, num_ranks), jnp.bool_)
    num_tokens_per_rank = ShapedArray((num_ranks,), jnp.int32)
    num_tokens_per_expert = ShapedArray((num_experts,), jnp.int32)
    rank_prefix_matrix = ShapedArray((num_ranks, num_ranks), jnp.int32)
    channel_prefix_matrix = ShapedArray((num_ranks, num_channels), jnp.int32)
    recv_channel_prefix_matrix = ShapedArray((num_ranks, num_channels), jnp.int32)
    recv_src_idx = ShapedArray((num_worst_tokens,), jnp.int32)
    send_head = ShapedArray((num_tokens, num_ranks), jnp.int32)

    return (
        recv_x,
        recv_x_scales,
        recv_topk_idx,
        recv_topk_weights,
        is_token_in_rank,
        num_tokens_per_rank,
        num_tokens_per_expert,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        recv_src_idx,
        send_head,
    )


def _moe_cached_dispatch_abstract_eval(
    x: jnp.ndarray,
    x_scales,
    is_token_in_rank,
    rank_prefix_matrix,
    channel_prefix_matrix,
    num_recv_tokens: int,
    expert_alignment: int,
    num_worst_tokens: int,
    ep_size: int,
    launch_mode: int,
    num_sms: int,
    num_max_nvl_chunked_send_tokens: int,
    num_max_nvl_chunked_recv_tokens: int,
    num_max_rdma_chunked_send_tokens: int,
    num_max_rdma_chunked_recv_tokens: int,
):
    del is_token_in_rank, rank_prefix_matrix, channel_prefix_matrix, expert_alignment
    del num_worst_tokens, launch_mode, num_max_nvl_chunked_send_tokens
    del num_max_nvl_chunked_recv_tokens, num_max_rdma_chunked_send_tokens
    del num_max_rdma_chunked_recv_tokens
    assert x.ndim == 2, "x must be a 2D array, but got {}".format(x.ndim)
    num_tokens, hidden_size = x.shape
    num_channels = num_sms // 2

    num_ranks = ep_size

    recv_x = ShapedArray((num_recv_tokens, hidden_size), x.dtype)
    recv_x_scales = _get_recv_x_scale_shape(x_scales, num_recv_tokens)
    recv_channel_prefix_matrix = ShapedArray((num_ranks, num_channels), jnp.int32)
    recv_src_idx = ShapedArray((num_recv_tokens,), jnp.int32)
    send_head = ShapedArray((num_tokens, num_ranks), jnp.int32)
    return recv_x, recv_x_scales, recv_channel_prefix_matrix, recv_src_idx, send_head


ABSTRACT_EVAL_TABLE[moe_dispatch_p] = _moe_dispatch_abstract_eval
ABSTRACT_EVAL_TABLE[moe_cached_dispatch_p] = _moe_cached_dispatch_abstract_eval


def _moe_internode_dispatch_abstract_eval(
    x: jnp.ndarray,
    x_scales,
    topk_idx,
    topk_weights,
    num_experts: int,
    expert_alignment: int,
    num_worst_tokens: int,
    ep_size: int,
    launch_mode: int,
    num_sms: int,
    num_max_nvl_chunked_send_tokens: int,
    num_max_nvl_chunked_recv_tokens: int,
    num_max_rdma_chunked_send_tokens: int,
    num_max_rdma_chunked_recv_tokens: int,
    source_meta_bytes: int,
):
    del expert_alignment, launch_mode, num_max_nvl_chunked_send_tokens
    del num_max_nvl_chunked_recv_tokens, num_max_rdma_chunked_send_tokens
    del num_max_rdma_chunked_recv_tokens

    assert x.ndim == 2
    assert topk_idx.ndim == 2

    num_ranks = ep_size
    num_rdma_ranks = _get_num_rdma_ranks(num_ranks)
    num_tokens, hidden_size = x.shape
    num_topk = topk_idx.shape[1]
    num_channels = num_sms // 2
    recv_x_scales = _get_recv_x_scale_shape(x_scales, num_worst_tokens)

    return (
        ShapedArray((num_worst_tokens, hidden_size), x.dtype),  # recv_x
        recv_x_scales,  # recv_x_scales
        ShapedArray((num_worst_tokens, num_topk), jnp.int32),  # recv_topk_idx
        ShapedArray((num_worst_tokens, num_topk), jnp.float32),  # recv_topk_weights
        ShapedArray((num_tokens, num_ranks), jnp.bool_),  # is_token_in_rank
        ShapedArray((num_ranks,), jnp.int32),  # num_tokens_per_rank
        ShapedArray((num_rdma_ranks,), jnp.int32),  # num_tokens_per_rdma_rank
        ShapedArray((num_experts,), jnp.int32),  # num_tokens_per_expert
        ShapedArray((num_rdma_ranks, num_channels), jnp.int32),  # rdma_channel_prefix_matrix
        ShapedArray((num_rdma_ranks,), jnp.int32),  # recv_rdma_rank_prefix_sum
        ShapedArray((num_ranks, num_channels), jnp.int32),  # gbl_channel_prefix_matrix
        ShapedArray((num_ranks,), jnp.int32),  # recv_gbl_rank_prefix_sum
        ShapedArray((num_worst_tokens, source_meta_bytes), jnp.uint8),  # recv_src_meta
        ShapedArray((num_rdma_ranks, num_channels), jnp.int32),  # recv_rdma_channel_prefix_matrix
        ShapedArray((num_ranks, num_channels), jnp.int32),  # recv_gbl_channel_prefix_matrix
        ShapedArray((num_tokens, num_rdma_ranks), jnp.int32),  # send_rdma_head
        ShapedArray((num_worst_tokens, NUM_MAX_NVL_PEERS), jnp.int32),  # send_nvl_head
    )


def _moe_internode_cached_dispatch_abstract_eval(
    x: jnp.ndarray,
    x_scales,
    is_token_in_rank,
    cached_rdma_channel_prefix_matrix,
    cached_recv_rdma_rank_prefix_sum,
    cached_gbl_channel_prefix_matrix,
    cached_recv_gbl_rank_prefix_sum,
    num_recv_tokens: int,
    num_rdma_recv_tokens: int,
    expert_alignment: int,
    num_worst_tokens: int,
    ep_size: int,
    launch_mode: int,
    num_sms: int,
    num_max_nvl_chunked_send_tokens: int,
    num_max_nvl_chunked_recv_tokens: int,
    num_max_rdma_chunked_send_tokens: int,
    num_max_rdma_chunked_recv_tokens: int,
):
    del is_token_in_rank, cached_rdma_channel_prefix_matrix
    del cached_recv_rdma_rank_prefix_sum, cached_gbl_channel_prefix_matrix
    del cached_recv_gbl_rank_prefix_sum, expert_alignment, num_worst_tokens
    del launch_mode, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens
    del num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens

    assert x.ndim == 2
    num_tokens, hidden_size = x.shape
    num_ranks = ep_size
    num_rdma_ranks = _get_num_rdma_ranks(num_ranks)
    num_channels = num_sms // 2
    recv_x_scales = _get_recv_x_scale_shape(x_scales, num_recv_tokens)

    return (
        ShapedArray((num_recv_tokens, hidden_size), x.dtype),  # recv_x
        recv_x_scales,  # recv_x_scales
        ShapedArray((num_rdma_ranks, num_channels), jnp.int32),  # recv_rdma_channel_prefix_matrix
        ShapedArray((num_ranks, num_channels), jnp.int32),  # recv_gbl_channel_prefix_matrix
        ShapedArray((num_tokens, num_rdma_ranks), jnp.int32),  # send_rdma_head
        ShapedArray((num_rdma_recv_tokens, NUM_MAX_NVL_PEERS), jnp.int32),  # send_nvl_head
    )


ABSTRACT_EVAL_TABLE[moe_internode_dispatch_p] = _moe_internode_dispatch_abstract_eval
ABSTRACT_EVAL_TABLE[moe_internode_cached_dispatch_p] = _moe_internode_cached_dispatch_abstract_eval


# ----------------------------------------
# Step-4: JIT Lowering
# ----------------------------------------
def _moe_dispatch_lowering(ctx, *args, **kwargs):
    target = deep_ep_runtime.get_target_name("moe_dispatch", launch_mode=kwargs.get("launch_mode"))
    return jax.ffi.ffi_lowering(target, has_side_effect=True)(ctx, *args, **kwargs)


def _moe_cached_dispatch_lowering(ctx, *args, **kwargs):
    target = deep_ep_runtime.get_target_name("moe_cached_dispatch", launch_mode=kwargs.get("launch_mode"))
    return jax.ffi.ffi_lowering(target, has_side_effect=True)(ctx, *args, **kwargs)


LOWERING_TABLE[moe_dispatch_p] = _moe_dispatch_lowering
LOWERING_TABLE[moe_cached_dispatch_p] = _moe_cached_dispatch_lowering


def _moe_internode_dispatch_lowering(ctx, *args, **kwargs):
    kwargs = dict(kwargs)
    # Used only by abstract eval to shape recv_src_meta; C++ derives the size itself.
    kwargs.pop("source_meta_bytes", None)
    return jax.ffi.ffi_lowering("moe_internode_dispatch_per_process", has_side_effect=True)(
        ctx, *args, **kwargs
    )


def _moe_internode_cached_dispatch_lowering(ctx, *args, **kwargs):
    return jax.ffi.ffi_lowering("moe_internode_cached_dispatch_per_process", has_side_effect=True)(
        ctx, *args, **kwargs
    )


LOWERING_TABLE[moe_internode_dispatch_p] = _moe_internode_dispatch_lowering
LOWERING_TABLE[moe_internode_cached_dispatch_p] = _moe_internode_cached_dispatch_lowering
# ----------------------------------------
# Step-5: batching
# ----------------------------------------
# Cross-rank IPC primitives: their per-rank outputs (12 for moe_dispatch, 5 for
# moe_cached_dispatch) cannot be merged across the batch axis, so the plain rule
# scans per microbatch.  See primus_turbo.jax.primitive._batching for the shared
# scan / SPMD-aware factory and the pipeline-parallel rationale.
from jax.interpreters import batching

from primus_turbo.jax.primitive._batching import make_ipc_batchers

(
    batching.primitive_batchers[moe_dispatch_p],
    batching.fancy_primitive_batchers[moe_dispatch_p],
) = make_ipc_batchers(moe_dispatch_p, num_outputs=12)

(
    batching.primitive_batchers[moe_cached_dispatch_p],
    batching.fancy_primitive_batchers[moe_cached_dispatch_p],
) = make_ipc_batchers(moe_cached_dispatch_p, num_outputs=5)


__all__ = [
    "moe_dispatch_p",
    "moe_cached_dispatch_p",
    "moe_internode_dispatch_p",
    "moe_internode_cached_dispatch_p",
]
