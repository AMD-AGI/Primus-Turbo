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

from primus_turbo.jax.primitive import ABSTRACT_EVAL_TABLE, IMPL_TABLE, LOWERING_TABLE

# ----------------------------------------
# Step-1: Primitive Define
# ----------------------------------------
moe_combine_p = Primitive("moe_combine")
moe_combine_p.multiple_results = True


# ----------------------------------------
# Step-2: Impl
# ----------------------------------------
IMPL_TABLE[moe_combine_p] = partial(xla.apply_primitive, moe_combine_p)
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
    num_sms: int,
    num_max_nvl_chunked_send_tokens: int,
    num_max_nvl_chunked_recv_tokens: int,
    num_max_rdma_chunked_send_tokens: int,
    num_max_rdma_chunked_recv_tokens: int,
):

    assert x.ndim == 2, f"x must be a 2D array, but got {x.ndim}"
    assert send_head.ndim == 2, f"send_head must be a 2D array, but got {send_head.ndim}"

    num_recv_tokens = send_head.shape[0]
    _, hidden_size = x.shape
    recv_x = ShapedArray((num_recv_tokens, hidden_size), x.dtype)
    shape = (num_recv_tokens, topk_weights.shape[1]) if topk_weights.size > 0 else topk_weights.shape
    recv_topk_weights = ShapedArray(shape, topk_weights.dtype)
    return recv_x, recv_topk_weights


ABSTRACT_EVAL_TABLE[moe_combine_p] = _moe_combine_abstract_eval

# ----------------------------------------
# Step-4: JIT Lowering
# ----------------------------------------
LOWERING_TABLE[moe_combine_p] = jax.ffi.ffi_lowering("moe_combine")
# ----------------------------------------
# Step-5: batching
# ----------------------------------------
# TODO


__all__ = ["moe_combine_p"]
