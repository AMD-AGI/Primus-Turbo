###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from functools import partial
from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from primus_turbo.jax.lax.moe_dispatch import _moe_dispatch_fwd
from primus_turbo.jax.primitive.moe_combine import moe_combine_p

from .moe_utils import Config

__all__ = ["get_combine_config", "moe_combine"]


_default_num_sms = 32
num_ranks = 8


def moe_combine(
    x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    handle: Optional[Tuple] = None,
    topk_weights: Optional[jnp.ndarray] = None,
    bias: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]] = None,
    config: Optional[Config] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    return _moe_combine(x, handle, topk_weights, bias, config)


@partial[Any](jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
def _moe_combine(
    x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    handle: Optional[Tuple] = None,
    topk_weights: Optional[jnp.ndarray] = None,
    bias: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]] = None,
    config: Optional[Config] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    out, _ = _moe_combine_fwd(x, handle, topk_weights, bias, config)
    return out


def get_combine_config(num_ranks: int = 8) -> Config:
    """
    Get a recommended dispatch config.

    Argument:
        num_ranks: the number of ranks.

    Returns:
        config: the recommended config.
    """
    global _default_num_sms
    return Config(_default_num_sms, 4, 256, 6, 128)


def _moe_combine_fwd(
    x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    handle: Optional[Tuple] = None,
    topk_weights: Optional[jnp.ndarray] = None,
    bias: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]] = None,
    config: Optional[Config] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:

    # default config
    config = get_combine_config(num_ranks) if config is None else config

    # unpack bias
    bias_0, bias_1 = None, None
    if isinstance(bias, jnp.ndarray):
        bias_0 = bias
    elif isinstance(bias, tuple):
        assert len(bias) == 2
        bias_0, bias_1 = bias

    rank_prefix_matrix, _, channel_prefix_matrix, src_idx, is_recv_token_in_rank, send_head = handle

    recv_x, recv_topk_weights = moe_combine_p.bind(
        x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head
    )
    return (recv_x, recv_topk_weights), handle


# input: nondiff_argnums, ctx, grad
# output: input grad
def _moe_combine_bwd(handle, topk_weights, bias, config, ctx, grad_x):
    (recv_grad_x, _, _, _), _ = _moe_dispatch_fwd(grad_x, handle=ctx)
    return recv_grad_x


_moe_combine.defvjp(_moe_combine_fwd, _moe_combine_bwd)
