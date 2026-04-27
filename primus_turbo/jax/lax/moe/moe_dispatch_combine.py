###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp

from primus_turbo.jax.primitive.moe.moe_combine import moe_combine_p
from primus_turbo.jax.primitive.moe.moe_dispatch import (
    moe_cached_dispatch_p,
    moe_dispatch_p,
)

from primus_turbo.jax._C import Config

__all__ = [
    "Config",
    "get_dispatch_config",
    "get_combine_config",
    "set_default_num_sms",
    "moe_dispatch",
    "moe_combine",
]


_default_num_sms = 64


def set_default_num_sms(num_sms: int):
    """Set the default number of SMS.
    Args:
        num_sms (int): The number of SMS.

    Note: 64 or 80 is recommended for single node dispatch/combine.
    """
    global _default_num_sms
    _default_num_sms = num_sms


def get_dispatch_config() -> Config:
    """
    Get a recommended dispatch config.
    Returns:
        config: the recommended config.
    """
    global _default_num_sms
    num_ranks = jax.local_device_count()
    assert num_ranks <= 8, "not support internode"
    config_map = {
        2: Config(_default_num_sms, 24, 512, 6, 512),
        4: Config(_default_num_sms, 6, 512, 6, 512),
        8: Config(_default_num_sms, 6, 512, 6, 512),
        16: Config(_default_num_sms, 36, 512, 20, 512),
        24: Config(_default_num_sms, 8, 512, 32, 512),
        32: Config(_default_num_sms, 32, 512, 32, 512),
        64: Config(_default_num_sms, 20, 512, 28, 512),
        128: Config(_default_num_sms, 20, 560, 32, 512),
        144: Config(_default_num_sms, 32, 720, 12, 512),
        160: Config(_default_num_sms, 28, 720, 12, 512),
    }
    assert num_ranks in config_map, f"Unsupported number of EP ranks: {num_ranks}"
    return config_map[num_ranks]


def get_combine_config() -> Config:
    """
    Get a recommended combine config.
    Returns:
        config: the recommended config.
    """
    global _default_num_sms
    num_ranks = jax.local_device_count()
    assert num_ranks <= 8, "not support internode"
    config_map = {
        2: Config(_default_num_sms, 10, 512, 6, 512),
        4: Config(_default_num_sms, 9, 512, 6, 512),
        8: Config(_default_num_sms, 4, 512, 6, 512),
        16: Config(_default_num_sms, 4, 512, 12, 512),
        24: Config(_default_num_sms, 1, 512, 8, 512),
        32: Config(_default_num_sms, 1, 512, 8, 512),
        64: Config(_default_num_sms, 1, 512, 20, 512),
        128: Config(_default_num_sms, 1, 560, 12, 512),
        144: Config(_default_num_sms, 2, 720, 8, 512),
        160: Config(_default_num_sms, 2, 720, 8, 512),
    }
    assert num_ranks in config_map, f"Unsupported number of EP ranks: {num_ranks}"
    return config_map[num_ranks]


def moe_dispatch(
    x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    topk_idx: jnp.ndarray,
    topk_weights: jnp.ndarray,
    num_experts: int,
    expert_alignment: int = 1,
    config: Optional[Config] = None,
) -> Tuple[Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], jnp.ndarray, jnp.ndarray, Tuple]:
    """
    Dispatch tokens to their selected experts in a Mixture of Experts (MoE) model.

    In MoE models, tokens dynamically select their top-k experts based on routing scores. This function
    executes the all-to-all communication required to dispatch tokens from all ranks to the specific ranks
    hosting their chosen experts. It leverages both intra-node communication (e.g., NVLink) and inter-node
    communication (e.g., RDMA) to facilitate efficient expert parallelism across multiple GPUs and nodes.

    Key functionalities of the dispatch process include:
    - Routing tokens based on top-k expert assignments.
    - Performing cross-rank communication to deliver tokens to their designated expert locations.
    - Computing and caching communication layouts to optimize performance.
    - Supporting both standard precision (bfloat16) and low precision (float8) data types.

    Args:
        x: A `jnp.ndarray` or a tuple of `jnp.ndarray`s.
            - If a single array, it must have a shape of `[num_tokens, hidden]` and a dtype of `jnp.bfloat16`.
            - If a tuple, the first element must be `[num_tokens, hidden]` with dtype `jnp.float8_e4m3fn`,
              and the second element must be `[num_tokens, hidden // 128]` (where hidden is divisible by 128)
              with dtype `jnp.float32`.
        topk_idx: A `jnp.ndarray` of shape `[num_tokens, num_topk]` and dtype `jnp.int32`, representing
            the expert indices selected by each token. A value of `-1` indicates no selection.
        topk_weights: A `jnp.ndarray` of shape `[num_tokens, num_topk]` and dtype `jnp.float32`,
            representing the routing weights for each token's selected experts.
        num_experts: The total number of experts across all ranks.
        expert_alignment: An integer specifying the alignment for the number of tokens received by each local expert.
        config: An optional performance tuning configuration. If `None`, the default configuration from
            `get_dispatch_config()` is used.

    Returns:
        recv_x: The received tokens, matching the type and structure of the input `x`, but with the
            token dimension updated to reflect the total number of received tokens.
        recv_topk_idx: The received expert indices with shape `[num_recv_tokens, num_topk]`, or `None`
            if `handle` was provided.
        recv_topk_weights: The received expert weights with shape `[num_recv_tokens, num_topk]`, or `None`
            if `handle` was provided.
        handle: A communication handle containing the computed layout information (e.g., `rank_prefix_matrix`,
            `channel_prefix_matrix`, `recv_channel_prefix_matrix`, `recv_src_idx`, `is_token_in_rank`,
            `send_head`). This handle can be passed to subsequent calls to bypass layout recomputation.
    """
    return _moe_dispatch(x, topk_idx, topk_weights, num_experts, expert_alignment, config)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _moe_dispatch(
    x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    topk_idx: jnp.ndarray,
    topk_weights: jnp.ndarray,
    num_experts: int,
    expert_alignment: int = 1,
    config: Optional[Config] = None,
) -> Tuple[Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], jnp.ndarray, jnp.ndarray, Tuple]:
    return _moe_dispatch_impl(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        expert_alignment=expert_alignment,
        num_experts=num_experts,
        config=config,
    )


def _moe_dispatch_impl(
    x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    handle: Optional[Tuple] = None,
    topk_idx: Optional[jnp.ndarray] = None,
    topk_weights: Optional[jnp.ndarray] = None,
    expert_alignment: int = 1,
    num_experts: Optional[int] = None,
    config: Optional[Config] = None,
) -> Tuple[
    Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    Optional[jnp.ndarray],
    Optional[jnp.ndarray],
    Optional[Tuple],
]:
    if isinstance(x, tuple):
        x, x_scales = x
    else:
        x_scales = jnp.array([], dtype=jnp.float32)

    assert x.ndim == 2, "x must be a 2D array, but got {}".format(x.ndim)
    num_tokens, _ = x.shape
    num_worst_tokens = num_tokens * jax.local_device_count()
    # default config
    config = get_dispatch_config() if config is None else config

    if handle is not None:
        assert topk_idx is None and topk_weights is None
        (
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            is_token_in_rank,
            send_head,
        ) = handle
        num_recv_tokens = recv_src_idx.shape[0]
        recv_x, recv_x_scales, _, _, _ = moe_cached_dispatch_p.bind(
            x,
            x_scales,
            is_token_in_rank,
            rank_prefix_matrix,
            channel_prefix_matrix,
            num_recv_tokens=num_recv_tokens,
            expert_alignment=expert_alignment,
            num_worst_tokens=num_worst_tokens,
            num_sms=config.num_sms,
            num_max_nvl_chunked_send_tokens=config.num_max_nvl_chunked_send_tokens,
            num_max_nvl_chunked_recv_tokens=config.num_max_nvl_chunked_recv_tokens,
            num_max_rdma_chunked_send_tokens=config.num_max_rdma_chunked_send_tokens,
            num_max_rdma_chunked_recv_tokens=config.num_max_rdma_chunked_recv_tokens,
        )
        recv = (recv_x, recv_x_scales) if x_scales.size > 0 else recv_x
        return recv, None, None, None
    else:
        assert topk_idx is not None and topk_weights is not None
        assert num_experts is not None

        (
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
        ) = moe_dispatch_p.bind(
            x,
            x_scales,
            topk_idx,
            topk_weights,
            num_experts=num_experts,
            expert_alignment=expert_alignment,
            num_worst_tokens=num_worst_tokens,
            num_sms=config.num_sms,
            num_max_nvl_chunked_send_tokens=config.num_max_nvl_chunked_send_tokens,
            num_max_nvl_chunked_recv_tokens=config.num_max_nvl_chunked_recv_tokens,
            num_max_rdma_chunked_send_tokens=config.num_max_rdma_chunked_send_tokens,
            num_max_rdma_chunked_recv_tokens=config.num_max_rdma_chunked_recv_tokens,
        )

        handle = (
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            is_token_in_rank,
            send_head,
        )
        return (
            (recv_x, recv_x_scales) if x_scales.size > 0 else recv_x,
            recv_topk_idx,
            recv_topk_weights,
            handle,
        )


def _moe_dispatch_fwd(
    x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    topk_idx: jnp.ndarray,
    topk_weights: jnp.ndarray,
    num_experts: int,
    expert_alignment: int = 1,
    config: Optional[Config] = None,
):
    recv_x, recv_topk_idx, recv_topk_weights, handle = _moe_dispatch_impl(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        expert_alignment=expert_alignment,
        config=config,
    )

    ctx = (handle,)
    return (recv_x, recv_topk_idx, recv_topk_weights, handle), ctx


def _moe_dispatch_bwd(num_experts, expert_alignment, config, ctx, grad_output):
    (handle,) = ctx
    grad_x, _, grad_topk_weights, _ = grad_output

    if isinstance(grad_x, tuple):
        grad_x_main, _ = grad_x
        grad_x_main, grad_topk_weights = _moe_combine_impl(
            grad_x_main, handle, topk_weights=grad_topk_weights, config=config
        )
        grad_x = (grad_x_main, None)
    else:
        grad_x, grad_topk_weights = _moe_combine_impl(
            grad_x, handle, topk_weights=grad_topk_weights, config=config
        )

    return grad_x, None, grad_topk_weights


_moe_dispatch.defvjp(_moe_dispatch_fwd, _moe_dispatch_bwd)


def moe_combine(
    x: jnp.ndarray,
    handle: Tuple,
    config: Optional[Config] = None,
) -> jnp.ndarray:
    """
    Combine (reduce) tokens from different experts back to their original ranks in a Mixture of Experts (MoE) model.

    After tokens are processed by their selected experts (via moe_dispatch), this function performs the reverse
    all-to-all communication to gather and aggregate results back to the original token locations. The aggregation
    is performed via addition across all ranks that processed each token. Supports both intranode communication
    via NVLink and internode communication via RDMA.

    This is the complement operation to moe_dispatch and must be called with the handle returned from moe_dispatch
    to ensure correct routing back to the original token positions.

    Arguments:
        x: `[num_recv_tokens, hidden]` with `jnp.bfloat16` or `jnp.float8_e4m3fn`, the expert output tokens
            to send back for reducing to their original ranks.
        handle: a required communication handle obtained from the moe_dispatch function, containing layout
            information (rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx,
            is_token_in_rank, send_head) needed for the reverse communication.
        config: the performance tuning config. If None, will use the default config from get_combine_config().

    Returns:
        combined_x: the reduced tokens from all expert ranks, gathered back to original token positions with
            shape `[num_tokens, hidden]`, aggregated via addition across all ranks.
    """
    return _moe_combine(x, handle, config)


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _moe_combine(
    x: jnp.ndarray,
    handle: Tuple,
    config: Optional[Config] = None,
) -> jnp.ndarray:
    combine_x, _ = _moe_combine_impl(x, handle, config=config)
    return combine_x


def _moe_combine_impl(
    x: jnp.ndarray,
    handle: Tuple,
    topk_weights: Optional[jnp.ndarray] = None,
    bias: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]] = None,
    config: Optional[Config] = None,
) -> Tuple[jnp.ndarray]:

    # default config
    config = get_combine_config() if config is None else config

    # unpack bias
    bias_0, bias_1 = None, None
    if isinstance(bias, jnp.ndarray):
        bias_0 = bias
    elif isinstance(bias, tuple):
        assert len(bias) == 2
        bias_0, bias_1 = bias

    if topk_weights is None:
        topk_weights = jnp.array([], dtype=jnp.float32)

    if bias_0 is None:
        bias_0 = jnp.array([], dtype=x.dtype)

    if bias_1 is None:
        bias_1 = jnp.array([], dtype=x.dtype)

    rank_prefix_matrix, _, channel_prefix_matrix, src_idx, is_recv_token_in_rank, send_head = handle

    combined_x, combined_topk_weights = moe_combine_p.bind(
        x,
        topk_weights,
        bias_0,
        bias_1,
        src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        send_head,
        num_sms=config.num_sms,
        num_max_nvl_chunked_send_tokens=config.num_max_nvl_chunked_send_tokens,
        num_max_nvl_chunked_recv_tokens=config.num_max_nvl_chunked_recv_tokens,
        num_max_rdma_chunked_send_tokens=config.num_max_rdma_chunked_send_tokens,
        num_max_rdma_chunked_recv_tokens=config.num_max_rdma_chunked_recv_tokens,
    )
    return combined_x, combined_topk_weights


def _moe_combine_fwd(
    x: jnp.ndarray,
    handle: Tuple,
    config: Optional[Config] = None,
) -> jnp.ndarray:
    combine_x, _ = _moe_combine_impl(x, handle, config=config)
    ctx = (handle,)
    return combine_x, ctx


def _moe_combine_bwd(config, ctx, grad_output):
    (handle,) = ctx

    recv_grad_x, _, _, _ = _moe_dispatch_impl(grad_output, handle=handle, config=config)
    handle_grad = jax.tree_util.tree_map(jnp.zeros_like, handle)
    return recv_grad_x, handle_grad


_moe_combine.defvjp(_moe_combine_fwd, _moe_combine_bwd)
