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

from .moe_utils import Config

__all__ = ["get_dispatch_config", "moe_dispatch", "get_combine_config", "moe_combine"]


_default_num_sms = 64

P = jax.sharding.PartitionSpec

# DeepEP topk_idx are int64, so we need to enable x64 precision.
jax.config.update("jax_enable_x64", True)


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
        2: Config(_default_num_sms, 24, 256, 6, 128),
        4: Config(_default_num_sms, 6, 256, 6, 128),
        8: Config(_default_num_sms, 6, 256, 6, 128),
        16: Config(_default_num_sms, 36, 288, 20, 128),
        24: Config(_default_num_sms, 8, 288, 32, 128),
        32: Config(_default_num_sms, 32, 288, 32, 128),
        64: Config(_default_num_sms, 20, 288, 28, 128),
        128: Config(_default_num_sms, 20, 560, 32, 128),
        144: Config(_default_num_sms, 32, 720, 12, 128),
        160: Config(_default_num_sms, 28, 720, 12, 128),
    }
    assert num_ranks in config_map, f"Unsupported number of EP ranks: {num_ranks}"
    return config_map[num_ranks]


def get_combine_config() -> Config:
    """
    Get a recommended dispatch config.
    Returns:
        config: the recommended config.
    """
    global _default_num_sms
    num_ranks = jax.local_device_count()
    assert num_ranks <= 8, "not support internode"
    config_map = {
        2: Config(_default_num_sms, 10, 256, 6, 128),
        4: Config(_default_num_sms, 9, 256, 6, 128),
        8: Config(_default_num_sms, 4, 256, 6, 128),
        16: Config(_default_num_sms, 4, 288, 12, 128),
        24: Config(_default_num_sms, 1, 288, 8, 128),
        32: Config(_default_num_sms, 1, 288, 8, 128),
        64: Config(_default_num_sms, 1, 288, 20, 128),
        128: Config(_default_num_sms, 1, 560, 12, 128),
        144: Config(_default_num_sms, 2, 720, 8, 128),
        160: Config(_default_num_sms, 2, 720, 8, 128),
    }
    assert num_ranks in config_map, f"Unsupported number of EP ranks: {num_ranks}"
    return config_map[num_ranks]


def moe_dispatch(
    x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    handle: Optional[Tuple] = None,
    topk_idx: Optional[jnp.ndarray] = None,
    topk_weights: Optional[jnp.ndarray] = None,
    expert_alignment: int = 1,
    num_experts: Optional[int] = None,
    config: Optional[Config] = None,
) -> Tuple[
    Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], Optional[jnp.ndarray], Optional[jnp.ndarray], Tuple
]:
    """
    Dispatch tokens to their selected experts in a Mixture of Experts (MoE) model.

    In MoE models, each token dynamically selects top-k experts based on routing scores. This function
    performs the all-to-all communication to dispatch tokens from all ranks to the ranks where their
    selected experts reside. It supports both intranode communication via NVLink and internode
    communication via RDMA, enabling efficient expert parallelism across multiple GPUs and nodes.

    The dispatch process handles:
    - Token routing based on top-k expert selection
    - Cross-rank communication to send tokens to appropriate expert locations
    - Layout computation and caching for performance optimization
    - Support for both standard (bfloat16) and low-precision (float8) datatypes

    Arguments:
        x: `jnp.ndarray` or tuple of `jnp.ndarray`, for the first type, the shape must be `[num_tokens, hidden]`,
            and type must be `jnp.bfloat16`; for the second type, the first element of the tuple must be shaped as
            `[num_tokens, hidden]` with type `jnp.float8_e4m3fn`, the second must be `[num_tokens, hidden // 128]`
             (requiring divisible) with type `jnp.float32`.
        handle: an optional communication handle, if set, the function will reuse the layout information to save
            computation time. When handle is provided, topk_idx and topk_weights must be None.
        topk_idx: `[num_tokens, num_topk]` with `jnp.int64`, the expert indices selected by each token,
            `-1` means no selections. Required when handle is None.
        topk_weights: `[num_tokens, num_topk]` with `jnp.float32`, the expert weights of each token to dispatch.
            Required when handle is None.
        expert_alignment: align the number of tokens received by each local expert to this variable.
        num_experts: total number of experts across all ranks. Required when handle is None.
        config: the performance tuning config. If None, will use the default config from get_dispatch_config().

    Returns:
        recv_x: received tokens, the same type and tuple as the input `x`, but the number of tokens equals to the
            received token count.
        recv_topk_idx: received expert indices with shape `[num_recv_tokens, num_topk]`, or None if handle is provided.
        recv_topk_weights: received expert weights with shape `[num_recv_tokens, num_topk]`, or None if handle is provided.
        handle: the communication handle containing layout information (rank_prefix_matrix, channel_prefix_matrix,
            recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head), which can be reused in subsequent
            calls to avoid recomputing the dispatch layout.
    """
    return _moe_dispatch(x, handle, topk_idx, topk_weights, expert_alignment, num_experts, config)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def _moe_dispatch(
    x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    handle: Optional[Tuple] = None,
    topk_idx: Optional[jnp.ndarray] = None,
    topk_weights: Optional[jnp.ndarray] = None,
    expert_alignment: int = 1,
    num_experts: Optional[int] = None,
    config: Optional[Config] = None,
) -> Tuple[
    Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], Optional[jnp.ndarray], Optional[jnp.ndarray], Tuple
]:
    out, _ = _moe_dispatch_fwd(x, handle, topk_idx, topk_weights, expert_alignment, num_experts, config)
    return out


def _moe_dispatch_fwd(
    x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    handle: Optional[Tuple] = None,
    topk_idx: Optional[jnp.ndarray] = None,
    topk_weights: Optional[jnp.ndarray] = None,
    expert_alignment: int = 1,
    num_experts: Optional[int] = None,
    config: Optional[Config] = None,
) -> Tuple[
    Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], Optional[jnp.ndarray], Optional[jnp.ndarray], Tuple
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
        return (
            (recv_x, recv_x_scales) if x_scales.size > 0 else recv_x,
            None,
            None,
            None,
        ), (None, None, None, None)
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
        ), (handle, expert_alignment, num_experts, config)


def _moe_dispatch_bwd(expert_alignment, num_experts, config, ctx, grad_output):
    handle, _, _, _ = ctx
    grad_x, _, grad_topk_weights, _ = grad_output

    if isinstance(grad_x, tuple):
        grad_x, _ = grad_x

    (grad_x, grad_topk_weights), _ = _moe_combine_fwd(grad_x, handle)
    return grad_x, None, grad_topk_weights, None


_moe_dispatch.defvjp(_moe_dispatch_fwd, _moe_dispatch_bwd)


def moe_combine(
    x: jnp.ndarray,
    handle: Tuple,
    topk_weights: Optional[jnp.ndarray] = None,
    bias: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]] = None,
    config: Optional[Config] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
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
        topk_weights: `[num_recv_tokens, num_topk]` with `jnp.float32`, the tokens' top-k weights for reducing
            back to their original ranks. If None, only tokens will be reduced without weights.
        bias: optional bias to add during combination. Can be a single `jnp.ndarray` or a tuple of two arrays
            for separate bias terms. The dtype should match the input `x`.
        config: the performance tuning config. If None, will use the default config from get_combine_config().

    Returns:
        combined_x: the reduced tokens from all expert ranks, gathered back to original token positions with
            shape `[num_tokens, hidden]`, aggregated via addition across all ranks.
        combined_topk_weights: the reduced top-k weights from all expert ranks with shape `[num_tokens, num_topk]`,
            or None if topk_weights was not provided.
    """
    return _moe_combine(x, handle, topk_weights, bias, config)


@jax.custom_vjp
def _moe_combine(
    x: jnp.ndarray,
    handle: Tuple,
    topk_weights: Optional[jnp.ndarray] = None,
    bias: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]] = None,
    config: Optional[Config] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    out, _ = _moe_combine_fwd(x, handle, topk_weights, bias, config)
    return out


def _moe_combine_fwd(
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
    return (combined_x, combined_topk_weights), (handle, bias, config)


def _moe_combine_bwd(residuals, grad_output):
    handle, _, _ = residuals
    grad_x, _ = grad_output

    (recv_grad_x, _, _, _), _ = _moe_dispatch_fwd(grad_x, handle=handle)
    return recv_grad_x, None, None, None, None


_moe_combine.defvjp(_moe_combine_fwd, _moe_combine_bwd)
