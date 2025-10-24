###############################################################################
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Tuple, Union

import torch

from primus_turbo.pytorch.core.float8_tensor import Float8Tensor
from primus_turbo.triton.moe import permutation

__all__ = ["token_permute", "token_unpermute", "TokenPermuteMaskMap", "TokenUnpermuteMaskMap"]


class TokenPermuteMaskMap(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        inp: Union[torch.Tensor, Float8Tensor],
        routing_map: torch.Tensor,
        num_out_tokens: int,
        probs: torch.Tensor,
        return_tokens_per_expert: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # store inp shape for boundary case
        ctx.inp_shape = inp.shape

        if not inp.numel():
            ctx.probs = probs
            return (
                inp,
                torch.tensor([], device=inp.device),
                torch.tensor([], device=inp.device),
                torch.tensor([], device=inp.device),
            )

        assert inp.is_cuda, "Primus-Turbo needs CUDA."
        assert routing_map.is_cuda, "Primus-Turbo needs CUDA."
        if probs is not None:
            assert probs.is_cuda, "Primus-Turbo needs CUDA."

        assert inp.size(0) == routing_map.size(0), "Permute not possible"
        num_tokens, hidden_size = inp.size()
        num_experts = routing_map.size(1)
        assert num_out_tokens is not None, "num_out_tokens must be provided to the fused permute function."

        row_id_map, tokens_per_experts = permutation.make_row_id_map(
            routing_map, num_tokens, num_experts, return_tokens_per_expert
        )

        if num_out_tokens < 0:
            num_out_tokens = tokens_per_experts.sum().item()

        use_fp8 = isinstance(inp, Float8Tensor)

        if use_fp8:
            raise ValueError("FP8 is not supported for now.")

        if use_fp8:
            fp8_scale = inp._scale
            fp8_dtype = inp._fp8_dtype
            scale_hidden_dim = fp8_scale.shape[1]
        else:
            fp8_scale = None
            fp8_dtype = None
            scale_hidden_dim = None

        output, permuted_scale, permuted_probs = permutation.permute_with_mask_map(
            inp,
            row_id_map,
            probs,
            fp8_scale,
            num_tokens,
            num_experts,
            num_out_tokens,
            hidden_size,
            scale_hidden_dim,
        )

        if use_fp8:
            output = Float8Tensor(
                data=output,
                scale=permuted_scale,
                orig_dtype=inp._orig_dtype,
                fp8_dtype=fp8_dtype,
                config=inp._config,
            )

        # For Backward, grad index 0 must be empty, but should return the shape like input
        if not output.numel():
            ctx.probs = probs

        ctx.save_for_backward(row_id_map)
        ctx.num_experts = num_experts
        ctx.num_tokens = num_tokens
        ctx.hidden_size = hidden_size
        return output, row_id_map, permuted_probs, tokens_per_experts

    @staticmethod
    def backward(
        ctx,
        permuted_act_grad: torch.Tensor,
        _,
        permuted_probs_grad: torch.Tensor,
        __,
    ) -> Tuple[torch.Tensor, ...]:
        if not permuted_act_grad.numel():
            return (
                torch.empty(ctx.inp_shape, dtype=permuted_act_grad.dtype, device=permuted_act_grad.device),
                None,
                None,
                ctx.probs,
                None,
            )

        act_grad = None
        probs_grad = None
        if ctx.needs_input_grad[0]:
            (row_id_map,) = ctx.saved_tensors
            assert not isinstance(
                permuted_act_grad, Float8Tensor
            ), "The backward of token_permute does not support FP8."
            act_grad, probs_grad = permutation.unpermute_with_mask_map(
                permuted_act_grad,
                row_id_map,
                None,
                permuted_probs_grad,
                ctx.num_tokens,
                ctx.num_experts,
                ctx.hidden_size,
            )
        if not ctx.needs_input_grad[3]:
            probs_grad = None
        return act_grad, None, None, probs_grad, None


class TokenUnpermuteMaskMap(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        row_id_map: torch.Tensor,
        merging_probs: Optional[torch.Tensor],
        restore_shape: Optional[torch.Size],
    ) -> torch.Tensor:

        ctx.input_is_empty = inp.numel() == 0

        if ctx.input_is_empty:
            ctx.inp_shape = inp.shape
            ctx.merging_probs = merging_probs
            # NOTE(huangzhen): should return restore_shape when restore_shape is not None, due to deepep combine accept restore_shape like tensor
            if restore_shape is not None:
                inp = torch.empty(restore_shape, dtype=inp.dtype, device=inp.device)

            return inp

        if restore_shape is None:
            restore_shape = inp.shape
        num_tokens, hidden_size = restore_shape
        num_experts = (row_id_map.size(1) - 1) // 2

        with_probs = merging_probs is not None
        if with_probs:
            assert merging_probs.is_cuda, "Tensor device needs be CUDA."

        # Device check
        assert inp.is_cuda, "Tensor device needs be CUDA."
        assert row_id_map.is_cuda, "Tensor device needs be CUDA."

        assert not isinstance(inp, Float8Tensor), "The forward of moe_unpermute does not support FP8."
        unpermuted_output, _ = permutation.unpermute_with_mask_map(
            inp,
            row_id_map,
            merging_probs,
            None,
            num_tokens,
            num_experts,
            hidden_size,
        )

        if with_probs:
            ctx.save_for_backward(inp, row_id_map, merging_probs)
        else:
            ctx.save_for_backward(row_id_map)
        ctx.num_experts = num_experts
        ctx.num_tokens = num_tokens
        ctx.num_permuted_tokens = inp.size(0)
        ctx.hidden_size = hidden_size
        ctx.with_probs = with_probs
        return unpermuted_output

    @staticmethod
    def backward(ctx, unpermuted_act_grad):
        if ctx.input_is_empty:
            empty_grad = torch.empty(
                ctx.inp_shape, dtype=unpermuted_act_grad.dtype, device=unpermuted_act_grad.device
            )
            return empty_grad, None, ctx.merging_probs, None

        act_grad = None
        probs_grad = None
        if ctx.needs_input_grad[0]:
            if ctx.with_probs:
                fwd_input, row_id_map, merging_probs = ctx.saved_tensors
            else:
                (row_id_map,) = ctx.saved_tensors

            use_fp8 = isinstance(unpermuted_act_grad, Float8Tensor)

            if use_fp8:
                fp8_scale = unpermuted_act_grad._scale
                fp8_dtype = unpermuted_act_grad._fp8_dtype
                scale_hidden_dim = fp8_scale.shape[1]
            else:
                fp8_scale = None
                fp8_dtype = None
                scale_hidden_dim = None

            if ctx.with_probs:
                assert (
                    not use_fp8
                ), "The backward of TokenUnpermutation with merging probs does not support FP8."
                act_grad, probs_grad = permutation.unpermute_with_mask_map_bwd_with_merging_probs(
                    unpermuted_act_grad,
                    row_id_map,
                    fwd_input,
                    merging_probs,
                    ctx.num_tokens,
                    ctx.num_experts,
                    ctx.num_permuted_tokens,
                    ctx.hidden_size,
                )
            else:
                act_grad, permuted_scale, _ = permutation.permute_with_mask_map(
                    unpermuted_act_grad,
                    row_id_map,
                    None,
                    fp8_scale,
                    ctx.num_tokens,
                    ctx.num_experts,
                    ctx.num_permuted_tokens,
                    ctx.hidden_size,
                    scale_hidden_dim,
                )

            if use_fp8:
                act_grad = Float8Tensor(
                    data=act_grad,
                    scale=permuted_scale,
                    orig_dtype=unpermuted_act_grad._orig_dtype,
                    fp8_dtype=fp8_dtype,
                    config=unpermuted_act_grad._config,
                )

        if not ctx.needs_input_grad[2]:
            probs_grad = None
        return act_grad, None, probs_grad, None


def token_permute(
    inp: torch.Tensor,
    num_out_tokens: int,
    probs: Optional[torch.Tensor] = None,
    routing_map: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
    drop_and_pad: bool = False,
    fused: bool = False,
    return_tokens_per_expert: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Permute the tokens based on the routing_map. Token with the same index will be grouped together.
    Tokens with the same designated expert will be grouped together.
    The routing_map indicates which experts were selected by each token.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    num_out_tokens: int
        The effective output token count, representing the number of tokens not dropped.
        By default, set to '-1', meaning no tokens are dropped.
    routing_map: torch.Tensor
        The token to expert mapping tensor.
        routing_map is of shape [num_tokens, num_experts] and dtype 'int32'.
        The values in it: 1 means the token is routed to this expert and 0 means not.
    topk_indices: torch.Tensor
        The token to expert mapping tensor.
        topk_indices is of shape [num_tokens, topK] and dtype 'int32'.
        The values in it are the routed expert indices.

    fused (bool, optional): Whether use the fused permute function.
    drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                   and pads the number of tokens to the expert capacity.
                                   If set to true, routing_map has a fixed number of non-zeros
                                   in each column.e
    return_tokens_per_expert (bool, optinal): Wheter compute tokens_per_expert by permute phase
    """
    if fused:
        if topk_indices != None:
            raise NotImplementedError("not support topk_indices")
        if routing_map != None:
            output, row_id_map, permuted_probs, tokens_per_experts = TokenPermuteMaskMap.apply(
                inp, routing_map, num_out_tokens, probs, return_tokens_per_expert
            )
            return output, permuted_probs, row_id_map, tokens_per_experts
        raise ValueError("must be set topk_indices or routing_map")

    num_tokens, _ = inp.shape
    num_experts = routing_map.shape[1]
    permuted_probs = None
    if drop_and_pad and not (num_out_tokens is None):
        capacity = num_out_tokens // num_experts
        assert not routing_map.requires_grad
        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.to(dtype=torch.int8).T.contiguous()
        # use argsort to put indices of all non-zeros in the beginning of list
        # and keep the first `capacity` number of indices
        sorted_indices = routing_map.argsort(dim=-1, descending=True, stable=True)[:, :capacity].contiguous()
        # flatten from [num_experts, capacity] to 1D
        sorted_indices = sorted_indices.view(-1)

        if probs is not None:
            # [num_tokens, num_experts] -> num_experts * num_tokens
            probs_T_1D = probs.T.contiguous().view(-1)
            # get 1D indices of the probs selected by routing_map
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_tokens + indices_dim1).view(-1)
            # get probs from indices
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
    else:
        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.bool().T.contiguous()

        # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
        token_indices = (
            torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
        )
        sorted_indices = token_indices.masked_select(routing_map)

        if probs is not None:
            permuted_probs = probs.T.contiguous().masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = inp.index_select(0, sorted_indices)

    return permuted_input, permuted_probs, sorted_indices, routing_map.sum(axis=0)


def token_unpermute(
    inp: torch.Tensor,
    sorted_indices: torch.Tensor,
    merging_probs: Optional[torch.Tensor] = None,
    restore_shape: Optional[torch.Size] = None,
    routing_map: Optional[torch.Tensor] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
) -> torch.Tensor:
    """
    Unpermute a tensor with permuted tokens, and optionally merge the tokens with their
    corresponding probabilities.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor with permuted tokens of shape `[num_tokens, hidden_size]` to be unpermuted.
    sorted_indices: torch.Tensor
        The indices used to sort the tokens.
    merging_probs: torch.Tensor, default = None
        The tensor of probabilities corresponding to the permuted tokens. If provided,
        the unpermuted tokens will be merged with their respective probabilities.
        By default, set to an empty tensor, which means that the tokens are directly merged by accumulation.
    restore_shape: torch.Size, default = None
        The output shape after the unpermute operation.
    fused (bool, optional): Whether use the fused unpermute function.
    drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                    and pads the number of tokens to the expert capacity.
    """
    if fused:
        return TokenUnpermuteMaskMap.apply(inp, sorted_indices, merging_probs, restore_shape)

    _, hidden = restore_shape
    input_dtype = inp.dtype

    if merging_probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        if drop_and_pad:
            num_experts = routing_map.size(1)
            num_permuted_tokens = sorted_indices.size(0)
            capacity = num_permuted_tokens // num_experts
            num_unpermuted_tokens = merging_probs.size(0)

            # [num_unpermuted_tokens, num_experts] -> num_experts * num_unpermuted_tokens
            probs_T_1D = merging_probs.T.contiguous().view(-1)

            # get 1D indices of the probs selected by routing_map
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_unpermuted_tokens + indices_dim1).view(-1)

            # get probs from indices
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
        else:
            permuted_probs = merging_probs.T.contiguous().masked_select(routing_map.T.contiguous())
        # Here may promote permuted_tokens to higher precision (fp32/fp64) if probs is in
        # higher precision due to moe_router_dtype being enabled. This can lead to
        # additional GPU memory usage. Use --moe-permute-fusion flag to avoid this extra memory
        # allocation.
        inp = inp * permuted_probs.unsqueeze(-1)

    # Create an output tensor filled with zeros
    output_tokens = torch.zeros(restore_shape, dtype=inp.dtype, device=inp.device)
    # Scatter add the permuted_input back to the original positions
    output_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), inp)
    return output_tokens.to(dtype=input_dtype)
