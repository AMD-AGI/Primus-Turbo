###############################################################################
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Tuple, Union

import torch

import primus_turbo.pytorch as turbo
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # store inp shape for boundary case
        ctx.inp_shape = inp.shape

        if not inp.numel():
            ctx.probs = probs
            return inp, torch.tensor([], device=inp.device), torch.tensor([], device=inp.device)

        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert routing_map.is_cuda, "TransformerEngine needs CUDA."
        if probs is not None:
            assert probs.is_cuda, "TransformerEngine needs CUDA."

        assert inp.size(0) == routing_map.size(0), "Permute not possible"
        num_tokens, hidden_size = inp.size()
        num_experts = routing_map.size(1)
        assert num_out_tokens is not None, "num_out_tokens must be provided to the fused permute function."

        row_id_map = permutation.make_row_id_map(routing_map, num_tokens, num_experts)

        use_fp8 = isinstance(inp, Float8Tensor)

        if use_fp8:
            fp8_scale = inp._scale
            fp8_dtype = inp._fp8_dtype
            scale_hidden_dim = fp8_scale.shape[1]
        else:
            fp8_scale = None
            fp8_dtype = None
            scale_hidden_dim = None

        turbo.modules.TurboDeepEPTokenDispatcher.maybe_cpu_sync()

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
        return output, row_id_map, permuted_probs

    @staticmethod
    def backward(
        ctx,
        permuted_act_grad: torch.Tensor,
        _,
        permuted_probs_grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        if not permuted_act_grad.numel():
            return (
                torch.empty(ctx.inp_shape, dtype=permuted_act_grad.dtype, device=permuted_act_grad.device),
                None,
                None,
                ctx.probs,
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
        return act_grad, None, None, probs_grad


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
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    """
    if topk_indices != None:
        raise NotImplementedError("not support topk_indices")
    if routing_map != None:
        output, row_id_map, permuted_probs = TokenPermuteMaskMap.apply(
            inp, routing_map, num_out_tokens, probs
        )
        return output, permuted_probs, row_id_map
    raise ValueError("must be set topk_indices or routing_map")


def token_unpermute(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: Optional[torch.Tensor] = None,
    restore_shape: Optional[torch.Size] = None,
) -> torch.Tensor:
    """
    Unpermute a tensor with permuted tokens, and optionally merge the tokens with their
    corresponding probabilities.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor with permuted tokens of shape `[num_tokens, hidden_size]` to be unpermuted.
    row_id_map: torch.Tensor
        The tensor of a mapping table for sorted indices used to unpermute the tokens,
        which is the second output tensor of `Permute`.
    merging_probs: torch.Tensor, default = None
        The tensor of probabilities corresponding to the permuted tokens. If provided,
        the unpermuted tokens will be merged with their respective probabilities.
        By default, set to an empty tensor, which means that the tokens are directly merged by accumulation.
    restore_shape: torch.Size, default = None
        The output shape after the unpermute operation.
    """
    return TokenUnpermuteMaskMap.apply(inp, row_id_map, merging_probs, restore_shape)
