###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from .attention_with_cp_a2a import (
    AttentionCKFunctionCPA2A,
    AttentionTritonFunctionCPA2A,
)


def dispatch_attention_cp_functions(
    q,
    k,
    v,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    bias,
    alibi_slopes,
    deterministic,
    return_lse,
    return_attn_probs,
    is_grad_enabled,
    backend_type,
    fp8,
    cp_group,
    cp_comm_type,
):
    if backend_type == "triton":
        if cp_comm_type == "a2a":
            return AttentionTritonFunctionCPA2A.apply(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                bias,
                alibi_slopes,
                return_lse,
                return_attn_probs,
                is_grad_enabled,
                fp8,
                cp_group,
            )
        else:
            raise NotImplementedError(
                f"not supported backend_type {backend_type} cp_comm_type {cp_comm_type} yet"
            )
    elif backend_type == "ck":
        if cp_comm_type == "a2a":
            return AttentionCKFunctionCPA2A.apply(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                bias,
                alibi_slopes,
                deterministic,
                return_lse,
                return_attn_probs,
                is_grad_enabled,
                cp_group,
            )
        else:
            raise NotImplementedError(
                f"not supported backend_type {backend_type} cp_comm_type {cp_comm_type} yet"
            )
    else:
        raise NotImplementedError(
            f"not supported backend_type {backend_type} cp_comm_type {cp_comm_type} yet"
        )
