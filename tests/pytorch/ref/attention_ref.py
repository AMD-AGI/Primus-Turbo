###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
from einops import repeat
from torch.nn.attention import SDPBackend, sdpa_kernel

ATTN_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


class AttnConfig:
    def __init__(self, seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v):
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.num_head_q = num_head_q
        self.num_head_kv = num_head_kv
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v


def attention_vanilla_forward_pytorch_ref_impl(q, k, v, sm_scale, causal, layout="bshd"):
    """Compute reference output and softmax_lse using PyTorch's built-in function"""

    if layout == "bshd":
        num_heads = q.shape[2]
        n_kv_heads = k.shape[2]
        n_rep = num_heads // n_kv_heads

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    else:
        raise ValueError(f"Unknown layout {layout}")

    with sdpa_kernel(ATTN_BACKENDS):
        o_ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=sm_scale, enable_gqa=n_rep > 1
        )
    if layout == "bshd":
        o_ref = o_ref.transpose(1, 2)
    return o_ref


def attention_with_sink_ref_impl(q, k, v, sink, sm_scale, causal):
    """Reference implementation of attention with sink."""

    dtype_og = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    sink = sink.float()

    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    # Expand k, v for GQA
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])

    # Use provided sm_scale
    scores = torch.einsum("bthd,bshd->bhts", q * sm_scale, k)

    # Apply causal mask
    if causal:
        row_idx = torch.arange(seqlen_q, device=q.device).view(-1, 1)
        col_idx = torch.arange(seqlen_k, device=q.device)
        causal_mask = col_idx > row_idx + seqlen_k - seqlen_q
        scores = scores.masked_fill(causal_mask, float("-inf"))

    # Concatenate sink scores
    batch_size = scores.shape[0]
    nheads = scores.shape[1]
    sink_expanded = sink.view(1, nheads, 1, 1).expand(batch_size, -1, seqlen_q, -1)
    scores = torch.cat([scores, sink_expanded], dim=-1)

    # Softmax
    attention = torch.softmax(scores, dim=-1).to(v.dtype)

    # Remove sink attention weights before computing output
    attention = attention[..., :-1]

    # Compute output
    output = torch.einsum("bhts,bshd->bthd", attention, v)

    return output.to(dtype=dtype_og)


class TurboAttentionRef(torch.nn.Module):
    def __init__(
        self,
        softmax_scale=None,
        causal=False,
    ):
        super().__init__()

        self.softmax_scale = softmax_scale
        self.causal = causal

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

        return attention_vanilla_forward_pytorch_ref_impl(
            q,
            k,
            v,
            sm_scale=self.softmax_scale,
            causal=self.causal,
        )
