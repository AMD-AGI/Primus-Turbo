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


def attention_vanilla_forward_pytorch_ref_impl(q, k, v, sm_scale, causal, qkv_format="bshd"):
    """Compute reference output and softmax_lse using PyTorch's built-in function"""

    if qkv_format == "bshd":
        num_heads = q.shape[2]
        n_kv_heads = k.shape[2]
        n_rep = num_heads // n_kv_heads

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    elif qkv_format == "sbhd":
        num_heads = q.shape[2]
        n_kv_heads = k.shape[2]
        n_rep = num_heads // n_kv_heads

        q = q.permute(1, 2, 0, 3).contiguous()
        k = k.permute(1, 2, 0, 3).contiguous()
        v = v.permute(1, 2, 0, 3).contiguous()
    elif qkv_format == "bhsd":
        num_heads = q.shape[1]
        n_kv_heads = k.shape[1]
        n_rep = num_heads // n_kv_heads
    else:
        raise ValueError(f"Unknown qkv format {qkv_format}")

    with sdpa_kernel(ATTN_BACKENDS):
        # NOTE: expect input layout is bhsd.
        o_ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=sm_scale, enable_gqa=n_rep > 1
        )
    if qkv_format == "bshd":
        o_ref = o_ref.transpose(1, 2).contiguous()
    elif qkv_format == "sbhd":
        o_ref = o_ref.permute(2, 0, 1, 3).contiguous()
    return o_ref


def _construct_local_mask(seqlen_q, seqlen_k, window_size, device):
    """Build the local attention mask using the same indexing rule as aiter."""
    row_idx = torch.arange(seqlen_q, device=device).view(-1, 1)
    col_idx = torch.arange(seqlen_k, device=device)
    shift = seqlen_k - seqlen_q

    if window_size[0] < 0:
        return col_idx > row_idx + shift + window_size[1]

    max_col_idx = torch.full_like(col_idx, seqlen_k)
    return torch.logical_or(
        col_idx > torch.minimum(row_idx + shift + window_size[1], max_col_idx),
        col_idx < row_idx + shift - window_size[0],
    )


def attention_with_sink_ref_impl(q, k, v, sink, sm_scale, causal, window_size=(-1, -1), qkv_format="bshd"):
    """Reference implementation of attention with sink and optional sliding window."""

    if qkv_format == "sbhd":
        q = q.permute(1, 0, 2, 3).contiguous()
        k = k.permute(1, 0, 2, 3).contiguous()
        v = v.permute(1, 0, 2, 3).contiguous()
    elif qkv_format != "bshd":
        raise ValueError(f"Unknown qkv format {qkv_format}")

    dtype_og = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    sink = sink.float()

    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    if causal:
        window_size = (window_size[0], 0)

    # Expand k, v for GQA
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])

    # Use provided sm_scale
    scores = torch.einsum("bthd,bshd->bhts", q * sm_scale, k)

    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = _construct_local_mask(seqlen_q, seqlen_k, window_size, q.device)
        scores = scores.masked_fill(local_mask.view(1, 1, seqlen_q, seqlen_k), float("-inf"))

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

    output = output.to(dtype=dtype_og)
    if qkv_format == "sbhd":
        output = output.permute(1, 0, 2, 3).contiguous()
    return output


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
