###############################################################################
# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func
from transformers import LlamaConfig

from .build_model import register_model
from .rope import apply_rotary_emb, precompute_freqs_cis


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# ===== Basic Transformer Layer (Torch Native) =====
class BasicAttention(torch.nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        bs, seqlen, _ = x.shape
        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # ROPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Attn
        attn_output = flash_attn_func(xq, xk, xv, causal=True)
        attn_output = attn_output.view(bs, seqlen, -1)
        return self.wo(attn_output)

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)


class BasicMLP(torch.nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
    ):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class BasicTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: LlamaConfig,
    ):
        super().__init__()
        self.attention = BasicAttention(config)
        self.mlp = BasicMLP(config)

        self.attention_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # TODO:
        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.mlp(self.mlp_norm(h))
        return out

    def init_weights(self):
        for norm in (self.attention_norm, self.mlp_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.mlp.init_weights(self.weight_init_std)


@register_model("llama", "torch")
class LlamaBasicModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
    ):
        super().__init__()
        self.config = config

        self.tok_embed = nn.Embedding(config.vocab_size, config.hidden_size)

        # TODO: persistent = False
        self.register_buffer(
            "freqs_cis",
            self._precompute_freqs_cis(
                config.hidden_size,
                config.num_attention_heads,
                config.max_position_embeddings,
                config.rope_theta,
            ),
            persistent=True,
        )

        # Only test 4 layers for now
        self.layers = nn.ModuleList([BasicTransformerBlock(layer_id, config) for layer_id in range(4)])
        self.norm = nn.RMSNorm(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.init_weights()

    def _precompute_freqs_cis(self, hidden_size, n_heads, max_seq_len, rope_theta) -> torch.Tensor:
        return precompute_freqs_cis(
            hidden_size // n_heads,
            max_seq_len,
            rope_theta,
        )

    def init_weights(self):
        if self.tok_embed is not None:
            nn.init.normal_(self.tok_embed.weight)
        for layer in self.layers:
            if layer is not None:
                layer.init_weights()
        if self.norm is not None:
            self.norm.reset_parameters()

        final_out_std = self.config.hidden_size**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def forward(self, input_ids):
        x = self.tok_embed(input_ids)  # (bs, seqlen, hidden)

        for layer in self.layers:
            x = layer(x, self.freqs_cis)

        x = self.norm(x)
        logits = self.output(x)  # (bs, seqlen, vocab)
        return logits
