###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
import torch.nn as nn

import primus_turbo.pytorch as turbo

from .basic_llama import (
    BasicAttention,
    BasicMLP,
    BasicTransformerBlock,
    LlamaBasicModel,
)
from .build_model import register_model
from .rope import apply_rotary_emb


class Attention(BasicAttention):
    def __init__(self, config):
        super().__init__(config)
        self.sdpa = turbo.modules.TurboAttention(causal=True, use_fp8=False, backend_type="ck")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Attn
        attn_output = self.sdpa(xq, xk, xv)
        attn_output = attn_output.view(bs, seqlen, -1)

        return self.wo(attn_output)


class TransformerBlock(BasicTransformerBlock):
    def __init__(self, layer_id: int, config):
        super().__init__(layer_id, config)
        self.attention = Attention(config)
        self.mlp = BasicMLP(config)


@register_model("llama", "turbo")
class LlamaTurboModel(LlamaBasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # Only test 4 layers for now
        self.layers = nn.ModuleList([TransformerBlock(layer_id, config) for layer_id in range(4)])
        self.init_weights()
