###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling attention as a flex-attention example, shared by the tests and the benchmark.

Inkling (thinkingmachines/Inkling, Inkling-Small) replaces positional embeddings with a
learned relative-position bias. Per TP rank, on T packed tokens:

    score    = (q . k) / head_dim                       # scale is 1/d, not 1/sqrt(d)
    score   += rel_logits[t, h, q - kv]   if 0 <= q - kv < RE
    keep     = causal & same document & (local layers: q - kv <= window - 1)

``rel_logits`` comes out of the fused QKV projection, so training needs its gradient.
Global layers use RE = 1024 and no window; local (sliding-window) layers RE = window = 512.

``miles_flex`` is how radixark/miles trains it (``InklingSelfAttention._seg_flex`` in
miles_plugins/models/inkling/layers.py @ 7890b29): one packed batch row, a document
mask over segment ids, and ``rel_logits`` captured by a PyTorch FlexAttention score_mod.
``turbo_flex`` is the Primus-Turbo equivalent on packed sequences.
"""

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import torch
from torch.nn.attention import flex_attention as torch_flex

import primus_turbo.pytorch as turbo

HEAD_DIM = 128
TP = 4

# Full-model (query, kv) heads from the HF text configs; layers are per TP rank.
_HEADS = {
    "inkling": {"global": (64, 8), "local": (64, 16)},
    "inkling_small": {"global": (32, 8), "local": (32, 8)},
}
# 8K packed tokens per rank: one document, or a typical RL packing of five samples.
_PACKINGS = {"1doc": (8192,), "5doc": (3072, 2048, 1536, 1024, 512)}


@dataclass(frozen=True)
class InklingLayer:
    name: str
    heads_q: int
    heads_kv: int
    rel_extent: int
    window: Optional[int]  # sliding_window_size for local layers, None for global
    seqlens: Tuple[int, ...]
    head_dim: int = HEAD_DIM

    @property
    def total_tokens(self) -> int:
        return sum(self.seqlens)

    @property
    def window_left(self) -> Optional[int]:
        return None if self.window is None else self.window - 1

    @property
    def scale(self) -> float:
        return 1.0 / self.head_dim

    def visible_pairs(self) -> int:
        total = 0
        for length in self.seqlens:
            w = length if self.window is None else min(self.window, length)
            total += w * (w + 1) // 2 + (length - w) * w
        return total

    def flops(self) -> int:
        """Forward FLOPs over the kept pairs (QK^T and PV)."""
        return 4 * self.heads_q * self.head_dim * self.visible_pairs()

    def shrunk(self, seqlens: Tuple[int, ...], rel_extent: int, window: Optional[int]) -> "InklingLayer":
        """Same heads and structure at a size an eager reference can afford."""
        return replace(self, name=f"{self.name}_small", seqlens=seqlens, rel_extent=rel_extent, window=window)


INKLING_LAYERS = tuple(
    InklingLayer(
        name=f"{model}_{kind}_{packing}",
        heads_q=heads[kind][0] // TP,
        heads_kv=max(1, heads[kind][1] // TP),
        rel_extent=512 if kind == "local" else 1024,
        window=512 if kind == "local" else None,
        seqlens=seqlens,
    )
    for model, heads in _HEADS.items()
    for kind in ("global", "local")
    for packing, seqlens in _PACKINGS.items()
)


@dataclass
class InklingInputs:
    q: torch.Tensor  # [T, Hq, D]
    k: torch.Tensor  # [T, Hkv, D]
    v: torch.Tensor  # [T, Hkv, D]
    rel_logits: torch.Tensor  # [T, Hq, RE] fp32, Turbo's layout
    rel_logits_htr: torch.Tensor  # [Hq, T, RE] fp32, Miles' layout (same values, own grad)
    cu_seqlens: torch.Tensor  # int32 [num_docs + 1]
    seg_id: torch.Tensor  # [T], document index of each token


def make_inputs(layer: InklingLayer, device="cuda", dtype=torch.bfloat16, seed=0, requires_grad=True):
    gen = torch.Generator().manual_seed(seed)
    T, Hq, Hkv, D = layer.total_tokens, layer.heads_q, layer.heads_kv, layer.head_dim

    def leaf(*shape, scale=1.0, dt=dtype):
        return (torch.randn(*shape, generator=gen) * scale).to(device, dt).requires_grad_(requires_grad)

    # q and k are RMS-normed in the model, so unit-variance rows are representative.
    q, k, v = leaf(T, Hq, D), leaf(T, Hkv, D), leaf(T, Hkv, D)
    rel = leaf(T, Hq, layer.rel_extent, scale=0.25, dt=torch.float32)
    rel_htr = rel.detach().permute(1, 0, 2).contiguous().requires_grad_(requires_grad)
    lengths = torch.tensor(layer.seqlens, device=device)
    cu = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)]).to(torch.int32)
    seg_id = torch.repeat_interleave(torch.arange(len(layer.seqlens), device=device), lengths)
    return InklingInputs(q, k, v, rel, rel_htr, cu, seg_id)


def miles_block_mask(layer: InklingLayer, inputs: InklingInputs, compile_mask: bool = True):
    """PyTorch BlockMask for Miles' document + causal (+ window) mask over one packed row."""
    seg_id, window_left = inputs.seg_id, layer.window_left

    def mask_mod(b, h, q_idx, kv_idx):
        keep = (q_idx >= kv_idx) & (seg_id[q_idx] == seg_id[kv_idx])
        return keep if window_left is None else keep & (q_idx - kv_idx <= window_left)

    T = layer.total_tokens
    return torch_flex.create_block_mask(
        mask_mod, None, None, T, T, device=inputs.q.device, _compile=compile_mask
    )


def miles_flex(layer: InklingLayer, inputs: InklingInputs, block_mask, flex_fn=torch_flex.flex_attention):
    """Miles' ``_seg_flex``: ``[T, H, D]`` in, ``[T, H, D]`` out, gradient on ``rel_logits_htr``."""
    rel, RE = inputs.rel_logits_htr, layer.rel_extent

    def score_mod(score, b, h, q_idx, kv_idx):
        rd = q_idx - kv_idx
        bias = rel[h, q_idx, torch.clamp(rd, 0, RE - 1)]
        return torch.where((rd >= 0) & (rd < RE), score + bias.to(score.dtype), score)

    q, k, v = (t.permute(1, 0, 2).unsqueeze(0) for t in (inputs.q, inputs.k, inputs.v))
    out = flex_fn(q, k, v, score_mod=score_mod, block_mask=block_mask, scale=layer.scale, enable_gqa=True)
    return out.squeeze(0).permute(1, 0, 2)


def turbo_flex(layer: InklingLayer, inputs: InklingInputs):
    """Primus-Turbo: ``[T, H, D]`` in, ``[T, H, D]`` out, gradient on ``rel_logits``."""
    return turbo.ops.flex_attention_varlen_rel_bias(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.rel_logits,
        inputs.cu_seqlens,
        max(layer.seqlens),
        scale=layer.scale,
        window_left=layer.window_left,
    )
