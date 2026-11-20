###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL launchers for RL attention on gfx950 (primus_turbo/flydsl/attention/rl_attention_fwd.py).

The forward runs one of four builds of the same kernel, chosen by the call, all giving the same bits
for a token (verified bitwise on MI355X):

* plain causal self-attention (no mask, no window, contiguous, cu_seqlens_q is cu_seqlens_k): the
  built-in bottom-right causal path;
* everything else (masks, windows, paged caches, chunked prefill, decode): visibility intervals,
  with causal / window attention converted to intervals first;
* with at most PACK_MAX_Q query tokens per sequence, the rows of a workgroup are (token, head)
  pairs of one query group, so a K/V tile is read once per KV head;
* when a sequence has more keys than SEGMENT_SIZE, the keys are reduced in fixed segments merged
  in order (single-segment rows give the same bits as without), and packed decode with too few
  workgroups for the GPU splits the segments across programs and merges them in a second launch.

The backward is the Triton RL backward, which recomputes from this forward's output and LSE.

Limits: bf16, head size 64 or 128, softmax_scale 1/sqrt(D), 64-key tiles (TILE_SIZE 0 or 64), pages
and SEGMENT_SIZE that are multiples of 64.

Compared with Turbo's stock FlyDSL forward, scores are scaled in fp32 after the QK MFMA instead of
pre-scaling Q in bf16, which keeps large logits accurate (max error vs fp32 0.015 instead of up to
1.2 on logits in the hundreds) at +8% (D=128) / +21% (D=64) causal forward time.
"""

import functools
from collections import OrderedDict
from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.kernels.attention.rl_attention_triton_impl import (
    rl_attention_bwd_triton,
)
from primus_turbo.triton.attention.rl_attention_vis import (
    causal_intervals_kernel,
    vis_group_bounds_kernel,
)

TILE = 64
PACK_MAX_Q = 16
ROWS_PER_WAVE = 32
DEFAULT_SEGMENT = 1024
MAX_THREADS = 256  # threads of the largest workgroup the kernel builds (block_m 128)


def can_handle(q, k, block_table=None, softmax_scale=None, tile_size=0, segment_size=0, **kwargs) -> bool:
    head_size = q.shape[-1]
    return (
        is_gfx950()
        and q.dtype == torch.bfloat16
        and k.dtype == q.dtype
        and head_size in (64, 128)
        and (softmax_scale is None or abs(softmax_scale - head_size**-0.5) < 1e-6)
        and tile_size in (0, TILE)
        and segment_size % TILE == 0
        and (block_table is None or k.shape[1] % TILE == 0)
    )


@functools.lru_cache(maxsize=1)
def _num_cus(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.lru_cache(maxsize=64)
def _module(
    num_q_heads, num_kv_heads, head_size, cross, vis, page_size, pack, sink, emit_lse, block_m, waves_per_eu,
    segment_tiles, split_mode,
):  # fmt: skip
    from primus_turbo.flydsl.attention.rl_attention_fwd import build_rl_flash_attn_module

    return build_rl_flash_attn_module(
        num_heads=num_q_heads,
        head_dim=head_size,
        causal=True,
        dtype_str="bf16",
        num_kv_heads=num_kv_heads,
        varlen=True,
        cross_seqlen=cross,
        emit_lse=emit_lse,
        window_left=-1,
        waves_per_eu=waves_per_eu,
        dualwave_swp_enable_stagger=False,
        block_m=block_m,
        has_vis=vis,
        page_size=page_size,
        gqa_pack=pack,
        has_sink=sink,
        segment_tiles=segment_tiles,
        split_mode=split_mode,
    )


class _StepCache:
    """Visibility metadata keyed on the identity and version of the tensors it is derived from, so
    every layer of a step reuses one computation. Entries hold those tensors, so their ids cannot be
    reused while cached. Bypassed during CUDA-graph capture (the graph must recompute it) and for
    inference tensors, which carry no version counter."""

    def __init__(self, size=16):
        self._size = size
        self._entries = OrderedDict()

    def get(self, tensors, extra, build):
        if torch.cuda.is_current_stream_capturing():
            return build()
        try:
            key = tuple((id(t), t._version) for t in tensors) + extra
        except RuntimeError:
            return build()
        hit = self._entries.get(key)
        if hit is not None:
            self._entries.move_to_end(key)
            return hit[1]
        value = build()
        self._entries[key] = (tensors, value)
        if len(self._entries) > self._size:
            self._entries.popitem(last=False)
        return value


_cache = _StepCache()


def _causal_intervals(cu_seqlens_q, key_starts, num_tokens, window_left):
    lo = torch.empty(num_tokens, dtype=torch.int32, device=cu_seqlens_q.device)
    hi = torch.empty_like(lo)
    causal_intervals_kernel[(num_tokens,)](
        cu_seqlens_q, key_starts, lo, hi, cu_seqlens_q.numel() - 1, WINDOW_LEFT=window_left
    )
    return lo, hi


def _group_bounds(cu_seqlens_q, lo, hi, tokens_per_group):
    out = torch.empty((lo.numel(), 4), dtype=torch.int32, device=lo.device)
    vis_group_bounds_kernel[(lo.numel(),)](
        cu_seqlens_q, lo, hi, out, cu_seqlens_q.numel() - 1, TOKENS=tokens_per_group
    )
    return out


def rl_attention_fwd_flydsl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor],
    seq_lens: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    window_left: int,
    vis_lo: Optional[torch.Tensor],
    vis_hi: Optional[torch.Tensor],
    sinks: Optional[torch.Tensor],
    return_lse: bool,
    mode: str = "auto",
    tile_size: int = 0,
    segment_size: int = 0,
    block_m: int = 0,
    num_warps: int = 0,
    num_stages: int = 0,
    waves_per_eu: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same contract as rl_attention_fwd_triton. Only ``block_m`` (64 or 128) of the schedule
    options applies; the others belong to the Triton kernel."""
    paged = block_table is not None
    num_tokens, num_q_heads, head_size = q.shape
    num_kv_heads = k.shape[-2]
    assert v.shape[-1] == head_size, "the FlyDSL RL forward needs equal q / v head sizes"
    assert k.is_contiguous() and v.is_contiguous(), "the FlyDSL RL forward needs contiguous K / V"
    q = q.contiguous()
    out = torch.empty_like(q)
    lse = torch.empty((num_tokens, num_q_heads) if return_lse else (0,), dtype=torch.float32, device=q.device)
    if num_tokens == 0:
        return out, lse
    num_seqs = cu_seqlens_q.numel() - 1
    if paged and cu_seqlens_k is None:
        cu_seqlens_k = _cache.get(
            (seq_lens,),
            ("starts",),
            lambda: torch.nn.functional.pad(torch.cumsum(seq_lens, 0, dtype=torch.int32), (1, 0)),
        )

    builtin = vis_lo is None and window_left < 0 and not paged and cu_seqlens_q is cu_seqlens_k
    if not builtin and vis_lo is None:
        vis_lo, vis_hi = _cache.get(
            (cu_seqlens_q, cu_seqlens_k),
            ("intervals", num_tokens, window_left),
            lambda: _causal_intervals(cu_seqlens_q, cu_seqlens_k, num_tokens, window_left),
        )
    group = num_q_heads // num_kv_heads
    pack = not builtin and int(max_seqlen_q) <= PACK_MAX_Q and ROWS_PER_WAVE % group == 0
    if block_m not in (0, 64, 128):
        raise ValueError(f"the FlyDSL RL forward supports BLOCK_M 64 or 128, got {block_m}")
    segment = segment_size or DEFAULT_SEGMENT
    segment_tiles = segment // TILE if int(max_seqlen_k) > segment else 0
    if segment_tiles and head_size == 128 and not pack:
        # The outer segment state is register heavy at D=128: one wave per SIMD and 64-row blocks
        # are the fastest measured configuration.
        block_m, wpe = block_m or 64, 1
    else:
        block_m, wpe = block_m or (128 if head_size == 128 else 64), 2
    num_q_blocks = (int(max_seqlen_q) * (group if pack else 1) + block_m - 1) // block_m
    split = (
        pack and segment_tiles > 0 and num_seqs * num_kv_heads * num_q_blocks < _num_cus(q.device.index or 0)
    )

    def build(split_mode):
        return _module(
            num_q_heads, num_kv_heads, head_size, not builtin, not builtin, k.shape[1] if paged else 0, pack,
            sinks is not None, bool(return_lse), block_m, wpe, segment_tiles, split_mode,
        )  # fmt: skip

    kw = {}
    if not builtin:
        tokens_per_group = ROWS_PER_WAVE // group if pack else ROWS_PER_WAVE
        kw.update(
            vis_lo=vis_lo,
            vis_hi=vis_hi,
            vis_group=_cache.get(
                (cu_seqlens_q, vis_lo, vis_hi),
                ("groups", tokens_per_group),
                lambda: _group_bounds(cu_seqlens_q, vis_lo, vis_hi, tokens_per_group),
            ),
        )
    if paged:
        kw.update(block_table=block_table, block_table_stride=block_table.stride(0))
    if return_lse:
        kw["debug_counts"] = lse
    kw.update(
        seq_len_kv=int(max_seqlen_k),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_k,
        sink=sinks,
        score_bound=None,
        stream=torch.cuda.current_stream(),
    )
    if split:
        num_segments = (int(max_seqlen_k) + segment - 1) // segment
        per_thread = 2 + 16 * (head_size // 32)
        workspace = torch.empty(
            num_segments * num_seqs * num_q_blocks * num_kv_heads * MAX_THREADS * per_thread,
            dtype=torch.float32,
            device=q.device,
        )
        build(1)(q, k, v, out, num_seqs, int(max_seqlen_q), partial=workspace, **kw)
        build(2)(q, k, v, out, num_seqs, int(max_seqlen_q), partial=workspace, **kw)
    else:
        build(0)(q, k, v, out, num_seqs, int(max_seqlen_q), **kw)
    return out, lse


rl_attention_bwd_flydsl = rl_attention_bwd_triton
