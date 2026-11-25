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
from typing import Optional, Tuple

import torch
import triton

from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.kernels.attention.rl_attention_triton_impl import (
    rl_attention_bwd_triton,
)
from primus_turbo.triton.attention.rl_attention_vis import rl_metadata_kernel

TILE = 64
PACK_MAX_Q = 16
DEFAULT_WAVE_ROWS = 16
DEFAULT_SEGMENT = 1024


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
    segment_tiles, split_mode, split_segs=1, wave_rows=32, seq_major=False, kv_cache_aux=0,
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
        split_segs=split_segs,
        rows_per_wave=wave_rows,
        seq_major=seq_major,
        kv_cache_aux=kv_cache_aux,
    )


# Packed decode builds: launch sequence-major, and load K / V with the non-temporal cache bits
# (gfx950 sc0 | nt): decode streams each key once, and default-cached loads were ~15% slower.
SEQ_MAJOR = True
KV_CACHE_AUX = 3


def _metadata(
    cu_seqlens_q, cu_seqlens_k, seq_lens, num_tokens, vis_lo, vis_hi, window_left, tokens_per_group
):
    """Key starts, intervals and group bounds in one launch: (cu_seqlens_k, lo, hi, group)."""
    device = cu_seqlens_q.device
    num_seqs = cu_seqlens_q.numel() - 1
    keys_are_lens = cu_seqlens_k is None
    starts = torch.empty(num_seqs + 1, dtype=torch.int32, device=device) if keys_are_lens else cu_seqlens_k
    has_mask = vis_lo is not None
    if not has_mask:
        vis_lo = torch.empty(num_tokens, dtype=torch.int32, device=device)
        vis_hi = torch.empty_like(vis_lo)
    group = torch.empty((num_tokens, 4), dtype=torch.int32, device=device)
    grid = max(num_tokens, num_seqs + 1 if keys_are_lens else 0)
    rl_metadata_kernel[(grid,)](
        cu_seqlens_q,
        seq_lens if keys_are_lens else cu_seqlens_k,
        vis_lo,
        vis_hi,
        starts,
        vis_lo,
        vis_hi,
        group,
        num_seqs,
        num_tokens,
        KEYS_ARE_LENS=keys_are_lens,
        HAS_MASK=has_mask,
        WINDOW_LEFT=window_left,
        TOKENS=tokens_per_group,
        SEQ_BLOCK=triton.next_power_of_2(num_seqs + 1),
    )
    return starts, vis_lo, vis_hi, group


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
    split_segs: int = 0,
    pack: str = "auto",
    wave_rows: int = 0,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same contract as rl_attention_fwd_triton. Schedule options (none changes the bits):
    ``block_m`` (64 / 128), ``waves_per_eu`` (1 / 2), ``mode`` ("fused" never splits the keys across
    programs, "split" splits packed decode whenever there are several segments), ``split_segs``
    (segments per split program; 0 picks up to 8 while at least 4 programs per CU remain) and ``pack`` ("on" / "off" forces (token, head) rows); 0 / "auto"
    keeps the measured defaults. ``num_warps`` and ``num_stages`` belong to the Triton kernel."""
    paged = block_table is not None
    num_tokens, num_q_heads, head_size = q.shape
    num_kv_heads = k.shape[-2]
    assert v.shape[-1] == head_size, "the FlyDSL RL forward needs equal q / v head sizes"
    assert k.is_contiguous() and v.is_contiguous(), "the FlyDSL RL forward needs contiguous K / V"
    q = q.contiguous()
    if out is None:
        out = torch.empty_like(q)
    elif not out.is_contiguous():
        raise ValueError("the FlyDSL RL forward writes out with q's contiguous [num_tokens, Hq, D] layout")
    lse = torch.empty((num_tokens, num_q_heads) if return_lse else (0,), dtype=torch.float32, device=q.device)
    if num_tokens == 0:
        return out, lse
    num_seqs = cu_seqlens_q.numel() - 1
    group = num_q_heads // num_kv_heads
    rows = wave_rows or DEFAULT_WAVE_ROWS
    if rows not in (16, 32):
        raise ValueError(f"WAVE_ROWS must be 16 or 32, got {wave_rows}")
    if pack == "on" and rows % group:
        raise ValueError(f"PACK=on needs a GQA group size dividing {rows}, got {group}")
    builtin = (
        vis_lo is None and window_left < 0 and not paged and cu_seqlens_q is cu_seqlens_k and pack != "on"
    )
    if pack == "auto":
        pack = not builtin and int(max_seqlen_q) <= PACK_MAX_Q and rows % group == 0
    else:
        pack = pack == "on"
    if block_m not in (0, 64, 128) and not (rows == 16 and block_m == 32):
        raise ValueError(f"the FlyDSL RL forward supports BLOCK_M 64 or 128 (and 32 with WAVE_ROWS=16), got {block_m}")
    segment = segment_size or DEFAULT_SEGMENT
    segment_tiles = segment // TILE if int(max_seqlen_k) > segment else 0
    if rows == 16:
        # Two waves of 16 rows for packed decode; otherwise as many waves as 64 rows give.
        packed_rows = int(max_seqlen_q) * group
        block_m = block_m or (32 if pack and packed_rows <= 32 else 64)
        wpe = waves_per_eu or 2
    elif segment_tiles and head_size == 128:
        # The outer segment state is register heavy at D=128: one wave per SIMD and 64-row blocks
        # are the fastest measured configuration, for packed decode as for prefill / training.
        block_m, wpe = block_m or 64, waves_per_eu or 1
    else:
        block_m, wpe = block_m or (128 if head_size == 128 else 64), waves_per_eu or 2
    num_q_blocks = (int(max_seqlen_q) * (group if pack else 1) + block_m - 1) // block_m
    window_fits = 0 <= window_left and window_left + int(max_seqlen_q) <= segment
    if mode == "fused" or not pack or not segment_tiles or (mode == "auto" and window_fits):
        split = False
    elif mode == "split":
        split = True
    else:
        split = num_seqs * num_kv_heads * num_q_blocks < 2 * _num_cus(q.device.index or 0)

    def build(split_mode, segs=1):
        return _module(
            num_q_heads, num_kv_heads, head_size, not builtin, not builtin, k.shape[1] if paged else 0, pack,
            sinks is not None, bool(return_lse), block_m, wpe, segment_tiles, split_mode, segs, rows,
            SEQ_MAJOR and pack and split_mode != 2, KV_CACHE_AUX if pack else 0,
        )  # fmt: skip

    kw = {}
    if not builtin:
        tokens_per_group = rows // group if pack else rows
        cu_seqlens_k, vis_lo, vis_hi, vis_group = _metadata(
            cu_seqlens_q, cu_seqlens_k, seq_lens if paged else None, num_tokens, vis_lo, vis_hi, window_left,
            tokens_per_group,
        )  # fmt: skip
        kw.update(vis_lo=vis_lo, vis_hi=vis_hi, vis_group=vis_group)
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
        if not split_segs:
            # Walking several segments per program pays only with many programs: keep at least
            # 4 per CU, up to 8 segments each.
            programs = num_seqs * num_kv_heads * num_q_blocks * num_segments
            split_segs = 1
            while split_segs < 8 and programs // (2 * split_segs) >= 4 * _num_cus(q.device.index or 0):
                split_segs *= 2
        split_segs = min(split_segs, num_segments)
        slots = -(-num_segments // split_segs) * split_segs + 1  # + the trash slot of multi-segment programs
        per_thread = 2 + head_size * rows // 64
        threads = 64 * block_m // rows
        # One row per program; 2D so no dimension passes the launcher's int32.
        workspace = torch.empty(
            (slots * num_seqs * num_q_blocks * num_kv_heads, threads * per_thread),
            dtype=torch.float32,
            device=q.device,
        )
        build(1, split_segs)(q, k, v, out, num_seqs, int(max_seqlen_q), partial=workspace, **kw)
        build(2)(q, k, v, out, num_seqs, int(max_seqlen_q), partial=workspace, **kw)
    else:
        build(0)(q, k, v, out, num_seqs, int(max_seqlen_q), **kw)
    return out, lse


rl_attention_bwd_flydsl = rl_attention_bwd_triton
