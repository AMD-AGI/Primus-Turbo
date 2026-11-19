###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton launchers for RL attention (primus_turbo/triton/attention/rl_attention_{fwd,bwd}.py).

Two kinds of parameters, kept apart on purpose:

* The reduction order (``RLAttentionReduceOrder``: the KV tile size and the KV segment length) sets
  the order of every floating-point reduction a row performs. It must be the same for rollout and
  training of one model, so it is a process-wide setting, never derived from the batch.
* Launch parameters (``_launch``: rows per program, warps, stages, waves, split vs fused mode) only
  schedule work. They may follow the shape freely; every choice gives the same bits. The one
  measured constraint is that warps tile only the rows of the score tile
  (num_warps <= BLOCK_M / 16 when BLOCK_M > 16): warps that also split the columns change the bits.
"""

import functools
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import triton

from primus_turbo.triton.attention import rl_attention_bwd, rl_attention_fwd

SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


@dataclass(frozen=True)
class RLAttentionReduceOrder:
    """KV tile size and KV segment length of the Triton RL attention forward. ``None`` picks the
    default (tile 64, or 32 for head size > 128; segment 1024)."""

    tile_size: Optional[int] = None
    segment_size: Optional[int] = None

    def resolve(self, head_size: int) -> Tuple[int, int]:
        tile = self.tile_size or (32 if head_size > 128 else 64)
        segment = self.segment_size or 1024
        if tile & (tile - 1) or segment % tile:
            raise ValueError(f"tile_size ({tile}) must be a power of 2 dividing segment_size ({segment})")
        return tile, segment


_reduce_order = RLAttentionReduceOrder()


def set_reduce_order(order: RLAttentionReduceOrder) -> None:
    global _reduce_order
    _reduce_order = order


def get_reduce_order() -> RLAttentionReduceOrder:
    return _reduce_order


@dataclass(frozen=True)
class _Launch:
    block_m: int
    num_warps: int
    num_stages: int
    waves_per_eu: int


@functools.lru_cache(maxsize=None)
def _num_cus(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _block_m(queries_per_kv: int, mode: str, max_seqlen_q: int) -> int:
    """Rows per program, a multiple of the GQA group so a block never splits a token."""
    group = triton.next_power_of_2(queries_per_kv)
    if mode == "split" or max_seqlen_q <= 16:
        return max(16, group)
    if max_seqlen_q < 1024:
        return max(128, group)
    return max(256, group)


def _launch(queries_per_kv: int, mode: str, max_seqlen_q: int, split_programs: int = 0) -> _Launch:
    """Tuned on MI355X (D=128, GQA 4, bf16), decode first."""
    block_m = _block_m(queries_per_kv, mode, max_seqlen_q)
    if block_m <= 16 and mode == "split" and split_programs > 512:
        launch = _Launch(block_m, num_warps=2, num_stages=1, waves_per_eu=2)
    elif block_m <= 16 and mode == "split":
        launch = _Launch(block_m, num_warps=4, num_stages=2, waves_per_eu=4)
    elif block_m <= 16:
        launch = _Launch(block_m, num_warps=2, num_stages=3, waves_per_eu=1)
    else:
        launch = _Launch(block_m, num_warps=8, num_stages=2, waves_per_eu=2 if block_m <= 128 else 1)
    if block_m > 16:
        launch = _Launch(
            block_m, min(launch.num_warps, block_m // 16), launch.num_stages, launch.waves_per_eu
        )
    return launch


def _choose_mode(
    mode: str, num_q_blocks: int, num_kv_heads: int, num_segments: int, max_seqlen_q: int, device
) -> str:
    if mode != "auto":
        return mode
    if (
        max_seqlen_q <= 16
        and num_segments > 1
        and num_q_blocks * num_kv_heads < 2 * _num_cus(device.index or 0)
    ):
        return "split"
    return "fused"


def _sliding_window(window_left: int) -> int:
    """Kernel constexpr: a row at position p sees keys (p - SLIDING_WINDOW, p]; 0 means no window."""
    return window_left + 1 if window_left is not None and window_left >= 0 else 0


def rl_attention_fwd_triton(
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward over a paged cache (``block_table`` and ``seq_lens``) or contiguous K/V
    (``cu_seqlens_k``). With visibility, ``cu_seqlens_k`` gives each sequence's first global key
    index in both layouts. Returns out and the natural-log LSE (empty unless ``return_lse``)."""
    paged = block_table is not None
    if paged:
        assert seq_lens is not None and k.ndim == 4
        _, block_size, num_kv_heads, head_size = k.shape
    else:
        assert cu_seqlens_k is not None and k.ndim == 3
        _, num_kv_heads, head_size = k.shape
        block_size = 1
    has_vis = vis_lo is not None
    assert not has_vis or cu_seqlens_k is not None, "visibility needs cu_seqlens_k (key starts)"
    num_tokens, num_q_heads, q_head_size = q.shape
    assert q_head_size == head_size and num_q_heads % num_kv_heads == 0
    assert q.dtype in SUPPORTED_DTYPES and k.dtype == q.dtype and v.dtype == q.dtype
    v_head_size = v.shape[-1]
    queries_per_kv = num_q_heads // num_kv_heads
    num_seqs = cu_seqlens_q.numel() - 1

    tile_size, segment_size = _reduce_order.resolve(head_size)
    num_segments = triton.cdiv(max(int(max_seqlen_k), 1), segment_size)
    probe = _launch(queries_per_kv, "fused", int(max_seqlen_q))
    probe_blocks = num_tokens // (probe.block_m // queries_per_kv) + num_seqs
    mode = _choose_mode(mode, probe_blocks, num_kv_heads, num_segments, int(max_seqlen_q), q.device)
    launch = _launch(
        queries_per_kv, mode, int(max_seqlen_q), split_programs=probe_blocks * num_kv_heads * num_segments
    )
    block_q = launch.block_m // queries_per_kv
    num_q_blocks = num_tokens // block_q + num_seqs

    out = torch.empty((num_tokens, num_q_heads, v_head_size), dtype=q.dtype, device=q.device)
    lse_shape = (num_tokens, num_q_heads) if return_lse else (0,)
    lse = torch.empty(lse_shape, dtype=torch.float32, device=q.device)
    if num_tokens == 0:
        return out, lse

    dummy = cu_seqlens_q
    bt = block_table if paged else dummy
    common = dict(
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        block_tables_ptr=bt,
        seq_lens_ptr=seq_lens if paged else dummy,
        cu_k_ptr=cu_seqlens_k if cu_seqlens_k is not None else dummy,
        cu_q_ptr=cu_seqlens_q,
        vis_lo_ptr=vis_lo if has_vis else dummy,
        vis_hi_ptr=vis_hi if has_vis else dummy,
        sink_ptr=sinks if sinks is not None else dummy,
        scale=float(softmax_scale),
        num_seqs=num_seqs,
        block_table_stride=bt.stride(0) if paged else 0,
        stride_q0=q.stride(0),
        stride_q1=q.stride(1),
        stride_k0=k.stride(0),
        stride_k1=k.stride(1) if paged else 0,
        stride_k2=k.stride(-2),
        stride_k3=k.stride(-1),
        stride_v0=v.stride(0),
        stride_v1=v.stride(1) if paged else 0,
        stride_v2=v.stride(-2),
        stride_v3=v.stride(-1),
        NUM_Q_HEADS=num_q_heads,
        QUERIES_PER_KV=queries_per_kv,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        V_HEAD_SIZE=v_head_size,
        V_HEAD_SIZE_PADDED=triton.next_power_of_2(v_head_size),
        PAGED=paged,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        TILES_PER_SEGMENT=segment_size // tile_size,
        SLIDING_WINDOW=_sliding_window(window_left),
        BLOCK_M=launch.block_m,
        BLOCK_Q=block_q,
        OFFSETS_64=max(k.numel(), v.numel()) >= 2**31,
        KV_CACHE_MODIFIER=".cg" if int(max_seqlen_q) <= 1 else "",
        HAS_VIS=has_vis,
    )
    opts = dict(
        num_warps=launch.num_warps,
        num_stages=launch.num_stages,
        waves_per_eu=launch.waves_per_eu,
        matrix_instr_nonkdim=16,
    )
    lse_arg = lse if return_lse else out

    if mode == "fused":
        rl_attention_fwd.attention_fwd_fused_kernel[(num_kv_heads, num_q_blocks)](
            out_ptr=out,
            lse_ptr=lse_arg,
            stride_o0=out.stride(0),
            stride_o1=out.stride(1),
            STORE_LSE=return_lse,
            HAS_SINK=sinks is not None,
            **common,
            **opts,
        )
    elif mode == "split":
        vd_pad = triton.next_power_of_2(v_head_size)
        segm_m = torch.empty((num_tokens, num_q_heads, num_segments), dtype=torch.float32, device=q.device)
        segm_l = torch.empty_like(segm_m)
        segm_acc = torch.empty(
            (num_tokens, num_q_heads, num_segments, vd_pad), dtype=torch.float32, device=q.device
        )
        rl_attention_fwd.attention_fwd_split_kernel[(num_q_blocks, num_kv_heads, num_segments)](
            segm_m_ptr=segm_m,
            segm_l_ptr=segm_l,
            segm_acc_ptr=segm_acc,
            NUM_SEGMENTS=num_segments,
            **common,
            **opts,
        )
        rl_attention_fwd.attention_fwd_reduce_kernel[(num_tokens, num_kv_heads)](
            out_ptr=out,
            lse_ptr=lse_arg,
            segm_m_ptr=segm_m,
            segm_l_ptr=segm_l,
            segm_acc_ptr=segm_acc,
            seq_lens_ptr=common["seq_lens_ptr"],
            cu_k_ptr=common["cu_k_ptr"],
            cu_q_ptr=cu_seqlens_q,
            vis_lo_ptr=common["vis_lo_ptr"],
            vis_hi_ptr=common["vis_hi_ptr"],
            sink_ptr=common["sink_ptr"],
            num_seqs=num_seqs,
            stride_o0=out.stride(0),
            stride_o1=out.stride(1),
            NUM_SEGMENTS=num_segments,
            NUM_Q_HEADS=num_q_heads,
            QUERIES_PER_KV=queries_per_kv,
            GROUP_PADDED=triton.next_power_of_2(queries_per_kv),
            V_HEAD_SIZE=v_head_size,
            V_HEAD_SIZE_PADDED=vd_pad,
            PAGED=paged,
            SEGMENT_SIZE=segment_size,
            SLIDING_WINDOW=_sliding_window(window_left),
            STORE_LSE=return_lse,
            HAS_VIS=has_vis,
            HAS_SINK=sinks is not None,
            num_warps=1,
        )
    else:
        raise ValueError(f"unknown RL attention mode {mode!r}")
    return out, lse


def rl_attention_bwd_triton(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    window_left: int,
    vis_lo: Optional[torch.Tensor],
    vis_hi: Optional[torch.Tensor],
    sinks: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Varlen backward: dq, dk, dv, and dsink ([num_q_heads] fp32, or None without sinks).
    Deterministic: no atomics, and dsink is a fixed-shape torch reduction."""
    dout = dout.contiguous()
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    rl_attention_bwd.flash_attn_onekernel_backward(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        None,
        softmax_scale,
        None,
        True,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        sliding_window=window_left if window_left is not None and window_left >= 0 else 0,
        visibility=(vis_lo, vis_hi) if vis_lo is not None else None,
    )
    dsink = None
    if sinks is not None:
        # A sink is a key with value 0: dsink_h = -sum_t exp(sink_h - lse_th) * rowsum(dO * O)_th.
        delta = (dout.float() * out.float()).sum(-1)
        dsink = -(torch.exp(sinks.float()[None, :] - lse) * delta).sum(0)
    return dq, dk, dv, dsink
