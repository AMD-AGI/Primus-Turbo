###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``SymLayout`` for the fused mega MoE -- one by-value struct + a ``get_*`` suite.

Carries BOTH symmetric heaps so the kernels stop threading scattered pointer
tables (``pool_ptrs`` / ``buffer_base`` / ``gate_addrs`` / ``scoreboard_ptrs`` ...):

  * cached ``main`` heap   -> ``buffer_base`` + ``buffer_offsets_ptr`` (per-peer deltas)
  * uncached ``signal`` pad -> ``signal_base`` + ``signal_offsets_ptr``

Layout shape derives from the DeepGEMM reference
(``DeepGEMM/deep_gemm/include/deep_gemm/layout/mega_moe.cuh``): the shared token
pool is sized by ``get_num_max_pool_tokens`` (worst-case received tokens + per-expert
``BLOCK_M`` padding, aligned to the candidate-BLOCK_M LCM), and the scoreboard
granularity is ``num_max_pool_blocks = num_max_pool_tokens / kMinCandidateBlockM``.

The byte layout lives in ONE place: ``_regions(dims)`` lists every sub-buffer (its
heap, flydsl dtype, element count) in order, and ``layout_spec(dims)`` packs each to
a 256B-aligned offset. Adding a sub-buffer is a one-line ``_regions`` edit.

Pass ``dst_rank`` (an fx i32) to ``get_ptr`` for the peer-translated address (one i64
delta load); pass ``None`` for the local address. All offset math constant-folds at
trace time (shape members are ``Constexpr[int]``), so a local access compiles to
``base + immediate (+ index<<log2(itemsize))``.
"""

import functools
from collections import namedtuple

import flydsl.expr as fx
from flydsl.expr import BFloat16, Float32, Int32, Int64, struct
from flydsl.expr.buffer_ops import buffer_load
from flydsl.expr.typing import Constexpr

from primus_turbo.flydsl.mega.prims import addr_buffer_resource

__all__ = [
    "SymLayout",
    "get_num_max_pool_tokens",
    "layout_spec",
    "num_bytes",
    "get_ptr",
    "map_to_peer",
]

# constants mirrored from layout/mega_moe.cuh
_MAX_BLOCK_M = 192  # kMaxCandidateBlockM
_MIN_BLOCK_M = 8  # kMinCandidateBlockM (finest scoreboard granularity)
_LCM_BLOCK_M = 384  # kLCMCandidateBlockM (pool alignment)
_BASE_ALIGNMENT = 256  # per-sub-buffer base alignment


def _align(x, a=_BASE_ALIGNMENT):
    return (x + a - 1) // a * a


def get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank):
    """Worst-case shared token-pool capacity (mega_moe.cuh ``get_num_max_pool_tokens``)."""
    num_max_recv_tokens = num_ranks * num_max_tokens_per_rank
    num_max_experts_per_token = min(num_topk, num_experts_per_rank)
    return _align(
        num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (_MAX_BLOCK_M - 1),
        _LCM_BLOCK_M,
    )


# ---------------------------------------------------------------------------
# Byte layout -- the single source of truth for both heaps.
# Edit ONE list to add / remove a sub-buffer; packing + get_ptr follow.
# ---------------------------------------------------------------------------
Region = namedtuple("Region", ("name", "heap", "dtype", "numel"))
Entry = namedtuple("Entry", ("heap", "offset", "itemsize", "numel"))


@functools.lru_cache(maxsize=None)
def layout_spec(
    num_ranks,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    num_max_pool_tokens,
    num_max_pool_blocks,
):
    """Pack every sub-buffer; return ``(entries{name: Entry}, totals{heap: bytes})``.

    Args are exactly the ``SymLayout`` Constexpr shape fields, in declaration order.
    Each heap is packed independently with 256B-aligned offsets; cached per shape so
    the ``get_*`` accessors reuse one result (trace-time only).

    main   = cached heap (cross-rank pool + local scratch + GEMM intermediates)
    signal = uncached pad (spin-wait flags + the cross-rank combine buffer)
    """
    cap = num_max_pool_tokens
    combine_slots = num_topk * num_max_tokens_per_rank
    regions = [
        Region("pool", "main", BFloat16, cap * hidden),
        Region("c_buffer", "main", Int32, num_ranks * num_experts),
        Region("signal", "main", Int32, num_ranks),
        Region("origin_rank", "main", Int32, cap),
        Region("origin_slot", "main", Int32, cap),
        Region("weight_recv_buf", "main", Float32, cap),
        Region("combine_gate", "main", Float32, combine_slots),
        Region("meta_scalars", "main", Int32, 8),
        Region("grid_barrier_state", "main", Int32, 2),
        Region("profile", "main", Int64, 8),
        Region("act", "main", BFloat16, cap * intermediate_hidden),
        Region("l2_token_buffer", "main", BFloat16, cap * hidden),
        Region("_ipc_barrier", "signal", Int32, num_ranks),
        Region("scoreboard", "signal", Int32, num_max_pool_blocks),
        Region("sb_consume", "signal", Int32, num_max_pool_blocks),
        Region("sb_l2", "signal", Int32, num_max_pool_blocks),
        Region("comb", "signal", BFloat16, combine_slots * hidden),
        Region("barrier_local", "signal", Int32, combine_slots),
    ]
    entries, cursors = {}, {"main": 0, "signal": 0}
    for name, heap, dtype, numel in regions:
        itemsize = dtype.width // 8
        offset = _align(cursors[heap])
        entries[name] = Entry(heap, offset, itemsize, numel)
        cursors[heap] = offset + numel * itemsize
    totals = {heap: _align(cursor) for heap, cursor in cursors.items()}
    return entries, totals


def num_bytes(*shape):
    """``(main_bytes, signal_bytes)`` for one rank (args as in ``layout_spec``)."""
    _, totals = layout_spec(*shape)
    return totals["main"], totals["signal"]


# ---------------------------------------------------------------------------
# The struct (passed to kernels by value)
# ---------------------------------------------------------------------------
@struct
class SymLayout:
    buffer_base: Int64  # cached main heap base (this rank)
    buffer_offsets_ptr: Int64  # i64* to main-heap per-peer delta table (peer_base[i] - base)
    signal_base: Int64  # uncached signal pad base (this rank)
    signal_offsets_ptr: Int64  # i64* to signal-pad per-peer delta table
    rank_idx: Int32
    num_ranks: Constexpr[int]
    num_experts: Constexpr[int]
    num_max_tokens_per_rank: Constexpr[int]
    num_topk: Constexpr[int]
    hidden: Constexpr[int]
    intermediate_hidden: Constexpr[int]
    num_max_pool_tokens: Constexpr[int]
    num_max_pool_blocks: Constexpr[int]


def _spec_of(sl):
    """``layout_spec`` keyed by a SymLayout instance's Constexpr shape fields."""
    return layout_spec(
        int(sl.num_ranks),
        int(sl.num_experts),
        int(sl.num_max_tokens_per_rank),
        int(sl.num_topk),
        int(sl.hidden),
        int(sl.intermediate_hidden),
        int(sl.num_max_pool_tokens),
        int(sl.num_max_pool_blocks),
    )


# ---------------------------------------------------------------------------
# Device address helpers
# ---------------------------------------------------------------------------
def _to_i64(value):
    """Sign-extend an fx i32 (or fold a python int) to an i64 ArithValue."""
    if isinstance(value, int):
        return fx.Int64(value)
    return fx.arith.ArithValue(fx.arith.extsi(fx.T.i64(), value.ir_value()), signed=True)


def get_ptr(sl, name, dst_rank=None, index=0):
    """i64 address of sub-buffer ``name`` at element ``index``.

    Routes ``name`` to its heap automatically. ``dst_rank`` (fx i32) returns the
    peer-translated address (one i64 delta load); ``None`` returns the local one.
    Constant terms fold to immediates; only a runtime ``index`` emits a shift."""
    entry = _spec_of(sl)[0][name]
    on_signal = entry.heap == "signal"
    addr = (sl.signal_base if on_signal else sl.buffer_base) + fx.Int64(entry.offset)
    if not (isinstance(index, int) and index == 0):
        addr = addr + _to_i64(index) * fx.Int64(entry.itemsize)
    if dst_rank is not None:
        addr = map_to_peer(sl, addr, dst_rank, on_signal=on_signal)
    return addr


def map_to_peer(sl, ptr, dst_rank, *, on_signal=False):
    """Translate a local ptr to peer ``dst_rank`` using the chosen heap's delta table."""
    delta_ptr = sl.signal_offsets_ptr if on_signal else sl.buffer_offsets_ptr
    res = addr_buffer_resource(delta_ptr, num_records_bytes=int(sl.num_ranks) * 8)
    return ptr + buffer_load(res, dst_rank, vec_width=1, dtype=fx.T.i64())


# ---------------------------------------------------------------------------
# Named accessors -- thin, discoverable sugar over get_ptr (one per sub-buffer)
# ---------------------------------------------------------------------------
def _named_accessor(name):
    def accessor(layout, dst_rank=None, index=0):
        return get_ptr(layout, name, dst_rank, index)

    accessor.__name__ = accessor.__qualname__ = f"get_{name.lstrip('_')}_ptr"
    accessor.__doc__ = f"i64 address of the ``{name}`` sub-buffer (see get_ptr)."
    return accessor


# cached main heap
get_pool_ptr = _named_accessor("pool")
get_c_buffer_ptr = _named_accessor("c_buffer")
get_signal_ptr = _named_accessor("signal")
get_origin_rank_ptr = _named_accessor("origin_rank")
get_origin_slot_ptr = _named_accessor("origin_slot")
get_weight_recv_ptr = _named_accessor("weight_recv_buf")
get_combine_gate_ptr = _named_accessor("combine_gate")
get_meta_scalars_ptr = _named_accessor("meta_scalars")
get_grid_barrier_ptr = _named_accessor("grid_barrier_state")
get_profile_ptr = _named_accessor("profile")
get_act_ptr = _named_accessor("act")
get_l2_token_ptr = _named_accessor("l2_token_buffer")

# uncached signal pad
get_ipc_barrier_ptr = _named_accessor("_ipc_barrier")
get_scoreboard_ptr = _named_accessor("scoreboard")
get_sb_consume_ptr = _named_accessor("sb_consume")
get_sb_l2_ptr = _named_accessor("sb_l2")
get_comb_ptr = _named_accessor("comb")
get_barrier_local_ptr = _named_accessor("barrier_local")

__all__ += [
    "get_pool_ptr",
    "get_c_buffer_ptr",
    "get_signal_ptr",
    "get_origin_rank_ptr",
    "get_origin_slot_ptr",
    "get_weight_recv_ptr",
    "get_combine_gate_ptr",
    "get_meta_scalars_ptr",
    "get_grid_barrier_ptr",
    "get_profile_ptr",
    "get_act_ptr",
    "get_l2_token_ptr",
    "get_ipc_barrier_ptr",
    "get_scoreboard_ptr",
    "get_sb_consume_ptr",
    "get_sb_l2_ptr",
    "get_comb_ptr",
    "get_barrier_local_ptr",
]
