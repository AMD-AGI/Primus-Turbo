###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import flydsl.expr as fx
import torch
from flydsl.expr import Int32, Int64, struct
from flydsl.expr.buffer_ops import buffer_load
from flydsl.expr.typing import Constexpr

from primus_turbo.flydsl.mega.prims import addr_buffer_resource

_BASE_ALIGNMENT = 256
_BF16, _I32, _F32, _I64 = torch.bfloat16, torch.int32, torch.float32, torch.int64

BLOCK_M = 256  # pool-block granularity (fixed policy)
_POOL_MULT = 2  # pool overprovision factor (fixed policy)


def _align(x, a=_BASE_ALIGNMENT):
    return (x + a - 1) // a * a


def _pool_sizes(num_ranks, num_experts, num_max_tokens_per_rank, num_topk):
    """Derive pool capacity scalars from the fixed block/pool policy."""
    experts_per_rank = num_experts // num_ranks
    avg_recv_tokens = num_max_tokens_per_rank * num_topk
    num_max_pool_tokens = _align(_POOL_MULT * avg_recv_tokens + experts_per_rank * BLOCK_M, BLOCK_M)
    return num_max_pool_tokens, num_max_pool_tokens // BLOCK_M, num_topk * num_max_tokens_per_rank


@struct
class SymLayout:
    base: Int64
    offsets_ptr: Int64
    signal_base: Int64
    signal_offsets_ptr: Int64
    rank_idx: Int32
    num_ranks: Constexpr[int]
    num_experts: Constexpr[int]
    num_experts_per_rank: Constexpr[int]
    num_max_tokens_per_rank: Constexpr[int]
    num_topk: Constexpr[int]
    hidden: Constexpr[int]
    intermediate_hidden: Constexpr[int]
    num_max_pool_tokens: Constexpr[int]
    num_max_pool_blocks: Constexpr[int]
    combine_slots: Constexpr[int]


# Region tables: (name, dtype, numel(sl)). Declaration order == memory order,
# the single source of truth for both device offsets and host allocation -- like
# deep_gemm's mega_moe Workspace. Sizes read SymLayout attrs directly (all
# Constexpr fields are plain ints on the instance).
_MAIN = [
    ("pool", _BF16, lambda s: s.num_max_pool_tokens * s.hidden),
    ("c_buffer", _I32, lambda s: s.num_ranks * s.num_experts),
    ("signal", _I32, lambda s: s.num_ranks),
    ("origin_rank", _I32, lambda s: s.num_max_pool_tokens),
    ("origin_slot", _I32, lambda s: s.num_max_pool_tokens),
    ("weight_recv_buf", _F32, lambda s: s.num_max_pool_tokens),
    ("dedup_src_row", _I32, lambda s: s.num_max_pool_tokens),
    ("combine_gate", _F32, lambda s: s.combine_slots),
    ("meta_scalars", _I32, lambda s: 8),
    ("grid_sync_count", _I32, lambda s: 2),
    ("profile", _I64, lambda s: 8),
    ("act", _BF16, lambda s: s.num_max_pool_tokens * s.intermediate_hidden),
    ("l2_token_buffer", _BF16, lambda s: s.num_max_pool_tokens * s.hidden),
    (
        "src_token_topk_idx",
        _I32,
        lambda s: s.num_experts_per_rank * s.num_ranks * s.num_ranks * s.num_max_tokens_per_rank,
    ),
]
_SIGNAL = [
    ("_ipc_barrier", _I32, lambda s: s.num_ranks),
    ("dispatch_flag", _I64, lambda s: 2 * s.num_max_pool_blocks),
    ("combine_flag", _I64, lambda s: 2 * s.num_max_pool_blocks),
    ("comb", _BF16, lambda s: s.combine_slots * s.hidden),
    ("reduce_flag", _I64, lambda s: 2 * s.combine_slots),
]

# Two IPC heaps -- data arena and signal-pad arena. Each names the SymLayout
# base/delta pointers it rides on plus its region table.
_HEAPS = {
    "main": ("base", "offsets_ptr", _MAIN),
    "signal": ("signal_base", "signal_offsets_ptr", _SIGNAL),
}
_HEAP_OF = {name: heap for heap, (_, _, table) in _HEAPS.items() for name, *_ in table}


def _pack(regions, sl):
    """256B-aligned packer -> ({name: (offset, dtype, numel)}, total_bytes)."""
    out, cursor = {}, 0
    for name, dtype, size in regions:
        numel = size(sl)
        cursor = _align(cursor)
        out[name] = (cursor, dtype, numel)
        cursor += numel * dtype.itemsize
    return out, _align(cursor)


def region_bytes(sl):
    """Return ``(main_off, sig_off, main_bytes, sig_bytes)``; each off maps name -> (offset, dtype, numel)."""
    main_off, main_bytes = _pack(_MAIN, sl)
    sig_off, sig_bytes = _pack(_SIGNAL, sl)
    return main_off, sig_off, main_bytes, sig_bytes


def _region_ptr(sl, name):
    base_attr, _, table = _HEAPS[_HEAP_OF[name]]
    offset = _pack(table, sl)[0][name][0]
    return getattr(sl, base_attr) + fx.Int64(offset)


def _remap(sl, heap, ptr, dst_rank):
    """Remap a pointer into ``heap`` to peer ``dst_rank`` via its IPC delta table."""
    _, delta_attr, _ = _HEAPS[heap]
    res = addr_buffer_resource(getattr(sl, delta_attr), num_records_bytes=int(sl.num_ranks) * 8)
    return ptr + buffer_load(res, dst_rank, vec_width=1, dtype=fx.T.i64())


def sym_map(sl, ptr, dst_rank):
    return _remap(sl, "main", ptr, dst_rank)


def map_signal(sl, ptr, dst_rank):
    return _remap(sl, "signal", ptr, dst_rank)


def build_sym_layout(
    num_ranks,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    *,
    base=0,
    offsets_ptr=0,
    signal_base=0,
    signal_offsets_ptr=0,
    rank_idx=0,
):
    """Build the device SymLayout from the MoE shape; pool sizes are derived here."""
    num_ranks, num_experts = int(num_ranks), int(num_experts)
    pool_tokens, pool_blocks, combine_slots = _pool_sizes(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk
    )
    return SymLayout(
        base=Int64(base),
        offsets_ptr=Int64(offsets_ptr),
        signal_base=Int64(signal_base),
        signal_offsets_ptr=Int64(signal_offsets_ptr),
        rank_idx=Int32(rank_idx),
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_experts_per_rank=num_experts // num_ranks,
        num_max_tokens_per_rank=int(num_max_tokens_per_rank),
        num_topk=int(num_topk),
        hidden=int(hidden),
        intermediate_hidden=int(intermediate_hidden),
        num_max_pool_tokens=pool_tokens,
        num_max_pool_blocks=pool_blocks,
        combine_slots=combine_slots,
    )


# Expose each region as a read-only ``<name>_ptr`` property returning the local
# device address. Peer addresses go through sym_map / map_signal.
for _name, *_ in (*_MAIN, *_SIGNAL):
    setattr(SymLayout, f"{_name}_ptr", property(lambda self, n=_name: _region_ptr(self, n)))
