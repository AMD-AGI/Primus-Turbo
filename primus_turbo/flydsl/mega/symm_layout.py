###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``SymLayout`` for the fused mega MoE -- one struct + a ``get_*`` suite.

Replaces the scattered cross-rank pointer tables that ``mega_moe_fused.py``'s
``SymmBuffer`` threads to its kernels (``pool_ptrs`` / ``buffer_base`` /
``buffer_offsets`` / ``gate_addrs`` / ``comb_addrs`` / ``barrier_addrs`` /
``scoreboard_ptrs``) with ONE by-value struct carrying both symmetric heaps:

  * cached ``main`` heap    -> ``base`` + ``main_delta_ptr`` (per-peer delta table)
  * uncached ``signal`` pad -> ``sig_base``  + ``sig_delta_ptr``

Design (readable + extensible):
  * each heap is declared ONCE as an ordered list of ``Region(name, itemsize, numel)``
    (the single source of truth for the byte layout, mirroring mega_moe_fused);
  * ``get_ptr(layout, name, dst_rank=None, index=0)`` is the generic accessor -- it
    routes ``name`` to its heap automatically, so adding a sub-buffer is a one-line
    Region edit and ``get_ptr`` works immediately;
  * the named ``get_<name>_ptr`` helpers are thin sugar over ``get_ptr``.

Pass ``dst_rank`` (an fx i32) to get the peer-translated address (SymBuffer::map
folded in via the heap's delta table). Pair the result with
``prims.addr_buffer_resource`` (vectorized) or ``prims.addr_elem_ptr_i32`` (scalar).

All offset math constant-folds at trace time (shape members are ``Constexpr[int]``),
so the accessors are a zero-cost abstraction -- a local access compiles to
``base_reg + immediate (+ index<<log2(itemsize))`` with no extra load, and a
cross-rank access adds exactly one i64 delta load (see [[project_flydsl_sym_layout]]).
"""

import functools
from collections import namedtuple

import flydsl.expr as fx
from flydsl.expr import Int32, Int64, struct
from flydsl.expr.buffer_ops import buffer_load
from flydsl.expr.typing import Constexpr

from primus_turbo.flydsl.mega.prims import addr_buffer_resource

__all__ = [
    "SymLayout",
    "MegaSymLayout",
    "derive_dims",
    "get_ptr",
    "map_to_peer",
    "main_offset_spec",
    "signal_offset_spec",
    "main_num_bytes",
    "signal_num_bytes",
]

# per-sub-buffer base alignment (matches mega_moe_fused._BASE_ALIGNMENT)
_BASE_ALIGNMENT = 256

# dtype byte sizes used by the sub-buffers
_SIZE_BF16, _SIZE_I32, _SIZE_F32, _SIZE_I64 = 2, 4, 4, 8

# one sub-buffer's layout entry: dtype byte size + element count (a function of dims)
Region = namedtuple("Region", ("name", "itemsize", "numel"))


def _align_up(nbytes, alignment=_BASE_ALIGNMENT):
    return (nbytes + alignment - 1) // alignment * alignment


# ---------------------------------------------------------------------------
# Shape derivation (mirrors mega_moe_fused.get_symm_buffer_size_for_mega_moe)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def derive_dims(
    num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, block_m, pool_mult
):
    """Derive every shape scalar the layout needs from the eight input dimensions.

    Cached: the layout is identical for a given shape, so each ``get_*`` accessor
    reuses one computed result instead of rebuilding it (trace-time only -- the
    compiled kernel sees folded constants either way)."""
    experts_per_rank = num_experts // num_ranks
    avg_recv_tokens = num_max_tokens_per_rank * num_topk
    pool_capacity = _align_up(pool_mult * avg_recv_tokens + experts_per_rank * block_m, block_m)
    num_pool_blocks = pool_capacity // block_m
    combine_slots = num_topk * num_max_tokens_per_rank
    return dict(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        block_m=block_m,
        pool_mult=pool_mult,
        experts_per_rank=experts_per_rank,
        pool_capacity=pool_capacity,
        num_pool_blocks=num_pool_blocks,
        combine_slots=combine_slots,
    )


# ---------------------------------------------------------------------------
# Heap declarations -- the single source of truth for the byte layout.
# To add / remove a sub-buffer, edit ONE Region list; routing + get_ptr follow.
# Order must match mega_moe_fused (a parity test guards against drift).
# ---------------------------------------------------------------------------
def _main_regions(dims):
    """Cached main heap: cross-rank pool + local scratch + GEMM intermediates."""
    capacity, hidden, intermediate = dims["pool_capacity"], dims["hidden"], dims["intermediate_hidden"]
    return [
        Region("pool", _SIZE_BF16, capacity * hidden),
        Region("c_buffer", _SIZE_I32, dims["num_ranks"] * dims["num_experts"]),
        Region("signal", _SIZE_I32, dims["num_ranks"]),
        Region("origin_rank", _SIZE_I32, capacity),
        Region("origin_slot", _SIZE_I32, capacity),
        Region("weight_recv_buf", _SIZE_F32, capacity),
        Region("combine_gate", _SIZE_F32, dims["combine_slots"]),
        Region("meta_scalars", _SIZE_I32, 8),
        Region("grid_barrier_state", _SIZE_I32, 2),
        Region("profile", _SIZE_I64, 8),
        Region("act", _SIZE_BF16, capacity * intermediate),
        Region("l2_token_buffer", _SIZE_BF16, capacity * hidden),
    ]


def _signal_regions(dims):
    """Uncached signal pad: spin-wait flags + the cross-rank combine buffer."""
    num_pool_blocks = dims["num_pool_blocks"]
    return [
        Region("_ipc_barrier", _SIZE_I32, dims["num_ranks"]),
        Region("scoreboard", _SIZE_I32, num_pool_blocks),
        Region("sb_consume", _SIZE_I32, num_pool_blocks),
        Region("sb_l2", _SIZE_I32, num_pool_blocks),
        Region("comb", _SIZE_BF16, dims["combine_slots"] * dims["hidden"]),
        Region("barrier_local", _SIZE_I32, dims["combine_slots"]),
    ]


# describes one symmetric heap: where its base/delta live on the struct + its regions
HeapSpec = namedtuple("HeapSpec", ("base_attr", "delta_attr", "regions_fn"))

_HEAPS = {
    "main": HeapSpec("base", "main_delta_ptr", _main_regions),
    "signal": HeapSpec("sig_base", "sig_delta_ptr", _signal_regions),
}

# name -> heap routing (names are dims-independent, so resolve once at import)
_CANON_DIMS = derive_dims(1, 1, 1, 1, 1, 1, 1, 1)
_HEAP_OF_BUFFER = {
    region.name: heap_name for heap_name, heap in _HEAPS.items() for region in heap.regions_fn(_CANON_DIMS)
}


# ---------------------------------------------------------------------------
# Offset spec: assign each region a 256B-aligned byte offset (cached per shape)
# ---------------------------------------------------------------------------
def _assign_offsets(regions):
    """Assign a 256B-aligned byte offset to each region.

    Returns ``(offset_spec{name: (byte_offset, itemsize, numel)}, total_bytes)``."""
    cursor, offset_spec = 0, {}
    for name, itemsize, numel in regions:
        cursor = _align_up(cursor)
        offset_spec[name] = (cursor, itemsize, numel)
        cursor += numel * itemsize
    return offset_spec, _align_up(cursor)


# the eight inputs to ``derive_dims``, in order -- a layout's identity
_SHAPE_KEYS = (
    "num_ranks",
    "num_experts",
    "num_max_tokens_per_rank",
    "num_topk",
    "hidden",
    "intermediate_hidden",
    "block_m",
    "pool_mult",
)

_MAX_BLOCK_M = 256


def _shape_key(dims):
    """Hashable 8-tuple identifying a layout (the inputs to ``derive_dims``)."""
    return tuple(dims[k] for k in _SHAPE_KEYS)


@functools.lru_cache(maxsize=None)
def _heap_layout(heap_name, shape_key):
    """Cached ``(offset_spec, total_bytes)`` for one heap at one shape."""
    dims = derive_dims(*shape_key)
    return _assign_offsets(_HEAPS[heap_name].regions_fn(dims))


def main_offset_spec(dims):
    return _heap_layout("main", _shape_key(dims))[0]


def signal_offset_spec(dims):
    return _heap_layout("signal", _shape_key(dims))[0]


def main_num_bytes(dims):
    """Total cached main heap bytes (== mega_moe_fused num_bytes)."""
    return _heap_layout("main", _shape_key(dims))[1]


def signal_num_bytes(dims):
    """Total uncached signal pad bytes (== mega_moe_fused signal_bytes)."""
    return _heap_layout("signal", _shape_key(dims))[1]


def get_num_max_pool_token(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank):
    num_max_recv_tokens = num_ranks * num_max_tokens_per_rank
    num_max_experts_per_token = min(num_topk, num_experts_per_rank)
    return num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (_MAX_BLOCK_M - 1)


# ---------------------------------------------------------------------------
# The struct (passed to kernels by value)
# ---------------------------------------------------------------------------


@struct
class SymLayout:
    buffer_base: Int64
    # i64* to cached-heap per-peer delta table (peer_base[i] - base)
    main_delta_ptr: Int64
    signal_base: Int64
    sig_delta_ptr: Int64  # i64* to signal-pad per-peer delta table
    rank_idx: Int32
    num_ranks: Constexpr[int]
    num_experts: Constexpr[int]
    num_max_tokens_per_rank: Constexpr[int]
    num_topk: Constexpr[int]
    hidden: Constexpr[int]
    intermediate_hidden: Constexpr[int]
    num_max_pool_tokens: Constexpr[int]
    num_max_pool_blocks: Constexpr[int]


def _dims_of(layout):
    """Read the Constexpr shape members off a SymLayout instance and derive the rest."""
    return derive_dims(
        int(layout.num_ranks),
        int(layout.num_experts),
        int(layout.num_max_tokens_per_rank),
        int(layout.num_topk),
        int(layout.hidden),
        int(layout.intermediate_hidden),
        int(layout.block_m),
        int(layout.pool_mult),
    )


# ---------------------------------------------------------------------------
# Device address helpers
# ---------------------------------------------------------------------------
def _to_i64(value):
    """Sign-extend an fx i32 (or fold a python int) to an i64 ArithValue."""
    if isinstance(value, int):
        return fx.Int64(value)
    return fx.arith.ArithValue(fx.arith.extsi(fx.T.i64(), value.ir_value()), signed=True)


def _sub_buffer_addr(base, delta_ptr, num_ranks, byte_offset, itemsize, index, dst_rank):
    """``base + byte_offset + index*itemsize`` (+ per-peer delta[dst_rank] if given), as i64.

    All-constant terms fold to immediates; only a runtime ``index`` emits a shift and
    only ``dst_rank`` emits the (unavoidable) one i64 delta load."""
    addr = base + fx.Int64(byte_offset)
    if not (isinstance(index, int) and index == 0):
        addr = addr + _to_i64(index) * fx.Int64(itemsize)
    if dst_rank is not None:
        delta_table = addr_buffer_resource(delta_ptr, num_records_bytes=num_ranks * 8)
        addr = addr + buffer_load(delta_table, dst_rank, vec_width=1, dtype=fx.T.i64())
    return addr


def get_ptr(layout, name, dst_rank=None, index=0):
    """Generic accessor: i64 address of sub-buffer ``name`` at element ``index``.

    Routes ``name`` to its heap automatically. ``dst_rank`` (fx i32) returns the
    peer-translated address; ``None`` returns the local address. This is the
    extensible entry point -- a new Region is reachable here with no extra code."""
    heap = _HEAPS[_HEAP_OF_BUFFER[name]]
    dims = _dims_of(layout)
    byte_offset, itemsize, _ = _heap_layout(_HEAP_OF_BUFFER[name], _shape_key(dims))[0][name]
    return _sub_buffer_addr(
        getattr(layout, heap.base_attr),
        getattr(layout, heap.delta_attr),
        dims["num_ranks"],
        byte_offset,
        itemsize,
        index,
        dst_rank,
    )


def map_to_peer(layout, ptr, dst_rank, *, on_signal=False):
    """Translate a local ptr to peer ``dst_rank`` using the chosen heap's delta table."""
    delta_ptr = layout.sig_delta_ptr if on_signal else layout.main_delta_ptr
    delta_table = addr_buffer_resource(delta_ptr, num_records_bytes=int(layout.num_ranks) * 8)
    return ptr + buffer_load(delta_table, dst_rank, vec_width=1, dtype=fx.T.i64())


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


# ---------------------------------------------------------------------------
# Host builder
# ---------------------------------------------------------------------------
class MegaSymLayout:
    """Host holder: builds a ``SymLayout`` from a ``SymmBuffer`` and keeps the two
    per-peer delta tables alive. Pass ``.layout`` to a kernel."""

    def __init__(self, layout, main_deltas, sig_deltas):
        self.layout = layout
        # keepalive: data_ptr referenced by layout.main_delta_ptr
        self.main_deltas = main_deltas
        self.sig_deltas = sig_deltas

    @classmethod
    def from_symm_buffer(cls, symm):
        import torch

        symm_mem = symm.sm
        rank = symm.rank

        def _delta_table(base_ptrs):
            return torch.tensor(
                [base_ptrs[peer] - base_ptrs[rank] for peer in range(symm.num_ranks)],
                dtype=torch.int64,
                device="cuda",
            )

        main_deltas = _delta_table(symm_mem.buffer_ptrs)
        sig_deltas = _delta_table(symm_mem.signal_pad_ptrs)
        layout = SymLayout(
            base=Int64(symm_mem.buffer_ptrs[rank]),
            main_delta_ptr=Int64(main_deltas.data_ptr()),
            sig_base=Int64(symm_mem.signal_pad_ptrs[rank]),
            sig_delta_ptr=Int64(sig_deltas.data_ptr()),
            rank_idx=Int32(rank),
            num_ranks=symm.num_ranks,
            num_experts=symm.num_experts,
            num_max_tokens_per_rank=symm.num_max_tokens_per_rank,
            num_topk=symm.num_topk,
            hidden=symm.hidden,
            intermediate_hidden=symm.intermediate_hidden,
            block_m=symm.block_m,
            # not stored on SymmBuffer; invert pool_capacity
            pool_mult=_recover_pool_mult(symm),
        )
        return cls(layout, main_deltas, sig_deltas)


def _recover_pool_mult(symm):
    """pool_mult isn't stored on SymmBuffer; recover it by inverting the pool_capacity formula."""
    avg_recv_tokens = symm.num_max_tokens_per_rank * symm.num_topk
    experts_per_rank = symm.num_experts // symm.num_ranks
    for pool_mult in range(1, 65):
        capacity = _align_up(pool_mult * avg_recv_tokens + experts_per_rank * symm.block_m, symm.block_m)
        if capacity == symm.pool_capacity:
            return pool_mult
    raise ValueError("cannot recover pool_mult from SymmBuffer shapes")
