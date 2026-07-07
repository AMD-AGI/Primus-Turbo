###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from collections import namedtuple
from functools import lru_cache
from typing import TYPE_CHECKING

import flydsl.expr as fx
import torch
from flydsl.expr import Int32, Int64, struct
from flydsl.expr.buffer_ops import buffer_load
from flydsl.expr.typing import Constexpr

from primus_turbo.flydsl.mega.prims import addr_buffer_resource

BLOCK_M = 256  # pool-block granularity (fixed policy)
HEAP_ALIGN = 256  # every region starts on a 256B boundary


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def get_num_max_pool_tokens(
    num_ranks: int, num_max_tokens_per_rank: int, num_topk: int, num_experts_per_rank: int
) -> int:
    """Worst-case pool capacity: all ranks send max tokens + per-expert BLOCK_M padding."""
    num_max_recv_tokens = num_ranks * num_max_tokens_per_rank
    num_max_experts_per_token = min(num_topk, num_experts_per_rank)
    return align_up(
        num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (BLOCK_M - 1), BLOCK_M
    )


# --- heap layout description ---------------------------------------------------


class RegionSpec:
    """Blueprint for one heap region, declared as a class field; ``shape=None`` keeps it flat.

    Sizes are concrete ints because SymLayoutMeta bakes the shape dims in as template
    params, so a spec reads as ``RegionSpec(torch.int32, num_ranks * num_experts)`` -- no
    lambda. ``dtype`` may be the dispatched-token dtype (bf16 / fp8) or a fixed dtype.
    A spec carries no address; assigning it an offset yields a :class:`PlacedRegion`.
    """

    def __init__(self, dtype: torch.dtype, numel: int, shape=None):
        self.dtype = dtype
        self.numel = numel
        self.shape = shape if shape is not None else (numel,)

    def __set_name__(self, owner, name):
        self.name = name


# A RegionSpec resolved to its 256B-aligned byte offset within the heap.
PlacedRegion = namedtuple("PlacedRegion", ["name", "offset", "dtype", "shape", "nbytes"])


class _LayoutMeta(type):
    """Gather the declared RegionSpec fields into ``_region_specs``, in declaration order."""

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        cls._region_specs = tuple(v for v in namespace.values() if isinstance(v, RegionSpec))
        return cls


class _SymLayoutMetaBase(metaclass=_LayoutMeta):
    """Region-introspection API shared by every SymLayoutMeta specialization.

    Declaration order == memory order. Concrete specializations (with sized regions)
    come from :func:`_sym_layout_meta`.
    """

    @classmethod
    def region_specs(cls):
        """All region blueprints, as declared (declaration order == memory order)."""
        return cls._region_specs

    @classmethod
    @lru_cache(maxsize=None)
    def placed_regions(cls):
        """Every region spec resolved to its 256B-aligned offset (cached per specialization)."""
        placed, cursor = [], 0
        for spec in cls._region_specs:
            cursor = align_up(cursor, HEAP_ALIGN)
            nbytes = spec.numel * spec.dtype.itemsize
            placed.append(PlacedRegion(spec.name, cursor, spec.dtype, spec.shape, nbytes))
            cursor += nbytes
        return tuple(placed)

    @classmethod
    def offset_of(cls, region_name: str) -> int:
        """Byte offset of one region within the heap (KeyError if unknown)."""
        for placed in cls.placed_regions():
            if placed.name == region_name:
                return placed.offset
        raise KeyError(region_name)

    @classmethod
    def num_nbytes(cls) -> int:
        """Total 256B-aligned heap size, for allocation."""
        last = cls.placed_regions()[-1]
        return align_up(last.offset + last.nbytes, HEAP_ALIGN)

    @classmethod
    def split_buffer(cls, buffer: "torch.Tensor"):
        """Split a flat int8 IPC heap into one non-owning tensor per region.

        Each view sits at its 256B-aligned offset, typed and reshaped to its region;
        each has its own Storage (like get_buffer) so it aliases the heap without
        tripping torch custom-op alias checks. Returned as a named tuple.
        """
        from primus_turbo.pytorch.core.symm_mem import _tensor_from_device_ptr

        placed = cls.placed_regions()
        total_bytes = cls.num_nbytes()
        buffer_nbytes = buffer.numel() * buffer.element_size()
        assert buffer_nbytes >= total_bytes, f"buffer too small: {buffer_nbytes} < {total_bytes} bytes"
        base_addr, device_index = buffer.data_ptr(), buffer.device.index
        views = [
            _tensor_from_device_ptr(base_addr + p.offset, p.shape, p.dtype, device_index) for p in placed
        ]
        RegionViews = _region_views_type(tuple(p.name for p in placed))
        return RegionViews(*views)


@lru_cache(maxsize=None)
def get_sym_layout_meta(
    dtype: torch.dtype,
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    num_max_pool_tokens: int,
    num_max_pool_blocks: int,
    num_combine_slots: int,
):
    """SymLayoutMeta specialized to these shape template params."""

    class SymLayoutMeta(_SymLayoutMetaBase):
        dispatch_token_pool = RegionSpec(dtype, num_max_pool_tokens * hidden, (num_max_pool_tokens, hidden))
        expert_count_buffer = RegionSpec(torch.int32, num_ranks * num_experts)
        signal = RegionSpec(torch.int32, num_ranks)
        pool_src_rank = RegionSpec(torch.int32, num_max_pool_tokens)
        pool_src_slot = RegionSpec(torch.int32, num_max_pool_tokens)
        weight_recv_buf = RegionSpec(torch.float32, num_max_pool_tokens)
        combine_gate = RegionSpec(torch.float32, num_combine_slots, (num_max_tokens_per_rank, num_topk))
        meta_scalars = RegionSpec(torch.int32, 8)
        grid_sync_count = RegionSpec(torch.int32, 2)
        l2_token_buffer = RegionSpec(dtype, num_max_pool_tokens * hidden, (num_max_pool_tokens, hidden))
        dispatch_flag = RegionSpec(torch.int64, 2 * num_max_pool_blocks)
        combine_flag = RegionSpec(torch.int64, 2 * num_max_pool_blocks)
        combine_token_buffer = RegionSpec(dtype, num_combine_slots * hidden, (num_combine_slots, hidden))
        reduce_flag = RegionSpec(torch.int64, 2 * num_combine_slots)
        # combine recv-segment table: one entry per (local_expert, source_rank).
        combine_recv_dst_rank = RegionSpec(torch.int32, num_experts)
        combine_recv_start_row = RegionSpec(torch.int32, num_experts)
        combine_recv_count = RegionSpec(torch.int32, num_experts)

    # expose derived pool dims so callers read them off the layout, not a kernel handle
    SymLayoutMeta.num_max_pool_tokens = num_max_pool_tokens
    SymLayoutMeta.num_max_pool_blocks = num_max_pool_blocks
    SymLayoutMeta.num_combine_slots = num_combine_slots
    return SymLayoutMeta


def get_sym_layout_meta(sl: "SymLayout", dtype: torch.dtype = torch.bfloat16):
    """The SymLayoutMeta for this handle (dims read off its Constexpr fields, cached per shape)."""
    return _sym_layout_meta(
        dtype,
        int(sl.num_ranks),
        int(sl.num_experts),
        int(sl.num_max_tokens_per_rank),
        int(sl.num_topk),
        int(sl.hidden),
        int(sl.num_max_pool_tokens),
        int(sl.num_max_pool_blocks),
        int(sl.num_combine_slots),
    )


def num_nbytes(sl: "SymLayout") -> int:
    """Total 256B-aligned byte size of the single IPC heap (for allocation)."""
    return get_sym_layout_meta(sl).num_nbytes()


@lru_cache(maxsize=None)
def _region_views_type(names: tuple):
    # Named view bundle: consumers may unpack by name; positional unpack still works.
    return namedtuple("RegionViews", names)


def sym_map(sl: "SymLayout", ptr: fx.Numeric, dst_rank: fx.Numeric) -> fx.Numeric:
    """Remap a local pointer to peer ``dst_rank`` via the IPC delta table."""
    resource = addr_buffer_resource(sl.offsets_ptr, num_records_bytes=int(sl.num_ranks) * 8)
    return ptr + buffer_load(resource, dst_rank, vec_width=1, dtype=fx.T.i64())


# Host-side layout API forwarded from the handle, auto-derived from the meta base so a
# new method there is delegated with no separate list to maintain. num_nbytes is shadowed
# by the handle's own property (attached below), so it resolves there, not via delegation.
_META_API = frozenset(n for n in vars(_SymLayoutMetaBase) if not n.startswith("_"))


# flydsl @struct rebuilds the class from annotations only, dropping any body method or
# property -- so accessors are attached after. ``sl.<region>_ptr`` -> local device address
# (base + compile-time offset); ``sl.map(ptr, dst)`` -> the peer address; the meta API is
# delegated to the resolved SymLayoutMeta. sl is the kernel PARAM (MLIR-backed), so all
# stay valid inside dynamic control flow. Only an explicit allow-list is forwarded, so
# dunder / field probes during @struct construction raise AttributeError, not recurse.
def _sym_layout_getattr(self, name: str):
    if name.endswith("_ptr"):
        try:
            offset = get_sym_layout_meta(self).offset_of(name[:-4])
        except KeyError:
            pass
        else:
            return self.base + fx.Int64(offset)
    if name in _META_API:
        return getattr(get_sym_layout_meta(self), name)
    raise AttributeError(name)


# The kernel PARAM handle: base / offsets_ptr / rank_idx (MLIR-backed) + Constexpr dims.
# @struct drops body methods (see above), so the accessors are attached right after.
@struct
class SymLayout:
    base: Int64
    offsets_ptr: Int64
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
    num_combine_slots: Constexpr[int]

    # Members attached after the class body (see below). Guarded by TYPE_CHECKING so
    # @struct never sees them as fields; this only teaches the type checker they exist.
    if TYPE_CHECKING:

        @property
        def num_nbytes(self) -> int: ...
        def map(self, ptr: fx.Numeric, dst_rank: fx.Numeric) -> fx.Numeric: ...

        # forwarded to the resolved SymLayoutMeta (see _META_API)
        def split_buffer(self, buffer: "torch.Tensor"): ...
        def offset_of(self, region_name: str) -> int: ...
        @classmethod
        def placed_regions(cls) -> tuple: ...
        @classmethod
        def region_specs(cls) -> tuple: ...
        def __getattr__(self, name: str) -> fx.Numeric: ...  # the ``<region>_ptr`` accessors


SymLayout.__getattr__ = _sym_layout_getattr
SymLayout.map = lambda self, ptr, dst_rank: sym_map(self, ptr, dst_rank)
# total 256B-aligned heap size for this handle's shape
SymLayout.num_nbytes = property(lambda self: get_sym_layout_meta(self).num_nbytes())


def get_sym_layout(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    *,
    base: int = 0,
    offsets_ptr: int = 0,
    rank_idx: int = 0,
) -> "SymLayout":

    num_ranks, num_experts = int(num_ranks), int(num_experts)
    num_experts_per_rank = num_experts // num_ranks
    pool_tokens = get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank)
    pool_blocks = pool_tokens // BLOCK_M
    num_combine_slots = num_max_tokens_per_rank * num_topk

    return SymLayout(
        base=Int64(base),
        offsets_ptr=Int64(offsets_ptr),
        rank_idx=Int32(rank_idx),
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_experts_per_rank=num_experts_per_rank,
        num_max_tokens_per_rank=int(num_max_tokens_per_rank),
        num_topk=int(num_topk),
        hidden=int(hidden),
        intermediate_hidden=int(intermediate_hidden),
        num_max_pool_tokens=pool_tokens,
        num_max_pool_blocks=pool_blocks,
        num_combine_slots=num_combine_slots,
    )
