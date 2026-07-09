import warnings
from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache
from math import prod

import flydsl.expr as fx
import torch
from flydsl.expr import Int32, Int64, struct
from flydsl.expr.buffer_ops import buffer_load
from flydsl.expr.typing import Constexpr

from primus_turbo.flydsl.mega.prims import addr_buffer_resource

__all__ = [
    "SymmBuffer",
    "get_symm_buffer_for_mega_moe",
    "SymLayout",
    "sym_map",
    "make_sym_layout_type",
    "make_sym_layout_meta",
    "LayoutConfig",
]

BLOCK_M = 256  # pool-block granularity (fixed policy)
TOKEN_DTYPE = torch.bfloat16  # dispatched-token dtype for the pool / L2 / combine buffers


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


# --- layout config: pure dimensions, decoupled from IPC allocation -------------


@dataclass(frozen=True)
class LayoutConfig:
    """All dims the RegionSpecs and SymLayout need. Pure data -> sizing is testable without IPC."""

    num_ranks: int
    num_experts: int
    num_experts_per_rank: int
    num_max_tokens_per_rank: int
    num_topk: int
    hidden: int
    intermediate_hidden: int
    num_max_pool_tokens: int
    num_max_pool_blocks: int
    num_combine_slots: int

    @classmethod
    def build(
        cls, num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden
    ) -> "LayoutConfig":
        """Derive the full config (pool capacity, block count, combine slots) from raw MoE dims."""
        assert num_experts % num_ranks == 0, f"num_experts {num_experts} not divisible by ranks {num_ranks}"
        num_experts_per_rank = num_experts // num_ranks
        num_max_pool_tokens = get_num_max_pool_tokens(
            num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank
        )
        return cls(
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_experts_per_rank=num_experts_per_rank,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            num_max_pool_tokens=num_max_pool_tokens,
            num_max_pool_blocks=num_max_pool_tokens // BLOCK_M,
            num_combine_slots=num_max_tokens_per_rank * num_topk,
        )


# --- heap layout description ---------------------------------------------------


class RegionSpec:

    def __init__(self, dtype: torch.dtype, shape: tuple):
        self.dtype = dtype
        self.shape = tuple(shape)
        self.numel = prod(self.shape)

    def __set_name__(self, owner, name):
        self.name = name


# A RegionSpec resolved to its 256B-aligned byte offset within a concrete heap.
PlacedRegion = namedtuple("PlacedRegion", ["name", "offset", "dtype", "shape", "nbytes"])


@lru_cache(maxsize=None)
def _region_views_type(names: tuple):
    # Named view bundle: consumers may unpack by name; positional unpack still works.
    return namedtuple("RegionViews", names)


class _LayoutMeta(type):
    """Gather the declared RegionSpec fields into ``_region_specs``, in declaration order."""

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        cls._region_specs = tuple(v for v in namespace.values() if isinstance(v, RegionSpec))
        return cls


class _SymLayoutMetaBase(metaclass=_LayoutMeta):
    """Region-introspection API shared by every layout meta (declaration order == memory order)."""

    @classmethod
    def region_names(cls) -> tuple:
        """Region field names, in declaration order."""
        return tuple(spec.name for spec in cls._region_specs)

    @classmethod
    def placed_regions(cls):
        """Regions resolved to 256B-aligned byte offsets (declaration order)."""
        placed, cursor = [], 0
        for spec in cls._region_specs:
            cursor = align_up(cursor, BLOCK_M)
            nbytes = spec.numel * spec.dtype.itemsize
            placed.append(PlacedRegion(spec.name, cursor, spec.dtype, spec.shape, nbytes))
            cursor += nbytes
        return tuple(placed)

    @classmethod
    def num_nbytes(cls) -> int:
        """Total 256B-aligned heap size, for allocation."""
        last = cls.placed_regions()[-1]
        return align_up(last.offset + last.nbytes, BLOCK_M)

    @classmethod
    def describe(cls) -> str:
        """Human-readable dump of the resolved heap layout (region / offset / bytes) -- for debugging IPC memory."""
        lines = [f"{cls.__name__} heap layout ({cls.num_nbytes()} bytes total):"]
        for p in cls.placed_regions():
            lines.append(f"  {p.name:<24} @ {p.offset:>10}  {p.nbytes:>12} B  {p.dtype} {tuple(p.shape)}")
        return "\n".join(lines)

    @classmethod
    def split_buffer(cls, buffer: "torch.Tensor"):
        """Split a flat int8 IPC heap into one non-owning typed tensor view per region."""
        from primus_turbo.pytorch.core.symm_mem import _tensor_from_device_ptr

        placed = cls.placed_regions()
        total_bytes = cls.num_nbytes()
        buffer_nbytes = buffer.numel() * buffer.element_size()
        assert buffer_nbytes >= total_bytes, f"buffer too small: {buffer_nbytes} < {total_bytes} bytes"
        base_addr, device_index = buffer.data_ptr(), buffer.device.index
        views = [
            _tensor_from_device_ptr(base_addr + p.offset, p.shape, p.dtype, device_index) for p in placed
        ]
        return _region_views_type(cls.region_names())(*views)


@lru_cache(maxsize=None)
def make_sym_layout_meta(token_dtype: torch.dtype, layout_config: LayoutConfig):
    """Build the heap-layout meta for a token dtype + a concrete LayoutConfig.

    Every region shape reads its dims from ``cfg`` directly (e.g. ``cfg.num_max_pool_tokens``) --
    no magic strings, so a reader sees exactly which LayoutConfig dim sizes each region. The three
    token buffers follow ``token_dtype``; every other region has a fixed dtype. Declaration
    order == memory order.
    """
    cfg = layout_config

    # regular class body: the metaclass collects RegionSpecs in declaration order (== memory order)
    class SymLayoutMeta(_SymLayoutMetaBase):
        # token buffers: dtype follows the template parameter
        dispatch_token_pool = RegionSpec(token_dtype, (cfg.num_max_pool_tokens, cfg.hidden))
        expert_count_buffer = RegionSpec(torch.int32, (cfg.num_ranks, cfg.num_experts))
        signal = RegionSpec(torch.int32, (cfg.num_ranks,))
        pool_src_rank = RegionSpec(torch.int32, (cfg.num_max_pool_tokens,))
        pool_src_slot = RegionSpec(torch.int32, (cfg.num_max_pool_tokens,))
        weight_recv_buf = RegionSpec(torch.float32, (cfg.num_max_pool_tokens,))
        combine_gate = RegionSpec(torch.float32, (cfg.num_max_tokens_per_rank, cfg.num_topk))
        meta_scalars = RegionSpec(torch.int32, (8,))
        grid_sync_count = RegionSpec(torch.int32, (2,))
        l2_token_buffer = RegionSpec(token_dtype, (cfg.num_max_pool_tokens, cfg.hidden))
        # flags: double-buffered, flat 1-D (host slices two banks of X out of 2*X)
        dispatch_flag = RegionSpec(torch.int64, (2 * cfg.num_max_pool_blocks,))
        combine_flag = RegionSpec(torch.int64, (2 * cfg.num_max_pool_blocks,))
        combine_token_buffer = RegionSpec(token_dtype, (cfg.num_combine_slots, cfg.hidden))
        reduce_flag = RegionSpec(torch.int64, (2 * cfg.num_combine_slots,))
        # combine recv-segment table: one entry per (local_expert, source_rank)
        combine_recv_dst_rank = RegionSpec(torch.int32, (cfg.num_experts,))
        combine_recv_start_row = RegionSpec(torch.int32, (cfg.num_experts,))
        combine_recv_count = RegionSpec(torch.int32, (cfg.num_experts,))

    SymLayoutMeta.token_dtype = token_dtype
    SymLayoutMeta.cfg = cfg
    return SymLayoutMeta


# --- SymLayout: the flydsl kernel handle over an allocated heap -----------------


@struct
class SymLayout:
    """Kernel handle over a heap: offsets table + rank + Constexpr shape dims (region ptrs added by make_sym_layout_type)."""

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


def sym_map(sl: SymLayout, ptr: fx.Numeric, dst_rank: fx.Numeric) -> fx.Numeric:
    """Remap a local pointer to peer ``dst_rank`` via the IPC delta table."""
    resource = addr_buffer_resource(sl.offsets_ptr, num_records_bytes=int(sl.num_ranks) * 8)
    return ptr + buffer_load(resource, dst_rank, vec_width=1, dtype=fx.T.i64())


# ptr -> peer-rank ptr mapping, callable as sym_layout.map(ptr, dst_rank)
SymLayout.map = lambda self, ptr, dst_rank: sym_map(self, ptr, dst_rank)


@lru_cache(maxsize=None)
def make_sym_layout_type(region_names: tuple):
    """Extend base SymLayout with one Int64 base+offset pointer field per region."""
    fields = [slice(fd.name, fd.type_spec) for fd in SymLayout.__dsl_field_defs__]
    fields += [slice(name, Int64) for name in region_names]
    layout_type = struct[tuple(fields)]
    layout_type.__dsl_display_name__ = "SymLayout"
    layout_type.map = SymLayout.map
    return layout_type


class SymmBuffer:

    def __init__(
        self,
        group,
        *,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        token_dtype=TOKEN_DTYPE,
    ):
        self.group = group
        self.rank = group.rank()
        self.world = group.size()
        self.token_dtype = token_dtype
        # dtype is part of the identity: a different token dtype needs a different heap
        self.key = (
            self.world,
            int(num_experts),
            int(num_max_tokens_per_rank),
            int(num_topk),
            int(hidden),
            int(intermediate_hidden),
            token_dtype,
        )

        # lazy import to avoid a circular import at module load time
        from primus_turbo.pytorch.core.symm_mem import SymmetricMemory

        # pure layout config: single source of truth for every region dim
        self.cfg = LayoutConfig.build(
            num_ranks=self.world,
            num_experts=int(num_experts),
            num_max_tokens_per_rank=int(num_max_tokens_per_rank),
            num_topk=int(num_topk),
            hidden=int(hidden),
            intermediate_hidden=int(intermediate_hidden),
        )
        # dtype-templated heap layout built from cfg (token buffers follow token_dtype)
        self.meta = make_sym_layout_meta(token_dtype, self.cfg)
        # expose dims as attributes for back-compat consumers (symm.hidden, symm.num_max_pool_tokens, ...)
        self.block_m = BLOCK_M
        self.num_ranks = self.cfg.num_ranks
        self.num_experts = self.cfg.num_experts
        self.num_experts_per_rank = self.cfg.num_experts_per_rank
        self.num_max_tokens_per_rank = self.cfg.num_max_tokens_per_rank
        self.num_topk = self.cfg.num_topk
        self.hidden = self.cfg.hidden
        self.intermediate_hidden = self.cfg.intermediate_hidden
        self.num_max_pool_tokens = self.cfg.num_max_pool_tokens
        self.num_max_pool_blocks = self.cfg.num_max_pool_blocks
        self.num_combine_slots = self.cfg.num_combine_slots
        self.num_tokens = self.cfg.num_max_tokens_per_rank  # back-compat alias

        self.num_bytes = self.meta.num_nbytes()

        # allocate the single IPC heap (SymmetricMemory already memsets it to 0 on alloc)
        self.symm_mem = SymmetricMemory(group, alloc_size=self.num_bytes, signal_pad_size=0)
        heap = self.symm_mem.get_buffer(self.rank, (self.num_bytes,), torch.int8)
        self.group.barrier()
        torch.cuda.synchronize()

        # split the heap into named region views and bind by name (no positional coupling)
        views = self.meta.split_buffer(heap)
        for name in self.meta.region_names():
            setattr(self, name, getattr(views, name))

        self.num_tokens_per_rank = torch.full(
            (self.world,), self.num_tokens, dtype=torch.int32, device="cuda"
        )

        # dispatch / combine use double-buffered parity signals
        self._disp_parity = 0
        self._disp_expected = [0, 0]
        self._combine_parity = 0
        self._combine_expected = [0, 0]
        self._reduce_expected = [0, 0]

        self._sym_layout = None

    def next_dispatch(self):

        self._disp_parity ^= 1
        p = self._disp_parity
        self._disp_expected[p] += int(self.world)
        return p, self._disp_expected[p]

    def next_combine(self, n_blocks):

        self._combine_parity ^= 1
        p = self._combine_parity
        self._combine_expected[p] += int(n_blocks)
        self._reduce_expected[p] += 1
        return p, self._combine_expected[p], self._reduce_expected[p]

    def get_sym_layout(self):

        if self._sym_layout is not None:
            return self._sym_layout

        # peer IPC deltas relative to this rank's own base pointer
        main = self.symm_mem.buffer_ptrs
        self._main_delta = torch.tensor([p - main[self.rank] for p in main], dtype=torch.int64, device="cuda")

        # extend the base SymLayout with this heap's regions, then fill dims + region base pointers
        base = main[self.rank]
        region_ptrs = {p.name: Int64(base + int(p.offset)) for p in self.meta.placed_regions()}
        layout_type = make_sym_layout_type(self.meta.region_names())
        self._sym_layout = layout_type(
            offsets_ptr=Int64(self._main_delta.data_ptr()),
            rank_idx=Int32(self.rank),
            num_ranks=self.num_ranks,
            num_experts=self.num_experts,
            num_experts_per_rank=self.num_experts_per_rank,
            num_max_tokens_per_rank=self.num_max_tokens_per_rank,
            num_topk=self.num_topk,
            hidden=self.hidden,
            intermediate_hidden=self.intermediate_hidden,
            num_max_pool_tokens=self.num_max_pool_tokens,
            num_max_pool_blocks=self.num_max_pool_blocks,
            num_combine_slots=self.num_combine_slots,
            **region_ptrs,
        )
        return self._sym_layout

    def assert_capacity(self):

        total_rows = int(self.meta_scalars[0].item())
        assert total_rows <= self.num_max_pool_tokens, (
            f"rank {self.rank}: dispatched rows {total_rows} exceed num_max_pool_tokens "
            f"{self.num_max_pool_tokens}; raise pool policy"
        )

    def describe(self) -> str:
        """Dump this heap's resolved region layout (offsets / bytes) -- for debugging IPC memory."""
        return self.meta.describe()

    def __repr__(self) -> str:
        return (
            f"SymmBuffer(rank={self.rank}/{self.world}, num_experts={self.num_experts}, "
            f"num_max_tokens_per_rank={self.num_max_tokens_per_rank}, num_topk={self.num_topk}, "
            f"hidden={self.hidden}, num_max_pool_tokens={self.num_max_pool_tokens}, "
            f"num_bytes={self.num_bytes})"
        )

    def destroy(self):
        global _CURRENT_SYMM_BUFFER
        if _CURRENT_SYMM_BUFFER is self:
            _CURRENT_SYMM_BUFFER = None
        try:
            self.symm_mem.destroy()
        except Exception as e:
            warnings.warn(f"SymmBuffer.destroy: symm_mem teardown failed: {e}")


_CURRENT_SYMM_BUFFER = None


def get_symm_buffer_for_mega_moe(
    group=None,
    *,
    num_experts=None,
    num_max_tokens_per_rank=None,
    num_topk=None,
    hidden=None,
    intermediate_hidden=None,
    token_dtype=TOKEN_DTYPE,  # dispatched-token dtype: bf16 (default) or fp8
) -> SymmBuffer:

    global _CURRENT_SYMM_BUFFER
    # note: block sizes are a fixed policy (BLOCK_M/BLOCK_N == 256), not a per-call knob
    if group is None:
        if _CURRENT_SYMM_BUFFER is None:
            raise RuntimeError(
                "no symmetric buffer is active; call get_symm_buffer_for_mega_moe(group, ...) first"
            )
        return _CURRENT_SYMM_BUFFER

    key = (
        group.size(),
        int(num_experts),
        int(num_max_tokens_per_rank),
        int(num_topk),
        int(hidden),
        int(intermediate_hidden),
        token_dtype,
    )
    symm = _CURRENT_SYMM_BUFFER
    if symm is None or symm.group is not group or symm.key != key:
        if symm is not None:
            symm.destroy()
        symm = SymmBuffer(
            group,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            token_dtype=token_dtype,
        )
        _CURRENT_SYMM_BUFFER = symm
    return symm
