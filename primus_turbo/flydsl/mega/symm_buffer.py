from collections import namedtuple
from functools import lru_cache

import flydsl.expr as fx
import torch
from flydsl.expr import Int32, Int64, struct
from flydsl.expr.buffer_ops import buffer_load
from flydsl.expr.typing import Constexpr

from primus_turbo.flydsl.mega.prims import addr_buffer_resource

__all__ = ["SymmBuffer", "get_symm_buffer_for_mega_moe", "SymLayout", "sym_map", "make_sym_layout_type"]

BLOCK_M = 256  # pool-block granularity (fixed policy)
HEAP_ALIGN = 256  # every region starts on a 256B boundary
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


# --- heap layout description ---------------------------------------------------


class RegionSpec:
    """Blueprint for one heap region, declared as a class field; ``shape=None`` keeps it flat.

    ``numel`` and ``shape`` are functions of a config exposing the shape dims as attributes (the
    owning SymmBuffer), since the dims aren't known until a buffer is sized -- a spec reads as
    ``RegionSpec(torch.int32, lambda c: c.num_ranks * c.num_experts)``. ``dtype`` may be the
    dispatched-token dtype or a fixed dtype. A spec carries no address; placing it against a
    config yields a :class:`PlacedRegion`.
    """

    def __init__(self, dtype: torch.dtype, numel, shape=None):
        self.dtype = dtype
        self.numel = numel
        self.shape = shape

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
        """Region field names, as declared -- available without a config (dims don't affect names)."""
        return tuple(spec.name for spec in cls._region_specs)

    @classmethod
    def placed_regions(cls, cfg):
        """Every region resolved to its 256B-aligned offset for ``cfg`` (declaration order).

        ``cfg`` is any object exposing the shape dims as attributes (the owning SymmBuffer).
        """
        placed, cursor = [], 0
        for spec in cls._region_specs:
            cursor = align_up(cursor, HEAP_ALIGN)
            numel = spec.numel(cfg)
            shape = spec.shape(cfg) if spec.shape is not None else (numel,)
            nbytes = numel * spec.dtype.itemsize
            placed.append(PlacedRegion(spec.name, cursor, spec.dtype, shape, nbytes))
            cursor += nbytes
        return tuple(placed)

    @classmethod
    def num_nbytes(cls, cfg) -> int:
        """Total 256B-aligned heap size for ``cfg``, for allocation."""
        last = cls.placed_regions(cfg)[-1]
        return align_up(last.offset + last.nbytes, HEAP_ALIGN)

    @classmethod
    def split_buffer(cls, buffer: "torch.Tensor", cfg):
        """Split a flat int8 IPC heap into one non-owning tensor per region.

        Each view sits at its 256B-aligned offset, typed and reshaped to its region; each has its
        own Storage (like get_buffer) so it aliases the heap without tripping torch alias checks.
        """
        from primus_turbo.pytorch.core.symm_mem import _tensor_from_device_ptr

        placed = cls.placed_regions(cfg)
        total_bytes = cls.num_nbytes(cfg)
        buffer_nbytes = buffer.numel() * buffer.element_size()
        assert buffer_nbytes >= total_bytes, f"buffer too small: {buffer_nbytes} < {total_bytes} bytes"
        base_addr, device_index = buffer.data_ptr(), buffer.device.index
        views = [
            _tensor_from_device_ptr(base_addr + p.offset, p.shape, p.dtype, device_index) for p in placed
        ]
        return _region_views_type(cls.region_names())(*views)


class SymLayoutMeta(_SymLayoutMetaBase):
    """The symmetric-memory heap layout -- the single source of truth for regions (name, dtype, size)."""

    dispatch_token_pool = RegionSpec(
        TOKEN_DTYPE, lambda c: c.num_max_pool_tokens * c.hidden, lambda c: (c.num_max_pool_tokens, c.hidden)
    )
    expert_count_buffer = RegionSpec(torch.int32, lambda c: c.num_ranks * c.num_experts)
    signal = RegionSpec(torch.int32, lambda c: c.num_ranks)
    pool_src_rank = RegionSpec(torch.int32, lambda c: c.num_max_pool_tokens)
    pool_src_slot = RegionSpec(torch.int32, lambda c: c.num_max_pool_tokens)
    weight_recv_buf = RegionSpec(torch.float32, lambda c: c.num_max_pool_tokens)
    combine_gate = RegionSpec(
        torch.float32, lambda c: c.num_combine_slots, lambda c: (c.num_max_tokens_per_rank, c.num_topk)
    )
    meta_scalars = RegionSpec(torch.int32, lambda c: 8)
    grid_sync_count = RegionSpec(torch.int32, lambda c: 2)
    l2_token_buffer = RegionSpec(
        TOKEN_DTYPE, lambda c: c.num_max_pool_tokens * c.hidden, lambda c: (c.num_max_pool_tokens, c.hidden)
    )
    dispatch_flag = RegionSpec(torch.int64, lambda c: 2 * c.num_max_pool_blocks)
    combine_flag = RegionSpec(torch.int64, lambda c: 2 * c.num_max_pool_blocks)
    combine_token_buffer = RegionSpec(
        TOKEN_DTYPE, lambda c: c.num_combine_slots * c.hidden, lambda c: (c.num_combine_slots, c.hidden)
    )
    reduce_flag = RegionSpec(torch.int64, lambda c: 2 * c.num_combine_slots)
    # combine recv-segment table: one entry per (local_expert, source_rank).
    combine_recv_dst_rank = RegionSpec(torch.int32, lambda c: c.num_experts)
    combine_recv_start_row = RegionSpec(torch.int32, lambda c: c.num_experts)
    combine_recv_count = RegionSpec(torch.int32, lambda c: c.num_experts)


# --- SymLayout: the flydsl kernel handle over an allocated heap -----------------


@struct
class SymLayout:
    """Kernel handle over an allocated heap: an offsets table + this rank + the Constexpr shape
    dims. :func:`make_sym_layout_type` extends a copy of this with one Int64 pointer field per
    heap region; kernels annotate with this base type (ABI keys off the actual value's type)."""

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
    """Extend the base SymLayout with one Int64 field per region (each holds its absolute
    ``base + offset`` pointer, read in a kernel as ``sym_layout.<region>``)."""
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
    ):
        self.group = group
        self.rank = group.rank()
        self.world = group.size()
        self.key = (
            self.world,
            int(num_experts),
            int(num_max_tokens_per_rank),
            int(num_topk),
            int(hidden),
            int(intermediate_hidden),
        )

        # lazy import to avoid a circular import at module load time
        from primus_turbo.pytorch.core.symm_mem import SymmetricMemory

        # shape dims (raw + derived); the region size lambdas and the SymLayout read these off self
        assert num_experts % self.world == 0, f"num_experts {num_experts} not divisible by ranks {self.world}"
        self.block_m = BLOCK_M
        self.num_ranks = self.world
        self.num_experts = int(num_experts)
        self.num_experts_per_rank = self.num_experts // self.world
        self.num_max_tokens_per_rank = int(num_max_tokens_per_rank)
        self.num_topk = int(num_topk)
        self.hidden = int(hidden)
        self.intermediate_hidden = int(intermediate_hidden)
        self.num_max_pool_tokens = get_num_max_pool_tokens(
            self.world, self.num_max_tokens_per_rank, self.num_topk, self.num_experts_per_rank
        )
        self.num_max_pool_blocks = self.num_max_pool_tokens // BLOCK_M
        self.num_combine_slots = self.num_max_tokens_per_rank * self.num_topk
        self.num_tokens = self.num_max_tokens_per_rank  # back-compat alias

        self.num_bytes = SymLayoutMeta.num_nbytes(self)

        # allocate the single IPC heap and zero it once
        self.symm_mem = SymmetricMemory(group, alloc_size=self.num_bytes, signal_pad_size=0)
        heap = self.symm_mem.get_buffer(self.rank, (self.num_bytes,), torch.int8)
        heap.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        # split the heap into named region views (declaration order == memory order)
        (
            self.dispatch_token_pool,
            self.expert_count_buffer,
            self.signal,
            self.pool_src_rank,
            self.pool_src_slot,
            self.weight_recv_buf,
            self.combine_gate,
            self.meta_scalars,
            self.grid_sync_count,
            self.l2_token_buffer,
            self.dispatch_flag,
            self.combine_flag,
            self.combine_token_buffer,
            self.reduce_flag,
            self.combine_recv_dst_rank,
            self.combine_recv_start_row,
            self.combine_recv_count,
        ) = SymLayoutMeta.split_buffer(heap, self)

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
        region_ptrs = {p.name: Int64(base + int(p.offset)) for p in SymLayoutMeta.placed_regions(self)}
        layout_type = make_sym_layout_type(SymLayoutMeta.region_names())
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

    def destroy(self):
        global _CURRENT_SYMM_BUFFER
        if _CURRENT_SYMM_BUFFER is self:
            _CURRENT_SYMM_BUFFER = None
        try:
            self.symm_mem.destroy()
        except Exception:
            pass


_CURRENT_SYMM_BUFFER = None


def get_symm_buffer_for_mega_moe(
    group=None,
    *,
    num_experts=None,
    num_max_tokens_per_rank=None,
    num_topk=None,
    hidden=None,
    intermediate_hidden=None,
    block_m=256,  # accepted for backward-compat; pool sizing is fixed policy
    block_n=256,
    pool_mult=2,
) -> SymmBuffer:

    global _CURRENT_SYMM_BUFFER
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
        )
        _CURRENT_SYMM_BUFFER = symm
    return symm
