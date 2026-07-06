import torch

from primus_turbo.flydsl.mega.sym_layout import BLOCK_M, build_sym_layout, region_bytes
from primus_turbo.pytorch.core.symm_mem import SymmetricMemory

__all__ = [
    "SymmBuffer",
    "get_symm_buffer_size_for_mega_moe",
    "get_symm_buffer_for_mega_moe",
]


def get_symm_buffer_size_for_mega_moe(
    world_size,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
):
    sl = build_sym_layout(
        world_size, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden
    )
    _, _, num_bytes, signal_bytes = region_bytes(sl)
    return num_bytes, signal_bytes


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

        # remember the MoE shape so make_sym_layout can rebuild the device layout
        self._shape = (
            self.world,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
        )
        sl = build_sym_layout(*self._shape)

        # scalars other methods / views need (pool sizes read back from the layout)
        self.block_m = BLOCK_M
        self.num_experts = num_experts
        self.num_tokens = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_max_pool_tokens = int(sl.num_max_pool_tokens)
        self.num_pool_blocks = int(sl.num_max_pool_blocks)
        self.combine_slots = int(sl.combine_slots)

        main_spec, signal_spec, self.num_bytes, self.signal_bytes = region_bytes(sl)

        # allocate the two IPC heaps and zero them once
        self.sm = SymmetricMemory(group, alloc_size=self.num_bytes, signal_pad_size=self.signal_bytes)
        self.sm.get_buffer(self.rank, (self.num_bytes,), torch.int8).zero_()
        self.sm.get_signal_pad(self.rank, (self.signal_bytes,), torch.int8).zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        # expose each region as an attribute view into its heap
        self._bind_views(main_spec, self.sm.get_buffer)
        self._bind_views(signal_spec, self.sm.get_signal_pad)

        # reshape the multi-dim regions to their logical shapes
        self.pool = self.pool.view(self.num_max_pool_tokens, self.hidden)
        self.act = self.act.view(self.num_max_pool_tokens, self.intermediate_hidden)
        self.l2_token_buffer = self.l2_token_buffer.view(self.num_max_pool_tokens, self.hidden)
        self.comb = self.comb.view(self.combine_slots, self.hidden)
        self.combine_gate = self.combine_gate.view(self.num_tokens, self.num_topk)

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

    def _bind_views(self, spec, get_heap):
        # map each region name to a flat view sliced out of its heap
        for name, (offset, dtype, numel) in spec.items():
            view = get_heap(self.rank, (numel,), dtype, storage_offset=offset // dtype.itemsize)
            setattr(self, name, view)

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

    def make_sym_layout(self):

        if self._sym_layout is not None:
            return self._sym_layout

        # peer IPC deltas relative to this rank's own base pointer
        main, signal = self.sm.buffer_ptrs, self.sm.signal_pad_ptrs
        self._main_delta = torch.tensor([p - main[self.rank] for p in main], dtype=torch.int64, device="cuda")
        self._signal_delta = torch.tensor(
            [p - signal[self.rank] for p in signal], dtype=torch.int64, device="cuda"
        )
        self._sym_layout = build_sym_layout(
            *self._shape,
            base=main[self.rank],
            offsets_ptr=self._main_delta.data_ptr(),
            signal_base=signal[self.rank],
            signal_offsets_ptr=self._signal_delta.data_ptr(),
            rank_idx=self.rank,
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
            self.sm.destroy()
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

    need_bytes, need_signal_bytes = get_symm_buffer_size_for_mega_moe(
        group.size(),
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
    )

    symm = _CURRENT_SYMM_BUFFER
    if (
        symm is None
        or symm.group is not group
        or symm.num_bytes < need_bytes
        or symm.signal_bytes < need_signal_bytes
    ):
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
