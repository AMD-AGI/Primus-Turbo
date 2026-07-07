import torch

from primus_turbo.flydsl.mega.sym_layout import BLOCK_M, get_sym_layout

__all__ = ["SymmBuffer", "get_symm_buffer_for_mega_moe"]


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

        # lazy import to avoid a circular import at module load time
        from primus_turbo.pytorch.core.symm_mem import SymmetricMemory

        meta = get_sym_layout(
            self.world, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden
        )

        # scalars other methods / views need (pool sizes read off the layout meta)
        self.block_m = BLOCK_M
        self.num_experts = num_experts
        self.num_tokens = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_max_pool_tokens = meta.num_max_pool_tokens
        self.num_pool_blocks = meta.num_max_pool_blocks
        self.num_combine_slots = meta.num_combine_slots

        self.num_bytes = meta.num_nbytes

        # allocate the single IPC heap and zero it once
        self.symm_mem = SymmetricMemory(group, alloc_size=self.num_bytes, signal_pad_size=0)
        heap = self.symm_mem.get_buffer(self.rank, (self.num_bytes,), torch.int8)
        heap.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        # split the heap into named region views (declaration order == _MAIN)
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
        ) = meta.split_buffer(heap)

        self.num_tokens_per_rank = torch.full(
            (self.world,), self.num_tokens, dtype=torch.int32, device="cuda"
        )

        # dispatch / combine use double-buffered parity signals
        self._disp_parity = 0
        self._disp_expected = [0, 0]
        self._combine_parity = 0
        self._combine_expected = [0, 0]
        self._reduce_expected = [0, 0]

        # MoE shape tuple build_sym_layout rebuilds the device handle from
        self._shape = (
            self.world,
            self.num_experts,
            self.num_tokens,
            self.num_topk,
            self.hidden,
            self.intermediate_hidden,
        )

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

    def make_sym_layout(self):

        if self._sym_layout is not None:
            return self._sym_layout

        # peer IPC deltas relative to this rank's own base pointer
        main = self.symm_mem.buffer_ptrs
        self._main_delta = torch.tensor([p - main[self.rank] for p in main], dtype=torch.int64, device="cuda")
        self._sym_layout = get_sym_layout(
            *self._shape,
            base=main[self.rank],
            offsets_ptr=self._main_delta.data_ptr(),
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

    need_bytes = get_sym_layout(
        group.size(), num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden
    ).num_nbytes
    symm = _CURRENT_SYMM_BUFFER
    if symm is None or symm.group is not group or symm.num_bytes < need_bytes:
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
