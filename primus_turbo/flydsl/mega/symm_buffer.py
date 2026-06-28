###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Single symmetric allocation carved into every buffer a fused mega MoE needs.

One cached ``SymmBuffer`` owns the cross-rank pool, all local scratch, the GEMM
intermediates (cached MAIN heap) AND the spin-wait flags + combine buffer
(uncached SIGNAL heap). The region order / 256B alignment mirror
``sym_layout.py`` byte-for-byte -- ``make_sym_layout()`` hands the device kernels a
single :class:`SymLayout` struct that recomputes the exact same offsets, so host
views and device addresses agree. Inspired by ``deep_gemm/mega``.

Moved out of ``primus_turbo.pytorch.ops.moe.mega_moe_fused`` (which re-exports
these names for backward compatibility) so the FlyDSL kernels can build the
symmetric workspace without importing the torch-op layer.
"""

import torch

from primus_turbo.flydsl.mega.sym_layout import build_sym_layout
from primus_turbo.flydsl.mega.sym_layout import layout as _sym_region_layout

# NOTE: SymmetricMemory is imported lazily inside SymmBuffer.__init__ to avoid a
# circular import (importing the pytorch package pulls in mega_moe_fused, which
# imports back from this module).

__all__ = [
    "SymmBuffer",
    "get_symm_buffer_size_for_mega_moe",
    "get_symm_buffer_for_mega_moe",
]

# each sub-buffer base aligned to 256B (matches the kernels' bump cursor)
_BASE_ALIGNMENT = 256


def _align(nbytes, alignment=_BASE_ALIGNMENT):
    return (nbytes + alignment - 1) // alignment * alignment


# --------------------------------------------------------------------------- #
# Symmetric-buffer layout: ``sym_layout`` owns the region order / sizes / 256B
# packing (single source of truth). Here we only map each region NAME to its torch
# dtype so the host views can be carved over the byte offsets ``sym_layout`` reports
# (itemsize alone cannot tell i32 from f32). Names MUST match ``sym_layout``.
# --------------------------------------------------------------------------- #
_MAIN_DTYPES = {
    "pool": torch.bfloat16,
    "c_buffer": torch.int32,
    "signal": torch.int32,
    "origin_rank": torch.int32,
    "origin_slot": torch.int32,
    "weight_recv_buf": torch.float32,
    "dedup_src_row": torch.int32,
    "combine_gate": torch.float32,
    "meta_scalars": torch.int32,
    "grid_sync_count": torch.int32,
    "profile": torch.int64,
    "act": torch.bfloat16,
    "l2_token_buffer": torch.bfloat16,
    "src_token_topk_idx": torch.int32,
}
_SIGNAL_DTYPES = {
    "_ipc_barrier": torch.int32,
    "scoreboard": torch.int32,
    "sb_consume": torch.int32,
    "sb_l2": torch.int32,
    "comb": torch.bfloat16,
    "barrier_local": torch.int32,
}


def _build_layout_spec(
    world_size,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    activation="swiglu",
    *,
    block_m=256,
    pool_mult=2,
):
    """Size both heaps directly from ``sym_layout``; attach torch dtypes for the views.

    Pool capacity / blocks / combine slots are the host allocation policy (driven by
    ``block_m`` / ``pool_mult``); the byte offsets + heap totals come from
    ``sym_layout.layout`` so host views and device addresses cannot drift apart.
    Returns ``(main_spec, signal_spec, num_bytes, signal_bytes, meta)`` where each
    spec maps ``name -> (offset, torch_dtype, numel)``."""
    experts_per_rank = num_experts // world_size
    avg_recv_tokens = num_max_tokens_per_rank * num_topk
    num_max_pool_tokens = _align(pool_mult * avg_recv_tokens + experts_per_rank * block_m, block_m)
    num_pool_blocks = num_max_pool_tokens // block_m
    combine_slots = num_topk * num_max_tokens_per_rank

    sl = build_sym_layout(
        world_size,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        num_max_pool_tokens,
        num_pool_blocks,
        combine_slots,
    )
    main_off, sig_off, num_bytes, signal_bytes = _sym_region_layout(sl)
    main_spec = {n: (off, _MAIN_DTYPES[n], numel) for n, (off, _it, numel) in main_off.items()}
    signal_spec = {n: (off, _SIGNAL_DTYPES[n], numel) for n, (off, _it, numel) in sig_off.items()}
    meta = dict(
        world_size=world_size,
        num_experts=num_experts,
        num_tokens=num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        activation=activation,
        block_m=block_m,
        num_max_pool_tokens=num_max_pool_tokens,
        num_pool_blocks=num_pool_blocks,
        combine_slots=combine_slots,
    )
    return main_spec, signal_spec, num_bytes, signal_bytes, meta


def get_symm_buffer_size_for_mega_moe(
    world_size,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    activation="swiglu",
    *,
    block_m=256,
    pool_mult=2,
):
    """Size the single symmetric buffer for one fused mega MoE forward.

    Returns ``(num_bytes, slice_input_buffers, signal_bytes, meta)`` (mirrors
    ``deep_gemm``'s ``get_symm_buffer_size_for_mega_moe``): ``num_bytes`` is the
    main (cached) HIP-IPC buffer total, ``slice_input_buffers`` maps each main
    sub-buffer name to ``(offset, dtype, numel)``, ``signal_bytes`` sizes the
    uncached signal pad, and ``meta`` carries the derived shape scalars. The main
    buffer holds the cross-rank pool, all local scratch, and the GEMM intermediates.
    The uncached signal pad holds the spin-wait flags (``scoreboard`` / ``sb_l2``)
    AND the cross-rank combine buffer (``comb``).

    ``activation`` is reserved (both gated ``swiglu`` and non-gated variants emit
    ``intermediate_hidden``, so it does not change sizing today).
    """
    slice_input_buffers, _signal_spec, num_bytes, signal_bytes, meta = _build_layout_spec(
        world_size,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        activation,
        block_m=block_m,
        pool_mult=pool_mult,
    )
    return num_bytes, slice_input_buffers, signal_bytes, meta


class SymmBuffer:
    """One symmetric allocation carved into every buffer a fused mega MoE forward needs.

    ``buffer`` (cached) + ``signal pad`` (uncached) come from a single
    ``SymmetricMemory``; each cross-rank sub-buffer exposes a per-rank pointer table
    (base_ptr + offset). Allocate once and reuse across steps (the kernels self-reset
    their counters)."""

    def __init__(
        self,
        group,
        *,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        block_m=256,
        block_n=256,
        pool_mult=2,
    ):
        self.group = group
        self.rank = group.rank()
        self.world = group.size()
        self.block_m = block_m
        self.block_n = block_n

        slice_input_buffers, signal_spec, num_bytes, signal_bytes, meta = _build_layout_spec(
            self.world,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            block_m=block_m,
            pool_mult=pool_mult,
        )
        # num_tokens / num_experts / hidden / num_max_pool_tokens / ...
        self.__dict__.update(meta)
        self.experts_per_rank = num_experts // self.world
        # keep the allocation sizes so the global getter can size-check + reuse
        self.num_bytes = num_bytes
        self.signal_bytes = signal_bytes

        # one symmetric allocation: cached main buffer + uncached signal pad
        from primus_turbo.pytorch.core.symm_mem import SymmetricMemory

        self.sm = SymmetricMemory(group, alloc_size=num_bytes, signal_pad_size=signal_bytes)
        self.sm.get_buffer(self.rank, (num_bytes,), torch.int8).zero_()
        self.sm.get_signal_pad(self.rank, (signal_bytes,), torch.int8).zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        self.signal_pad = self.sm.get_signal_pad(self.rank)
        self._slice_input_buffers = slice_input_buffers
        self._signal_spec = signal_spec

        # ---- carve out local views (zero-copy slices of the single buffer) ----
        def _main_view(name):
            offset, dtype, numel = slice_input_buffers[name]
            return self.sm.get_buffer(self.rank, (numel,), dtype, storage_offset=offset // dtype.itemsize)

        def _signal_view(name):
            offset, dtype, numel = signal_spec[name]
            return self.sm.get_signal_pad(self.rank, (numel,), dtype, storage_offset=offset // dtype.itemsize)

        for name in slice_input_buffers:
            setattr(self, name, _main_view(name))
        for name in signal_spec:
            setattr(self, name, _signal_view(name))
        # back-compat alias: the old layout named the grid-sync counter grid_barrier_state
        self.grid_barrier_state = self.grid_sync_count
        # reshape the matrix-shaped views
        self.pool = self.pool.view(self.num_max_pool_tokens, self.hidden)
        self.act = self.act.view(self.num_max_pool_tokens, self.intermediate_hidden)
        self.l2_token_buffer = self.l2_token_buffer.view(self.num_max_pool_tokens, self.hidden)
        self.comb = self.comb.view(self.combine_slots, self.hidden)
        # d_topk_w push slots, slot = token*topk + k -> view [num_tokens, num_topk]
        self.combine_gate = self.combine_gate.view(self.num_tokens, self.num_topk)
        # combine reduce reads only num_tokens_per_rank[rank]; it's the fixed per-rank token
        # count -> build it ONCE here (was a per-call torch.full in both fwd + bwd).
        self.num_tokens_per_rank = torch.full(
            (self.world,), self.num_tokens, dtype=torch.int32, device="cuda"
        )

        # ---- per-rank pointer tables for the cross-rank sub-buffers ----
        buffer_ptrs, signal_ptrs = self.sm.buffer_ptrs, self.sm.signal_pad_ptrs

        def _peer_ptr_table(base_ptrs, offset):
            return torch.tensor(
                [base_ptrs[peer] + offset for peer in range(self.world)],
                dtype=torch.int64,
                device="cuda",
            )

        self.pool_ptrs = _peer_ptr_table(buffer_ptrs, slice_input_buffers["pool"][0])
        # prologue addressing via prims.symm_at: one [world] i64 peer heap-base table +
        # a [5] i64 byte-offset table (rows = c_buffer / signal / origin_rank / origin_slot
        # / weight_recv_buf). Replaces the old [5, world] pre-offset peer_ptrs table.
        self.buffer_base = _peer_ptr_table(buffer_ptrs, 0)
        self.buffer_offsets = torch.tensor(
            [
                slice_input_buffers["c_buffer"][0],
                slice_input_buffers["signal"][0],
                slice_input_buffers["origin_rank"][0],
                slice_input_buffers["origin_slot"][0],
                slice_input_buffers["weight_recv_buf"][0],
            ],
            dtype=torch.int64,
            device="cuda",
        )
        # comb + barrier_local live in the uncached signal pad -> peer tables from signal_pad_ptrs
        self.comb_addrs = _peer_ptr_table(signal_ptrs, signal_spec["comb"][0])
        self.barrier_addrs = _peer_ptr_table(signal_ptrs, signal_spec["barrier_local"][0])
        self.scoreboard_ptrs = _peer_ptr_table(signal_ptrs, signal_spec["scoreboard"][0])
        # combine_gate (cached main) peer table -> backward gate-grad (d_topk_w) scatter
        self.gate_addrs = _peer_ptr_table(buffer_ptrs, slice_input_buffers["combine_gate"][0])

        # the SymLayout struct + its delta tables are built lazily on first request
        self._sym_layout = None

    def make_sym_layout(self):
        """Build (once, cached) the :class:`SymLayout` struct the FlyDSL kernels take by value.

        Computes the two per-peer base-DELTA tables (``peer_base - my_base``) for the
        cached MAIN and uncached SIGNAL heaps and packs them, with this rank's heap
        bases + dims, into a ``SymLayout``. The struct recomputes every region offset
        from its Constexpr dims; those offsets match this buffer's host views because
        both pack the same region order with the same 256B alignment."""
        if self._sym_layout is not None:
            return self._sym_layout
        buffer_ptrs, signal_ptrs = self.sm.buffer_ptrs, self.sm.signal_pad_ptrs
        my_main, my_signal = buffer_ptrs[self.rank], signal_ptrs[self.rank]
        # keep the delta tables alive on self (build_sym_layout stores their data_ptr())
        self._main_delta = torch.tensor(
            [buffer_ptrs[p] - my_main for p in range(self.world)], dtype=torch.int64, device="cuda"
        )
        self._signal_delta = torch.tensor(
            [signal_ptrs[p] - my_signal for p in range(self.world)], dtype=torch.int64, device="cuda"
        )
        self._sym_layout = build_sym_layout(
            self.world,
            self.num_experts,
            self.num_tokens,
            self.num_topk,
            self.hidden,
            self.intermediate_hidden,
            self.num_max_pool_tokens,
            self.num_pool_blocks,
            self.combine_slots,
            base=my_main,
            offsets_ptr=self._main_delta.data_ptr(),
            signal_base=my_signal,
            signal_offsets_ptr=self._signal_delta.data_ptr(),
            rank_idx=self.rank,
        )
        return self._sym_layout

    def assert_capacity(self):
        """Guard against silent pool overflow (bounded buffer_store drops OOB rows)."""
        total_rows = int(self.meta_scalars[0].item())
        assert total_rows <= self.num_max_pool_tokens, (
            f"rank {self.rank}: dispatched rows {total_rows} exceed num_max_pool_tokens "
            f"{self.num_max_pool_tokens}; raise pool_mult"
        )

    def destroy(self):
        global _CURRENT_SYMM_BUFFER
        if _CURRENT_SYMM_BUFFER is self:
            _CURRENT_SYMM_BUFFER = None
        try:
            self.sm.destroy()
        except Exception:
            pass


# The single live symmetric buffer, exposed globally so kernels can fetch the
# active symmetric workspace without threading it through every call.
_CURRENT_SYMM_BUFFER = None


def get_symm_buffer_for_mega_moe(
    group=None,
    *,
    num_experts=None,
    num_max_tokens_per_rank=None,
    num_topk=None,
    hidden=None,
    intermediate_hidden=None,
    block_m=256,
    block_n=256,
    pool_mult=2,
) -> SymmBuffer:
    """Get (allocate or reuse) the single global symmetric buffer for a fused mega MoE.

    Only one symmetric buffer is kept alive. The requested shape/tiling is sized via
    ``get_symm_buffer_size_for_mega_moe``; if the live buffer is missing or too small
    (main or signal heap) it is released and a fresh one is rendezvous-allocated.
    Otherwise the existing buffer is reused as-is.

    Called with no ``group`` it returns the live buffer -- kernels fetch the workspace
    this way instead of receiving it as a parameter; raises if none exists yet."""
    global _CURRENT_SYMM_BUFFER
    if group is None:
        if _CURRENT_SYMM_BUFFER is None:
            raise RuntimeError(
                "no symmetric buffer is active; call get_symm_buffer_for_mega_moe(group, ...) first"
            )
        return _CURRENT_SYMM_BUFFER

    need_bytes, _, need_signal_bytes, _ = get_symm_buffer_size_for_mega_moe(
        group.size(),
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        block_m=block_m,
        pool_mult=pool_mult,
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
            block_m=block_m,
            block_n=block_n,
            pool_mult=pool_mult,
        )
        _CURRENT_SYMM_BUFFER = symm
    return symm
