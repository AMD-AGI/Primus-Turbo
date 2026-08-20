###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""The fused mega MoE's symmetric workspace, from both sides.

One cached ``SymmBuffer`` owns the cross-rank pool, all local scratch and the GEMM
intermediates (cached MAIN heap) plus the spin-wait flags and combine buffer
(uncached SIGNAL heap). ``SymLayout`` is the same carving as seen from the device:
kernels take it by value and recompute every region offset themselves.

The two halves MUST agree byte for byte -- same region order, same 256B alignment --
or host views and device addresses point at different memory. They live in one file
because that agreement is the whole contract between them; split across two, it was
a comment rather than something a reader can check.

Each heap's ``offsets_ptr`` is an ``i64[num_ranks]`` table of per-peer base deltas
(``peer_base[i] - my_base``), so adding a delta translates a local address into that
peer's. Inspired by ``deep_gemm/mega``.
"""

import flydsl.expr as fx
import torch
from flydsl.expr import Int32, Int64, struct
from flydsl.expr.buffer_ops import buffer_load
from flydsl.expr.typing import Constexpr

from primus_turbo.flydsl.mega.prims import addr_buffer_resource

# NOTE: SymmetricMemory is imported lazily inside SymmBuffer.__init__ to avoid a
# circular import (importing the pytorch package pulls in fused_mega_moe, which
# imports back from this module).

__all__ = [
    "SymLayout",
    "SymmBuffer",
    "build_sym_layout",
    "sym_map",
    "get_symm_buffer_size_for_mega_moe",
    "get_symm_buffer_for_mega_moe",
]

# ---- element byte sizes (from layout/mega_moe.cuh) ----
_BF16, _I32, _F32, _I64 = 2, 4, 4, 8

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
    # mxfp8 forward-only (present only when use_mxfp8=1)
    "pool_fp8": torch.float8_e4m3fn,
    "pool_scale": torch.uint8,  # raw E8M0 byte
    "act_fp8": torch.float8_e4m3fn,
    "act_scale": torch.uint8,  # raw E8M0 byte
    "pool_scale_ps": torch.int32,  # E8M0 in ScaleS2R broadcast layout (fused quant-in-push)
}
_SIGNAL_DTYPES = {
    "_ipc_barrier": torch.int32,
    "dispatch_flag": torch.int64,  # 2 banks x num_pool_blocks (comm->preshuffle epoch gate)
    "preshuffle_flag": torch.int64,  # 2 banks x num_pool_blocks (preshuffle->gemm epoch gate)
    "combine_flag": torch.int64,  # 2 banks x num_pool_blocks x (H//block_n) scatter release slots
    "comb": torch.bfloat16,
    "reduce_flag": torch.int64,  # 2 banks x combine_slots (combine->reduce epoch gate)
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
    block_n=256,
    pool_mult=2,
    use_mxfp8=False,
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
        block_n=block_n,
        use_mxfp8=int(bool(use_mxfp8)),
    )
    main_off, sig_off, num_bytes, signal_bytes = layout(sl)
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
        use_mxfp8=bool(use_mxfp8),
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
    block_n=256,
    pool_mult=2,
    use_mxfp8=False,
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
        block_n=block_n,
        pool_mult=pool_mult,
        use_mxfp8=use_mxfp8,
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
        use_mxfp8=False,
    ):
        self.group = group
        self.rank = group.rank()
        self.world = group.size()
        self.block_m = block_m
        self.block_n = block_n
        self.use_mxfp8 = bool(use_mxfp8)

        slice_input_buffers, signal_spec, num_bytes, signal_bytes, meta = _build_layout_spec(
            self.world,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            block_m=block_m,
            block_n=block_n,
            pool_mult=pool_mult,
            use_mxfp8=use_mxfp8,
        )
        # num_tokens / num_experts / hidden / num_max_pool_tokens / ...
        self.__dict__.update(meta)
        self.experts_per_rank = num_experts // self.world
        # keep the allocation sizes so the global getter can size-check + reuse
        self.num_bytes = num_bytes
        self.signal_bytes = signal_bytes

        # one symmetric allocation: cached main buffer + uncached signal pad
        from primus_turbo.pytorch.core.symm_mem import SymmetricMemory

        # uncached: the signal heap holds the combine buffer a peer PUSHes and the reduce reads,
        # not just flags -- a cached line there shows the reduce a stale E8M0.
        self.sm = SymmetricMemory(
            group, alloc_size=num_bytes, signal_pad_size=signal_bytes, uncached_signal_pad=True
        )
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
        if self.use_mxfp8:
            # fp8 token/act pool + raw E8M0 block scales (block=32 along the hidden/K dim).
            self.pool_fp8 = self.pool_fp8.view(self.num_max_pool_tokens, self.hidden)
            self.pool_scale = self.pool_scale.view(self.num_max_pool_tokens, self.hidden // 32)
            self.act_fp8 = self.act_fp8.view(self.num_max_pool_tokens, self.intermediate_hidden)
            self.act_scale = self.act_scale.view(self.num_max_pool_tokens, self.intermediate_hidden // 32)
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
        if self.use_mxfp8:
            # peer tables for the fp8 dispatch push (token bytes + raw E8M0 scale bytes)
            self.pool_fp8_ptrs = _peer_ptr_table(buffer_ptrs, slice_input_buffers["pool_fp8"][0])
            self.pool_scale_ptrs = _peer_ptr_table(buffer_ptrs, slice_input_buffers["pool_scale"][0])
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
        # comb + the epoch flags live in the uncached signal pad -> peer tables from signal_pad_ptrs
        # (kept for parity/debug; the kernels reach peers via the sym_layout delta tables).
        self.comb_addrs = _peer_ptr_table(signal_ptrs, signal_spec["comb"][0])
        self.reduce_flag_addrs = _peer_ptr_table(signal_ptrs, signal_spec["reduce_flag"][0])
        self.dispatch_flag_addrs = _peer_ptr_table(signal_ptrs, signal_spec["dispatch_flag"][0])
        # combine_gate (cached main) peer table -> backward gate-grad (d_topk_w) scatter
        self.gate_addrs = _peer_ptr_table(buffer_ptrs, slice_input_buffers["combine_gate"][0])

        # ---- device epoch state (bf16-style self-reset) for the cross-rank spin flags ----
        # parity (0/1) picks the flag bank; expected[parity] is the cumulative spin target.
        # Bumped on-device by each op's epoch_bump kernel -> the flags need NO host reset
        # (removes the per-call synchronize()+barrier() rendezvous) and carry no cross-call reset
        # race. LOCAL tensors (each rank bumps/reads its own; lockstep keeps every rank's
        # parity/expected identical, so cross-rank flag writes hit the right bank).
        #   dispatch: dispatch_flag(+num_ranks) comm->preshuffle, preshuffle_flag(+1) preshuffle->gemm
        self._disp_parity = torch.zeros(1, dtype=torch.int64, device="cuda")
        self._disp_expected = torch.zeros(2, dtype=torch.int64, device="cuda")
        self._ps_expected = torch.zeros(2, dtype=torch.int64, device="cuda")
        #   combine: combine_flag(+1) per-epoch expected; GEMM st per (block_m,block_n) slot
        self._combine_parity = torch.zeros(1, dtype=torch.int64, device="cuda")
        self._combine_expected = torch.zeros(2, dtype=torch.int64, device="cuda")
        self._reduce_expected = torch.zeros(2, dtype=torch.int64, device="cuda")

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
            block_n=self.block_n,
            use_mxfp8=int(self.use_mxfp8),
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
    use_mxfp8=False,
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
        block_n=block_n,
        pool_mult=pool_mult,
        use_mxfp8=use_mxfp8,
    )

    symm = _CURRENT_SYMM_BUFFER

    if (
        symm is None
        or symm.group is not group
        or symm.num_bytes < need_bytes
        or symm.signal_bytes < need_signal_bytes
        or bool(getattr(symm, "use_mxfp8", False)) != bool(use_mxfp8)
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
            use_mxfp8=use_mxfp8,
        )
        _CURRENT_SYMM_BUFFER = symm
    return symm


# ============================================================================
# Device-side view of the same regions: SymLayout recomputes these offsets in the kernel.
# ============================================================================


# ---------------------------------------------------------------------------
# The struct (passed to kernels by value) -- the single source of truth.
# ---------------------------------------------------------------------------
@struct
class SymLayout:
    base: Int64  # this rank's MAIN (cached) heap base address
    offsets_ptr: Int64  # i64[num_ranks] MAIN per-peer delta table (peer_base - base)
    signal_base: Int64  # this rank's SIGNAL (uncached) heap base address
    signal_offsets_ptr: Int64  # i64[num_ranks] SIGNAL per-peer delta table
    rank_idx: Int32
    num_ranks: Constexpr[int]
    num_experts: Constexpr[int]
    num_experts_per_rank: Constexpr[int]
    num_max_tokens_per_rank: Constexpr[int]
    num_topk: Constexpr[int]
    hidden: Constexpr[int]
    intermediate_hidden: Constexpr[int]
    num_max_pool_tokens: Constexpr[int]  # pool capacity (rows)
    num_max_pool_blocks: Constexpr[int]  # num_max_pool_tokens // block_m
    combine_slots: Constexpr[int]  # num_topk * num_max_tokens_per_rank
    block_n: Constexpr[int]  # L2 GEMM N-tile width (combine_flag slots per block_m = hidden // block_n)
    # mxfp8 forward: append fp8 pool/act data + raw E8M0 block-scale regions (1 = on).
    # Appended AFTER every bf16 region so the bf16 (use_mxfp8=0) byte layout is unchanged.
    use_mxfp8: Constexpr[int]


# ---------------------------------------------------------------------------
# Memory layout: two heaps, each a 256B-aligned region packer. The region order
# MUST mirror ``fused_mega_moe`` (``main`` list and ``_signal_regions``).
# ---------------------------------------------------------------------------
def _main_regions(sl):
    R, E = int(sl.num_ranks), int(sl.num_experts)
    P, H, I = int(sl.num_max_pool_tokens), int(sl.hidden), int(sl.intermediate_hidden)
    CS = int(sl.combine_slots)
    EPR, T = int(sl.num_experts_per_rank), int(sl.num_max_tokens_per_rank)
    regions = [
        ("pool", _BF16, P * H),
        ("c_buffer", _I32, R * E),
        ("signal", _I32, R),
        ("origin_rank", _I32, P),
        ("origin_slot", _I32, P),
        ("weight_recv_buf", _F32, P),
        # token-dedup map2 (dense_to_expert): secondary dest slot -> primary slot to
        # copy from (-1 = primary). Source rank writes it cross-rank -> symmetric.
        ("dedup_src_row", _I32, P),
        ("combine_gate", _F32, CS),
        ("meta_scalars", _I32, 8),
        ("grid_sync_count", _I32, 2),
        ("profile", _I64, 8),
        ("act", _BF16, P * I),
        ("l2_token_buffer", _BF16, P * H),
        # DG dispatch index (dest-side): src_token_topk_idx[le, src_rank, slot] = token*K+k.
        # Source rank scatters cross-rank into the dest -> symmetric. Appended LAST so
        # every preceding region keeps its byte offset.
        ("src_token_topk_idx", _I32, EPR * R * R * T),
    ]
    # mxfp8 forward-only regions (fp8 = 1B/elem, E8M0 scale = 1B / 32 K-elems). Appended
    # last (offset-stable). Pushed cross-rank by dispatch (pool_*) / written by SwiGLU
    # (act_*); read as A operands by the grouped mxfp8 GEMM (which preshuffles internally).
    if int(getattr(sl, "use_mxfp8", 0)):
        # pool_scale_ps: fused quant-in-push writes the pool E8M0 scale directly in the
        # ScaleS2R broadcast layout-1 (int32, ceildiv(P,64)*(H//128)*256), so the fused L1
        # GEMM reads it with ScaleS2R (no preshuffle pass). The raw ``pool_scale`` is kept
        # for the decoupled fp8 path (push raw -> grouped GEMM preshuffles internally).
        ps_i32 = ((P + 63) // 64) * (H // 128) * 256
        regions += [
            ("pool_fp8", 1, P * H),
            ("pool_scale", 1, P * (H // 32)),
            ("act_fp8", 1, P * I),
            ("act_scale", 1, P * (I // 32)),
            ("pool_scale_ps", _I32, ps_i32),
        ]
    return regions


def _signal_regions(sl):
    R, NPB = int(sl.num_ranks), int(sl.num_max_pool_blocks)
    CS, H = int(sl.combine_slots), int(sl.hidden)
    BN = int(sl.block_n)
    n_l2_n = H // BN
    # All epoch flags (bf16-style self-reset): 2 banks (parity) x length, i64. Never host-reset;
    # each spins on a cumulative per-bank expected -> no consuming store, no cross-call reset race.
    return [
        ("_ipc_barrier", _I32, R),
        ("dispatch_flag", _I64, 2 * NPB),  # cross-rank comm->preshuffle gate (per-expert, atomic_add)
        ("preshuffle_flag", _I64, 2 * NPB),  # local preshuffle->gemm gate (per-block, st=expected)
        ("combine_flag", _I64, 2 * NPB * n_l2_n),  # L2 GEMM->combine: per (block_m, block_n) release st
        ("comb", _BF16, CS * H),
        ("reduce_flag", _I64, 2 * CS),  # L2 combine->reduce gate (per-slot, st=expected)
    ]


def _pack(regions):
    """256B-aligned packer -> ({name: (offset, itemsize, numel)}, total_bytes)."""
    offsets, cursor = {}, 0
    for name, item, numel in regions:
        cursor = _align(cursor)
        offsets[name] = (cursor, item, numel)
        cursor += numel * item
    return offsets, _align(cursor)


def layout(sl):
    """Both heaps' offset maps + totals: (main_off, sig_off, main_bytes, sig_bytes).

    Each offset map is ``{name: (byte_offset, itemsize, numel)}`` -- the single source
    of truth for region order / sizes (the host ``SymmBuffer`` builds its views from it)."""
    main_off, main_total = _pack(_main_regions(sl))
    sig_off, sig_total = _pack(_signal_regions(sl))
    return main_off, sig_off, main_total, sig_total


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------
def _as_i64(x):
    """Sign-extend an fx i32 (or fold a python int) to an i64 ArithValue."""
    if isinstance(x, int):
        return fx.Int64(x)
    return fx.arith.ArithValue(fx.arith.extsi(fx.T.i64(), x.ir_value()), signed=True)


def _region_ptr(sl, name, index=0, dst_rank=None):
    """Address (i64) of ``region[index]``; ``dst_rank`` translates into that peer."""
    main_off, sig_off, _, _ = layout(sl)
    if name in main_off:
        off, item, _ = main_off[name]
        base, offsets_ptr = sl.base, sl.offsets_ptr
    else:
        off, item, _ = sig_off[name]
        base, offsets_ptr = sl.signal_base, sl.signal_offsets_ptr
    addr = base + fx.Int64(off)
    if not (isinstance(index, int) and index == 0):
        addr = addr + _as_i64(index) * fx.Int64(item)
    if dst_rank is not None:
        res = addr_buffer_resource(offsets_ptr, num_records_bytes=int(sl.num_ranks) * 8)
        addr = addr + buffer_load(res, dst_rank, vec_width=1, dtype=fx.T.i64())
    return addr


def sym_map(sl, ptr, dst_rank):
    """Translate a local MAIN-heap ptr ``ptr`` (i64) into peer ``dst_rank``."""
    res = addr_buffer_resource(sl.offsets_ptr, num_records_bytes=int(sl.num_ranks) * 8)
    return ptr + buffer_load(res, dst_rank, vec_width=1, dtype=fx.T.i64())


# ---------------------------------------------------------------------------
# Host builder
# ---------------------------------------------------------------------------
def build_sym_layout(
    num_ranks,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    num_max_pool_tokens,
    num_max_pool_blocks,
    combine_slots,
    *,
    block_n=256,
    use_mxfp8=0,
    base=0,
    offsets_ptr=0,
    signal_base=0,
    signal_offsets_ptr=0,
    rank_idx=0,
):
    """Build a ``SymLayout`` from concrete dims + the two heaps' base/delta-table ptrs.

    Pool sizes (``num_max_pool_tokens`` / ``num_max_pool_blocks`` / ``combine_slots``)
    are passed explicitly so the layout matches whatever the host ``SymmBuffer``
    actually allocated (which derives them from ``block_m`` / ``pool_mult``)."""
    return SymLayout(
        base=Int64(base),
        offsets_ptr=Int64(offsets_ptr),
        signal_base=Int64(signal_base),
        signal_offsets_ptr=Int64(signal_offsets_ptr),
        rank_idx=Int32(rank_idx),
        num_ranks=int(num_ranks),
        num_experts=int(num_experts),
        num_experts_per_rank=int(num_experts) // int(num_ranks),
        num_max_tokens_per_rank=int(num_max_tokens_per_rank),
        num_topk=int(num_topk),
        hidden=int(hidden),
        intermediate_hidden=int(intermediate_hidden),
        num_max_pool_tokens=int(num_max_pool_tokens),
        num_max_pool_blocks=int(num_max_pool_blocks),
        combine_slots=int(combine_slots),
        block_n=int(block_n),
        use_mxfp8=int(use_mxfp8),
    )


# ---------------------------------------------------------------------------
# Convenience accessors: ``sym_layout.<region>_ptr`` -> this rank's i64 base ptr of that
# region (a read-only property, so it is safe to read inside scf if/while regions -- a
# struct method call would otherwise be treated as a state variable). Peer translation
# uses ``sym_map(sym_layout, sym_layout.<region>_ptr, dst_rank)`` for MAIN-heap regions;
# SIGNAL-heap regions are translated in-kernel against a ``signal_offsets_ptr`` delta
# resource. (_specialize_type subclasses SymLayout, so these properties are inherited
# by the per-shape specialized instances.) Names must match _main_regions / _signal_regions.
# ---------------------------------------------------------------------------
_REGION_ACCESSORS = (
    "pool",
    "c_buffer",
    "signal",
    "origin_rank",
    "origin_slot",
    "weight_recv_buf",
    "dedup_src_row",
    "combine_gate",
    "meta_scalars",
    "grid_sync_count",
    "profile",
    "act",
    "l2_token_buffer",
    "src_token_topk_idx",
    # mxfp8-only (the property raises if read when use_mxfp8=0, i.e. the region is absent)
    "pool_fp8",
    "pool_scale",
    "act_fp8",
    "act_scale",
    "pool_scale_ps",
    "dispatch_flag",
    "preshuffle_flag",
    "combine_flag",
    "comb",
    "reduce_flag",
)


def _make_region_accessor(region_name):
    def _get(self):
        return _region_ptr(self, region_name)

    _get.__name__ = region_name
    _get.__doc__ = f"This rank's i64 base ptr of the '{region_name}' region."
    return property(_get)


for _region_name in _REGION_ACCESSORS:
    setattr(SymLayout, f"{_region_name}_ptr", _make_region_accessor(_region_name))
