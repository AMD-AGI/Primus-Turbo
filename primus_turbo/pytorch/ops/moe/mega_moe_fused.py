###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fully fused mega MoE forward (FlyDSL) over a single symmetric buffer.

Pipeline (EP intra-node, mirrors the EP test):

  1. mega_moe_prologue_impl     -- build the cross-rank dispatch plan from topk
  2. dispatch_grouped_gemm_impl -- cross-rank dispatch PUSH + grouped L1 GEMM (NT)
  3. swiglu                     -- fused SwiGLU activation
  4. grouped_gemm_combine_impl  -- grouped L2 GEMM (NT) + cross-rank combine PUSH
  5. weighted topk reduce       -- y[token] = sum_k w[token, k] * comb[k, token]

Cross-rank pool + scratch memory is carved out of ONE cached symmetric main
buffer (``SymmBuffer``, sized by ``get_symm_buffer_size_for_mega_moe`` --
inspired by ``deep_gemm/mega``); the spin-wait flags AND the combine buffer
(``comb``) live in the uncached signal pad -- the reduce reads ``comb`` through
the cache, so a cached ``comb`` would serve stale locally-zeroed rows. The stage
kernels are torch custom ops from ``primus_turbo.pytorch.kernels.mega_moe`` and are
called directly in the forward.

``MegaMoEFusedFunction`` wraps the forward for autograd; backward (conjugate via
Dispatch<->Combine duality, mirroring ``MegaKernelFlyDSL/ops/mega_moe.py``) returns
grads for x / w1 / w2 / topk_weights:

  1. dispatch_grouped_gemm_impl(layout="nn") -- dispatch dy + L2 dgrad (d_swiglu = d_l2y @ w2)
  2. swiglu_backward(scale=dispatch_weight_in_buf, grad_gate) -- re-inject routing weight + gate grad
  3. dW2 = grouped_gemm_variable_k(d_l2y, w*act, trans_a) ; dW1 = variable_k(grad_l1, pool_x, trans_a)
  4. grouped_gemm_combine_impl(layout="nn") -- 3-role L1 dgrad GEMM + combine PUSH + dx reduce
  5. d_topk_w[t,k] = grad_gate scattered cross-rank into combine_gate (the gate ride-along of step 4)

Naming mirrors triton_dist (``ep_a2a_fused_layer`` / ``ep_moe_fused``): the per-pool-row
routing weight lives in ``weight_recv_buf`` (used as ``dispatch_weight_in_buf``); the prologue
rides each token's routing weight cross-rank into it (no all_gather). Step 4 reuses the forward's
fused 3-role kernel (``grouped_gemm_combine_impl``), with the combine role also scattering
``grad_gate`` -> ``combine_gate`` for ``d_topk_w`` (mirrors ``MegaKernelFlyDSL/ops/mega_moe.py``).
"""

import os

import torch

from primus_turbo.flydsl.mega.mega_moe_epilogue import (
    ACTIVATION_CLAMP,
    swiglu,
    swiglu_backward,
)
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.symm_mem import SymmetricMemory
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
    grouped_gemm_variable_k_impl,
)
from primus_turbo.pytorch.kernels.mega_moe.dispatch_grouped_gemm_impl import (
    dispatch_grouped_gemm_impl,
)
from primus_turbo.pytorch.kernels.mega_moe.grouped_gemm_combine_impl import (
    grouped_gemm_combine_impl,
)
from primus_turbo.pytorch.kernels.mega_moe.mega_moe_prologue_impl import (
    mega_moe_prologue_impl,
)

__all__ = [
    "SymmBuffer",
    "get_symm_buffer_size_for_mega_moe",
    "get_symm_buffer_for_mega_moe",
    "MegaMoEFusedFunction",
    "mega_moe_fused",
]

_FLYDSL = BackendType.FLYDSL.value
# each sub-buffer base aligned to 256B (matches the kernels' bump cursor)
_BASE_ALIGNMENT = 256


def _align(nbytes, alignment=_BASE_ALIGNMENT):
    return (nbytes + alignment - 1) // alignment * alignment


# --------------------------------------------------------------------------- #
# Symmetric-buffer layout: one allocation carved into all sub-buffers.
# --------------------------------------------------------------------------- #
def _layout(regions):
    """Assign a 256B-aligned byte offset to each (name, dtype, numel); return (spec, total)."""
    offset, spec = 0, {}
    for name, dtype, numel in regions:
        offset = _align(offset)
        spec[name] = (offset, dtype, numel)
        offset += numel * dtype.itemsize
    return spec, _align(offset)


def _signal_regions(num_pool_blocks, combine_slots, hidden, world):
    """Uncached signal-pad regions (spin-wait flags + the cross-rank combine buffer).

    Everything here lives in the signal pad (``hipMallocUncached`` -> L1/L2-bypass).
    The flags must be uncached so the scoreboard handshake never spins on a stale
    cached value; ``comb`` must be uncached because it is written cross-rank and then
    read by the FlyDSL topk reduce -- a cached ``comb`` is locally zeroed before peers
    push into it, so the reduce would read stale zeros for the still-cached rows."""
    i32, bf16 = torch.int32, torch.bfloat16
    return [
        # reserved signal_pad[0:world] for a device IPC barrier (sm.barrier()); currently unused
        # (fwd/bwd are barrier-free) but kept at offset 0 so re-enabling the barrier is drop-in.
        ("_ipc_barrier", i32, world),
        ("scoreboard", i32, num_pool_blocks),  # cross-rank dispatch scoreboard
        ("sb_consume", i32, num_pool_blocks),  # GEMM last-reader counter (kernel self-resets)
        ("sb_l2", i32, num_pool_blocks),  # local L2 spin-wait gate
        ("comb", bf16, combine_slots * hidden),  # cross-rank combine buffer (read by reduce)
        ("barrier_local", i32, combine_slots),  # per-slot ready flags for the 3-role reduce
    ]


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
    AND the cross-rank combine buffer (``comb``) -- ``comb`` is read by the FlyDSL
    topk reduce, so it must bypass the cache to avoid stale reads.

    ``activation`` is reserved (both gated ``swiglu`` and non-gated variants emit
    ``intermediate_hidden``, so it does not change sizing today).
    """
    experts_per_rank = num_experts // world_size
    avg_recv_tokens = num_max_tokens_per_rank * num_topk
    pool_capacity = _align(pool_mult * avg_recv_tokens + experts_per_rank * block_m, block_m)
    num_pool_blocks = pool_capacity // block_m
    combine_slots = num_topk * num_max_tokens_per_rank
    i32, f32, i64, bf16 = torch.int32, torch.float32, torch.int64, torch.bfloat16

    # Only buffers the kernels actually touch. The prologue's dispatch-plan tables
    # (destination/start/count/.../tile_to_group/expected/source_*) are allocated +
    # returned by ``mega_moe_prologue_impl`` itself, so we don't carve them here.
    # c_buffer / signal back the cross-rank count exchange via ``peer_ptrs``.
    main = [
        # cross-rank data (reached via per-rank pointer tables)
        ("pool", bf16, pool_capacity * hidden),
        ("c_buffer", i32, world_size * num_experts),
        ("signal", i32, world_size),
        ("origin_rank", i32, pool_capacity),
        ("origin_slot", i32, pool_capacity),
        # per-pool-row routing weight (prologue rides it cross-rank from weight_send_buf/src_w)
        # -> bwd swiglu scale. triton_dist name: weight_recv_buf (used as dispatch_weight_in_buf).
        ("weight_recv_buf", f32, pool_capacity),
        # backward d_topk_w push slots (gate grad), slot = token*topk + k; written cross-rank
        ("combine_gate", f32, combine_slots),
        # prologue device scalars / barrier / profile (caller-owned, kernel-written)
        ("meta_scalars", i32, 8),
        ("grid_barrier_state", i32, 2),
        ("profile", i64, 8),
        # intermediates (swiglu output / L2 GEMM output)
        ("act", bf16, pool_capacity * intermediate_hidden),
        ("l2_token_buffer", bf16, pool_capacity * hidden),
    ]
    # spin-wait flags + combine buffer kept in the UNCACHED signal pad (see helper)
    slice_input_buffers, num_bytes = _layout(main)
    _, signal_bytes = _layout(_signal_regions(num_pool_blocks, combine_slots, hidden, world_size))
    meta = dict(
        world_size=world_size,
        num_experts=num_experts,
        num_tokens=num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        activation=activation,
        block_m=block_m,
        pool_capacity=pool_capacity,
        num_pool_blocks=num_pool_blocks,
        combine_slots=combine_slots,
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

        num_bytes, slice_input_buffers, signal_bytes, meta = get_symm_buffer_size_for_mega_moe(
            self.world,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            block_m=block_m,
            pool_mult=pool_mult,
        )
        # num_tokens / num_experts / hidden / pool_capacity / ...
        self.__dict__.update(meta)

        # spin-wait flags + combine buffer carved from the UNCACHED signal pad
        signal_spec, _ = _layout(
            _signal_regions(self.num_pool_blocks, self.combine_slots, self.hidden, self.world)
        )

        # one symmetric allocation: cached main buffer + uncached signal pad
        self.sm = SymmetricMemory(group, alloc_size=num_bytes, signal_pad_size=signal_bytes)
        self.sm.get_buffer(self.rank, (num_bytes,), torch.int8).zero_()
        self.sm.get_signal_pad(self.rank, (signal_bytes,), torch.int8).zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        self.signal_pad = self.sm.get_signal_pad(self.rank)

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
        # reshape the matrix-shaped views
        self.pool = self.pool.view(self.pool_capacity, self.hidden)
        self.act = self.act.view(self.pool_capacity, self.intermediate_hidden)
        self.l2_token_buffer = self.l2_token_buffer.view(self.pool_capacity, self.hidden)
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

    def assert_capacity(self):
        """Guard against silent pool overflow (bounded buffer_store drops OOB rows)."""
        total_rows = int(self.meta_scalars[0].item())
        assert total_rows <= self.pool_capacity, (
            f"rank {self.rank}: dispatched rows {total_rows} exceed pool_capacity "
            f"{self.pool_capacity}; raise pool_mult"
        )

    def destroy(self):
        _SYMM_BUFFER_CACHE.pop(getattr(self, "_cache_key", None), None)
        try:
            self.sm.destroy()
        except Exception:
            pass


# Cache the single symmetric allocation per (group, shape, tiling) so repeated
# forwards reuse it -- each build is a collective rendezvous, so we allocate once.
_SYMM_BUFFER_CACHE = {}


def get_symm_buffer_for_mega_moe(
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
) -> SymmBuffer:
    """Get (allocate or reuse) the single symmetric buffer for a fused mega MoE forward.

    Cached per (group, shape, tiling): the first call rendezvous-allocates, later
    calls with the same key return the same buffer (allocate-once / reuse-per-step)."""
    key = (
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        block_m,
        block_n,
        pool_mult,
    )
    symm = _SYMM_BUFFER_CACHE.get(key)
    if symm is None:
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
        symm._cache_key = key
        _SYMM_BUFFER_CACHE[key] = symm
    return symm


# --------------------------------------------------------------------------- #
# Forward / backward: call the stage kernels directly over the symmetric buffer.
# --------------------------------------------------------------------------- #
class MegaMoEFusedFunction(torch.autograd.Function):
    """Wraps the fused mega MoE forward so its output joins the autograd graph.

    Backward (conjugate via Dispatch<->Combine duality) returns grads for x, w1,
    w2, and topk_weights; topk_idx / group / tiling args are non-differentiable.

    NOTE: the symmetric buffer is cached/shared per (group, shape, tiling), and
    ``backward`` mutates it in place. A second ``forward`` with the SAME shape
    before this op's ``backward`` (e.g. another same-shape layer, grad
    accumulation, or activation recompute) would clobber the shared buffer; only
    a single forward->backward per shape between collectives is safe."""

    @staticmethod
    def forward(ctx, x, topk_idx, topk_weights, w1, w2, group, block_m, block_n, pool_mult):
        num_tokens, hidden_size = x.shape
        num_topk = topk_idx.shape[-1]

        ctx.set_materialize_grads(False)
        # derive the shape config from the tensors, then get the cached symmetric buffer
        experts_per_rank = w1.shape[0]
        symm = get_symm_buffer_for_mega_moe(
            group,
            num_experts=experts_per_rank * group.size(),
            num_max_tokens_per_rank=num_tokens,
            num_topk=num_topk,
            hidden=hidden_size,
            intermediate_hidden=w1.shape[1] // 2,
            block_m=block_m,
            block_n=block_n,
            pool_mult=pool_mult,
        )
        num_experts = symm.num_experts

        # 1) prologue: build the cross-rank dispatch plan; returns the plan output tables
        # (source_topk_slot / source_weight are unused here -- weight is applied at the reduce)
        # The prologue resets scoreboard -> 0 and barrier_local -> -1 in-kernel before its
        # final cross-rank barrier (folds out those two host launches). sb_l2 / comb are still
        # host-zeroed below (comb is large -> faster as a full-grid memset + the host barrier).
        # plan = bundled dispatch plan (kept as one list, passed straight into dispatch)
        plan, tile_to_group, expected = mega_moe_prologue_impl(
            topk_idx,
            topk_weights,
            symm.buffer_base,
            symm.buffer_offsets,
            symm.origin_rank,
            symm.origin_slot,
            symm.meta_scalars,
            symm.grid_barrier_state,
            symm.profile,
            symm.scoreboard,
            symm.barrier_local,
            num_tokens,
            num_topk,
            num_experts,
            symm.world,
            symm.rank,
            symm.block_m,
            symm.pool_capacity,
            _FLYDSL,
            no_cpu_sync=True,
        )
        num_tile_blocks = symm.meta_scalars[1:2]  # device real-tile count

        # sb_l2 (local L2 scoreboard) self-resets in the combine push role and is rank-local
        # (no cross-rank), so no host zero is needed. comb is fully overwritten by the combine
        # PUSH (reduce skips dropped slots). scoreboard/barrier_local reset in-kernel by prologue.

        # 2) cross-rank dispatch PUSH + grouped L1 GEMM (NT): pool[M,H] @ w1[g,2I,H] -> l1_out
        l1_out = dispatch_grouped_gemm_impl(
            x,
            plan,
            symm.pool,
            symm.pool_ptrs,
            w1,
            tile_to_group,
            symm.scoreboard,
            symm.scoreboard_ptrs,
            expected,
            num_tile_blocks,
            symm.sb_consume,
            num_experts,
            _FLYDSL,
            layout="nt",
            BM=symm.block_m,
            BN=symm.block_n,
        )

        # 3) fused SwiGLU activation: l1_out[M,2I] -> act[M,I]
        swiglu(
            l1_out,
            symm.act,
            symm.intermediate_hidden,
            symm.pool_capacity,
            num_tile_blocks=num_tile_blocks,
            BM=symm.block_m,
        )

        # 4) grouped L2 GEMM (NT) + cross-rank combine PUSH + fused 3-role topk reduce:
        #    act[M,I] @ w2[g,H,I] -> comb (cross-rank), then per-slot flags gate the
        #    weighted reduce comb[token*topk+k] -> y[token] (no host barrier needed).
        y = torch.empty((num_tokens, symm.hidden), dtype=torch.bfloat16, device=x.device)
        # fixed per-rank token count, cached on the symm buffer (reduce reads only [rank])
        num_tokens_per_rank = symm.num_tokens_per_rank
        # CU split tunable via env; reduce runs on empty GEMM blocks so dedicated region defaults to 0
        _combine_cu = int(os.environ.get("MEGA_COMBINE_CU", "64"))
        _reduce_cu = int(os.environ.get("MEGA_REDUCE_CU", "0"))
        grouped_gemm_combine_impl(
            symm.act,
            w2,
            symm.l2_token_buffer,
            tile_to_group,
            symm.sb_l2,
            symm.origin_rank,
            symm.origin_slot,
            num_tile_blocks,
            y,
            symm.barrier_local,
            symm.combine_slots,
            _FLYDSL,
            comb_addrs=symm.comb_addrs,
            comb_local=symm.comb,
            barrier_addrs=symm.barrier_addrs,
            topk_indices=topk_idx.to(torch.int32).contiguous().view(-1),
            num_tokens_per_rank=num_tokens_per_rank,
            topk_weights=topk_weights.to(torch.float32).contiguous().view(-1),
            topk=num_topk,
            num_experts=num_experts,
            rank=symm.rank,
            num_combine_cu=_combine_cu,
            num_reduce_cu=_reduce_cu,
            layout="nt",
            BM=symm.block_m,
            BN=symm.block_n,
        )

        # ---- stash everything backward needs (clone the persistent symm buffers) ----
        if any(ctx.needs_input_grad):
            # per-pool-row routing weight (triton_dist: dispatch_weight_in_buf): the prologue
            # rode each token's routing weight cross-rank into weight_recv_buf[dest_row] (no all_gather).
            dispatch_weight_in_buf = symm.weight_recv_buf.clone()

            ctx.symm = symm
            ctx.num_tokens = num_tokens
            ctx.num_topk = num_topk
            ctx.num_experts = num_experts
            ctx.inter = symm.intermediate_hidden
            ctx.hidden = symm.hidden
            # group metadata for the variable-K wgrads (block_m-granular, padded rows -> 0)
            eo = torch.zeros((experts_per_rank + 1,), dtype=torch.int32, device=x.device)
            eo[1:] = (
                (
                    torch.bincount(tile_to_group.to(torch.int64), minlength=experts_per_rank)[
                        :experts_per_rank
                    ]
                    * symm.block_m
                )
                .to(torch.int32)
                .cumsum(0)
            )
            group_offs = eo.to(torch.int64)
            group_lens = (eo[1:] - eo[:-1]).to(torch.int64)
            # plan tensors (prologue outputs are fresh) + cloned persistent symm buffers
            # d_topk_w is produced in-kernel in backward (swiglu_backward grad_gate +
            # combine gate-scatter), so the forward combine buffer is NOT saved.
            ctx.save_for_backward(
                *plan,  # bundled dispatch plan (7 tensors), rebuilt in backward
                tile_to_group,
                expected,
                symm.origin_rank.clone(),
                symm.origin_slot.clone(),
                num_tile_blocks.clone(),  # device real-tile count
                symm.pool.clone(),  # dispatched x in pool order (dW1 B)
                l1_out,  # swiglu input (swiglu_backward; dW2 B = act_w recomputed here)
                dispatch_weight_in_buf,
                group_offs,
                group_lens,
                w1,
                w2,
                topk_idx,
            )
        return y

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_y):
        """Conjugate of forward via Dispatch<->Combine duality; grads for x / w1 / w2 / topk_w."""
        # set_materialize_grads(False) -> grad_y is None when the output got no grad
        if grad_y is None:
            return (None,) * 9
        (
            d0,
            d1,
            d2,
            d3,
            d4,
            d5,
            d6,
            tile_to_group,
            expected,
            origin_rank,
            origin_slot,
            num_tile_blocks,
            saved_pool,
            l1_out,
            dispatch_weight_in_buf,
            group_offs,
            group_lens,
            w1,
            w2,
            topk_idx,
        ) = ctx.saved_tensors
        plan = [d0, d1, d2, d3, d4, d5, d6]  # rebuilt bundled dispatch plan
        symm = ctx.symm
        T, K, H, I = ctx.num_tokens, ctx.num_topk, ctx.hidden, ctx.inter
        num_experts = ctx.num_experts
        dy = grad_y.contiguous().to(torch.bfloat16)
        triton_be = BackendType.TRITON.value

        # ===== STEP 1: combine^T (dispatch dy) + L2 dgrad — fused dispatch + NN GEMM =====
        # No cross-rank barrier: scoreboard self-resets EARLY in the forward dispatch (its last
        # GEMM reader) while the backward dispatch's peer signal is LATE (a full fwd-pass after the
        # prologue barrier), so the reuse does not race; pool/comb are likewise freed by forward
        # long before backward reuses them. (Holds while per-iter rank skew < ~one pass.)
        # dispatch dy into the pool (-> d_l2y, unweighted), then d_swiglu = d_l2y @ w2 (NN).
        d_swiglu = dispatch_grouped_gemm_impl(
            dy,
            plan,
            symm.pool,
            symm.pool_ptrs,
            w2,
            tile_to_group,
            symm.scoreboard,
            symm.scoreboard_ptrs,
            expected,
            num_tile_blocks,
            symm.sb_consume,
            num_experts,
            _FLYDSL,
            layout="nn",
            BM=symm.block_m,
            BN=symm.block_n,
        )
        d_l2y = symm.pool  # pool now holds the dispatched dy rows

        # ===== STEP 2: SwiGLU^T (re-inject routing weight) + gate grad (= d_topk_w/row) =====
        # grad_gate[r] = <d_swiglu_unweighted[r], act_unweighted[r]> = d_topk_w of pair (t,k);
        # from the UNSCALED dact (independent of weight placement). swiglu_backward allocates
        # grad_gate internally (torch.empty, single store/row) and returns it (triton_dist dscale).
        # act_w = (recomputed fwd act) * routing weight is emitted by swiglu_backward (folds the
        # host saved_act*weight mul AND removes the forward saved_act clone).
        grad_l1, grad_gate, act_w = swiglu_backward(
            d_swiglu,
            l1_out,
            I,
            clamp=ACTIVATION_CLAMP,
            scale=dispatch_weight_in_buf,
            num_tile_blocks=num_tile_blocks,
            BM=symm.block_m,
            return_gate=True,
            return_act_w=True,
        )

        # ===== dW2 = d_l2y^T @ act_w (variable-K wgrad; weight already folded into act_w) =====
        dW2 = grouped_gemm_variable_k_impl(
            d_l2y,
            act_w,
            group_lens,
            group_offs,
            trans_a=True,
            trans_b=False,
            trans_c=False,
            num_cu=None,
            default_backend=triton_be,
        )

        # ===== STEP 3: L1 dgrad (grad_pool = grad_l1 @ w1, NN) + combine PUSH + dx reduce =====
        # 3-role fused (mirrors forward STEP 4); the combine role also scatters grad_gate ->
        # origin combine_gate[token*topk+k] (d_topk_w). grad_l1 carries the weight -> unweighted reduce.
        # sb_l2 self-resets in the combine push role; comb is fully overwritten by the push
        # (reduce skips dropped slots); combine_gate's scatter is a per-slot overwrite (dropped
        # pairs masked to 0 below); barrier_local self-resets to 0 in the reduce (forward consumed
        # the same non-dropped slots -> already 0 here) -> none need a host zero. No STEP3 barrier:
        # the STEP1 barrier already drained all ranks' forward (frees comb/barrier_local), and the
        # STEP1 cross-rank dispatch couples ranks enough for the no-wait reduce's push-before-read.
        dx = torch.empty((T, H), dtype=torch.bfloat16, device=dy.device)
        # fixed per-rank token count, cached on the symm buffer (reduce reads only [rank])
        num_tokens_per_rank = symm.num_tokens_per_rank
        # CU split tunable via env; reduce runs on empty GEMM blocks so dedicated region defaults to 0
        _combine_cu = int(os.environ.get("MEGA_COMBINE_CU", "64"))
        _reduce_cu = int(os.environ.get("MEGA_REDUCE_CU", "0"))
        # the fused combine scatters grad_gate -> symm.combine_gate, and the reduce folds the
        # masked d_topk_w (= combine_gate * (0<=topk_idx<E)) into a fresh returned buffer.
        _, d_topk_w_flat = grouped_gemm_combine_impl(
            grad_l1,
            w1,
            symm.l2_token_buffer,
            tile_to_group,
            symm.sb_l2,
            origin_rank,
            origin_slot,
            num_tile_blocks,
            dx,
            symm.barrier_local,
            symm.combine_slots,
            _FLYDSL,
            comb_addrs=symm.comb_addrs,
            comb_local=symm.comb,
            barrier_addrs=symm.barrier_addrs,
            topk_indices=topk_idx.to(torch.int32).contiguous().view(-1),
            num_tokens_per_rank=num_tokens_per_rank,
            topk_weights=None,
            grad_gate=grad_gate,
            gate_addrs=symm.gate_addrs,
            gate_local=symm.combine_gate,
            topk=K,
            num_experts=num_experts,
            rank=symm.rank,
            num_combine_cu=_combine_cu,
            num_reduce_cu=_reduce_cu,
            layout="nn",
            BM=symm.block_m,
            BN=symm.block_n,
        )

        # ===== dW1 = grad_l1^T @ pool(x) (variable-K wgrad) =====
        dW1 = grouped_gemm_variable_k_impl(
            grad_l1,
            saved_pool,
            group_lens,
            group_offs,
            trans_a=True,
            trans_b=False,
            trans_c=False,
            num_cu=None,
            default_backend=triton_be,
        )

        # d_topk_w[t,k]: produced (masked + decoupled into a fresh buffer) by the combine reduce;
        # just reshape the [combine_slots] output to [num_tokens, num_topk].
        d_topk_w = d_topk_w_flat.view(T, K)
        # grads for (x, topk_idx, topk_weights, w1, w2, group, block_m, block_n, pool_mult)
        return (dx, None, d_topk_w, dW1.to(w1.dtype), dW2.to(w2.dtype), None, None, None, None)


def mega_moe_fused(
    group, x, topk_idx, topk_weights, w1, w2, *, block_m=256, block_n=256, pool_mult=2
) -> torch.Tensor:
    """One fully fused mega MoE forward; the symmetric buffer is fetched (and cached)
    internally via ``get_symm_buffer_for_mega_moe`` from the tensor shapes + ``group``.

    ``x`` [num_tokens, hidden] bf16, ``topk_idx`` / ``topk_weights`` [num_tokens, num_topk],
    ``w1`` [experts_per_rank, 2*intermediate_hidden, hidden],
    ``w2`` [experts_per_rank, hidden, intermediate_hidden]. Returns y [num_tokens, hidden].
    The buffer is allocated once per (group, shape, tiling) and reused on later calls."""
    return MegaMoEFusedFunction.apply(x, topk_idx, topk_weights, w1, w2, group, block_m, block_n, pool_mult)
