###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Intra-node EP communication layer (FlyDSL), precision-agnostic.

The cross-rank dispatch PUSH + per-pool-block scoreboard handshake, factored out
of the fused dispatch+GEMM kernels so both the fp8 and bf16 paths share ONE comm
implementation. Nothing here is GEMM- or precision-specific:

  * scoreboard / fence / spin prims (system-scope atomics over peer pointers),
  * ``_make_dispatch_tile`` -- the byte-agnostic warp-per-token XGMI push (token
    rows pushed as i32 words; ``signal=True`` adds the release-fence + scoreboard
    signal of the pushed pool blocks),
  * ``dispatch_bf16_tile`` -- the bf16 push closure (wraps the byte push with
    bf16 geometry: 2 bytes/element).

Both ``dispatch_grouped_gemm_fp8`` and ``dispatch_grouped_gemm_bf16_kernel``
import from here; this module depends only on ``common.tile_spec`` (for
``_emit_if_then``)."""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ASTRewriter, InsertEmptyYieldForSCFFor
from flydsl.expr.buffer_ops import (
    _unwrap_value,
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
    create_llvm_ptr,
    extract_base_index,
    get_element_ptr,
)

from primus_turbo.flydsl.common.tile_spec import _emit_if_then

_VEC = 16  # fp8 bytes per lane per push step (v4i32 = dwordx4 / b128 = XGMI-wide)
_WARP = 64  # wavefront size (gfx950); per-token warp copy uses ALL warps
# 8 waves (wave_m x wave_n = 2 x 4) — the tile-spec block size
_BLOCK_THREADS = 512

# ── scoreboard / cross-rank prims (the comm handshake; gemm_helper has none) ──
_ORD = _llvm.AtomicOrdering
_I4 = 4  # int32 byte stride / atomic alignment


def _elem_ptr_i32(tensor, idx):
    """LLVM ptr to int32 element ``tensor[idx]``."""
    base = create_llvm_ptr(extract_base_index(tensor, address_space=1), 1)
    byte_off = _unwrap_value(idx * fx.Int32(_I4))
    return get_element_ptr(base, byte_offset=byte_off, elem_type=fx.T.i8())


def _elem_ptr_i32_from_addr(addr_i64, idx):
    """LLVM ptr to int32 element at ``(addr_i64)[idx]`` (runtime peer base addr)."""
    base = create_llvm_ptr(_unwrap_value(addr_i64), 1)
    byte_off = _unwrap_value(idx * fx.Int32(_I4))
    return get_element_ptr(base, byte_offset=byte_off, elem_type=fx.T.i8())


def _atomic_add_addr(addr_i64, idx, val):
    """Relaxed atomic int32 add to ``(addr_i64)[idx]`` (peer scoreboard, system scope)."""
    ptr = _elem_ptr_i32_from_addr(addr_i64, idx)
    _llvm.atomicrmw(
        _llvm.AtomicBinOp.add, ptr, _unwrap_value(val), _ORD.monotonic, syncscope=None, alignment=_I4
    )


def _atomic_add_local(tensor, idx, val):
    """Relaxed atomic int32 add to ``tensor[idx]`` (local scoreboard, no peer ptr)."""
    ptr = _elem_ptr_i32(tensor, idx)
    _llvm.atomicrmw(
        _llvm.AtomicBinOp.add, ptr, _unwrap_value(val), _ORD.monotonic, syncscope=None, alignment=_I4
    )


def _atomic_add_local_ret(tensor, idx, val):
    """Relaxed atomic int32 add to ``tensor[idx]``; returns the OLD value.
    Used by the GEMM gate's last-reader scoreboard self-reset (count consumers)."""
    ptr = _elem_ptr_i32(tensor, idx)
    res = _llvm.atomicrmw(
        _llvm.AtomicBinOp.add, ptr, _unwrap_value(val), _ORD.monotonic, syncscope=None, alignment=_I4
    )
    return fx.arith.ArithValue(res, signed=True)


def _st_relaxed(tensor, idx, val):
    """Relaxed atomic int32 store of ``val`` into ``tensor[idx]`` (uncached signal reset)."""
    ptr = _elem_ptr_i32(tensor, idx)
    _llvm.StoreOp(_unwrap_value(val), ptr, ordering=_ORD.monotonic, syncscope=None, alignment=_I4)


def _ld_relaxed(tensor, idx):
    """Relaxed atomic int32 load of ``tensor[idx]`` for spin polling.
    Coherence comes from the scoreboard/sb_copy living in UNCACHED memory (every
    access bypasses L1/L2 -> always fresh); scope is just ordering here."""
    ptr = _elem_ptr_i32(tensor, idx)
    op = _llvm.LoadOp(fx.T.i32(), ptr, ordering=_ORD.monotonic, syncscope=None, alignment=_I4)
    return fx.arith.ArithValue(op.result, signed=True)


def _fence_release():
    """System-scope release fence: peer-pushed pool rows visible before the signal."""
    _llvm.fence(_ORD.release, syncscope=None)


def _fence_acquire():
    """System-scope acquire fence: read fresh peer-pushed pool after the signal."""
    _llvm.fence(_ORD.acquire, syncscope=None)


def _emit_for(stop, body):
    """Runtime ``for i in range(stop)`` from a module-level helper (calls ``body(i)``).
    The @flyc.kernel AST rewrite is body-only, so a shared helper must drive scf.for
    directly (mirrors ``_emit_if_then`` for if). ``i`` is the wrapped i32 index."""
    InsertEmptyYieldForSCFFor.scf_for_dispatch(
        fx.Int32(0), stop, fx.Int32(1), lambda iv, _names: body(fx.arith.ArithValue(iv, signed=True))
    )


# ──────────────────────────────────────────────────────────────────────
# Byte-agnostic warp-per-token XGMI push (shared by fp8 byte path + bf16).
# ──────────────────────────────────────────────────────────────────────
def _make_dispatch_tile(
    *,
    thread_index,
    n_warps,
    hidden_i32,
    cols_per_warp_i32,
    vec_i32,
    chunk_count,
    pool_record_bytes,
    input_resource,
    destination_resource,
    comm_start_resource,
    comm_count_resource,
    comm_source_offset_resource,
    source_tokens_resource,
    pool_address_resource,
    signal=False,
    scoreboard_address_resource=None,
    block_m=0,
    source_dedup_resource=None
):
    """Unified comm push, shared by the fused kernel and dispatch_only. One block (all
    warps) pushes a token slice of one task to its peer pool; ``n_sub`` (compile-time,
    per call) splits a task across n_sub blocks (coarse-task BW path; n_sub=1 = whole
    task). Per-task metadata is read ONCE. ``signal=True`` adds the release fence +
    scoreboard signal of the pushed pool blocks (the fused handshake).

    ``source_dedup_resource`` (token dedup): source-indexed flags; when a row's flag
    is 1 it is a SECONDARY (its token already crossed XGMI for this dest rank), so its
    hidden stores are skipped here and the dest pool row is filled by a dest-local copy
    in the consumer. Block signaling is unchanged (dest row range still covers it)."""
    warp_id = thread_index // fx.Int32(_WARP)
    lane_id = thread_index % fx.Int32(_WARP)

    def load_task(task_index):
        # read ALL per-task metadata ONCE (uniform scalars + dependent peer_pool)
        destination_rank = buffer_load(destination_resource, task_index, vec_width=1, dtype=fx.T.i32())
        dest_row_start = buffer_load(comm_start_resource, task_index, vec_width=1, dtype=fx.T.i32())
        source_offset = buffer_load(comm_source_offset_resource, task_index, vec_width=1, dtype=fx.T.i32())
        token_count = buffer_load(comm_count_resource, task_index, vec_width=1, dtype=fx.T.i32())
        pool_address = buffer_load(pool_address_resource, destination_rank, vec_width=1, dtype=fx.T.i64())
        peer_pool = create_buffer_resource_from_addr(pool_address, num_records_bytes=pool_record_bytes)
        return destination_rank, dest_row_start, source_offset, token_count, peer_pool

    def copy_slice(dest_row_start, source_offset, peer_pool, tok_lo, tok_hi):
        # warp-per-token copy of [tok_lo, tok_hi); only source_row is per-token
        local_count = (tok_hi - tok_lo - warp_id + fx.Int32(n_warps - 1)) // fx.Int32(n_warps)

        def _row(i):
            row_index = tok_lo + warp_id + i * fx.Int32(n_warps)

            def _push_row():
                source_row = buffer_load(
                    source_tokens_resource, source_offset + row_index, vec_width=1, dtype=fx.T.i32()
                )
                dest_row = dest_row_start + row_index
                chunk_values = []
                for chunk_index in fx.range_constexpr(chunk_count):
                    column = fx.Int32(chunk_index * cols_per_warp_i32) + lane_id * fx.Int32(vec_i32)
                    chunk_values.append(
                        buffer_load(
                            input_resource,
                            source_row * fx.Int32(hidden_i32) + column,
                            vec_width=vec_i32,
                            dtype=fx.T.i32(),
                        )
                    )
                for chunk_index in fx.range_constexpr(chunk_count):
                    column = fx.Int32(chunk_index * cols_per_warp_i32) + lane_id * fx.Int32(vec_i32)
                    buffer_store(
                        chunk_values[chunk_index], peer_pool, dest_row * fx.Int32(hidden_i32) + column
                    )

            if source_dedup_resource is None:
                _push_row()
            else:
                # skip the XGMI push for secondary rows (filled by dest-local copy)
                flag = buffer_load(
                    source_dedup_resource, source_offset + row_index, vec_width=1, dtype=fx.T.i32()
                )
                _emit_if_then(flag != fx.Int32(1), _push_row)

        _emit_for(local_count, _row)

    def dispatch_tile(task_index, sub, n_sub):
        destination_rank, dest_row_start, source_offset, token_count, peer_pool = load_task(task_index)
        if n_sub == 1:
            tok_lo = fx.Int32(0)
            tok_hi = token_count
        else:
            slice_tokens = (token_count + fx.Int32(n_sub - 1)) // fx.Int32(n_sub)
            tok_lo = sub * slice_tokens
            tok_hi = fx.arith.select(tok_lo + slice_tokens < token_count, tok_lo + slice_tokens, token_count)
        copy_slice(dest_row_start, source_offset, peer_pool, tok_lo, tok_hi)
        if signal:
            fx.rocdl.s_waitcnt(0)
            fx.gpu.barrier()

            def _signal():
                _fence_release()  # pushed rows visible cross-rank before the signal
                scoreboard_address = buffer_load(
                    scoreboard_address_resource, destination_rank, vec_width=1, dtype=fx.T.i64()
                )
                first_block = (dest_row_start + tok_lo) // fx.Int32(block_m)
                last_block = (dest_row_start + tok_hi - fx.Int32(1)) // fx.Int32(block_m)
                _emit_for(
                    last_block - first_block + fx.Int32(1),
                    lambda bo: _atomic_add_addr(scoreboard_address, first_block + bo, fx.Int32(1)),
                )

            _emit_if_then(thread_index == fx.Int32(0), _signal)

    return dispatch_tile


# ──────────────────────────────────────────────────────────────────────
# bf16 push closure: wraps the byte push with bf16 geometry (2 bytes/element).
# Each token row pushed as i32 words; only the byte geometry differs from fp8.
# ──────────────────────────────────────────────────────────────────────
def _bf16_push_geom(hidden_size):
    hidden_bytes = hidden_size * 2
    assert (
        hidden_bytes % (_WARP * _VEC) == 0
    ), "hidden*2 must be a multiple of 1024 bytes (warp push step) -> hidden % 512 == 0"
    vec_i32 = _VEC // 4
    hidden_i32 = hidden_bytes // 4
    n_warps = _BLOCK_THREADS // _WARP
    cols_per_warp_i32 = _WARP * vec_i32
    chunk_count = hidden_i32 // cols_per_warp_i32
    return vec_i32, hidden_i32, n_warps, cols_per_warp_i32, chunk_count


# ──────────────────────────────────────────────────────────────────────
# Stage-2 "permute" copy: fill this rank's dedup SECONDARY pool rows from their
# PRIMARY rows (dest-local copy). dispatch_bf16_tile's stage-1 pushed only the
# primaries over XGMI (source_dedup skip); the redundant rows of a token routed
# to >=2 experts on this dest rank are filled here -> each lands in its expert's
# pool slot. Module-level + ASTRewriter so the primary-scoreboard spin (while) is
# lowered to scf (same pattern as the prologue's grid_barrier). Called from the
# fused kernel's comm role; the GEMM role then gates on sb_copy.
# ──────────────────────────────────────────────────────────────────────
def permute_token_tile(
    thread_index,
    block_index,
    comm_block_count,
    num_tile_blocks_resource,
    dedup_src_row_resource,
    expected_resource,
    scoreboard_tensor,
    pool_i32_resource,
    sb_copy_resource,
    block_m,
    n_warps,
    rows_per_warp,
    chunk_count,
    cols_per_warp_i32,
    vec_i32,
    hidden_i32,
):
    """Round-robin this rank's real pool blocks over the comm blocks; each block
    warp-per-row copies its secondaries from the primary (gated on the primary
    block's scoreboard), then flips ``sb_copy[blk]``. Copy waits only on PRIMARY
    blocks (never on copies) -> acyclic, deadlock-free."""
    warp_id = thread_index // fx.Int32(_WARP)
    lane_id = thread_index % fx.Int32(_WARP)
    real_tiles = buffer_load(num_tile_blocks_resource, fx.Int32(0), vec_width=1, dtype=fx.T.i32())
    # Per-pool-block: copy this block's secondaries, then signal sb_copy[blk] -> the GEMM
    # role's block_m starts as soon as ITS block is copied (fine-grained overlap, not a
    # coarse all-primaries barrier). Per-row wait is lane-0 only (uncached scoreboard);
    # the copy loads ALL chunks then stores all -> latency-pipelined, bandwidth-bound.
    copy_count = (real_tiles - block_index + comm_block_count - fx.Int32(1)) // comm_block_count
    for copy_iteration in range(copy_count):
        blk = block_index + copy_iteration * comm_block_count
        for jj in range(fx.Int32(rows_per_warp)):
            dst_row = blk * fx.Int32(block_m) + warp_id + jj * fx.Int32(n_warps)
            primary = buffer_load(dedup_src_row_resource, dst_row, vec_width=1, dtype=fx.T.i32())
            if primary >= fx.Int32(0):
                p_block = primary // fx.Int32(block_m)
                exp_p = buffer_load(expected_resource, p_block, vec_width=1, dtype=fx.T.i32())
                # lane 0 only spins on the UNCACHED scoreboard until the primary landed
                if lane_id == fx.Int32(0):
                    sig_p = _ld_relaxed(scoreboard_tensor, p_block)
                    while sig_p < exp_p:
                        fx.rocdl.s_sleep(fx.Int32(2))
                        sig_p = _ld_relaxed(scoreboard_tensor, p_block)
                # load ALL chunks first (latency pipelined), then store all -> BW-bound
                chunk_values = []
                for chunk in fx.range_constexpr(chunk_count):
                    col = fx.Int32(chunk * cols_per_warp_i32) + lane_id * fx.Int32(vec_i32)
                    chunk_values.append(
                        buffer_load(
                            pool_i32_resource,
                            primary * fx.Int32(hidden_i32) + col,
                            vec_width=vec_i32,
                            dtype=fx.T.i32(),
                        )
                    )
                for chunk in fx.range_constexpr(chunk_count):
                    col = fx.Int32(chunk * cols_per_warp_i32) + lane_id * fx.Int32(vec_i32)
                    buffer_store(chunk_values[chunk], pool_i32_resource, dst_row * fx.Int32(hidden_i32) + col)
        fx.rocdl.s_waitcnt(fx.Int32(0))  # local copies committed to L2
        fx.gpu.barrier()
        if thread_index == fx.Int32(0):
            _fence_release()  # POOL is cached: copied rows visible before the sb_copy gate flip
            # sb_copy is UNCACHED -> the store itself is immediately visible to the GEMM gate
            buffer_store(fx.Int32(1), sb_copy_resource, blk)


# Lower the spin-while / runtime for to scf (called inside the fused @flyc.kernel).
permute_token_tile = ASTRewriter.transform(permute_token_tile)


def dispatch_bf16_tile(
    *,
    thread_index,
    hidden_size,
    pool_capacity,
    input_resource,
    destination_resource,
    comm_start_resource,
    comm_count_resource,
    comm_source_offset_resource,
    source_tokens_resource,
    pool_address_resource,
    signal=False,
    scoreboard_address_resource=None,
    block_m=0,
    source_dedup_resource=None
):
    """bf16 comm PUSH closure (wraps the shared byte push with bf16 geometry)."""
    vec_i32, hidden_i32, n_warps, cols_per_warp_i32, chunk_count = _bf16_push_geom(hidden_size)
    pool_record_bytes = pool_capacity * hidden_size * 2
    return _make_dispatch_tile(
        thread_index=thread_index,
        n_warps=n_warps,
        hidden_i32=hidden_i32,
        cols_per_warp_i32=cols_per_warp_i32,
        vec_i32=vec_i32,
        chunk_count=chunk_count,
        pool_record_bytes=pool_record_bytes,
        input_resource=input_resource,
        destination_resource=destination_resource,
        comm_start_resource=comm_start_resource,
        comm_count_resource=comm_count_resource,
        comm_source_offset_resource=comm_source_offset_resource,
        source_tokens_resource=source_tokens_resource,
        pool_address_resource=pool_address_resource,
        signal=signal,
        scoreboard_address_resource=scoreboard_address_resource,
        block_m=block_m,
        source_dedup_resource=source_dedup_resource,
    )
