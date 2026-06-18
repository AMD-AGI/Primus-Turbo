###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused cross-rank dispatch PUSH + grouped FP8 GEMM (NT), FlyDSL.

Role-specialized single kernel: ``block_index < comm_blocks`` blocks push token
rows to peer pools over XGMI and signal a per-pool-block scoreboard; the
remaining blocks each compute one NT output tile of the grouped FP8 GEMM
(``A=pool[M,K]`` fp8, per-expert ``B=weight[G,N,K]`` fp8 -> ``C=out[M,N]`` bf16),
spinning on the scoreboard until their pool block is filled. The comm latency is
hidden under the MFMA-bound GEMM.

Built DIRECTLY on ``common.tile_spec.DenseFp8TileSpec``: ``DispatchGroupFP8TileSpec``
swaps in the LINEAR no-sync tile-id scheduler (offset past the comm blocks) and
reuses every stock per-stage hook + ``spec.emit`` unchanged; the per-expert B slab
rides the stock ``emit(..., group_base=...)`` scalar. The fused launcher in
``_compile`` emits BOTH roles in one kernel -- the GEMM tile (via ``spec.emit``) and
the comm push (``dispatch_tile``, the ONLY kernel-specific code). The comm handshake
prims + host tensor helpers are vendored here, so this module depends only on
``common.tile_spec`` (not the grouped-spec module).

Per-tensor A/B scale (matches the dense tile spec's ``StoreCPerTensor``)."""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import InsertEmptyYieldForSCFFor
from flydsl.expr.buffer_ops import (
    _unwrap_value,
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
    create_llvm_ptr,
    extract_base_index,
    get_element_ptr,
)

from primus_turbo.flydsl.common.tile_spec import (
    BLOCK_K,
    DenseFp8TileSpec,
    LinearNoSyncScheduler,
    _emit_if_then,
)
from primus_turbo.flydsl.utils.gemm_helper import ceildiv, make_value_attrs


_VEC = 16              # fp8 bytes per lane per push step (v4i32 = dwordx4 / b128 = XGMI-wide)
_WARP = 64             # wavefront size (gfx950); per-token warp copy uses ALL warps
# 8 waves (wave_m x wave_n = 2 x 4) — the tile-spec block size
_BLOCK_THREADS = 512
# TN inplace MFMA needs a pinned AGPR budget.
_LAYOUT_AGPR = {"nt": 0, "nn": 0, "tn": 128}

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
    _llvm.atomicrmw(_llvm.AtomicBinOp.add, ptr, _unwrap_value(val),
                    _ORD.monotonic, syncscope=None, alignment=_I4)


def _ld_relaxed(tensor, idx):
    """Relaxed atomic int32 load of ``tensor[idx]`` for spin polling."""
    ptr = _elem_ptr_i32(tensor, idx)
    op = _llvm.LoadOp(fx.T.i32(), ptr, ordering=_ORD.monotonic,
                      syncscope=None, alignment=_I4)
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
        fx.Int32(0), stop, fx.Int32(1),
        lambda iv, _names: body(fx.arith.ArithValue(iv, signed=True)))


# ── host-side tensor helpers ──
def _as_i8_flat(t):
    """Zero-copy flat int8 byte view (fp8 -> int8)."""
    if t.element_size() == 1 and t.dtype != torch.int8:
        return t.contiguous().view(torch.int8).view(-1)
    return t.contiguous().view(-1)


def _scalar(scale, device):
    return scale.to(dtype=torch.float32, device=device).reshape(1)


def _weight_layout(layout, weight_fp8):
    """Per-layout weight unpack -> (G, K_contract, N_out, flat_byte_view).
    NT: W[G,N,K] flat [G*N,K]; NN/TN: W[G,K,N] flat [G*K,N]."""
    G = weight_fp8.shape[0]
    if layout == "nt":
        _, N, K = weight_fp8.shape
        flat = weight_fp8.reshape(G * N, K)
    elif layout in ("nn", "tn"):
        _, K, N = weight_fp8.shape
        flat = weight_fp8.reshape(G * K, N)
    else:
        raise NotImplementedError(f"layout {layout!r} not implemented yet")
    return G, K, N, _as_i8_flat(flat)


def _group_base(group_res, block_m, K, c_n):
    """Per-expert B slab scalar for ``emit(group_base=...)``: ``g * K * c_n`` (fp8
    elems), g = tile_to_group[block_m]. Same slab for NT / NN / TN (the per-layout B
    base / b_k_mult absorb the row-major-vs-K-strided difference)."""
    g_idx = buffer_load(group_res, block_m, vec_width=1, dtype=fx.T.i32())
    return g_idx * fx.Int32(K) * c_n


def _make_grouped_spec(layout, K, BLOCK_M, BLOCK_N, nt_vmcnt, *,
                       GROUP_M=8, num_xcd=8, out_fp16=False, act=None):
    """A plain ``DenseFp8TileSpec`` (so it KEEPS the dense XCD-swizzle scheduler) for
    the grouped GEMM-only launcher. The group lookup stays consistent because the
    launcher derives block_m from ``spec.scheduler_spec.map`` -- the SAME map ``emit``
    re-derives. The grouped-only grid is exact, so no no-sync linear map is needed; the
    XCD swizzle recovers the dense L2-reuse the linear map gave up."""
    vh = 3 if layout == "tn" else 2
    return DenseFp8TileSpec(
        layout=layout, K=K, block_tile=(BLOCK_M, BLOCK_N, BLOCK_K),
        warp_tile=(BLOCK_M // 4, BLOCK_N // 8, BLOCK_K),
        GROUP_M=GROUP_M, num_xcd=num_xcd, group_n=0, nt_vmcnt=nt_vmcnt, vmcnt_hint=vh,
        b_inline_asm_load=False, cbsz=0, blgp=0, out_fp16=out_fp16, act=act)


# ──────────────────────────────────────────────────────────────────────
# Fused dispatch + grouped GEMM, built directly on DenseFp8TileSpec.
# ──────────────────────────────────────────────────────────────────────
class DispatchGroupFP8TileSpec(DenseFp8TileSpec):
    """Fused cross-rank dispatch PUSH + grouped FP8 GEMM. Subclasses
    ``DenseFp8TileSpec`` directly: swaps in the LINEAR no-sync ``LinearNoSyncScheduler``
    and reuses every per-stage hook + ``spec.emit`` unchanged. The fused launcher
    (in ``_compile``) emits BOTH roles -- the GEMM tile (via ``spec.emit``) and the
    comm push (``dispatch_tile``). The per-expert B slab rides ``emit(group_base=...)``.

    Construction config: ``num_comm_blocks`` (front comm blocks the tile-id map
    skips), ``out_features`` (-> n_blocks / grid), ``pool_capacity`` (-> the no-sync
    over-launch bound + peer-pool record bytes), ``num_comm`` (the comm-task count)."""

    def __init__(self, *, num_comm_blocks, out_features, pool_capacity, num_comm, **kw):
        super().__init__(**kw)
        self.num_comm_blocks = num_comm_blocks
        self.out_features = out_features
        self.pool_capacity = pool_capacity
        self.num_comm = num_comm
        self.kernel_name = "dispatch_grouped_" + self.layout
        # swap the dense XCD scheduler for the fused LINEAR no-sync tile-id map
        self.scheduler_spec = LinearNoSyncScheduler(
            num_comm_blocks=num_comm_blocks)
        # rebuild cache_tag (folds the swapped scheduler key) + the dispatch config
        self.cache_tag = self._assemble_cache_tag(
        ) + (out_features, pool_capacity, num_comm)


@functools.lru_cache(maxsize=256)
def make_dispatch_tile_spec(*, layout, K, out_features, pool_capacity, num_comm,
                            BLOCK_M=256, BLOCK_N=256, nt_vmcnt=3, num_comm_blocks=0,
                            vmcnt_hint=2):
    """Cached ``DispatchGroupFP8TileSpec`` factory. TN uses a deeper tr8 drain hint."""
    vh = 3 if layout == "tn" else vmcnt_hint
    return DispatchGroupFP8TileSpec(
        out_features=out_features, pool_capacity=pool_capacity, num_comm=num_comm,
        num_comm_blocks=num_comm_blocks, layout=layout, K=K,
        block_tile=(BLOCK_M, BLOCK_N, BLOCK_K),
        warp_tile=(BLOCK_M // 4, BLOCK_N // 8, BLOCK_K),
        GROUP_M=1, num_xcd=1, group_n=0, nt_vmcnt=nt_vmcnt, vmcnt_hint=vh,
        b_inline_asm_load=False, cbsz=0, blgp=0, out_fp16=False)


def _make_dispatch_tile(*, thread_index, n_warps, hidden_i32, cols_per_warp_i32, vec_i32,
                        chunk_count, pool_record_bytes, input_resource, destination_resource,
                        comm_start_resource, comm_count_resource, comm_source_offset_resource,
                        source_tokens_resource, pool_address_resource,
                        signal=False, scoreboard_address_resource=None, block_m=0):
    """Unified comm push, shared by the fused kernel and dispatch_only. One block (all
    warps) pushes a token slice of one task to its peer pool; ``n_sub`` (compile-time,
    per call) splits a task across n_sub blocks (coarse-task BW path; n_sub=1 = whole
    task). Per-task metadata is read ONCE. ``signal=True`` adds the release fence +
    scoreboard signal of the pushed pool blocks (the fused handshake)."""
    warp_id = thread_index // fx.Int32(_WARP)
    lane_id = thread_index % fx.Int32(_WARP)

    def load_task(task_index):
        # read ALL per-task metadata ONCE (uniform scalars + dependent peer_pool)
        destination_rank = buffer_load(
            destination_resource, task_index, vec_width=1, dtype=fx.T.i32())
        dest_row_start = buffer_load(
            comm_start_resource, task_index, vec_width=1, dtype=fx.T.i32())
        source_offset = buffer_load(
            comm_source_offset_resource, task_index, vec_width=1, dtype=fx.T.i32())
        token_count = buffer_load(
            comm_count_resource, task_index, vec_width=1, dtype=fx.T.i32())
        pool_address = buffer_load(
            pool_address_resource, destination_rank, vec_width=1, dtype=fx.T.i64())
        peer_pool = create_buffer_resource_from_addr(
            pool_address, num_records_bytes=pool_record_bytes)
        return destination_rank, dest_row_start, source_offset, token_count, peer_pool

    def copy_slice(dest_row_start, source_offset, peer_pool, tok_lo, tok_hi):
        # warp-per-token copy of [tok_lo, tok_hi); only source_row is per-token
        local_count = (tok_hi - tok_lo - warp_id +
                       fx.Int32(n_warps - 1)) // fx.Int32(n_warps)

        def _row(i):
            row_index = tok_lo + warp_id + i * fx.Int32(n_warps)
            source_row = buffer_load(
                source_tokens_resource, source_offset + row_index, vec_width=1, dtype=fx.T.i32())
            dest_row = dest_row_start + row_index
            chunk_values = []
            for chunk_index in fx.range_constexpr(chunk_count):
                column = fx.Int32(
                    chunk_index * cols_per_warp_i32) + lane_id * fx.Int32(vec_i32)
                chunk_values.append(buffer_load(input_resource, source_row * fx.Int32(
                    hidden_i32) + column, vec_width=vec_i32, dtype=fx.T.i32()))
            for chunk_index in fx.range_constexpr(chunk_count):
                column = fx.Int32(
                    chunk_index * cols_per_warp_i32) + lane_id * fx.Int32(vec_i32)
                buffer_store(
                    chunk_values[chunk_index], peer_pool, dest_row * fx.Int32(hidden_i32) + column)

        _emit_for(local_count, _row)

    def dispatch_tile(task_index, sub, n_sub):
        destination_rank, dest_row_start, source_offset, token_count, peer_pool = load_task(task_index)
        if n_sub == 1:
            tok_lo = fx.Int32(0)
            tok_hi = token_count
        else:
            slice_tokens = (token_count + fx.Int32(n_sub - 1)) // fx.Int32(n_sub)
            tok_lo = sub * slice_tokens
            tok_hi = fx.arith.select(
                tok_lo + slice_tokens < token_count, tok_lo + slice_tokens, token_count)
        copy_slice(dest_row_start, source_offset, peer_pool, tok_lo, tok_hi)
        if signal:
            fx.rocdl.s_waitcnt(0)
            fx.gpu.barrier()

            def _signal():
                _fence_release()   # pushed rows visible cross-rank before the signal
                scoreboard_address = buffer_load(
                    scoreboard_address_resource, destination_rank, vec_width=1, dtype=fx.T.i64())
                first_block = (dest_row_start + tok_lo) // fx.Int32(block_m)
                last_block = (dest_row_start + tok_hi -
                              fx.Int32(1)) // fx.Int32(block_m)
                _emit_for(last_block - first_block + fx.Int32(1),
                          lambda bo: _atomic_add_addr(scoreboard_address, first_block + bo, fx.Int32(1)))

            _emit_if_then(thread_index == fx.Int32(0), _signal)

    return dispatch_tile


@functools.lru_cache(maxsize=256)
def _compile(layout, out_features, hidden_size, pool_capacity, BLOCK_M, BLOCK_N, comm_blocks,
             num_comm, nt_vmcnt=3, waves_per_eu=2, agpr_alloc=0):
    spec = make_dispatch_tile_spec(layout=layout, K=hidden_size, out_features=out_features,
                                   pool_capacity=pool_capacity, num_comm=num_comm,
                                   BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, nt_vmcnt=nt_vmcnt,
                                   num_comm_blocks=int(comm_blocks))
    layout = spec.layout
    hidden_size = spec.K
    BLOCK_M, BLOCK_N, _ = spec.block_tile
    out_features = spec.out_features
    pool_capacity = spec.pool_capacity
    num_comm = spec.num_comm
    num_comm_blocks = spec.num_comm_blocks
    if agpr_alloc is None:
        agpr_alloc = _LAYOUT_AGPR[layout]
    assert hidden_size % (
        _WARP * _VEC) == 0, "hidden must be a multiple of 1024 (warp push step)"
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert out_features % BLOCK_N == 0, "out_features must be a multiple of BLOCK_N"
    n_blocks = out_features // BLOCK_N
    # no-sync over-launch bound (LINEAR map)
    worst_case_tiles = pool_capacity // BLOCK_M

    # comm-role push geometry. fp8 token bytes are pushed as i32 words (v8i8 raw
    # buffer_load does not legalize; v4i32 = 16 bytes/lane = a legal dwordx4/b128 --
    # the widest transaction, needed to saturate XGMI).
    # Per-token WARP copy: each of the n_warps warps copies whole token rows (ALL 512
    # threads active, vs the old 128); its _WARP lanes cover _WARP*_VEC bytes per step.
    _VEC_I32 = _VEC // 4
    hidden_i32 = hidden_size // 4
    n_warps = _BLOCK_THREADS // _WARP
    cols_per_warp_i32 = _WARP * _VEC_I32             # i32 cols one warp copies per step
    chunk_count = hidden_i32 // cols_per_warp_i32    # warp-steps to copy one token row
    pool_record_bytes = pool_capacity * hidden_size  # fp8 = 1 byte

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def dispatch_grouped(INPUT_TOKENS: fx.Tensor, COMM_DESTINATION: fx.Tensor, COMM_START: fx.Tensor,
                         COMM_COUNT: fx.Tensor, COMM_SOURCE_OFFSET: fx.Tensor, SOURCE_TOKENS: fx.Tensor,
                         POOL_PTRS: fx.Tensor, SCOREBOARD_PTRS: fx.Tensor, POOL: fx.Tensor,
                         WEIGHTS: fx.Tensor, OUTPUT: fx.Tensor, A_SCALE: fx.Tensor, B_SCALE: fx.Tensor,
                         TILE_TO_GROUP: fx.Tensor, SCOREBOARD: fx.Tensor, EXPECTED: fx.Tensor,
                         NUM_TILE_BLOCKS: fx.Tensor, c_n: fx.Int32):
        _ = spec.cache_tag  # JIT cache-key discriminator; emits no IR
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        comm_block_count = fx.Int32(num_comm_blocks)
        # NT/TN: LDS at the top (unconditional); NN: lds=None -> emit allocs it
        # inside the guard. Ternary (not an if-statement). Per-layout codegen sensitivity.
        lds = fx.SharedAllocator().allocate(
            spec.shared_storage).peek() if layout != "nn" else None

        # ===== COMM role resources =====
        input_resource = create_buffer_resource(INPUT_TOKENS, max_size=True)
        destination_resource = create_buffer_resource(
            COMM_DESTINATION, max_size=True)
        comm_start_resource = create_buffer_resource(COMM_START, max_size=True)
        comm_count_resource = create_buffer_resource(COMM_COUNT, max_size=True)
        comm_source_offset_resource = create_buffer_resource(
            COMM_SOURCE_OFFSET, max_size=True)
        source_tokens_resource = create_buffer_resource(
            SOURCE_TOKENS, max_size=True)
        pool_address_resource = create_buffer_resource(
            POOL_PTRS, max_size=True)
        scoreboard_address_resource = create_buffer_resource(
            SCOREBOARD_PTRS, max_size=True)
        # ===== GEMM role resources =====
        group_resource = create_buffer_resource(TILE_TO_GROUP, max_size=True)
        expected_resource = create_buffer_resource(EXPECTED, max_size=True)
        num_tile_blocks_resource = create_buffer_resource(
            NUM_TILE_BLOCKS, max_size=True)

        # the shared comm push (load-once metadata + all-warp copy + fence/signal)
        dispatch_tile = _make_dispatch_tile(
            thread_index=thread_index, n_warps=n_warps, hidden_i32=hidden_i32,
            cols_per_warp_i32=cols_per_warp_i32, vec_i32=_VEC_I32, chunk_count=chunk_count,
            pool_record_bytes=pool_record_bytes, input_resource=input_resource,
            destination_resource=destination_resource, comm_start_resource=comm_start_resource,
            comm_count_resource=comm_count_resource,
            comm_source_offset_resource=comm_source_offset_resource,
            source_tokens_resource=source_tokens_resource,
            pool_address_resource=pool_address_resource, signal=True,
            scoreboard_address_resource=scoreboard_address_resource, block_m=BLOCK_M)

        if block_index < comm_block_count:
            # COMM role: round-robin share of the comm tasks (whole task per block)
            local_task_count = (fx.Int32(
                num_comm) - block_index + comm_block_count - fx.Int32(1)) // comm_block_count
            for task_iteration in range(local_task_count):
                dispatch_tile(block_index + task_iteration * comm_block_count, fx.Int32(0), 1)
        else:
            # GEMM role: LINEAR tile-id map; no-sync over-launch self-bound by
            # num_tile_blocks (padding tiles early-exit). block_m only -> the
            # scoreboard spin; the stock spec.emit re-derives (block_m, block_n).
            tile_index = block_index - comm_block_count
            block_m = tile_index // fx.Int32(n_blocks)
            real_tiles = buffer_load(num_tile_blocks_resource, fx.Int32(
                0), vec_width=1, dtype=fx.T.i32())
            if block_m < real_tiles:
                # c_m = host-known pool extent (compile-time scalar), NOT real_tiles*BM
                # from the HBM load: a runtime c_m pessimizes emit's epilogue (~8%). The
                # over-launch self-bound stays -- real_tiles still gates the tiles above.
                c_m_real = fx.Int32(pool_capacity)
                # spin until every comm task on this pool block has signalled
                expected_count = buffer_load(
                    expected_resource, block_m, vec_width=1, dtype=fx.T.i32())
                if thread_index == fx.Int32(0):
                    signal = _ld_relaxed(SCOREBOARD, block_m)
                    while signal < expected_count:
                        fx.rocdl.s_sleep(fx.Int32(2))
                        signal = _ld_relaxed(SCOREBOARD, block_m)
                fx.gpu.barrier()
                _fence_acquire()   # read fresh peer-pushed pool before the GEMM
                # standard immutable emit; per-expert B slab via the scalar seam
                gbase = _group_base(group_resource, block_m, hidden_size, c_n)
                spec.emit(A=POOL, B=WEIGHTS, C=OUTPUT, A_scale=A_SCALE, B_scale=B_SCALE,
                          c_m=c_m_real, c_n=c_n, lds=lds, group_base=gbase)

    @flyc.jit
    def launch(INPUT_TOKENS, COMM_DESTINATION, COMM_START, COMM_COUNT, COMM_SOURCE_OFFSET,
               SOURCE_TOKENS, POOL_PTRS, SCOREBOARD_PTRS, POOL, WEIGHTS, OUTPUT, A_SCALE, B_SCALE,
               TILE_TO_GROUP, SCOREBOARD, EXPECTED, NUM_TILE_BLOCKS, c_n: int,
               stream: fx.Stream = fx.Stream(None)):
        grid_size = num_comm_blocks + worst_case_tiles * n_blocks
        dispatch_grouped(INPUT_TOKENS, COMM_DESTINATION, COMM_START, COMM_COUNT, COMM_SOURCE_OFFSET,
                         SOURCE_TOKENS, POOL_PTRS, SCOREBOARD_PTRS, POOL, WEIGHTS, OUTPUT, A_SCALE, B_SCALE,
                         TILE_TO_GROUP, SCOREBOARD, EXPECTED, NUM_TILE_BLOCKS, c_n,
                         value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512")).launch(
            grid=(grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


@functools.lru_cache(maxsize=256)
def _compile_dispatch_only(hidden_size, pool_capacity, comm_blocks, num_comm, waves_per_eu=2):
    """Build & cache a comm-PUSH-only launcher (no GEMM, no scoreboard/fence) for
    measuring raw dispatch bandwidth. Every block runs the same warp-per-token push
    as the fused comm role (round-robin over the comm tasks); grid = comm_blocks."""
    assert hidden_size % (
        _WARP * _VEC) == 0, "hidden must be a multiple of 1024 (warp push step)"
    _VEC_I32 = _VEC // 4
    hidden_i32 = hidden_size // 4
    n_warps = _BLOCK_THREADS // _WARP
    cols_per_warp_i32 = _WARP * _VEC_I32
    chunk_count = hidden_i32 // cols_per_warp_i32
    pool_record_bytes = pool_capacity * hidden_size
    # When there are FEWER tasks than comm blocks (coarse all-to-all), one block per
    # task = one CU per XGMI link, which can't saturate it. Split each task's tokens
    # into contiguous slices across blocks_per_task blocks (several CUs/link).
    blocks_per_task = max(1, comm_blocks // num_comm)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def dispatch_only_k(INPUT_TOKENS: fx.Tensor, COMM_DESTINATION: fx.Tensor, COMM_START: fx.Tensor,
                        COMM_COUNT: fx.Tensor, COMM_SOURCE_OFFSET: fx.Tensor, SOURCE_TOKENS: fx.Tensor,
                        POOL_PTRS: fx.Tensor):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        comm_block_count = fx.Int32(comm_blocks)
        input_resource = create_buffer_resource(INPUT_TOKENS, max_size=True)
        destination_resource = create_buffer_resource(
            COMM_DESTINATION, max_size=True)
        comm_start_resource = create_buffer_resource(COMM_START, max_size=True)
        comm_count_resource = create_buffer_resource(COMM_COUNT, max_size=True)
        comm_source_offset_resource = create_buffer_resource(
            COMM_SOURCE_OFFSET, max_size=True)
        source_tokens_resource = create_buffer_resource(
            SOURCE_TOKENS, max_size=True)
        pool_address_resource = create_buffer_resource(POOL_PTRS, max_size=True)

        # same shared comm push, push-only (no fence/signal)
        dispatch_tile = _make_dispatch_tile(
            thread_index=thread_index, n_warps=n_warps, hidden_i32=hidden_i32,
            cols_per_warp_i32=cols_per_warp_i32, vec_i32=_VEC_I32, chunk_count=chunk_count,
            pool_record_bytes=pool_record_bytes, input_resource=input_resource,
            destination_resource=destination_resource, comm_start_resource=comm_start_resource,
            comm_count_resource=comm_count_resource,
            comm_source_offset_resource=comm_source_offset_resource,
            source_tokens_resource=source_tokens_resource,
            pool_address_resource=pool_address_resource, signal=False)

        if blocks_per_task > 1:
            # contiguous multi-block-per-task: blocks_per_task CUs per peer/XGMI link
            task_index = block_index // fx.Int32(blocks_per_task)
            sub = block_index % fx.Int32(blocks_per_task)
            if task_index < fx.Int32(num_comm):
                dispatch_tile(task_index, sub, blocks_per_task)
        else:
            # more tasks than blocks: round-robin whole tasks (each push is contiguous)
            if block_index < comm_block_count:
                local_task_count = (fx.Int32(num_comm) - block_index +
                                    comm_block_count - fx.Int32(1)) // comm_block_count
                for it in range(local_task_count):
                    dispatch_tile(block_index + it * comm_block_count, fx.Int32(0), 1)

    @flyc.jit
    def launch(INPUT_TOKENS, COMM_DESTINATION, COMM_START, COMM_COUNT, COMM_SOURCE_OFFSET,
               SOURCE_TOKENS, POOL_PTRS, stream: fx.Stream = fx.Stream(None)):
        dispatch_only_k(INPUT_TOKENS, COMM_DESTINATION, COMM_START, COMM_COUNT,
                        COMM_SOURCE_OFFSET, SOURCE_TOKENS, POOL_PTRS,
                        value_attrs=make_value_attrs(waves_per_eu, 0, "512,512")).launch(
            grid=(comm_blocks, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def dispatch_only(x_fp8, comm, pool_ptrs, pool_fp8, *, comm_blocks=32):
    """Comm-PUSH only (no GEMM) — pushes ``x_fp8`` token rows to peer pools over XGMI.
    For measuring dispatch bandwidth: bytes pushed per rank = num_src_tokens * hidden."""
    hidden_size = x_fp8.size(1)
    pool_capacity = pool_fp8.size(0)
    x_i32 = x_fp8.contiguous().view(torch.int32).view(-1)
    launch = _compile_dispatch_only(hidden_size, pool_capacity,
                                    int(comm_blocks), int(comm.num_comm))
    launch(x_i32, comm.dest, comm.start, comm.cnt, comm.srcoff, comm.src_tokens,
           pool_ptrs, stream=torch.cuda.current_stream())


@functools.lru_cache(maxsize=256)
def compile_grouped_gemm(layout, K, BLOCK_M=256, BLOCK_N=256, nt_vmcnt=3,
                         waves_per_eu=2, agpr_alloc=None, act=None):
    """Build & cache the standalone grouped GEMM-only launcher (the compute-peak
    baseline; the mega analog of ``compile_dense_nt_tiled``). Keeps the dense
    XCD-swizzle scheduler (L2 reuse) and passes ``c_m`` as a host SCALAR (not the
    runtime ``NUM_TILE_BLOCKS`` load -> ~8% faster epilogue). ``NUM_TILE_BLOCKS`` is
    still read, but only for the early-exit guard (the over-launch self-bound). A is
    pool[M,K] (NT/NN) or [K,M] (TN); B the flat per-expert weight; C out[M,N] bf16."""
    spec = _make_grouped_spec(layout, K, BLOCK_M, BLOCK_N, nt_vmcnt, act=act)
    spec.kernel_name = "grouped_" + layout
    tag = spec.cache_tag
    # NT/TN: top alloc; NN: alloc inside emit (under the guard)
    lds_top = layout != "nn"
    if agpr_alloc is None:
        agpr_alloc = _LAYOUT_AGPR[layout]

    def kernel(A, B, C, A_scale, B_scale, TILE_TO_GROUP: fx.Tensor,
               NUM_TILE_BLOCKS: fx.Tensor, c_m: fx.Int32, c_n: fx.Int32):
        _ = tag  # JIT cache-key discriminator; emits no IR
        n_blocks = ceildiv(c_n, BLOCK_N)
        group_res = create_buffer_resource(TILE_TO_GROUP, max_size=True)
        ntb = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
        real_tiles = buffer_load(ntb, fx.Int32(
            0), vec_width=1, dtype=fx.T.i32())
        lds = fx.SharedAllocator().allocate(
            spec.shared_storage).peek() if lds_top else None
        # XCD-swizzled (block_m, block_n); emit re-derives the SAME map, so the
        # group_base lookup stays consistent with the tile emit computes.
        block_m, _bn = spec.scheduler_spec.map(
            spec.geom, c_m=c_m, c_n=c_n, n_blocks=n_blocks)

        def _emit_one():
            gbase = _group_base(group_res, block_m, spec.K, c_n)
            spec.emit(A=A, B=B, C=C, A_scale=A_scale, B_scale=B_scale,
                      c_m=c_m, c_n=c_n, lds=lds, group_base=gbase)

        _emit_if_then(block_m < real_tiles, _emit_one)

    kernel.__name__ = spec.kernel_name
    kernel.__qualname__ = kernel.__name__
    kernel = flyc.kernel(kernel, known_block_size=[512, 1, 1])

    @flyc.jit
    def launch(A, B, C, A_scale, B_scale, TILE_TO_GROUP, NUM_TILE_BLOCKS,
               c_m: int, c_n: int, stream: fx.Stream = fx.Stream(None)):
        grid_x = spec.scheduler_spec.grid(spec.geom, c_m, c_n)
        kernel(A, B, C, A_scale, B_scale, TILE_TO_GROUP, NUM_TILE_BLOCKS, c_m, c_n,
               value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512")).launch(
            grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch


# Per-shape autotune candidates (comm_blocks, nt_vmcnt, agpr_alloc, waves_per_eu).
# The comm/GEMM CU split dominates (each comm CU pushes at HBM/XGMI, each GEMM CU
# does MFMA); sweep it widely. nt_vmcnt (G2S drain) / waves are secondary.
_DISPATCH_CANDIDATES = [
    (16, 3, 0, 2), (24, 3, 0, 2), (32, 3, 0, 2), (40, 3, 0, 2),
    (48, 3, 0, 2), (56, 3, 0, 2), (64, 3, 0, 2), (80,
                                                  # CU-split sweep
                                                  3, 0, 2), (96, 3, 0, 2),
    # G2S drain depth
    (48, 4, 0, 2), (48, 2, 0, 2), (64, 4, 0, 2),
    # occupancy probes
    (48, 3, 0, 1), (48, 3, -256, 2),
]
_DISPATCH_AUTOTUNE_CACHE: dict = {}


def dispatch_grouped_gemm_fp8(x_fp8, comm, pool_ptrs, scoreboard_ptrs, pool_fp8, weight_fp8,
                              output, tile_to_group, scoreboard, expected, mblk_dev, *,
                              a_scale, b_scale, layout="nt", BM=256, BN=256, comm_blocks=32,
                              nt_vmcnt=3, autotune=False, autotune_reset=None):
    """Fused cross-rank dispatch PUSH + grouped FP8 GEMM.

    ``layout`` selects the GEMM: ``nt`` (fwd L1, weight [G,N,K]) or ``nn`` (bwd
    dgrad, weight [G,K,N]). ``x_fp8`` [num_src_tokens, K] fp8 source tokens;
    ``pool_fp8`` [pool_cap, K] fp8 landing pool; ``output`` [pool_cap, N] bf16;
    per-tensor ``a_scale`` / ``b_scale``. ``comm`` carries
    dest/start/cnt/srcoff/src_tokens/num_comm. Scoreboard must be zeroed first."""
    G, K_contract, out_features, weight_bytes = _weight_layout(
        layout, weight_fp8)
    hidden_size = x_fp8.size(1)
    assert K_contract == hidden_size, f"weight K={K_contract} != activation K={hidden_size}"
    pool_capacity = pool_fp8.size(0)
    c_n = out_features
    device = x_fp8.device

    sa = _scalar(a_scale, device)
    sb = _scalar(b_scale, device)
    # source tokens pushed as i32 words (legal dwordx2 load); pool/weight read as fp8 bytes
    x_i32 = x_fp8.contiguous().view(torch.int32).view(-1)
    pool_bytes = _as_i8_flat(pool_fp8)
    output_flat = output.contiguous().view(-1)

    pos_args = (x_i32, comm.dest, comm.start, comm.cnt, comm.srcoff, comm.src_tokens,
                pool_ptrs, scoreboard_ptrs, pool_bytes, weight_bytes, output_flat, sa, sb,
                tile_to_group, scoreboard, expected, mblk_dev, c_n)

    if autotune:
        key = (layout, out_features, hidden_size,
               pool_capacity, BM, BN, int(comm.num_comm))
        cached = _DISPATCH_AUTOTUNE_CACHE.get(key)
        if cached is None:
            cached = _autotune(layout, pos_args, output_flat, out_features, hidden_size,
                               pool_capacity, BM, BN, int(comm.num_comm), scoreboard, autotune_reset)
            _DISPATCH_AUTOTUNE_CACHE[key] = cached
        launch, _cfg = cached
    else:
        launch = _compile(layout, out_features, hidden_size, pool_capacity, BM, BN,
                          int(comm_blocks), int(comm.num_comm), int(nt_vmcnt))
    launch(*pos_args, stream=torch.cuda.current_stream())
    return output


def _autotune(layout, pos_args, finite_view, out_features, hidden_size, pool_capacity, BM, BN,
              num_comm, scoreboard, reset):
    """Bench the candidates with a per-iter scoreboard reset; return (launch, cfg)."""
    if reset is None:
        reset = scoreboard.zero_
    stream = torch.cuda.current_stream()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    best_us, best = float("inf"), None
    for comm_blocks, nt_vmcnt, agpr, waves in _DISPATCH_CANDIDATES:
        try:
            launch = _compile(layout, out_features, hidden_size, pool_capacity, BM, BN,
                              int(comm_blocks), num_comm, int(nt_vmcnt), int(waves), int(agpr))
            reset()
            launch(*pos_args, stream=stream)
            torch.cuda.synchronize()
            if not torch.isfinite(finite_view.view(-1)[:1024].float()).all().item():
                continue
            for _ in range(2):
                reset()
                launch(*pos_args, stream=stream)
            torch.cuda.synchronize()
            us_total = 0.0
            for _ in range(20):
                reset()
                e0.record()
                launch(*pos_args, stream=stream)
                e1.record()
                torch.cuda.synchronize()
                us_total += e0.elapsed_time(e1) * 1000.0
            us = us_total / 20
            if us < best_us:
                best_us, best = us, (launch, (comm_blocks,
                                     nt_vmcnt, agpr, waves))
        except Exception:
            continue
    if best is None:
        raise RuntimeError(
            "dispatch_grouped_gemm_fp8 autotune found no working cfg")
    return best


def grouped_gemm_fp8_only(pool_fp8, weight_fp8, output, tile_to_group, mblk_dev, *,
                          a_scale, b_scale, layout="nt", BM=256, BN=256, nt_vmcnt=3,
                          waves_per_eu=2, agpr_alloc=None, act=None):
    """Pure grouped FP8 GEMM (no dispatch) — the compute-peak baseline.

    NT/NN: ``pool_fp8`` is A=[M,K] (output rows = M = pool rows). TN: ``pool_fp8``
    is A=[K,M] (output rows = M = A's columns); ``output`` is [M,N]."""
    G, K_contract, out_features, weight_bytes = _weight_layout(
        layout, weight_fp8)
    if layout == "tn":
        hidden_size = pool_fp8.size(0)        # A=[K,M]
        pool_capacity = pool_fp8.size(1)      # output rows = M
    else:
        hidden_size = pool_fp8.size(1)        # A=[M,K]
        pool_capacity = pool_fp8.size(0)      # output rows = M
    assert K_contract == hidden_size, f"weight K={K_contract} != A K={hidden_size}"
    if agpr_alloc is None:
        agpr_alloc = _LAYOUT_AGPR[layout]     # TN inplace MFMA needs 128 AGPRs
    device = pool_fp8.device
    sa = _scalar(a_scale, device)
    sb = _scalar(b_scale, device)
    pool_bytes = _as_i8_flat(pool_fp8)
    output_flat = output.contiguous().view(-1)
    # grouped GEMM-only launcher; c_m = pool rows (grid bound), mblk_dev = real
    # tile-blocks (runtime self-bound), c_n = out_features.
    launch = compile_grouped_gemm(layout, hidden_size, BM, BN, int(nt_vmcnt),
                                  int(waves_per_eu), int(agpr_alloc), act=act)
    launch(pool_bytes, weight_bytes, output_flat, sa, sb, tile_to_group, mblk_dev,
           pool_capacity, out_features, stream=torch.cuda.current_stream())
    return output
