###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused cross-rank dispatch PUSH + grouped BF16 GEMM (NT), FlyDSL.

Mega analog of ``dispatch_grouped_gemm_fp8`` for bf16. The GEMM is NOT built on
the fp8 tile-spec; instead it reuses the shared ``gemm_bf16_nt_tile`` (the dense
bf16 software pipeline, per-tile compute closure) from ``gemm_bf16_kernel.py``,
and the cross-rank comm PUSH from the fp8 path is encapsulated as
``dispatch_bf16_tile`` (byte-agnostic, reused for bf16 by treating each token row
as i32 words).

Role-specialized single kernel (mirrors the fp8 fused kernel):
  * ``block_index < comm_blocks`` blocks push token rows to peer pools over XGMI
    and signal a per-pool-block scoreboard (``dispatch_bf16_tile``).
  * the remaining blocks each compute ONE NT output tile of the grouped GEMM
    (``A=pool[M,K]`` bf16, per-expert ``B=weight[G,N,K]`` bf16 -> ``C=out[M,N]``
    bf16) via ``gemm_bf16_nt_tile``, spinning on the scoreboard until their pool
    block is filled. The comm latency hides under the MFMA-bound GEMM.

NT only (the bf16 GEMM kernel implements NT only). K must be a multiple of
BLOCK_K (DSv3 shapes qualify) and hidden a multiple of 512 (warp push step).
"""

import functools
import os as _os

import torch

# [DIAG] skip the role3 sb_copy gate (measures the scatter-dependency cost; breaks correctness)
_DIAG_SKIP_SBCOPY = int(_os.environ.get("MEGA_SKIP_SBCOPY", "0"))

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr.buffer_ops import buffer_load, create_buffer_resource

# shared bf16 GEMM tile + geometry/LDS helpers from the dense bf16 kernel
from primus_turbo.flydsl.mega.gemm_bf16_kernel import (
    _make_shared_storage,
    gemm_bf16_nn_tile,
    gemm_bf16_nt_tile,
    gemm_bf16_tn_tile,
)

# per-tile GEMM closure by layout (NT forward, NN dgrad, TN wgrad); all grouped via b_group_base
_GEMM_TILE = {"nt": gemm_bf16_nt_tile, "nn": gemm_bf16_nn_tile, "tn": gemm_bf16_tn_tile}
from primus_turbo.flydsl.common.tile_spec import _emit_if_then

# comm-PUSH + scoreboard prims (byte-agnostic; shared intra-node EP layer)
from primus_turbo.flydsl.mega.ep_intranode import (
    _BLOCK_THREADS,
    _bf16_push_geom,
    _fence_acquire,
    _ld_relaxed,
    dispatch_bf16_tile,
    permute_token_tile,
)
from primus_turbo.flydsl.utils.gemm_helper import (
    ceildiv,
    make_value_attrs,
    xcd_remap_pid,
)


# ───────────────────────────────────────────────────────────────────────
# Grouped GEMM-only launcher (the compute-peak baseline). Dense XCD-swizzle
# scheduler + GROUP_M, per-expert B slab via tile_to_group.
# ───────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=256)
def compile_grouped_gemm_bf16(
    K,
    BLOCK_M=256,
    BLOCK_N=256,
    GROUP_M=1,
    num_xcd=8,
    nt_vmcnt=3,
    waves_per_eu=2,
    agpr_alloc=0,
    out_fp16=False,
    layout="nt",
):
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)
    gemm_tile = _GEMM_TILE[layout]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_grouped(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        TILE_TO_GROUP: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        n_blocks = ceildiv(c_n, BLOCK_N)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        group_res = create_buffer_resource(TILE_TO_GROUP, max_size=True)
        ntb = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
        real_tiles = buffer_load(ntb, fx.Int32(0), vec_width=1, dtype=fx.T.i32())
        # The pool is over-allocated (capacity >> real tiles), so the launch grid is
        # mostly padding. Map/XCD-swizzle over the REAL tile range only (front-loaded),
        # so real work stays DENSE across CUs with L2 reuse and padding tiles early-exit
        # at the tail. (Swizzling over the full pool grid scatters real tiles into the
        # padding -> ~half-idle waves -> ~2x slower; that was the gemm_only perf bug.)
        real_grid = real_tiles * n_blocks

        def _emit():
            pid = xcd_remap_pid(fx.block_idx.x, real_grid, num_xcd)
            num_pid_m = real_tiles
            num_pid_in_group = GROUP_M * n_blocks
            group_id = pid // num_pid_in_group
            pid_in_group = pid % num_pid_in_group
            first_pid_m = group_id * GROUP_M
            remaining_m = num_pid_m - first_pid_m
            group_size_m = arith.select(remaining_m < GROUP_M, remaining_m, fx.Int32(GROUP_M))
            block_m = first_pid_m + (pid_in_group % group_size_m)
            block_n = pid_in_group // group_size_m
            g_idx = buffer_load(group_res, block_m, vec_width=1, dtype=fx.T.i32())
            gbase = g_idx * fx.Int32(K) * c_n
            gemm_tile(
                A,
                B,
                C,
                c_m,
                c_n,
                lds,
                block_m,
                block_n,
                K=K,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                out_fp16=out_fp16,
                nt_vmcnt=nt_vmcnt,
                b_group_base=gbase,
            )

        _emit_if_then(fx.block_idx.x < real_grid, _emit)

    @flyc.jit
    def launch(
        A, B, C, TILE_TO_GROUP, NUM_TILE_BLOCKS, c_m: int, c_n: int, stream: fx.Stream = fx.Stream(None)
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_grouped(
            A,
            B,
            C,
            TILE_TO_GROUP,
            NUM_TILE_BLOCKS,
            c_m,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch


# ───────────────────────────────────────────────────────────────────────
# Fused dispatch PUSH + grouped GEMM. LINEAR no-sync tile-id map offset past the
# comm blocks; GEMM blocks spin on the scoreboard before computing.
# ───────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=256)
def _compile(
    out_features,
    hidden_size,
    pool_capacity,
    BLOCK_M,
    BLOCK_N,
    comm_blocks,
    num_comm,
    nt_vmcnt=3,
    waves_per_eu=2,
    agpr_alloc=0,
    out_fp16=False,
    GROUP_M=1,
    dedup=False,
    permute_blocks=32,
    layout="nt",
):
    K = hidden_size
    gemm_tile = _GEMM_TILE[layout]
    assert out_features % BLOCK_N == 0, "out_features must be a multiple of BLOCK_N"
    assert pool_capacity % BLOCK_M == 0, "pool_capacity must be a multiple of BLOCK_M"
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)
    n_blocks = out_features // BLOCK_N
    worst_case_tiles = pool_capacity // BLOCK_M
    # dest-local copy geometry (bf16 token row pushed/copied as i32 words)
    vec_i32, hidden_i32, n_warps, cols_per_warp_i32, chunk_count = _bf16_push_geom(hidden_size)
    assert BLOCK_M % n_warps == 0, f"BLOCK_M={BLOCK_M} must be a multiple of n_warps={n_warps}"
    rows_per_warp = BLOCK_M // n_warps

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def dispatch_grouped(
        INPUT_TOKENS: fx.Tensor,
        COMM_DESTINATION: fx.Tensor,
        COMM_START: fx.Tensor,
        COMM_COUNT: fx.Tensor,
        COMM_SOURCE_OFFSET: fx.Tensor,
        SOURCE_TOKENS: fx.Tensor,
        POOL_PTRS: fx.Tensor,
        SCOREBOARD_PTRS: fx.Tensor,
        POOL: fx.Tensor,
        WEIGHTS: fx.Tensor,
        OUTPUT: fx.Tensor,
        TILE_TO_GROUP: fx.Tensor,
        SCOREBOARD: fx.Tensor,
        EXPECTED: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        SOURCE_DEDUP: fx.Tensor,
        DEDUP_SRC_ROW: fx.Tensor,
        SB_COPY: fx.Tensor,
        POOL_I32: fx.Tensor,
        c_n: fx.Int32,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        comm_block_count = fx.Int32(comm_blocks)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        # ===== COMM role resources =====
        input_resource = create_buffer_resource(INPUT_TOKENS, max_size=True)
        destination_resource = create_buffer_resource(COMM_DESTINATION, max_size=True)
        comm_start_resource = create_buffer_resource(COMM_START, max_size=True)
        comm_count_resource = create_buffer_resource(COMM_COUNT, max_size=True)
        comm_source_offset_resource = create_buffer_resource(COMM_SOURCE_OFFSET, max_size=True)
        source_tokens_resource = create_buffer_resource(SOURCE_TOKENS, max_size=True)
        pool_address_resource = create_buffer_resource(POOL_PTRS, max_size=True)
        scoreboard_address_resource = create_buffer_resource(SCOREBOARD_PTRS, max_size=True)
        # ===== GEMM role resources =====
        group_resource = create_buffer_resource(TILE_TO_GROUP, max_size=True)
        expected_resource = create_buffer_resource(EXPECTED, max_size=True)
        num_tile_blocks_resource = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
        # ===== dedup resources (token-dedup XGMI saving) =====
        if fx.const_expr(dedup):
            source_dedup_resource = create_buffer_resource(SOURCE_DEDUP, max_size=True)
            dedup_src_row_resource = create_buffer_resource(DEDUP_SRC_ROW, max_size=True)
            sb_copy_resource = create_buffer_resource(SB_COPY, max_size=True)
            pool_i32_resource = create_buffer_resource(POOL_I32, max_size=True)
        else:
            source_dedup_resource = None

        dispatch_tile = dispatch_bf16_tile(
            thread_index=thread_index,
            hidden_size=hidden_size,
            pool_capacity=pool_capacity,
            input_resource=input_resource,
            destination_resource=destination_resource,
            comm_start_resource=comm_start_resource,
            comm_count_resource=comm_count_resource,
            comm_source_offset_resource=comm_source_offset_resource,
            source_tokens_resource=source_tokens_resource,
            pool_address_resource=pool_address_resource,
            signal=True,
            scoreboard_address_resource=scoreboard_address_resource,
            block_m=BLOCK_M,
            source_dedup_resource=source_dedup_resource,
        )

        # ── 3-role (dedup): role1 dispatch [0,D) ‖ role2 permute [D,D+P) ‖ role3 gemm.
        #    2-role (no dedup): role1 dispatch [0,D) ‖ role2 gemm. ───────────────────
        gemm_offset = fx.Int32(comm_blocks + (permute_blocks if dedup else 0))
        if block_index < comm_block_count:
            # role1: dispatch PUSH (primaries over XGMI; secondaries skipped) + signal.
            local_task_count = (
                fx.Int32(num_comm) - block_index + comm_block_count - fx.Int32(1)
            ) // comm_block_count
            for task_iteration in range(local_task_count):
                dispatch_tile(block_index + task_iteration * comm_block_count, fx.Int32(0), 1)
        else:
            run_gemm = True
            tile_index = block_index - gemm_offset
            if fx.const_expr(dedup):
                # role2: dest-local permute copy (concurrent with role1 + role3)
                if block_index < gemm_offset:
                    permute_token_tile(
                        thread_index,
                        block_index - comm_block_count,
                        fx.Int32(permute_blocks),
                        num_tile_blocks_resource,
                        dedup_src_row_resource,
                        expected_resource,
                        SCOREBOARD,
                        pool_i32_resource,
                        sb_copy_resource,
                        BLOCK_M,
                        n_warps,
                        rows_per_warp,
                        chunk_count,
                        cols_per_warp_i32,
                        vec_i32,
                        hidden_i32,
                    )
                    run_gemm = False
            if run_gemm:
                # role3 (dedup) / role2 (no dedup): GROUP_M tile-id map over the REAL grid.
                real_tiles = buffer_load(num_tile_blocks_resource, fx.Int32(0), vec_width=1, dtype=fx.T.i32())
                real_grid = real_tiles * fx.Int32(n_blocks)
                if tile_index < real_grid:
                    num_pid_in_group = fx.Int32(GROUP_M * n_blocks)
                    group_id = tile_index // num_pid_in_group
                    pid_in_group = tile_index % num_pid_in_group
                    first_pid_m = group_id * fx.Int32(GROUP_M)
                    remaining_m = real_tiles - first_pid_m
                    group_size_m = arith.select(
                        remaining_m < fx.Int32(GROUP_M), remaining_m, fx.Int32(GROUP_M)
                    )
                    block_m = first_pid_m + (pid_in_group % group_size_m)
                    block_n = pid_in_group // group_size_m
                    c_m_real = fx.Int32(pool_capacity)
                    # GEMM gate: wait this block's primaries (scoreboard) and, under dedup,
                    # the role2 secondary copies (sb_copy). No copy work here.
                    expected_count = buffer_load(expected_resource, block_m, vec_width=1, dtype=fx.T.i32())
                    if thread_index == fx.Int32(0):
                        signal = _ld_relaxed(SCOREBOARD, block_m)
                        while signal < expected_count:
                            fx.rocdl.s_sleep(fx.Int32(2))
                            signal = _ld_relaxed(SCOREBOARD, block_m)
                        if fx.const_expr(dedup and not _DIAG_SKIP_SBCOPY):
                            copy_done = _ld_relaxed(SB_COPY, block_m)
                            while copy_done < fx.Int32(1):
                                fx.rocdl.s_sleep(fx.Int32(2))
                                copy_done = _ld_relaxed(SB_COPY, block_m)
                    fx.gpu.barrier()
                    _fence_acquire()  # read fresh peer-pushed + locally-copied pool rows

                    g_idx = buffer_load(group_resource, block_m, vec_width=1, dtype=fx.T.i32())
                    gbase = g_idx * fx.Int32(K) * c_n
                    gemm_tile(
                        POOL,
                        WEIGHTS,
                        OUTPUT,
                        c_m_real,
                        c_n,
                        lds,
                        block_m,
                        block_n,
                        K=K,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        out_fp16=out_fp16,
                        nt_vmcnt=nt_vmcnt,
                        b_group_base=gbase,
                    )

    @flyc.jit
    def launch(
        INPUT_TOKENS,
        COMM_DESTINATION,
        COMM_START,
        COMM_COUNT,
        COMM_SOURCE_OFFSET,
        SOURCE_TOKENS,
        POOL_PTRS,
        SCOREBOARD_PTRS,
        POOL,
        WEIGHTS,
        OUTPUT,
        TILE_TO_GROUP,
        SCOREBOARD,
        EXPECTED,
        NUM_TILE_BLOCKS,
        SOURCE_DEDUP,
        DEDUP_SRC_ROW,
        SB_COPY,
        POOL_I32,
        c_n: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_size = comm_blocks + (permute_blocks if dedup else 0) + worst_case_tiles * n_blocks
        dispatch_grouped(
            INPUT_TOKENS,
            COMM_DESTINATION,
            COMM_START,
            COMM_COUNT,
            COMM_SOURCE_OFFSET,
            SOURCE_TOKENS,
            POOL_PTRS,
            SCOREBOARD_PTRS,
            POOL,
            WEIGHTS,
            OUTPUT,
            TILE_TO_GROUP,
            SCOREBOARD,
            EXPECTED,
            NUM_TILE_BLOCKS,
            SOURCE_DEDUP,
            DEDUP_SRC_ROW,
            SB_COPY,
            POOL_I32,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


# ───────────────────────────────────────────────────────────────────────
# Host-side helpers.
# ───────────────────────────────────────────────────────────────────────
def _bf16_flat(t):
    return t.contiguous().view(-1)


def grouped_gemm_bf16_only(
    # ── operands (A=pool, B=weight, C=output) ─────────────────────────
    pool,  # [M, K] bf16   A operand (pre-filled activation)
    weight,  # [G, N, K] bf16   per-expert B
    output,  # [M, N] bf16   C
    tile_to_group,  # [n_mblk] i32   expert id per BM pool block
    num_tile_blocks,  # [1] i32        real tile-block count (device)
    # ── tile / schedule config (compile-time scalars) ─────────────────
    *,
    layout="nt",
    BM=256,
    BN=256,
    GROUP_M=4,
    num_xcd=8,
    nt_vmcnt=3,
    waves_per_eu=2,
    agpr_alloc=0,
):
    """Pure grouped BF16 GEMM (no dispatch) — the compute-peak baseline.

    ``pool`` is A=[M,K] bf16, ``output`` is [M,N] bf16, ``tile_to_group`` maps each
    BM pool block -> expert. Weight layout: NT (forward) ``weight`` [G,N,K];
    NN (dgrad) / TN (wgrad) ``weight`` [G,K,N]. ``num_tile_blocks`` is the real
    tile-block count (runtime over-launch self-bound)."""
    assert layout in ("nt", "nn", "tn"), f"unknown layout {layout}"
    assert pool.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    if layout == "tn":  # A is [K, M] (K-major)
        hidden_size, c_m = pool.shape
    else:  # A is [M, K]
        c_m, hidden_size = pool.shape
    if layout == "nt":  # weight [G, N, K]
        G, N, K = weight.shape
        weight_flat = weight.reshape(G * N, K).contiguous().view(-1)
    else:  # NN / TN: weight [G, K, N]
        G, K, N = weight.shape
        weight_flat = weight.reshape(G * K, N).contiguous().view(-1)
    assert K == hidden_size, f"weight K={K} != activation K={hidden_size}"
    out_features = N
    launch = compile_grouped_gemm_bf16(
        K=hidden_size,
        BLOCK_M=BM,
        BLOCK_N=BN,
        GROUP_M=GROUP_M,
        num_xcd=num_xcd,
        nt_vmcnt=int(nt_vmcnt),
        waves_per_eu=int(waves_per_eu),
        agpr_alloc=int(agpr_alloc),
        layout=layout,
    )
    launch(
        _bf16_flat(pool),
        weight_flat,
        _bf16_flat(output),
        tile_to_group,
        num_tile_blocks,
        c_m,
        out_features,
        stream=torch.cuda.current_stream(),
    )
    return output


# ───────────────────────────────────────────────────────────────────────
# Dispatch-PUSH-only launcher (no GEMM, no scoreboard) — raw dispatch bandwidth.
# bf16 mirror of the fp8 dispatch_only; every block pushes a round-robin share of
# the comm tasks to peer pools over XGMI.
# ───────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=256)
def _compile_dispatch_only(hidden_size, pool_capacity, comm_blocks, num_comm, waves_per_eu=2):
    # split each task across blocks_per_task blocks when tasks < blocks (saturate XGMI)
    blocks_per_task = max(1, comm_blocks // num_comm)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def dispatch_only_k(
        INPUT_TOKENS: fx.Tensor,
        COMM_DESTINATION: fx.Tensor,
        COMM_START: fx.Tensor,
        COMM_COUNT: fx.Tensor,
        COMM_SOURCE_OFFSET: fx.Tensor,
        SOURCE_TOKENS: fx.Tensor,
        POOL_PTRS: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        comm_block_count = fx.Int32(comm_blocks)
        input_resource = create_buffer_resource(INPUT_TOKENS, max_size=True)
        destination_resource = create_buffer_resource(COMM_DESTINATION, max_size=True)
        comm_start_resource = create_buffer_resource(COMM_START, max_size=True)
        comm_count_resource = create_buffer_resource(COMM_COUNT, max_size=True)
        comm_source_offset_resource = create_buffer_resource(COMM_SOURCE_OFFSET, max_size=True)
        source_tokens_resource = create_buffer_resource(SOURCE_TOKENS, max_size=True)
        pool_address_resource = create_buffer_resource(POOL_PTRS, max_size=True)

        dispatch_tile = dispatch_bf16_tile(
            thread_index=thread_index,
            hidden_size=hidden_size,
            pool_capacity=pool_capacity,
            input_resource=input_resource,
            destination_resource=destination_resource,
            comm_start_resource=comm_start_resource,
            comm_count_resource=comm_count_resource,
            comm_source_offset_resource=comm_source_offset_resource,
            source_tokens_resource=source_tokens_resource,
            pool_address_resource=pool_address_resource,
            signal=False,
        )

        if blocks_per_task > 1:
            task_index = block_index // fx.Int32(blocks_per_task)
            sub = block_index % fx.Int32(blocks_per_task)
            if task_index < fx.Int32(num_comm):
                dispatch_tile(task_index, sub, blocks_per_task)
        else:
            if block_index < comm_block_count:
                local_task_count = (
                    fx.Int32(num_comm) - block_index + comm_block_count - fx.Int32(1)
                ) // comm_block_count
                for it in range(local_task_count):
                    dispatch_tile(block_index + it * comm_block_count, fx.Int32(0), 1)

    @flyc.jit
    def launch(
        INPUT_TOKENS,
        COMM_DESTINATION,
        COMM_START,
        COMM_COUNT,
        COMM_SOURCE_OFFSET,
        SOURCE_TOKENS,
        POOL_PTRS,
        stream: fx.Stream = fx.Stream(None),
    ):
        dispatch_only_k(
            INPUT_TOKENS,
            COMM_DESTINATION,
            COMM_START,
            COMM_COUNT,
            COMM_SOURCE_OFFSET,
            SOURCE_TOKENS,
            POOL_PTRS,
            value_attrs=make_value_attrs(waves_per_eu, 0, "512,512"),
        ).launch(grid=(comm_blocks, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def dispatch_only(
    # ── source activation (pushed over XGMI) ──────────────────────────
    x,  # [num_src_tokens, K] bf16
    # ── dispatch comm plan (flat per-task metadata) ───────────────────
    comm_dest,  # [num_comm] i32   peer rank per task
    comm_start,  # [num_comm] i32   dst pool-row start on peer
    comm_count,  # [num_comm] i32   rows per task
    comm_src_offset,  # [num_comm] i32   offset into comm_src_tokens
    comm_src_tokens,  # [total]    i32   this rank's token ids, row order
    num_comm,  # int              number of comm tasks
    # ── symmetric activation pool (push target) ───────────────────────
    pool,  # [pool_capacity, K] bf16   local landing pool
    pool_ptrs,  # [world] i64   peer pool base ptrs
    # ── schedule config ───────────────────────────────────────────────
    *,
    comm_blocks=32,
):
    """Cross-rank dispatch PUSH only (no GEMM) — pushes ``x`` token rows to peer
    pools over XGMI. Bytes pushed per rank = (sum comm_count) * hidden * 2."""
    assert x.dtype == torch.bfloat16
    hidden_size = x.size(1)
    pool_capacity = pool.size(0)
    x_i32 = x.contiguous().view(torch.int32).view(-1)
    launch = _compile_dispatch_only(hidden_size, pool_capacity, int(comm_blocks), int(num_comm))
    launch(
        x_i32,
        comm_dest,
        comm_start,
        comm_count,
        comm_src_offset,
        comm_src_tokens,
        pool_ptrs,
        stream=torch.cuda.current_stream(),
    )


# ───────────────────────────────────────────────────────────────────────
# Dispatch-PUSH-only WITH token dedup (no GEMM): push PRIMARY rows over XGMI
# (secondaries skipped) + dest-local permute copy of secondaries. Mirrors the
# fused comm role exactly, minus the GEMM -> isolates the dispatch-side cost of
# dedup (XGMI saving vs the local-copy + scoreboard overhead).
# ───────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=256)
def _compile_dispatch_dedup_only(
    hidden_size, pool_capacity, dispatch_blocks, permute_blocks, num_comm, BLOCK_M, waves_per_eu=2
):
    # 2-role: role1 (block < dispatch_blocks) pushes PRIMARY rows + signals scoreboard;
    # role2 (block >= dispatch_blocks) permute-copies secondaries CONCURRENTLY (gated on
    # the primary scoreboard) -> permute hides under the dispatch push (triton_dist 2-stage).
    vec_i32, hidden_i32, n_warps, cols_per_warp_i32, chunk_count = _bf16_push_geom(hidden_size)
    assert BLOCK_M % n_warps == 0, f"BLOCK_M={BLOCK_M} must be a multiple of n_warps={n_warps}"
    rows_per_warp = BLOCK_M // n_warps

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def dispatch_dedup_k(
        INPUT_TOKENS: fx.Tensor,
        COMM_DESTINATION: fx.Tensor,
        COMM_START: fx.Tensor,
        COMM_COUNT: fx.Tensor,
        COMM_SOURCE_OFFSET: fx.Tensor,
        SOURCE_TOKENS: fx.Tensor,
        POOL_PTRS: fx.Tensor,
        SCOREBOARD_PTRS: fx.Tensor,
        SCOREBOARD: fx.Tensor,
        EXPECTED: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        SOURCE_DEDUP: fx.Tensor,
        DEDUP_SRC_ROW: fx.Tensor,
        SB_COPY: fx.Tensor,
        POOL_I32: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        dispatch_block_count = fx.Int32(dispatch_blocks)
        permute_block_count = fx.Int32(permute_blocks)
        input_resource = create_buffer_resource(INPUT_TOKENS, max_size=True)
        destination_resource = create_buffer_resource(COMM_DESTINATION, max_size=True)
        comm_start_resource = create_buffer_resource(COMM_START, max_size=True)
        comm_count_resource = create_buffer_resource(COMM_COUNT, max_size=True)
        comm_source_offset_resource = create_buffer_resource(COMM_SOURCE_OFFSET, max_size=True)
        source_tokens_resource = create_buffer_resource(SOURCE_TOKENS, max_size=True)
        pool_address_resource = create_buffer_resource(POOL_PTRS, max_size=True)
        scoreboard_address_resource = create_buffer_resource(SCOREBOARD_PTRS, max_size=True)
        expected_resource = create_buffer_resource(EXPECTED, max_size=True)
        num_tile_blocks_resource = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
        source_dedup_resource = create_buffer_resource(SOURCE_DEDUP, max_size=True)
        dedup_src_row_resource = create_buffer_resource(DEDUP_SRC_ROW, max_size=True)
        sb_copy_resource = create_buffer_resource(SB_COPY, max_size=True)
        pool_i32_resource = create_buffer_resource(POOL_I32, max_size=True)

        if block_index < dispatch_block_count:
            # role1: push primary rows + signal scoreboard (secondaries skipped)
            dispatch_tile = dispatch_bf16_tile(
                thread_index=thread_index,
                hidden_size=hidden_size,
                pool_capacity=pool_capacity,
                input_resource=input_resource,
                destination_resource=destination_resource,
                comm_start_resource=comm_start_resource,
                comm_count_resource=comm_count_resource,
                comm_source_offset_resource=comm_source_offset_resource,
                source_tokens_resource=source_tokens_resource,
                pool_address_resource=pool_address_resource,
                signal=True,
                scoreboard_address_resource=scoreboard_address_resource,
                block_m=BLOCK_M,
                source_dedup_resource=source_dedup_resource,
            )
            local_task_count = (
                fx.Int32(num_comm) - block_index + dispatch_block_count - fx.Int32(1)
            ) // dispatch_block_count
            for task_iteration in range(local_task_count):
                dispatch_tile(block_index + task_iteration * dispatch_block_count, fx.Int32(0), 1)
        else:
            # role2: dest-local permute copy of secondaries (concurrent with role1)
            permute_block_index = block_index - dispatch_block_count
            permute_token_tile(
                thread_index,
                permute_block_index,
                permute_block_count,
                num_tile_blocks_resource,
                dedup_src_row_resource,
                expected_resource,
                SCOREBOARD,
                pool_i32_resource,
                sb_copy_resource,
                BLOCK_M,
                n_warps,
                rows_per_warp,
                chunk_count,
                cols_per_warp_i32,
                vec_i32,
                hidden_i32,
            )

    @flyc.jit
    def launch(
        INPUT_TOKENS,
        COMM_DESTINATION,
        COMM_START,
        COMM_COUNT,
        COMM_SOURCE_OFFSET,
        SOURCE_TOKENS,
        POOL_PTRS,
        SCOREBOARD_PTRS,
        SCOREBOARD,
        EXPECTED,
        NUM_TILE_BLOCKS,
        SOURCE_DEDUP,
        DEDUP_SRC_ROW,
        SB_COPY,
        POOL_I32,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_size = dispatch_blocks + permute_blocks
        dispatch_dedup_k(
            INPUT_TOKENS,
            COMM_DESTINATION,
            COMM_START,
            COMM_COUNT,
            COMM_SOURCE_OFFSET,
            SOURCE_TOKENS,
            POOL_PTRS,
            SCOREBOARD_PTRS,
            SCOREBOARD,
            EXPECTED,
            NUM_TILE_BLOCKS,
            SOURCE_DEDUP,
            DEDUP_SRC_ROW,
            SB_COPY,
            POOL_I32,
            value_attrs=make_value_attrs(waves_per_eu, 0, "512,512"),
        ).launch(grid=(grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def dispatch_dedup_only(
    x,
    comm_dest,
    comm_start,
    comm_count,
    comm_src_offset,
    comm_src_tokens,
    num_comm,
    pool,
    pool_ptrs,
    scoreboard,
    scoreboard_ptrs,
    expected_count,
    num_tile_blocks,
    source_dedup,
    dedup_src_row,
    sb_copy,
    *,
    BM=256,
    dispatch_blocks=16,
    permute_blocks=16,
):
    """2-role cross-rank dispatch PUSH with token dedup (no GEMM): role1 pushes PRIMARY
    rows over XGMI (secondaries skipped) + signals scoreboard; role2 dest-local
    ``permute_token_tile`` copies secondaries CONCURRENTLY. Scoreboard + sb_copy zeroed first."""
    assert x.dtype == torch.bfloat16
    hidden_size = x.size(1)
    pool_capacity = pool.size(0)
    x_i32 = x.contiguous().view(torch.int32).view(-1)
    pool_i32 = pool.contiguous().view(torch.int32).view(-1)
    launch = _compile_dispatch_dedup_only(
        hidden_size, pool_capacity, int(dispatch_blocks), int(permute_blocks), int(num_comm), int(BM)
    )
    launch(
        x_i32,
        comm_dest,
        comm_start,
        comm_count,
        comm_src_offset,
        comm_src_tokens,
        pool_ptrs,
        scoreboard_ptrs,
        scoreboard,
        expected_count,
        num_tile_blocks,
        source_dedup,
        dedup_src_row,
        sb_copy,
        pool_i32,
        stream=torch.cuda.current_stream(),
    )


# ───────────────────────────────────────────────────────────────────────
# permute_token ONLY (role2 standalone) — measures the dest-local copy bandwidth
# in isolation (caller pre-fills the scoreboard so there is no wait).
# ───────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=256)
def _compile_permute_only(hidden_size, pool_capacity, permute_blocks, BLOCK_M, waves_per_eu=2):
    vec_i32, hidden_i32, n_warps, cols_per_warp_i32, chunk_count = _bf16_push_geom(hidden_size)
    assert BLOCK_M % n_warps == 0
    rows_per_warp = BLOCK_M // n_warps

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def permute_only_k(
        SCOREBOARD: fx.Tensor,
        EXPECTED: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        DEDUP_SRC_ROW: fx.Tensor,
        SB_COPY: fx.Tensor,
        POOL_I32: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        expected_resource = create_buffer_resource(EXPECTED, max_size=True)
        num_tile_blocks_resource = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
        dedup_src_row_resource = create_buffer_resource(DEDUP_SRC_ROW, max_size=True)
        sb_copy_resource = create_buffer_resource(SB_COPY, max_size=True)
        pool_i32_resource = create_buffer_resource(POOL_I32, max_size=True)
        permute_token_tile(
            thread_index,
            block_index,
            fx.Int32(permute_blocks),
            num_tile_blocks_resource,
            dedup_src_row_resource,
            expected_resource,
            SCOREBOARD,
            pool_i32_resource,
            sb_copy_resource,
            BLOCK_M,
            n_warps,
            rows_per_warp,
            chunk_count,
            cols_per_warp_i32,
            vec_i32,
            hidden_i32,
        )

    @flyc.jit
    def launch(
        SCOREBOARD,
        EXPECTED,
        NUM_TILE_BLOCKS,
        DEDUP_SRC_ROW,
        SB_COPY,
        POOL_I32,
        stream: fx.Stream = fx.Stream(None),
    ):
        permute_only_k(
            SCOREBOARD,
            EXPECTED,
            NUM_TILE_BLOCKS,
            DEDUP_SRC_ROW,
            SB_COPY,
            POOL_I32,
            value_attrs=make_value_attrs(waves_per_eu, 0, "512,512"),
        ).launch(grid=(permute_blocks, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def permute_only(
    scoreboard, expected_count, num_tile_blocks, dedup_src_row, sb_copy, pool, *, BM=256, permute_blocks=32
):
    """Run only the dest-local permute copy (role2). Pre-fill ``scoreboard`` >= expected so
    there is no wait -> measures pure copy bandwidth."""
    pool_capacity = pool.size(0)
    hidden_size = pool.size(1)
    pool_i32 = pool.contiguous().view(torch.int32).view(-1)
    launch = _compile_permute_only(hidden_size, pool_capacity, int(permute_blocks), int(BM))
    launch(
        scoreboard,
        expected_count,
        num_tile_blocks,
        dedup_src_row,
        sb_copy,
        pool_i32,
        stream=torch.cuda.current_stream(),
    )


# Per-shape autotune candidates (comm_blocks, nt_vmcnt, agpr_alloc, waves_per_eu).
# The comm/GEMM CU split dominates; the local push is HBM-roofline-bound so sweep
# comm_blocks widely (incl. past 96). nt_vmcnt (G2S drain) / waves are secondary.
_DISPATCH_CANDIDATES = [
    (16, 3, 0, 2),
    (24, 3, 0, 2),
    (32, 3, 0, 2),
    (40, 3, 0, 2),
    (48, 3, 0, 2),
    (56, 3, 0, 2),
    (64, 3, 0, 2),
    (80, 3, 0, 2),
    (96, 3, 0, 2),
    (112, 3, 0, 2),
    (128, 3, 0, 2),
    (160, 3, 0, 2),
    (48, 4, 0, 2),
    (48, 2, 0, 2),
    (64, 4, 0, 2),
    (48, 3, 0, 1),
]
_DISPATCH_AUTOTUNE_CACHE: dict = {}


def dispatch_grouped_gemm_bf16(
    # ── source activation (pushed over XGMI) ──────────────────────────
    x,  # [num_src_tokens, K] bf16
    # ── dispatch comm plan (flat per-task metadata) ───────────────────
    comm_dest,  # [num_comm] i32   peer rank per task
    comm_start,  # [num_comm] i32   dst pool-row start on peer
    comm_count,  # [num_comm] i32   rows per task
    comm_src_offset,  # [num_comm] i32   offset into comm_src_tokens
    comm_src_tokens,  # [total]    i32   this rank's token ids, row order
    num_comm,  # int              number of comm tasks
    # ── symmetric activation pool (A operand) ─────────────────────────
    pool,  # [pool_capacity, K] bf16   local landing pool
    pool_ptrs,  # [world] i64   peer pool base ptrs
    # ── per-expert weight (B) + output (C) ────────────────────────────
    weight,  # [G, N, K] bf16
    output,  # [pool_capacity, N] bf16
    tile_to_group,  # [n_mblk] i32   expert id per pool block
    # ── scoreboard handshake ──────────────────────────────────────────
    scoreboard,  # [n_mblk] i32   local fill counter (caller zeroes)
    scoreboard_ptrs,  # [world] i64   peer scoreboard base ptrs
    expected_count,  # [n_mblk] i32   expected push count per block
    num_tile_blocks,  # [1] i32        real tile-block count (device)
    # ── tile / schedule config (compile-time scalars) ─────────────────
    *,
    layout="nt",
    BM=256,
    BN=256,
    GROUP_M=4,
    comm_blocks=32,
    nt_vmcnt=3,
    autotune=False,
    autotune_reset=None,
    permute_blocks=32,
    # ── token dedup (optional XGMI saving) ────────────────────────────
    source_dedup=None,  # [pool_cap] i32  source-indexed: 1 = secondary (skip push)
    dedup_src_row=None,  # [pool_cap] i32  dest-indexed: -1 = primary, >=0 = copy-from row
    sb_copy=None,  # [n_mblk] i32    local copy-done gate (caller zeroes)
):
    """Fused cross-rank dispatch PUSH + grouped BF16 GEMM.

    ``x`` [num_src_tokens, K] bf16 source tokens; ``pool`` [pool_cap, K] bf16
    landing pool; ``output`` [pool_cap, N] bf16. ``layout`` selects the GEMM tile:
    NT (forward) weight [G,N,K]; NN (dgrad) weight [G,K,N]. The dispatch fills the
    pool M-major, so only NT/NN are valid here -- TN needs a K-major A, use the
    standalone ``grouped_gemm_bf16_only``. The dispatch comm plan is flat
    (``comm_dest/start/count/src_offset/src_tokens`` + ``num_comm``). Scoreboard zeroed first.

    Token dedup (all of ``source_dedup``/``dedup_src_row``/``sb_copy`` given): a token
    routed to >=2 experts on this dest rank crosses XGMI once (primary push); the
    redundant pool rows are filled by a dest-local ``permute_token_tile`` copy gated on
    the primary's scoreboard, then the GEMM additionally waits ``sb_copy``. Output same."""
    assert layout in (
        "nt",
        "nn",
    ), f"fused dispatch+GEMM supports nt/nn (pool is M-major); for tn use grouped_gemm_bf16_only, got {layout}"
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    hidden_size = x.size(1)
    if layout == "nt":  # weight [G, N, K]
        G, N, K = weight.shape
        weight_flat = weight.reshape(G * N, K).contiguous().view(-1)
    else:  # NN / TN: weight [G, K, N]
        G, K, N = weight.shape
        weight_flat = weight.reshape(G * K, N).contiguous().view(-1)
    assert K == hidden_size, f"weight K={K} != activation K={hidden_size}"
    pool_capacity = pool.size(0)
    out_features = N
    c_n = out_features

    # source tokens pushed as i32 words (bf16 viewed as int32); pool/weight read as bf16
    x_i32 = x.contiguous().view(torch.int32).view(-1)
    pool_flat = _bf16_flat(pool)
    output_flat = _bf16_flat(output)

    # dedup: pool also viewed as i32 for the dest-local copy; dummies when disabled
    dedup = source_dedup is not None
    assert (dedup_src_row is None) == (source_dedup is None) and (sb_copy is None) == (
        source_dedup is None
    ), "token dedup needs all of source_dedup/dedup_src_row/sb_copy or none"
    pool_i32 = pool.contiguous().view(torch.int32).view(-1)
    if dedup:
        source_dedup_arg, dedup_src_row_arg, sb_copy_arg = source_dedup, dedup_src_row, sb_copy
    else:
        _dummy = x_i32[:1]  # tiny i32; never read (resources gated on dedup)
        source_dedup_arg = dedup_src_row_arg = sb_copy_arg = _dummy

    pos_args = (
        x_i32,
        comm_dest,
        comm_start,
        comm_count,
        comm_src_offset,
        comm_src_tokens,
        pool_ptrs,
        scoreboard_ptrs,
        pool_flat,
        weight_flat,
        output_flat,
        tile_to_group,
        scoreboard,
        expected_count,
        num_tile_blocks,
        source_dedup_arg,
        dedup_src_row_arg,
        sb_copy_arg,
        pool_i32,
        c_n,
    )

    if autotune:
        key = (out_features, hidden_size, pool_capacity, BM, BN, int(num_comm), dedup, layout)
        cached = _DISPATCH_AUTOTUNE_CACHE.get(key)
        if cached is None:
            cached = _autotune(
                pos_args,
                output_flat,
                out_features,
                hidden_size,
                pool_capacity,
                BM,
                BN,
                int(num_comm),
                scoreboard,
                autotune_reset,
                dedup,
                layout,
            )
            _DISPATCH_AUTOTUNE_CACHE[key] = cached
        launch, _cfg = cached
    else:
        launch = _compile(
            out_features,
            hidden_size,
            pool_capacity,
            BM,
            BN,
            int(comm_blocks),
            int(num_comm),
            int(nt_vmcnt),
            GROUP_M=int(GROUP_M),
            dedup=dedup,
            permute_blocks=int(permute_blocks),
            layout=layout,
        )
    launch(*pos_args, stream=torch.cuda.current_stream())
    return output


def _autotune(
    pos_args,
    finite_view,
    out_features,
    hidden_size,
    pool_capacity,
    BM,
    BN,
    num_comm,
    scoreboard,
    reset,
    dedup=False,
    layout="nt",
):
    """Bench the candidates with a per-iter scoreboard reset; return (launch, cfg)."""
    if reset is None:
        reset = scoreboard.zero_
    stream = torch.cuda.current_stream()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    best_us, best = float("inf"), None
    for comm_blocks, nt_vmcnt, agpr, waves in _DISPATCH_CANDIDATES:
        try:
            launch = _compile(
                out_features,
                hidden_size,
                pool_capacity,
                BM,
                BN,
                int(comm_blocks),
                num_comm,
                int(nt_vmcnt),
                int(waves),
                int(agpr),
                dedup=dedup,
                layout=layout,
            )
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
                best_us, best = us, (launch, (comm_blocks, nt_vmcnt, agpr, waves))
        except Exception:
            continue
    if best is None:
        raise RuntimeError("dispatch_grouped_gemm_bf16 autotune found no working cfg")
    return best
