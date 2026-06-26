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
  * ``block_index < num_dispatch_cu`` blocks push token rows to peer pools over XGMI
    and signal a per-pool-block scoreboard (``dispatch_bf16_tile``).
  * the remaining blocks each compute ONE NT output tile of the grouped GEMM
    (``A=pool[M,K]`` bf16, per-expert ``B=weight[G,N,K]`` bf16 -> ``C=out[M,N]``
    bf16) via ``gemm_bf16_nt_tile``, spinning on the scoreboard until their pool
    block is filled. The comm latency hides under the MFMA-bound GEMM.

NT only (the bf16 GEMM kernel implements NT only). K must be a multiple of
BLOCK_K (DSv3 shapes qualify) and hidden a multiple of 512 (warp push step).
"""

import functools
from typing import Optional, Tuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith
from flydsl.expr.buffer_ops import (
    buffer_load,
    create_buffer_resource,
    create_buffer_resource_from_addr,
)
from flydsl.expr.typing import AddressSpace, PointerType

from primus_turbo.flydsl.common.tile_spec import _emit_if_then
from primus_turbo.flydsl.mega.dispatch_prologue_kernel import dispatch_prologue
from primus_turbo.flydsl.mega.ep_intranode import _BLOCK_THREADS, dispatch_bf16_tile

# shared bf16 GEMM tile + geometry/LDS helpers from the dense bf16 kernel
from primus_turbo.flydsl.mega.gemm_bf16_kernel import (
    _make_shared_storage,
    gemm_bf16_nn_tile,
    gemm_bf16_nt_tile,
    gemm_bf16_tn_tile,
)
from primus_turbo.flydsl.mega.prims import atomic_add, ld, memory_fence, st
from primus_turbo.flydsl.mega.sym_layout import SymLayout
from primus_turbo.flydsl.mega.symm_buffer import get_symm_buffer_for_mega_moe
from primus_turbo.flydsl.utils.gemm_helper import (
    ceildiv,
    make_value_attrs,
    xcd_remap_pid,
)

# per-tile GEMM closure by layout (NT forward, NN dgrad, TN wgrad); all grouped via b_group_base
_GEMM_TILE = {"nt": gemm_bf16_nt_tile, "nn": gemm_bf16_nn_tile, "tn": gemm_bf16_tn_tile}

_BLOCK_M = 256

# comm-PUSH + scoreboard prims (byte-agnostic; shared intra-node EP layer)

# fused dispatch-prologue: builds the DeepEP-style handle when handle=None


# ───────────────────────────────────────────────────────────────────────
# Grouped GEMM-only launcher (the compute-peak baseline). Dense XCD-swizzle
# scheduler + GROUP_M, per-expert B slab via tile_to_expert.
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
    num_dispatch_cu,
    num_comm,
    nt_vmcnt=3,
    waves_per_eu=2,
    agpr_alloc=0,
    out_fp16=False,
    GROUP_M=1,
    layout="nt",
):
    K = hidden_size
    gemm_tile = _GEMM_TILE[layout]
    assert out_features % BLOCK_N == 0, "out_features must be a multiple of BLOCK_N"
    assert pool_capacity % BLOCK_M == 0, "pool_capacity must be a multiple of BLOCK_M"
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)
    n_blocks = out_features // BLOCK_N
    worst_case_tiles = pool_capacity // BLOCK_M

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def dispatch_grouped_gemm_kernel(
        INPUT_TOKENS: fx.Tensor,
        EXPERT_SEND_DST_RANK: fx.Tensor,
        EXPERT_SEND_DST_ROW: fx.Tensor,
        EXPERT_SEND_COUNT: fx.Tensor,
        EXPERT_SEND_OFFSET: fx.Tensor,
        DISPATCHED_TOKEN_IDX: fx.Tensor,
        sym_layout: SymLayout,
        WEIGHTS: fx.Tensor,
        OUTPUT: fx.Tensor,
        TILE_TO_GROUP: fx.Tensor,
        EXPECTED: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        c_n: fx.Int32,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        comm_block_count = fx.Int32(num_dispatch_cu)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        # ===== COMM role resources =====
        input_resource = create_buffer_resource(INPUT_TOKENS, max_size=True)
        expert_send_dst_rank_resource = create_buffer_resource(EXPERT_SEND_DST_RANK, max_size=True)
        expert_send_dst_row_resource = create_buffer_resource(EXPERT_SEND_DST_ROW, max_size=True)
        expert_send_count_resource = create_buffer_resource(EXPERT_SEND_COUNT, max_size=True)
        expert_send_offset_resource = create_buffer_resource(EXPERT_SEND_OFFSET, max_size=True)
        dispatched_token_idx_resource = create_buffer_resource(DISPATCHED_TOKEN_IDX, max_size=True)
        # pool base / peer pool / peer scoreboard all come from the SymLayout now
        # ===== GEMM role resources =====
        group_resource = create_buffer_resource(TILE_TO_GROUP, max_size=True)
        expected_resource = create_buffer_resource(EXPECTED, max_size=True)
        num_tile_blocks_resource = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)

        dispatch_tile = dispatch_bf16_tile(
            thread_index=thread_index,
            hidden_size=hidden_size,
            pool_capacity=pool_capacity,
            input_resource=input_resource,
            expert_send_dst_rank_resource=expert_send_dst_rank_resource,
            expert_send_dst_row_resource=expert_send_dst_row_resource,
            expert_send_count_resource=expert_send_count_resource,
            expert_send_offset_resource=expert_send_offset_resource,
            dispatched_token_idx_resource=dispatched_token_idx_resource,
            pool_address_resource=None,
            signal=True,
            scoreboard_address_resource=None,
            block_m=BLOCK_M,
            # two-heap delta path: local base ptr + i64[world] per-peer delta table
            pool_base=sym_layout.pool_ptr,
            pool_offsets_resource=create_buffer_resource_from_addr(
                sym_layout.offsets_ptr, num_records_bytes=int(sym_layout.num_ranks) * 8
            ),
            scoreboard_base=sym_layout.scoreboard_ptr,
            scoreboard_offsets_resource=create_buffer_resource_from_addr(
                sym_layout.signal_offsets_ptr, num_records_bytes=int(sym_layout.num_ranks) * 8
            ),
        )

        # ── 2-role: role1 dispatch [0,D) ‖ role2 gemm [D, grid). ───────────────────
        if block_index < comm_block_count:
            # role1: dispatch PUSH over XGMI + signal.
            local_task_count = (
                fx.Int32(num_comm) - block_index + comm_block_count - fx.Int32(1)
            ) // comm_block_count
            for task_iteration in range(local_task_count):
                dispatch_tile(block_index + task_iteration * comm_block_count, fx.Int32(0), 1)
        else:
            # role2: GROUP_M tile-id map over the REAL grid.
            tile_index = block_index - comm_block_count
            real_tiles = buffer_load(num_tile_blocks_resource, fx.Int32(0), vec_width=1, dtype=fx.T.i32())
            real_grid = real_tiles * fx.Int32(n_blocks)
            if tile_index < real_grid:
                num_pid_in_group = fx.Int32(GROUP_M * n_blocks)
                group_id = tile_index // num_pid_in_group
                pid_in_group = tile_index % num_pid_in_group
                first_pid_m = group_id * fx.Int32(GROUP_M)
                remaining_m = real_tiles - first_pid_m
                group_size_m = arith.select(remaining_m < fx.Int32(GROUP_M), remaining_m, fx.Int32(GROUP_M))
                block_m = first_pid_m + (pid_in_group % group_size_m)
                block_n = pid_in_group // group_size_m
                c_m_real = fx.Int32(pool_capacity)
                # GEMM gate: wait this block's primaries (scoreboard). No copy work here.
                # local scoreboard reachable via the SymLayout base (same memory the
                # dispatch role signals on peers); no separate SCOREBOARD arg needed.
                sb_base = sym_layout.scoreboard_ptr
                expected_count = buffer_load(expected_resource, block_m, vec_width=1, dtype=fx.T.i32())
                if thread_index == fx.Int32(0):
                    signal = ld(sb_base, block_m, scope="sys")
                    while signal < expected_count:
                        fx.rocdl.s_sleep(fx.Int32(2))
                        signal = ld(sb_base, block_m, scope="sys")
                fx.gpu.barrier()
                memory_fence("acquire", scope="sys")  # read fresh peer-pushed pool rows

                g_idx = buffer_load(group_resource, block_m, vec_width=1, dtype=fx.T.i32())
                gbase = g_idx * fx.Int32(K) * c_n
                # A operand: bf16 view over this rank's pool from the SymLayout base ptr
                pool_ptr_ty = PointerType.get(
                    elem_ty=fx.BFloat16.ir_type, address_space=AddressSpace.Global, alignment=16
                )
                pool_tensor = fx.make_view(
                    fx.inttoptr(pool_ptr_ty, sym_layout.pool_ptr),
                    fx.make_layout(pool_capacity * K, 1),
                )
                gemm_tile(
                    pool_tensor,
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
                # No SB_CONSUME: fold the reader count into the scoreboard. After each
                # GEMM tile finishes, bump scoreboard[block_m]; the last of the n_blocks
                # N-tiles (count reaches expected + n_blocks - 1) resets the slot to 0
                # for the next launch. The gate stays >= expected, so readers that arrive
                # after earlier bumps still pass; reset only fires once all have passed.
                if thread_index == fx.Int32(0):
                    prev = atomic_add(sb_base, block_m, fx.Int32(1), scope="sys")
                    if prev == expected_count + fx.Int32(n_blocks - 1):
                        st(sb_base, block_m, fx.Int32(0), scope="sys")

    @flyc.jit
    def launch(
        INPUT_TOKENS,
        EXPERT_SEND_DST_RANK,
        EXPERT_SEND_DST_ROW,
        EXPERT_SEND_COUNT,
        EXPERT_SEND_OFFSET,
        DISPATCHED_TOKEN_IDX,
        sym_layout,
        WEIGHTS,
        OUTPUT,
        TILE_TO_GROUP,
        EXPECTED,
        NUM_TILE_BLOCKS,
        c_n: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_size = num_dispatch_cu + worst_case_tiles * n_blocks
        dispatch_grouped_gemm_kernel(
            INPUT_TOKENS,
            EXPERT_SEND_DST_RANK,
            EXPERT_SEND_DST_ROW,
            EXPERT_SEND_COUNT,
            EXPERT_SEND_OFFSET,
            DISPATCHED_TOKEN_IDX,
            sym_layout,
            WEIGHTS,
            OUTPUT,
            TILE_TO_GROUP,
            EXPECTED,
            NUM_TILE_BLOCKS,
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
    tile_to_expert,  # [n_mblk] i32   expert id per BM pool block
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

    ``pool`` is A=[M,K] bf16, ``output`` is [M,N] bf16, ``tile_to_expert`` maps each
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
        tile_to_expert,
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
def _compile_dispatch_only(hidden_size, pool_capacity, num_dispatch_cu, num_comm, waves_per_eu=2):
    # split each task across blocks_per_task blocks when tasks < blocks (saturate XGMI)
    blocks_per_task = max(1, num_dispatch_cu // num_comm)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def dispatch_only_k(
        INPUT_TOKENS: fx.Tensor,
        EXPERT_SEND_DST_RANK: fx.Tensor,
        EXPERT_SEND_DST_ROW: fx.Tensor,
        EXPERT_SEND_COUNT: fx.Tensor,
        EXPERT_SEND_OFFSET: fx.Tensor,
        DISPATCHED_TOKEN_IDX: fx.Tensor,
        POOL_PTRS: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        comm_block_count = fx.Int32(num_dispatch_cu)
        input_resource = create_buffer_resource(INPUT_TOKENS, max_size=True)
        expert_send_dst_rank_resource = create_buffer_resource(EXPERT_SEND_DST_RANK, max_size=True)
        expert_send_dst_row_resource = create_buffer_resource(EXPERT_SEND_DST_ROW, max_size=True)
        expert_send_count_resource = create_buffer_resource(EXPERT_SEND_COUNT, max_size=True)
        expert_send_offset_resource = create_buffer_resource(EXPERT_SEND_OFFSET, max_size=True)
        dispatched_token_idx_resource = create_buffer_resource(DISPATCHED_TOKEN_IDX, max_size=True)
        pool_address_resource = create_buffer_resource(POOL_PTRS, max_size=True)

        dispatch_tile = dispatch_bf16_tile(
            thread_index=thread_index,
            hidden_size=hidden_size,
            pool_capacity=pool_capacity,
            input_resource=input_resource,
            expert_send_dst_rank_resource=expert_send_dst_rank_resource,
            expert_send_dst_row_resource=expert_send_dst_row_resource,
            expert_send_count_resource=expert_send_count_resource,
            expert_send_offset_resource=expert_send_offset_resource,
            dispatched_token_idx_resource=dispatched_token_idx_resource,
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
        EXPERT_SEND_DST_RANK,
        EXPERT_SEND_DST_ROW,
        EXPERT_SEND_COUNT,
        EXPERT_SEND_OFFSET,
        DISPATCHED_TOKEN_IDX,
        POOL_PTRS,
        stream: fx.Stream = fx.Stream(None),
    ):
        dispatch_only_k(
            INPUT_TOKENS,
            EXPERT_SEND_DST_RANK,
            EXPERT_SEND_DST_ROW,
            EXPERT_SEND_COUNT,
            EXPERT_SEND_OFFSET,
            DISPATCHED_TOKEN_IDX,
            POOL_PTRS,
            value_attrs=make_value_attrs(waves_per_eu, 0, "512,512"),
        ).launch(grid=(num_dispatch_cu, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def dispatch_only(
    # ── source activation (pushed over XGMI) ──────────────────────────
    x,  # [num_src_tokens, K] bf16
    # ── dispatch handle handle (DeepEP-style; expanded to ABI below) ─────
    # dispatch handle tuple: (dst_rank, dst_offset, count, src_offset, src_tokens, ...)
    handle,
    # ── symmetric activation pool (push target) ───────────────────────
    pool,  # [pool_capacity, K] bf16   local landing pool
    pool_ptrs,  # [world] i64   peer pool base ptrs
    # ── schedule config ───────────────────────────────────────────────
    *,
    num_dispatch_cu=32,
):
    """Cross-rank dispatch PUSH only (no GEMM) — pushes ``x`` token rows to peer
    pools over XGMI. Bytes pushed per rank = (sum expert_send_count) * hidden * 2."""
    # expand the handle to the device ABI (order is correctness-critical)
    # dispatch handle (DeepEP-style tuple); num_comm derived from dst_rank
    expert_send_dst_rank, expert_send_dst_row, expert_send_count, expert_send_offset, dispatched_token_idx = (
        handle[:5]
    )
    num_comm = expert_send_dst_rank.numel()
    assert x.dtype == torch.bfloat16
    hidden_size = x.size(1)
    pool_capacity = pool.size(0)
    x_i32 = x.contiguous().view(torch.int32).view(-1)
    launch = _compile_dispatch_only(hidden_size, pool_capacity, int(num_dispatch_cu), int(num_comm))
    launch(
        x_i32,
        expert_send_dst_rank,
        expert_send_dst_row,
        expert_send_count,
        expert_send_offset,
        dispatched_token_idx,
        pool_ptrs,
        stream=torch.cuda.current_stream(),
    )


# Per-shape autotune candidates (num_dispatch_cu, nt_vmcnt, agpr_alloc, waves_per_eu).
# The comm/GEMM CU split dominates; the local push is HBM-roofline-bound so sweep
# num_dispatch_cu widely (incl. past 96). nt_vmcnt (G2S drain) / waves are secondary.
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
]
_DISPATCH_AUTOTUNE_CACHE: dict = {}


def dispatch_grouped_gemm_bf16(
    # ── source activation (pushed over XGMI) ──────────────────────────
    x: torch.Tensor,  # [num_src_tokens, K] bf16
    # ── per-expert weight (B) + output (C) ────────────────────────────
    l1_weights: torch.Tensor,  # [G, N, K] bf16
    group: torch.distributed.group,
    handle: Optional[Tuple] = None,
    topk_idx: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    layout: str = "nt",
    num_dispatched_tokens: Optional[int] = None,
    num_dispatch_cu: int = 16,
    autotune=False,  # remove in the future
    BM=256,
    BN=256,
    GROUP_M=4,
):
    """Fused cross-rank dispatch PUSH + grouped BF16 GEMM.

    ``x`` [num_src_tokens, K] bf16 source tokens; ``pool`` [pool_cap, K] bf16
    landing pool; ``output`` [pool_cap, N] bf16. ``layout`` selects the GEMM tile:
    NT (forward) weight [G,N,K]; NN (dgrad) weight [G,K,N]. The dispatch fills the
    pool M-major, so only NT/NN are valid here -- TN needs a K-major A, use the
    standalone ``grouped_gemm_bf16_only``. The dispatch comm handle is flat
    (``expert_send_dst_rank/start/count/src_offset/src_tokens`` + ``num_comm``). Scoreboard zeroed first.

    The symmetric workspace (pool GEMM A operand + peer pool / peer scoreboard delta
    tables) is fetched from the active ``get_symm_buffer_for_mega_moe()`` (no-group call)
    -- the buffer the caller already built -- so it is no longer passed as an argument.
    ``tile_to_expert`` / ``expected_count`` are read from the handle
    (``handle[-2]`` / ``handle[-1]``)."""
    # active symmetric workspace (built earlier by get_symm_buffer_for_mega_moe); names
    # the pool + signal heaps and the peer pool/scoreboard delta tables
    symm = get_symm_buffer_for_mega_moe()
    sym_layout = symm.make_sym_layout()
    # no handle given -> build the handle from topk via the fused prologue. The
    # prologue returns tile_to_expert / expected separately (not appended to handle).
    if handle is None:
        assert topk_idx is not None, "handle=None requires topk_idx to run the prologue"
        assert layout == "nt", "handle=None auto-prologue is forward-only (nt); pass handle for nn/tn"
        num_tokens = x.size(0)
        handle, tile_to_expert, expected_count, *_ = dispatch_prologue(
            topk_idx,
            topk_weights,
            sym_layout=sym_layout,
            num_tokens=num_tokens,
            num_topk=int(sym_layout.num_topk),
            num_experts=int(sym_layout.num_experts),
            world_size=int(sym_layout.num_ranks),
            rank=int(sym_layout.rank_idx),
            experts_per_rank=int(sym_layout.num_experts_per_rank),
            block_m=_BLOCK_M,
            pool_capacity=int(sym_layout.num_max_pool_tokens),
        )
    else:
        # caller-supplied extended handle: tile_to_expert / expected ride at the tail
        tile_to_expert, expected_count = handle[-2], handle[-1]
    # expand the handle to the device ABI (order is correctness-critical)
    expert_send_dst_rank, expert_send_dst_row, expert_send_count, expert_send_offset, dispatched_token_idx = (
        handle[:5]
    )
    num_comm = expert_send_dst_rank.numel()
    assert layout in (
        "nt",
        "nn",
    ), f"fused dispatch+GEMM supports nt/nn (pool is M-major); for tn use grouped_gemm_bf16_only, got {layout}"
    assert x.dtype == torch.bfloat16 and l1_weights.dtype == torch.bfloat16
    hidden_size = x.size(1)
    if layout == "nt":  # weight [G, N, K]
        G, N, K = l1_weights.shape
        weight_flat = l1_weights.reshape(G * N, K).contiguous().view(-1)
    else:  # NN: weight [G, K, N]
        G, K, N = l1_weights.shape
        weight_flat = l1_weights.reshape(G * K, N).contiguous().view(-1)
    assert K == hidden_size, f"weight K={K} != activation K={hidden_size}"
    # pool capacity is a compile-time scalar named by the SymLayout
    pool_capacity = int(sym_layout.num_max_pool_tokens)
    out_features = N
    c_n = out_features

    # real tile count from the active symm buffer; meta_scalars[1] = total_rows // BM.
    # The local scoreboard is read/reset inside the kernel via sym_layout.scoreboard_ptr
    # (gate + folded reader-count self-reset), so neither scoreboard nor sb_consume are
    # passed. scoreboard is kept only as the autotune per-iter reset target.
    scoreboard = symm.scoreboard
    num_tile_blocks = symm.meta_scalars[1:2]
    nt_vmcnt = 3

    is_nosync = num_dispatched_tokens is None
    if is_nosync:
        output = torch.empty((pool_capacity, out_features), dtype=x.dtype, device=x.device)
    else:
        assert False

    # source tokens pushed as i32 words (bf16 viewed as int32); pool/weight read as bf16
    x_i32 = x.contiguous().view(torch.int32).view(-1)
    output_flat = _bf16_flat(output)

    pos_args = (
        x_i32,
        expert_send_dst_rank,
        expert_send_dst_row,
        expert_send_count,
        expert_send_offset,
        dispatched_token_idx,
        sym_layout,
        weight_flat,
        output_flat,
        tile_to_expert,
        expected_count,
        num_tile_blocks,
        c_n,
    )

    if autotune:
        key = (out_features, hidden_size, pool_capacity, BM, BN, int(num_comm), layout)
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
                None,  # reset=None -> _autotune uses scoreboard.zero_
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
            int(num_dispatch_cu),
            int(num_comm),
            int(nt_vmcnt),
            GROUP_M=int(GROUP_M),
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
    layout="nt",
):
    """Bench the candidates with a per-iter scoreboard reset; return (launch, cfg)."""
    if reset is None:
        reset = scoreboard.zero_
    stream = torch.cuda.current_stream()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    best_us, best = float("inf"), None
    for num_dispatch_cu, nt_vmcnt, agpr, waves in _DISPATCH_CANDIDATES:
        try:
            launch = _compile(
                out_features,
                hidden_size,
                pool_capacity,
                BM,
                BN,
                int(num_dispatch_cu),
                num_comm,
                int(nt_vmcnt),
                int(waves),
                int(agpr),
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
                best_us, best = us, (launch, (num_dispatch_cu, nt_vmcnt, agpr, waves))
        except Exception:
            continue
    if best is None:
        raise RuntimeError("dispatch_grouped_gemm_bf16 autotune found no working cfg")
    return best
