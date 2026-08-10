###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused cross-rank dispatch PUSH (fp8) + grouped MXFP8 GEMM (NT), FlyDSL — 3-stage grid.

One kernel per rank; a block picks its role from its index, and all three overlap in the one grid:

  * COMM (first ``num_dispatch_cu`` blocks): pushes one comm task's PRE-QUANTIZED fp8 rows and RAW
    E8M0 scales into the peer's ``pool_fp8`` / ``pool_scale`` over XGMI, then signals its
    scoreboard. Quantizing once on the source keeps the push coalesced and XGMI-bound.
  * PRESHUFFLE (next ``num_preshuffle_cu``): waits for a pool-block, transposes its A-scale raw ->
    ScaleS2R broadcast into ``pool_scale_ps`` once, stamps a SENTINEL.
  * GEMM (the rest): one NT output tile each via ``gemm_mxfp8_nt_tile``, spinning on that SENTINEL.

Token quant and the weight B-scale preshuffle are host-side, before the launch. The sys-scope
scoreboard plus each role's own cache fence carry all visibility, so no host sync is needed.

NT only. Constraints: hidden % 1024 == 0 (fp8 warp push), N % BLOCK_N == 0,
num_max_pool_tokens % BLOCK_M == 0, K % 128 == 0 and K >= 256 (mxfp8 MMA).
"""

import functools
import json
import os
import pathlib as _pathlib
from typing import Optional, Tuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
)
from flydsl.expr.typing import AddressSpace, PointerType
from torch.distributed import ProcessGroup

from primus_turbo.flydsl.mega.fp8.dispatch_prologue import dispatch_prologue
from primus_turbo.flydsl.mega.fp8.ep_fp8 import (
    _BLOCK_THREADS,
    dispatch_fp8_copy_tile,
)
from primus_turbo.flydsl.mega.fp8.gemm_mxfp8_tile import (
    BLOCK_K,
    gemm_mxfp8_nt_tile,
)
from primus_turbo.flydsl.mega.fp8.prims import (
    l2_invalidate,
    ld,
    st,
)
from primus_turbo.flydsl.mega.fp8.quant import quantize_rowwise_mxfp8_flydsl
from primus_turbo.flydsl.mega.fp8.sym_layout import SymLayout
from primus_turbo.flydsl.mega.fp8.symm_buffer import get_symm_buffer_for_mega_moe
from primus_turbo.flydsl.mega.prims import cast, read_clock, spin_timed_out
from primus_turbo.flydsl.utils.gemm_helper import (
    ceildiv,
    emit_lds_repack,
    make_value_attrs,
)

_H_NUM_TILE_BLOCKS = 11  # appended by this module: the per-call real-tile count
_H_ORIGIN_RANK = 12  # appended by this module: per-call pool row -> owning rank
_H_ORIGIN_SLOT = 13  # appended by this module: per-call pool row -> owning topk slot

_FUSED_COMPILED: dict = {}  # (shape key) -> flyc.compile'd launch (eager; skip per-call @flyc.jit dispatch)

# The COMM/PRESHUFFLE split of the shared grid is a tuning knob, and the best pair moves with the
# GEMM's N: the L2-dgrad's N=I has half the forward's FLOPs against the same push bytes, so it wants
# more comm. A LOOKUP, not a search -- timing candidates on the first call is not reproducible (the
# ranks are still compiling, and the two forward candidates measure within 0.8% there against a ~5%
# steady-state gap), which made fwd-only bimodal at 5.09-5.37 ms. benchmark/ops/tune_cu_split_fp8.py
# builds the table offline; no tuning code lives here.
_CU_SPLIT_CANDIDATES = ((16, 16), (24, 8))  # the scanner's search space; never timed at runtime
# Measured steady-state winners on EP8 T=8192 DSv3 (gfx950): N=2I forward, N=I L2 dgrad.
_CU_SPLIT_DEFAULTS = {4096: (16, 16), 2048: (24, 8)}
_CU_SPLIT_TABLE: Optional[dict] = None  # lazily loaded from disk
# PT_MEGA_CU_SPLIT="d,p" pins the split for every shape (the scanner uses this).
_CU_SPLIT_ENV = os.environ.get("PT_MEGA_CU_SPLIT", "")
_CU_SPLIT_DEBUG = os.environ.get("PT_MEGA_CU_SPLIT_DEBUG", "0") != "0"
_CU_SPLIT_LOGGED: set = set()
# Written by the scanner; keys are "N,K,pool,BM,BN,ranks,G" strings -> [dispatch_cu, preshuffle_cu].
_CU_SPLIT_TABLE_PATH = os.environ.get(
    "PT_MEGA_CU_SPLIT_TABLE",
    str(_pathlib.Path(__file__).with_name("cu_split_table.json")),
)


def extend_handle(prologue_handle, symm) -> tuple:
    """Append this module's three per-call handle slots (indices 11-13) to a ``dispatch_prologue``
    handle. Every caller of ``dispatch_grouped_gemm_mxfp8`` must do this first; the plain prologue
    handle stops at index 10.

    All three are snapshots of shared symm scratch, so they belong to THIS call: the next prologue
    overwrites meta_scalars AND resets the whole origin_rank / origin_slot region, while the backward
    reuses this handle long after that. num_tile_blocks is the real-tile count; origin_rank /
    origin_slot map each pool row back to the rank and topk slot that sent it, which is what the
    L1-dgrad push needs to return the row to its owner. Stream-ordered D2D copies, no host sync.
    (bf16 does the same with pool_src_slot.)
    """
    return tuple(prologue_handle) + (
        symm.meta_scalars[1:2].clone(),
        symm.origin_rank.clone(),
        symm.origin_slot.clone(),
    )


def cu_split_key(N, K, num_max_pool_tokens, BM, BN, num_ranks, G) -> str:
    """The shape key both this module and the offline scanner use for the split table."""
    return f"{N},{K},{num_max_pool_tokens},{BM},{BN},{num_ranks},{G}"


def _cu_split_table() -> dict:
    global _CU_SPLIT_TABLE
    if _CU_SPLIT_TABLE is None:
        try:
            _CU_SPLIT_TABLE = json.loads(_pathlib.Path(_CU_SPLIT_TABLE_PATH).read_text())
        except Exception:  # absent or unreadable -> built-in defaults only
            _CU_SPLIT_TABLE = {}
    return _CU_SPLIT_TABLE


def _resolve_cu_split(key: str, N: int):
    """``(split, source)`` from env pin > offline table > per-N default. Never times anything."""
    if _CU_SPLIT_ENV:
        d, p = (int(v) for v in _CU_SPLIT_ENV.split(","))
        return (d, p), "env"
    if key in _cu_split_table():
        return tuple(_cu_split_table()[key]), "table"
    if N in _CU_SPLIT_DEFAULTS:
        return _CU_SPLIT_DEFAULTS[N], "default"
    # Unmeasured shape: borrow the nearest measured N, since the split tracks how much GEMM work
    # rides on the same push bytes. Scan it to replace this guess with a measurement.
    return _CU_SPLIT_DEFAULTS[min(_CU_SPLIT_DEFAULTS, key=lambda n: abs(n - N))], "guess"


def _log_cu_split(key: str, split, src: str):
    """Once per shape, not once per launch. The scanner reads the shape key off these lines, so this
    also fires for a caller-pinned split -- that is the case it is scanning."""
    if _CU_SPLIT_DEBUG and key not in _CU_SPLIT_LOGGED:
        _CU_SPLIT_LOGGED.add(key)
        print(f"[mega fp8] cu split {tuple(split)} from {src} for {key}", flush=True)


def _make_fwd_shared_storage_coalesce(BLOCK_M, BLOCK_N, tile_ps):
    """fp8 8-buffer LDS ping-pong (A/B cur/next x0/1, gemm role) + a ps_tile int32 scratch for the
    preshuffle role's coalesced LDS transpose. flydsl allows only ONE SharedAllocator per kernel, so
    both regions share one struct; gemm & preshuffle are distinct blocks so never use both at once
    (extra ~14 KB @ K=7168 -> still 1 block/CU). Mirrors the bwd fork's coalesce storage."""
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    a_lds = LDS_BLOCK_M * BLOCK_K
    b_lds = LDS_BLOCK_N * BLOCK_K

    @fx.struct
    class SharedStorageCoalesce:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds, 16]
        ps_tile: fx.Array[fx.Int32, tile_ps, 16]

    return SharedStorageCoalesce


@functools.lru_cache(maxsize=4)
def _make_epoch_bump(add_dispatch, add_ps):
    """Single-block device kernel: flip the dispatch flag parity, bump dispatch/preshuffle
    expected[new_parity]. Launched on the dispatch stream just before the main kernel so the
    comm->preshuffle (dispatch_flag) and preshuffle->gemm (preshuffle_flag) gates self-reset (no
    host synchronize()+barrier(), no cross-call reset race). Mirrors the bf16 dispatch epoch bump,
    plus a second (preshuffle) counter for the fp8-only preshuffle role."""

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def epoch_bump_kernel(PARITY: fx.Tensor, DISP_EXP: fx.Tensor, PS_EXP: fx.Tensor):
        if fx.thread_idx.x == fx.Int32(0):
            parity_res = create_buffer_resource(PARITY, max_size=True)
            disp_res = create_buffer_resource(DISP_EXP, max_size=True)
            ps_res = create_buffer_resource(PS_EXP, max_size=True)
            new_parity = buffer_load(parity_res, fx.Int32(0), vec_width=1, dtype=fx.T.i64()) ^ fx.Int64(1)
            buffer_store(new_parity, parity_res, fx.Int32(0))
            idx = cast(new_parity, fx.T.i32())
            new_disp = buffer_load(disp_res, idx, vec_width=1, dtype=fx.T.i64()) + fx.Int64(add_dispatch)
            buffer_store(new_disp, disp_res, idx)
            new_ps = buffer_load(ps_res, idx, vec_width=1, dtype=fx.T.i64()) + fx.Int64(add_ps)
            buffer_store(new_ps, ps_res, idx)

    return epoch_bump_kernel


@functools.lru_cache(maxsize=64)
def _compile(
    out_features,
    hidden_size,
    num_max_pool_tokens,
    BLOCK_M,
    BLOCK_N,
    num_dispatch_cu,
    num_preshuffle_cu,
    num_comm,
    num_ranks,
    G,
    blgp=0,
    nt_vmcnt=3,
    waves_per_eu=2,
    agpr_alloc=0,
    out_fp16=False,
    GROUP_M=4,
    push_only=0,
    gemm_only=0,
):
    K = hidden_size
    N = out_features
    assert num_max_pool_tokens % BLOCK_M == 0, "num_max_pool_tokens must be a multiple of BLOCK_M"
    assert N % BLOCK_N == 0, "out_features must be a multiple of BLOCK_N"
    assert K % 128 == 0 and K >= 256, f"mxfp8 needs K % 128 == 0 and K >= 256, got K={K}"
    assert K % 1024 == 0, f"clean fp8 push needs hidden % 1024 == 0, got K={K}"
    K128 = K // 128
    _push_only = bool(push_only)
    _gemm_only = bool(gemm_only)
    KT_PS = K128 if (K128 % 8 == 0 and 64 * K128 <= 16384) else 8
    assert (64 * KT_PS) % _BLOCK_THREADS == 0, f"PS tile {64 * KT_PS} not divisible by {_BLOCK_THREADS}"
    assert BLOCK_M % 64 == 0, "coalesced preshuffle needs BLOCK_M % 64 == 0"
    _n_ps_chunks = ceildiv(K128, KT_PS)
    _n_ps_groups = BLOCK_M // 64
    TILE_PS = 64 * KT_PS
    SharedStorage = _make_fwd_shared_storage_coalesce(BLOCK_M, BLOCK_N, TILE_PS)
    n_blocks = N // BLOCK_N
    worst_case_tiles = num_max_pool_tokens // BLOCK_M
    _comm_cu = num_dispatch_cu
    _gemm_base = _comm_cu + num_preshuffle_cu  # gemm tiles come after comm+ps
    _grid_size = _gemm_base + worst_case_tiles * n_blocks
    pool_scale_bytes_raw = num_max_pool_tokens * (K // 32)  # raw E8M0 pool region bytes
    _ps_rounds = (worst_case_tiles + num_preshuffle_cu - 1) // num_preshuffle_cu

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def dispatch_grouped_gemm_mxfp8_kernel(
        XQ: fx.Tensor,  # fp8 tokens int32 view [T, K//4] flattened (comm reads)
        XS: fx.Tensor,  # raw E8M0 scales int32 view [T, K//128] (comm reads)
        EXPERT_SEND_DST_RANK: fx.Tensor,
        EXPERT_SEND_DST_ROW: fx.Tensor,
        EXPERT_SEND_COUNT: fx.Tensor,
        EXPERT_SEND_OFFSET: fx.Tensor,
        DISPATCHED_TOKEN_IDX: fx.Tensor,
        sym_layout: SymLayout,
        WEIGHTS: fx.Tensor,  # fp8 weights viewed int8 [G*N*K] flattened
        WEIGHT_SCALE_PS: fx.Tensor,  # host-preshuffled weight E8M0 (ScaleBComb b_sp, int32)
        POOL_SCALE_PS: fx.Tensor,  # local pool E8M0 in ScaleS2R broadcast layout a_sp (int32)
        OUTPUT: fx.Tensor,  # bf16 [num_max_pool_tokens, N] flattened
        TILE_TO_GROUP: fx.Tensor,
        EXPECTED: fx.Tensor,  # (unused after epoch migration; kept for handle-plumbing stability)
        NUM_TILE_BLOCKS: fx.Tensor,
        DISP_PARITY: fx.Tensor,
        DISP_EXPECTED: fx.Tensor,
        PS_EXPECTED: fx.Tensor,
        c_n: fx.Int32,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        comm_block_count = fx.Int32(_comm_cu)
        gemm_base = fx.Int32(_gemm_base)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        # ---- epoch: parity picks the flag bank; expected[parity] is the cumulative spin target ----
        disp_parity_res = create_buffer_resource(DISP_PARITY, max_size=True)
        disp_expected_res = create_buffer_resource(DISP_EXPECTED, max_size=True)
        ps_expected_res = create_buffer_resource(PS_EXPECTED, max_size=True)
        disp_parity = cast(
            buffer_load(disp_parity_res, fx.Int32(0), vec_width=1, dtype=fx.T.i64()), fx.T.i32()
        )
        bank_offset = disp_parity * fx.Int32(worst_case_tiles)
        expected_dispatch = buffer_load(disp_expected_res, disp_parity, vec_width=1, dtype=fx.T.i64())
        expected_ps = buffer_load(ps_expected_res, disp_parity, vec_width=1, dtype=fx.T.i64())
        dispatch_flag_local = sym_layout.dispatch_flag_ptr
        preshuffle_flag_local = sym_layout.preshuffle_flag_ptr

        xq_res = create_buffer_resource(XQ, max_size=True)
        xs_res = create_buffer_resource(XS, max_size=True)
        esr = create_buffer_resource(EXPERT_SEND_DST_RANK, max_size=True)
        esrow = create_buffer_resource(EXPERT_SEND_DST_ROW, max_size=True)
        escnt = create_buffer_resource(EXPERT_SEND_COUNT, max_size=True)
        esoff = create_buffer_resource(EXPERT_SEND_OFFSET, max_size=True)
        dti = create_buffer_resource(DISPATCHED_TOKEN_IDX, max_size=True)
        group_resource = create_buffer_resource(TILE_TO_GROUP, max_size=True)
        num_tile_blocks_resource = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)

        dispatch_tile = dispatch_fp8_copy_tile(
            thread_index=thread_index,
            hidden_size=hidden_size,
            num_max_pool_tokens=num_max_pool_tokens,
            xq_resource=xq_res,
            xs_resource=xs_res,
            expert_send_dst_rank_resource=esr,
            expert_send_dst_row_resource=esrow,
            expert_send_count_resource=escnt,
            expert_send_offset_resource=esoff,
            dispatched_token_idx_resource=dti,
            pool_fp8_base=sym_layout.pool_fp8_ptr,
            pool_scale_base=sym_layout.pool_scale_ptr,  # RAW E8M0 region
            pool_offsets_resource=create_buffer_resource_from_addr(
                sym_layout.offsets_ptr, num_records_bytes=num_ranks * 8
            ),
            dispatch_flag_base=sym_layout.dispatch_flag_ptr,
            dispatch_flag_offsets_resource=create_buffer_resource_from_addr(
                sym_layout.signal_offsets_ptr, num_records_bytes=num_ranks * 8
            ),
            bank=bank_offset,
            world_size=num_ranks,
        )

        pipeline_idx = block_index

        if pipeline_idx < comm_block_count:
            if not _gemm_only:
                # ---- COMM role: this block owns tasks {comm_idx, comm_idx+comm_cu, ...} ----
                comm_idx = pipeline_idx
                local_task_count = (
                    fx.Int32(num_comm) - comm_idx + comm_block_count - fx.Int32(1)
                ) // comm_block_count
                for task_iteration in range(local_task_count):
                    dispatch_tile(comm_idx + task_iteration * comm_block_count, fx.Int32(0), 1)
        elif block_index < gemm_base:
            if not _push_only:
                # ---- PRESHUFFLE role ----
                ps_index = pipeline_idx - comm_block_count
                real_tiles = buffer_load(num_tile_blocks_resource, fx.Int32(0), vec_width=1, dtype=fx.T.i32())
                a_scale_raw_res = create_buffer_resource_from_addr(
                    sym_layout.pool_scale_ptr, num_records_bytes=pool_scale_bytes_raw
                )
                ps_res = create_buffer_resource(POOL_SCALE_PS, max_size=True)
                for _r in range(_ps_rounds):
                    block_m_ps = ps_index + fx.Int32(_r * num_preshuffle_cu)
                    if block_m_ps < real_tiles:
                        expert_ps = buffer_load(group_resource, block_m_ps, vec_width=1, dtype=fx.T.i32())
                        if (not _gemm_only) and thread_index == fx.Int32(0):
                            spin_start = read_clock()
                            sig = ld(dispatch_flag_local, bank_offset + expert_ps, scope="sys", dtype=fx.T.i64())
                            while sig != expected_dispatch:
                                fx.rocdl.s_sleep(fx.Int32(2))
                                if spin_timed_out(spin_start):
                                    fx.printf(
                                        "MEGA mxfp8 preshuffle gate timeout: block={} expert={} sig={} exp={}\n",
                                        block_m_ps, expert_ps, sig, expected_dispatch,
                                    )
                                    spin_start = read_clock()
                                sig = ld(dispatch_flag_local, bank_offset + expert_ps, scope="sys", dtype=fx.T.i64())
                        fx.gpu.barrier()
                        # The transpose doubles as this role's fence (-16.8% when ported from the
                        # bwd fork): rd_cm=1 acquires the peer-pushed pool_scale with a coherent
                        # read instead of a whole-L2 buffer_inv, st_cm=16 releases pool_scale_ps by
                        # writing through, so no device-wide l2_writeback handoff is needed.
                        for _g in range(_n_ps_groups):
                            grp = block_m_ps * fx.Int32(_n_ps_groups) + fx.Int32(_g)
                            for _c in range(_n_ps_chunks):
                                emit_lds_repack(
                                    True, grp, fx.Int32(_c * KT_PS), lds.ps_tile,
                                    a_scale_raw_res, ps_res, num_max_pool_tokens,
                                    K128, KT_PS, thread_index, _BLOCK_THREADS,
                                    rd_cm=1, st_cm=16,
                                )
                                fx.gpu.barrier()
                        fx.rocdl.s_waitcnt(fx.Int32(0))
                        fx.gpu.barrier()
                        if thread_index == fx.Int32(0):
                            st(preshuffle_flag_local, bank_offset + block_m_ps, expected_ps, scope="sys")
        else:
            if not _push_only:
                # ---- GEMM role ----
                tile_index = block_index - gemm_base
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
                    c_m_real = fx.Int32(num_max_pool_tokens)
                    if (not _gemm_only) and thread_index == fx.Int32(0):
                        spin_start = read_clock()
                        signal = ld(preshuffle_flag_local, bank_offset + block_m, scope="sys", dtype=fx.T.i64())
                        while signal != expected_ps:
                            fx.rocdl.s_sleep(fx.Int32(2))
                            if spin_timed_out(spin_start):
                                fx.printf(
                                    "MEGA mxfp8 GEMM gate timeout: block={} signal={} exp={}\n",
                                    block_m, signal, expected_ps,
                                )
                                spin_start = read_clock()
                            signal = ld(preshuffle_flag_local, bank_offset + block_m, scope="sys", dtype=fx.T.i64())
                    fx.gpu.barrier()
                    # ACQUIRE for the peer-pushed pool rows / preshuffled A-scale. `buffer_inv sc1`
                    # invalidates the CU's vector L1 and the XCD's L2, both shared by the whole
                    # workgroup, so ONE lane covers every wave; the barrier then releases the rest.
                    # Per-wave instead costs 2.362 vs 2.169 ms (2.149 incoherent bound) -- the price
                    # is issue serialization on the L2 port, not eviction of the B weights.
                    if thread_index == fx.Int32(0):
                        l2_invalidate()
                        fx.rocdl.s_waitcnt(fx.Int32(0))
                    fx.gpu.barrier()

                    g_idx = buffer_load(group_resource, block_m, vec_width=1, dtype=fx.T.i32())
                    pool_ptr_ty = PointerType.get(
                        elem_ty=fx.T.i8(), address_space=AddressSpace.Global, alignment=16
                    )
                    pool_fp8 = fx.make_view(
                        fx.inttoptr(pool_ptr_ty, sym_layout.pool_fp8_ptr),
                        fx.make_layout(num_max_pool_tokens * K, 1),
                    )
                    gemm_mxfp8_nt_tile(
                        pool_fp8,
                        POOL_SCALE_PS,
                        WEIGHTS,
                        WEIGHT_SCALE_PS,
                        OUTPUT,
                        c_m_real,
                        c_n,
                        lds,
                        block_m,
                        block_n,
                        K=K,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        G=G,
                        group_idx=g_idx,
                        blgp=blgp,
                        out_fp16=out_fp16,
                        nt_vmcnt=nt_vmcnt,
                        scale_pack=1,
                    )

    @flyc.jit
    def launch(
        XQ,
        XS,
        EXPERT_SEND_DST_RANK,
        EXPERT_SEND_DST_ROW,
        EXPERT_SEND_COUNT,
        EXPERT_SEND_OFFSET,
        DISPATCHED_TOKEN_IDX,
        sym_layout,
        WEIGHTS,
        WEIGHT_SCALE_PS,
        POOL_SCALE_PS,
        OUTPUT,
        TILE_TO_GROUP,
        EXPECTED,
        NUM_TILE_BLOCKS,
        DISP_PARITY,
        DISP_EXPECTED,
        PS_EXPECTED,
        c_n: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        # Same-stream ordering is what makes the bumped parity/expected visible to the kernel.
        _make_epoch_bump(int(num_ranks), 1)(DISP_PARITY, DISP_EXPECTED, PS_EXPECTED).launch(
            grid=(1, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream
        )
        dispatch_grouped_gemm_mxfp8_kernel(
            XQ,
            XS,
            EXPERT_SEND_DST_RANK,
            EXPERT_SEND_DST_ROW,
            EXPERT_SEND_COUNT,
            EXPERT_SEND_OFFSET,
            DISPATCHED_TOKEN_IDX,
            sym_layout,
            WEIGHTS,
            WEIGHT_SCALE_PS,
            POOL_SCALE_PS,
            OUTPUT,
            TILE_TO_GROUP,
            EXPECTED,
            NUM_TILE_BLOCKS,
            DISP_PARITY,
            DISP_EXPECTED,
            PS_EXPECTED,
            fx.Int32(c_n),
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(_grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def dispatch_grouped_gemm_mxfp8(
    x: torch.Tensor,
    w1_fp8: tuple,
    handle,
    sym_layout,
    symm,
    *,
    num_dispatch_cu: Optional[int] = None,
    num_preshuffle_cu: Optional[int] = None,
    BM: int = 256,
    BN: int = 256,
    GROUP_M: int = 4,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fused fp8 dispatch PUSH + grouped mxfp8 L1 GEMM (comm/preshuffle/gemm pipeline).

    Rowwise-quantizes the bf16 ``x`` [T, K] in one host-side launch, then runs the
    comm|preshuffle|gemm overlap. Leave ``num_dispatch_cu`` / ``num_preshuffle_cu`` at None to take
    the split from the offline table (``_resolve_cu_split``); pass a pair to pin it. The gates
    self-reset via the device epoch bump (``_make_epoch_bump``), whose tensors ride on ``symm``."""
    assert x.dtype == torch.bfloat16, f"activation must be bf16, got {x.dtype}"
    # All four come from prepare_dispatch_weight_fp8 at the op layer: this kernel derives no weight
    # state and caches none, so it cannot go stale behind an optimizer step.
    w1q, w1s, WEIGHTS, weight_scale_ps = w1_fp8
    (
        expert_send_dst_rank,
        expert_send_dst_row,
        expert_send_count,
        expert_send_offset,
        dispatched_token_idx,
        *_rest,
    ) = handle
    tile_to_expert = handle[7]
    expected_count = handle[8]

    G, N, K = w1q.shape
    T, Kx = x.shape
    assert Kx == K, f"token K={Kx} != weight K={K}"
    num_comm = int(expert_send_dst_rank.numel())
    num_ranks = int(sym_layout.num_ranks)
    num_max_pool_tokens = int(sym_layout.num_max_pool_tokens)
    blgp = 1 if w1q.dtype == torch.float8_e5m2 else 0
    out_fp16 = out_dtype == torch.float16
    c_n = N
    # Breakdown switches: compile only the push leg or only the gemm leg, so a bench can time
    # them separately. Either one on makes the output WRONG.
    push_only = int(os.environ.get("PT_DISPATCH_PUSH_ONLY", "0") == "1")
    gemm_only = int(os.environ.get("PT_DISPATCH_GEMM_ONLY", "0") == "1")

    dev = x.device
    xq, xs = quantize_rowwise_mxfp8_flydsl(x)
    xs = xs.view(torch.float8_e8m0fnu)
    xq_c = xq if xq.is_contiguous() else xq.contiguous()
    xs_c = xs if xs.is_contiguous() else xs.contiguous()
    if xs_c.dtype != torch.uint8:
        xs_c = xs_c.view(torch.uint8)
    XQ = xq_c.view(torch.int32)
    XS = xs_c.view(torch.int32)
    pool_scale_ps = symm.pool_scale_ps  # local broadcast a_sp (preshuffle role writes it)

    # The real-tile count must come off the handle, not the shared symm scratch the prologue wrote
    # it into: a call that reuses a handle (the backward) would otherwise pair its own tile table
    # with whatever count the most recent prologue left there.
    num_tile_blocks = handle[_H_NUM_TILE_BLOCKS]
    # Fresh per call, not shared scratch: l1 is the SwiGLU backward's input, so it must survive to
    # this layer's backward, and sharing lets the next layer's forward overwrite it first. Invisible
    # with one MoE layer -- 2 layers over 20 steps drifted 1.52 in loss against bf16 vs 0.07 for one.
    output = torch.empty((num_max_pool_tokens, N), dtype=out_dtype, device=dev)
    output_flat = output.view(-1)

    # _compile is lru_cached, so re-asking per candidate split costs one compile each, then nothing.
    def _make_raw(dc, pc):
        return _compile(
            N,
            K,
            num_max_pool_tokens,
            BM,
            BN,
            int(dc),
            int(pc),
            int(num_comm),
            int(num_ranks),
            int(G),
            blgp=blgp,
            out_fp16=out_fp16,
            GROUP_M=int(GROUP_M),
            push_only=push_only,
            gemm_only=gemm_only,
        )
    args = (
        XQ,
        XS,
        expert_send_dst_rank,
        expert_send_dst_row,
        expert_send_count,
        expert_send_offset,
        dispatched_token_idx,
        sym_layout,
        WEIGHTS,
        weight_scale_ps,
        pool_scale_ps,
        output_flat,
        tile_to_expert,
        expected_count,
        num_tile_blocks,
        symm._disp_parity,
        symm._disp_expected,
        symm._ps_expected,
        c_n,
        torch.cuda.current_stream(),
    )
    def _run(dc, pc):
        ck = (N, K, num_max_pool_tokens, BM, BN, int(dc), int(pc),
              int(num_comm), int(num_ranks), int(G), blgp, out_fp16, int(GROUP_M),
              push_only, gemm_only)
        compiled = _FUSED_COMPILED.get(ck)
        if compiled is None:
            compiled = flyc.compile(_make_raw(dc, pc), *args)
            _FUSED_COMPILED[ck] = compiled
        compiled(*args)

    cu_key = cu_split_key(N, K, num_max_pool_tokens, BM, BN, num_ranks, G)
    if num_dispatch_cu is None or num_preshuffle_cu is None:
        (num_dispatch_cu, num_preshuffle_cu), cu_src = _resolve_cu_split(cu_key, N)
    else:
        cu_src = "caller"
    _log_cu_split(cu_key, (num_dispatch_cu, num_preshuffle_cu), cu_src)
    _run(num_dispatch_cu, num_preshuffle_cu)
    return output




def dispatch_grouped_gemm_mxfp8_flydsl_kernel(
    x: torch.Tensor,
    w1_fp8: tuple,
    group: ProcessGroup,
    handle: Optional[tuple] = None,
    topk_idx: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    BM: int = 256,
    BN: int = 256,
    num_dispatch_cu: Optional[int] = None,
    num_preshuffle_cu: Optional[int] = None,
) -> Tuple[torch.Tensor, tuple, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Self-contained fp8 dispatch + grouped mxfp8 NT GEMM; fp8 sibling of
    ``dispatch_grouped_gemm_bf16_flydsl_kernel``.

    Drives BOTH the forward L1 (dispatch x + fc1) and the backward L2 dgrad (dispatch dy +
    fc2-dgrad): the same NT op, only the input and weight differ. ``w1_fp8`` is the 4-tuple from
    ``prepare_dispatch_weight_fp8`` -- the fc1 weight for the forward, ``w2^T`` for the dgrad.
    ``handle is None`` marks the forward and builds the symm workspace + prologue handle from
    ``topk_idx``; the backward passes that handle back and must, so its dy retraces x's routing.

    Returns ``(l1, handle, dispatch_weights, pool_x_fp8)``. ``dispatch_weights`` and ``pool_x_fp8``
    are LIVE views into the shared symm pool -- clone them before a later stage overwrites them.
    ``handle[_H_NUM_TILE_BLOCKS]`` is the device real-tile count, which bounds the SwiGLU
    epilogue's rows."""
    w1q = w1_fp8[0]
    if handle is None:
        assert topk_idx is not None, "handle=None requires topk_idx to run the prologue"
        assert group is not None, "handle=None requires group to build the symm workspace"
        G, world = w1q.shape[0], group.size()
        T, H = x.shape
        I = w1q.shape[1] // 2
        K = topk_idx.shape[-1]
        symm = get_symm_buffer_for_mega_moe(
            group, num_experts=G * world, num_max_tokens_per_rank=T, num_topk=K,
            hidden=H, intermediate_hidden=I, block_m=BM, block_n=BN, use_mxfp8=True,
        )
        sym_layout = symm.make_sym_layout()
        handle = extend_handle(
            dispatch_prologue(
                topk_idx, topk_weights, sym_layout=sym_layout, num_tokens=T, num_topk=K,
                num_experts=G * world, world_size=world, rank=symm.rank, experts_per_rank=G,
                block_m=BM, num_max_pool_tokens=symm.num_max_pool_tokens,
            ),
            symm,
        )
    else:
        symm = get_symm_buffer_for_mega_moe()  # live buffer from a prior forward
        sym_layout = symm.make_sym_layout()
    l1 = dispatch_grouped_gemm_mxfp8(
        x, w1_fp8, handle, sym_layout, symm,
        num_dispatch_cu=num_dispatch_cu, num_preshuffle_cu=num_preshuffle_cu, BM=BM, BN=BN,
    )
    _Px, _Hx = symm.pool_fp8.shape
    return l1, handle, symm.weight_recv_buf, (symm.pool_fp8, symm.pool_scale.reshape(_Px, _Hx // 32))
