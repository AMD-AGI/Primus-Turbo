###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused cross-rank dispatch PUSH (fp8) + grouped MXFP8 GEMM (NT), FlyDSL — 3-stage pipeline.

One role-specialized kernel launched on every rank, with a comm -> preshuffle -> gemm
software pipeline gated per pool-block by the sys-scope scoreboard:

  * COMM role (``block_index < num_dispatch_cu``): each block CLEAN-pushes a comm task's
    PRE-QUANTIZED fp8 token rows + their RAW E8M0 block scales into the peer ``pool_fp8`` /
    ``pool_scale`` regions over XGMI (coalesced, XGMI-saturating; no in-push quant), drains
    with a device-scope L2 write-back, then signals the peer per-pool-block scoreboard.
  * PRESHUFFLE role (next ``num_preshuffle_cu`` blocks): each block waits for a pool-block's
    tokens (scoreboard >= expected), invalidates L2 to see the peer-written raw scale,
    transposes that block's A-scale raw->broadcast into the local ``pool_scale_ps`` ONCE
    (non-redundant), writes it back, then stamps a SENTINEL on the scoreboard.
  * GEMM role (remaining blocks): each computes ONE NT output tile of the grouped L1 GEMM
    (A = ``pool_fp8`` + ``pool_scale_ps`` broadcast E8M0, per-expert B = ``weight_fp8`` +
    host-preshuffled ``weight_scale``) via ``gemm_mxfp8_nt_tile`` (ScaleS2R / ScaleBComb,
    fast MMA), spinning until its pool-block's SENTINEL is set.

Comm / preshuffle / gemm all overlap; the scoreboard sys-scope acquire/release + device-
scope L2 fences carry cross-rank/cross-XCD visibility (no host sync + standalone L2
invalidate). Tokens are quantized ONCE on the source before the push.

NT only. Constraints: hidden % 1024 == 0 (fp8 warp push), N % BLOCK_N == 0,
num_max_pool_tokens % BLOCK_M == 0, K % 128 == 0 and K >= 256 (mxfp8 MMA).
"""

import functools
import os
from typing import Optional, Tuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from torch.distributed import ProcessGroup
from flydsl.expr import arith
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
)
from flydsl.expr.typing import AddressSpace, PointerType

from primus_turbo.flydsl.mega.fp8.ep_fp8 import (
    _BLOCK_THREADS,
    dispatch_fp8_copy_tile,
    preshuffle_a_scale_tile,
)
from primus_turbo.flydsl.mega.fp8.gemm_mxfp8_tile import (
    BLOCK_K,
    gemm_mxfp8_nt_tile,
    make_mxfp8_shared_storage,
)
from primus_turbo.flydsl.mega.fp8.quant_flydsl import (
    preshuffle_b_scale,
    quantize_rowwise_mxfp8_flydsl,
)
from primus_turbo.flydsl.mega.prims import cast
from primus_turbo.flydsl.mega.fp8.prims import (
    l2_invalidate,
    l2_writeback,
    ld,
    read_clock,
    spin_timed_out,
    st,
)
from primus_turbo.flydsl.mega.fp8.sym_layout import SymLayout
from primus_turbo.flydsl.mega.fp8.gemm_helper import _emit_lds_repack, ceildiv, make_value_attrs
from primus_turbo.flydsl.mega.fp8.dispatch_prologue import dispatch_prologue
from primus_turbo.flydsl.mega.fp8.symm_buffer import get_symm_buffer_for_mega_moe

_FUSED_COMPILED: dict = {}  # (shape key) -> flyc.compile'd launch (eager; skip per-call @flyc.jit dispatch)
_BSP_CACHE: dict = {}  # (weight data_ptr, G, N, K) -> preshuffled weight scale b_sp (weights static)
_PS_SENTINEL = 1 << 20  # scoreboard value the preshuffle role stamps when a pool-block is ready


def _make_fwd_shared_storage_coalesce(BLOCK_M, BLOCK_N, tile_ps):
    """fp8 ping-pong (== make_mxfp8_shared_storage, gemm role) + a ps_tile int32 scratch for the
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
    cbsz=0,
    blgp=0,
    nt_vmcnt=3,
    waves_per_eu=2,
    agpr_alloc=0,
    out_fp16=False,
    GROUP_M=4,
    ps_read_cm=1,
    ps_coalesce=1,
    ps_release=1,
):
    K = hidden_size
    N = out_features
    assert num_max_pool_tokens % BLOCK_M == 0, "num_max_pool_tokens must be a multiple of BLOCK_M"
    assert N % BLOCK_N == 0, "out_features must be a multiple of BLOCK_N"
    assert K % 128 == 0 and K >= 256, f"mxfp8 needs K % 128 == 0 and K >= 256, got K={K}"
    assert K % 1024 == 0, f"clean fp8 push needs hidden % 1024 == 0, got K={K}"
    K128 = K // 128
    # Preshuffle-role fence optimization (ported from the bwd fork R2/R3), DEFAULT ON:
    #   ps_read_cm=1  : ACQUIRE via glc coherent read of the peer-pushed raw pool_scale (skip buffer_inv).
    #   ps_coalesce=1 : write pool_scale_ps via the coalesced LDS transpose (_emit_lds_repack, b128).
    #   ps_release=1  : sc1 WRITE-THROUGH release + DROP the whole-L2 l2_writeback (gemm l2_invalidate
    #                   still acquires it). See the bwd fork for the profiler evidence (-16.8% there).
    PS_COALESCE = bool(ps_coalesce)
    PS_RELEASE = bool(ps_release)
    KT_PS = K128 if (K128 % 8 == 0 and 64 * K128 <= 16384) else 8
    assert (64 * KT_PS) % _BLOCK_THREADS == 0, f"PS tile {64 * KT_PS} not divisible by {_BLOCK_THREADS}"
    assert BLOCK_M % 64 == 0, "coalesced preshuffle needs BLOCK_M % 64 == 0"
    _n_ps_chunks = ceildiv(K128, KT_PS)
    _n_ps_groups = BLOCK_M // 64
    TILE_PS = 64 * KT_PS
    SharedStorage = (
        _make_fwd_shared_storage_coalesce(BLOCK_M, BLOCK_N, TILE_PS)
        if PS_COALESCE else make_mxfp8_shared_storage(BLOCK_M, BLOCK_N)
    )
    n_blocks = N // BLOCK_N
    worst_case_tiles = num_max_pool_tokens // BLOCK_M
    _comm_cu = num_dispatch_cu
    _gemm_base = _comm_cu + num_preshuffle_cu  # gemm tiles start after comm + preshuffle roles
    _grid_size = _gemm_base + worst_case_tiles * n_blocks
    pool_scale_bytes_raw = num_max_pool_tokens * (K // 32)  # raw E8M0 pool region bytes
    _ps_rounds = (worst_case_tiles + num_preshuffle_cu - 1) // num_preshuffle_cu

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def dispatch_grouped_gemm_mxfp8_kernel(
        XQ: fx.Tensor,  # pre-quantized fp8 tokens int32 view [T, K//4] flattened
        XS: fx.Tensor,  # raw E8M0 scales int32 view [T, K//128] flattened
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
        expected_resource = create_buffer_resource(EXPECTED, max_size=True)
        num_tile_blocks_resource = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)

        # COMM role closure: clean-push pre-quantized fp8 + RAW scale to the peer pool, + signal.
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
            signal=True,
            dispatch_flag_base=sym_layout.dispatch_flag_ptr,
            dispatch_flag_offsets_resource=create_buffer_resource_from_addr(
                sym_layout.signal_offsets_ptr, num_records_bytes=num_ranks * 8
            ),
            bank=bank_offset,
            world_size=num_ranks,
        )

        if block_index < comm_block_count:
            # COMM: this block owns comm tasks {block_index, block_index+comm_cu, ...}.
            local_task_count = (
                fx.Int32(num_comm) - block_index + comm_block_count - fx.Int32(1)
            ) // comm_block_count
            for task_iteration in range(local_task_count):
                dispatch_tile(block_index + task_iteration * comm_block_count, fx.Int32(0), 1)
        elif block_index < fx.Int32(_gemm_base):
            # PRESHUFFLE role: this block owns pool-blocks {ps_index, ps_index+ps_cu, ...}.
            # For each: gate on the comm scoreboard (tokens arrived), invalidate to see the
            # peer-written raw pool_scale, transpose that block's A-scale raw->broadcast into
            # pool_scale_ps (ONCE, non-redundant), write it back, then stamp a SENTINEL so the
            # block's gemm tiles can proceed.
            ps_index = block_index - comm_block_count
            real_tiles = buffer_load(num_tile_blocks_resource, fx.Int32(0), vec_width=1, dtype=fx.T.i32())
            a_scale_raw_res = create_buffer_resource_from_addr(
                sym_layout.pool_scale_ptr, num_records_bytes=pool_scale_bytes_raw
            )
            ps_res = create_buffer_resource(POOL_SCALE_PS, max_size=True)
            for _r in range(_ps_rounds):
                block_m_ps = ps_index + fx.Int32(_r * num_preshuffle_cu)
                if block_m_ps < real_tiles:
                    # epoch gate: wait this block's expert to receive all num_ranks arrivals
                    # (dispatch_flag[bank + expert] == cumulative expected). expert = tile_to_group.
                    expert_ps = buffer_load(group_resource, block_m_ps, vec_width=1, dtype=fx.T.i32())
                    if thread_index == fx.Int32(0):
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
                    # ACQUIRE the peer-pushed raw pool_scale. Default ps_read_cm=1: glc coherent read
                    # (in the transpose loads below), skip buffer_inv (comm already l2_writeback'd the
                    # push to the coherent point before the sys signal). ps_read_cm=0 -> buffer_inv.
                    if ps_read_cm == 0:
                        l2_invalidate()
                    fx.gpu.barrier()
                    if PS_COALESCE:
                        # coalesced LDS transpose raw pool_scale -> broadcast pool_scale_ps (b128 both
                        # sides). st_cm=16 (sc1 write-through) publishes to the coherent point when
                        # PS_RELEASE, so the whole-L2 l2_writeback below is dropped.
                        _ps_stcm = 16 if PS_RELEASE else 0
                        for _g in range(_n_ps_groups):
                            grp = block_m_ps * fx.Int32(_n_ps_groups) + fx.Int32(_g)
                            for _c in range(_n_ps_chunks):
                                _emit_lds_repack(
                                    True, grp, fx.Int32(_c * KT_PS), lds.ps_tile,
                                    a_scale_raw_res, ps_res, num_max_pool_tokens,
                                    K128, KT_PS, thread_index, _BLOCK_THREADS,
                                    rd_cm=ps_read_cm, st_cm=_ps_stcm,
                                )
                                fx.gpu.barrier()  # ps_tile reused next (grp,chunk): finish LDS reads
                    else:
                        preshuffle_a_scale_tile(
                            a_scale_raw_res, ps_res, block_m_ps * fx.Int32(BLOCK_M),
                            BLOCK_M, K128, thread_index, _BLOCK_THREADS,
                            read_cache_modifier=ps_read_cm,
                        )
                    fx.rocdl.s_waitcnt(fx.Int32(0))
                    # RELEASE: default = coalesced sc1 write-through (published above); else whole-L2
                    # l2_writeback. The gemm role's l2_invalidate acquire is UNCHANGED.
                    if not (PS_COALESCE and PS_RELEASE):
                        l2_writeback()
                    fx.gpu.barrier()
                    if thread_index == fx.Int32(0):
                        # epoch handoff to gemm: preshuffle_flag[bank + block] = cumulative expected_ps
                        st(preshuffle_flag_local, bank_offset + block_m_ps, expected_ps, scope="sys")
        else:
            # GEMM role: one NT output tile (block_m, block_n) of the grouped L1 GEMM.
            tile_index = block_index - fx.Int32(_gemm_base)
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
                # wait for the preshuffle role's epoch handoff (block_m's A-scale is in pool_scale_ps).
                if thread_index == fx.Int32(0):
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
                l2_invalidate()  # acquire: see peer-written pool_fp8 + role-written pool_scale_ps
                fx.gpu.barrier()

                g_idx = buffer_load(group_resource, block_m, vec_width=1, dtype=fx.T.i32())
                pool_ptr_ty = PointerType.get(
                    elem_ty=fx.T.i8(), address_space=AddressSpace.Global, alignment=16
                )
                pool_fp8 = fx.make_view(
                    fx.inttoptr(pool_ptr_ty, sym_layout.pool_fp8_ptr),
                    fx.make_layout(num_max_pool_tokens * K, 1),
                )
                # A from the local broadcast pool_scale_ps (preshuffle-role-written), B from the
                # host-preshuffled weight scale -> ScaleS2R / ScaleBComb (fast MMA load).
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
                    cbsz=cbsz,
                    blgp=blgp,
                    out_fp16=out_fp16,
                    nt_vmcnt=nt_vmcnt,
                    preshuffled=True,
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
        # bump epoch on device (dispatch += num_ranks, preshuffle += 1) before the kernel;
        # same-stream ordering makes the bumped parity/expected visible to the kernel.
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
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(_grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def dispatch_grouped_gemm_mxfp8(
    xq: torch.Tensor,
    xs: torch.Tensor,
    w1q: torch.Tensor,
    w1s: torch.Tensor,
    handle,
    sym_layout,
    symm,
    *,
    num_dispatch_cu: int = 16,
    num_preshuffle_cu: int = 16,
    BM: int = 256,
    BN: int = 256,
    GROUP_M: int = 4,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fused fp8 dispatch PUSH + grouped mxfp8 L1 GEMM (3-stage comm/preshuffle/gemm pipeline).

    Token quant lives HERE: pass a bf16 activation as ``xq`` with ``xs=None`` and this op does ONE
    global rowwise mxfp8 quant (T rows, a separate launch on this stream -> naturally ordered
    before the push, no explicit sync) then runs the pipeline. This is a single global quant, NOT
    a per-push quant (a token routed to K experts is quantized once, pushed K times). Pass
    pre-quantized ``xq`` [T, K] fp8 + ``xs`` [T, K//32] raw E8M0 to skip the internal quant.
    The comm role clean-pushes them (coalesced) to the peer ``pool_fp8`` / ``pool_scale``; a
    preshuffle role transposes each pool-block's A-scale raw->broadcast into ``pool_scale_ps``
    once; the gemm role reads ``pool_scale_ps`` + the host-preshuffled ``w1s`` (ScaleS2R /
    ScaleBComb). ``w1q`` [G, N, K] fp8 weights + ``w1s`` [G, N, K//32] raw E8M0.
    Returns L1 out [num_max_pool_tokens, N] bf16.

    Self-resetting: the comm->preshuffle (``dispatch_flag``) and preshuffle->gemm
    (``preshuffle_flag``) gates are double-banked + device epoch-bumped, so no host scoreboard
    reset / rendezvous is needed (the epoch tensors ride on ``symm``)."""
    # bf16 activation -> one global rowwise mxfp8 quant here (matches the per-forward cost living
    # inside this op); pre-quantized fp8 tokens skip it.
    if xq.dtype == torch.bfloat16:
        assert xs is None, "bf16 activation path computes xs internally; pass xs=None"
        xq, xs = quantize_rowwise_mxfp8_flydsl(xq)
        xs = xs.view(torch.float8_e8m0fnu)
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
    T, Kx = xq.shape
    assert Kx == K, f"token K={Kx} != weight K={K}"
    assert xq.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), "fused kernel takes pre-quantized fp8 tokens"
    num_comm = int(expert_send_dst_rank.numel())
    num_ranks = int(sym_layout.num_ranks)
    num_max_pool_tokens = int(sym_layout.num_max_pool_tokens)
    cbsz = 1 if xq.dtype == torch.float8_e5m2 else 0
    blgp = 1 if w1q.dtype == torch.float8_e5m2 else 0
    out_fp16 = out_dtype == torch.float16
    c_n = N
    # Preshuffle-role fence optimization (bwd-fork R2/R3), DEFAULT ON. glc coherent acquire + coalesced
    # write-through release (drop whole-L2 l2_writeback). Set PT_MXFP8_PS_COALESCE=0 / PS_RELEASE=0 /
    # PS_READ_CM=0 to restore the scattered write / whole-L2 writeback / buffer_inv acquire.
    ps_read_cm = int(os.environ.get("PT_MXFP8_PS_READ_CM", "1"))
    ps_coalesce = int(os.environ.get("PT_MXFP8_PS_COALESCE", "1"))
    ps_release = int(os.environ.get("PT_MXFP8_PS_RELEASE", "1"))

    XQ = xq.contiguous().view(torch.uint8).view(torch.int32).reshape(-1)
    XS = xs.contiguous().view(torch.uint8).view(torch.int32).reshape(-1)
    WEIGHTS = w1q.contiguous().reshape(G * N, K).view(torch.int8).reshape(-1)
    # weights are static -> preshuffle the weight scale ONCE (ScaleBComb b_sp), cached.
    _bk = (w1s.data_ptr(), G, N, K)
    weight_scale_ps = _BSP_CACHE.get(_bk)
    if weight_scale_ps is None:
        weight_scale_ps = preshuffle_b_scale(w1s, G, N, K)
        _BSP_CACHE[_bk] = weight_scale_ps
    pool_scale_ps = symm.pool_scale_ps  # local broadcast a_sp (preshuffle role writes it)

    num_tile_blocks = symm.meta_scalars[1:2]
    output = torch.empty((num_max_pool_tokens, N), dtype=out_dtype, device=xq.device)
    output_flat = output.contiguous().view(-1)

    raw = _compile(
        N,
        K,
        num_max_pool_tokens,
        BM,
        BN,
        int(num_dispatch_cu),
        int(num_preshuffle_cu),
        int(num_comm),
        int(num_ranks),
        int(G),
        cbsz=cbsz,
        blgp=blgp,
        out_fp16=out_fp16,
        GROUP_M=int(GROUP_M),
        ps_read_cm=ps_read_cm,
        ps_coalesce=ps_coalesce,
        ps_release=ps_release,
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
    ck = (N, K, num_max_pool_tokens, BM, BN, int(num_dispatch_cu), int(num_preshuffle_cu),
          int(num_comm), int(num_ranks), int(G), cbsz, blgp, out_fp16, int(GROUP_M),
          ps_read_cm, ps_coalesce, ps_release)
    if torch.cuda.is_current_stream_capturing():
        raw(*args)
    else:
        compiled = _FUSED_COMPILED.get(ck)
        if compiled is None:
            compiled = flyc.compile(raw, *args)
            _FUSED_COMPILED[ck] = compiled
        compiled(*args)
    return output


def _host_rendezvous(group) -> None:
    """Cross-rank publish barrier: drain this rank's GPU work, then all-rank barrier, so a
    scoreboard/flag reset is visible on every peer before any rank signals it. (Full mode;
    the source op gates these behind PT_MEGA_BARRIER_MODE -- kept always-on here for safety.)"""
    torch.cuda.synchronize()
    group.barrier()


def dispatch_grouped_gemm_mxfp8_flydsl_kernel(
    x: torch.Tensor,
    w1q: torch.Tensor,
    w1s: torch.Tensor,
    group: ProcessGroup,
    handle: Optional[tuple] = None,
    topk_idx: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    BM: int = 256,
    BN: int = 256,
    num_dispatch_cu: int = 16,
    num_preshuffle_cu: int = 16,
) -> Tuple[torch.Tensor, tuple, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Self-contained fp8 dispatch + grouped mxfp8 NT GEMM; fp8 sibling of
    ``dispatch_grouped_gemm_bf16_flydsl_kernel``. Drives BOTH the forward L1 (dispatch x + fc1) and
    the backward STEP1 (dispatch dy + fc2-dgrad) -- both are the SAME NT op, only the input/weight
    and the comm/preshuffle CU split (``num_dispatch_cu`` / ``num_preshuffle_cu``) differ.

    Takes the pre-quantized weight (``w1q`` [G,*,K] fp8 + ``w1s`` raw E8M0; prepared version-keyed by
    the caller -- fc1 weight for forward, ``w2^T`` for the STEP1 dgrad). When ``handle is None``
    (forward), builds the symmetric workspace + dispatch-prologue handle from ``topk_idx`` /
    ``topk_weights``; otherwise reuses the live symm buffer + the given handle (backward). Runs the
    fused dispatch-PUSH + grouped mxfp8 GEMM (token quant folded in via the bf16-x path); the comm
    gates self-reset via the device epoch (no host scoreboard rendezvous).

    Returns ``(l1, handle, dispatch_weights, pool_x_fp8)`` where ``l1`` is the GEMM output (fc1 out
    for forward, grad_swiglu for STEP1), ``dispatch_weights`` is ``symm.weight_recv_buf`` (per-pool-row
    routing weight; unused by STEP1), and ``pool_x_fp8`` is ``(symm.pool_fp8 [P,H] fp8, symm.pool_scale
    [P,H//32] E8M0)`` -- both LIVE views into the shared symm pool (no clone). The caller keeps
    ``handle`` (L2 + backward reuse it); ``handle[-1]`` is the device ``num_tile_blocks`` (real-tile
    count), the SwiGLU-epilogue row bound (mirrors bf16's ``handle[_H_NUM_TILE_BLOCKS]``). It can
    re-fetch the live symm buffer via ``get_symm_buffer_for_mega_moe()`` (e.g. the L2 combine flag reset).
    """
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
        handle = tuple(
            dispatch_prologue(
                topk_idx, topk_weights, sym_layout=sym_layout, num_tokens=T, num_topk=K,
                num_experts=G * world, world_size=world, rank=symm.rank, experts_per_rank=G,
                block_m=BM, num_max_pool_tokens=symm.num_max_pool_tokens,
            )
        ) + (symm.meta_scalars[1:2],)  # handle[-1] = num_tile_blocks (device real-tile count)
    else:
        symm = get_symm_buffer_for_mega_moe()  # live buffer from a prior forward
        sym_layout = symm.make_sym_layout()
    # epoch self-reset: dispatch_flag/preshuffle_flag are double-banked + device epoch-bumped, so
    # NO host rendezvous + scoreboard zero (that per-call synchronize()+barrier() is gone).
    l1 = dispatch_grouped_gemm_mxfp8(
        x, None, w1q, w1s, handle, sym_layout, symm,
        num_dispatch_cu=num_dispatch_cu, num_preshuffle_cu=num_preshuffle_cu, BM=BM, BN=BN,
    )
    # backward saves (clone BEFORE backward STEP1's dispatch(dy) overwrites the symm pool):
    #  * dispatch_weights: the per-pool-row routing weight (prologue-scattered) -- swiglu_backward
    #    re-injects it as the SwiGLU^T scale + gate grad.
    #  * pool_x_fp8: THIS dispatched fc1-input pool in native rowwise-fp8 -- lets dW1 be a LOCAL
    #    variable-K wgrad (grad_l1^T @ pool_x) with NO cross-rank re-dispatch (mirrors dW2's reuse
    #    of the STEP1 pool). fp8 (1B) [P,H] + E8M0 [P,H//32].
    _Px, _Hx = symm.pool_fp8.shape
    return l1, handle, symm.weight_recv_buf, (symm.pool_fp8, symm.pool_scale.reshape(_Px, _Hx // 32))
