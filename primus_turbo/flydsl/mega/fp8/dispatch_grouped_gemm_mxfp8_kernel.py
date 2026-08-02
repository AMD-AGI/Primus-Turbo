###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused cross-rank dispatch PUSH (fp8) + grouped MXFP8 GEMM (NT), FlyDSL — 4-stage grid.

One role-specialized kernel launched on every rank:

  * SETUP role (first ``num_setup_cu`` blocks): rowwise mxfp8 quant of bf16 activations
    into scratch ``xq``/``xs`` and (when needed) weight B-scale preshuffle into
    ``weight_scale_ps``. A device-scope gate releases the pipeline only after every setup
    block finishes — no separate host quant / ``preshuffle_b_scale`` launches.
  * Then comm -> preshuffle -> gemm software pipeline gated per pool-block by the sys-scope
    scoreboard:

  * COMM role: each block CLEAN-pushes a comm task's
    PRE-QUANTIZED fp8 token rows + their RAW E8M0 block scales into the peer ``pool_fp8`` /
    ``pool_scale`` regions over XGMI (coalesced, XGMI-saturating; no in-push quant), drains
    with a device-scope L2 write-back, then signals the peer per-pool-block scoreboard.
  * PRESHUFFLE role (next ``num_preshuffle_cu`` blocks): each block waits for a pool-block's
    tokens (scoreboard >= expected), invalidates L2 to see the peer-written raw scale,
    transposes that block's A-scale raw->broadcast into the local ``pool_scale_ps`` ONCE
    (non-redundant), writes it back, then stamps a SENTINEL on the scoreboard.
  * GEMM role (remaining blocks): each computes ONE NT output tile of the grouped L1 GEMM
    (A = ``pool_fp8`` + ``pool_scale_ps`` broadcast E8M0, per-expert B = ``weight_fp8`` +
    preshuffled ``weight_scale``) via ``gemm_mxfp8_nt_tile`` (ScaleS2R / ScaleBComb,
    fast MMA), spinning until its pool-block's SENTINEL is set.

Comm / preshuffle / gemm all overlap after setup completes; the scoreboard sys-scope
acquire/release + device-scope L2 fences carry cross-rank/cross-XCD visibility (no host
sync + standalone L2 invalidate). Tokens are quantized ONCE on the source before the push.

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
from flydsl.expr import arith, range_constexpr
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
)
from primus_turbo.flydsl.mega.fp8.gemm_mxfp8_tile import (
    BLOCK_K,
    gemm_mxfp8_nt_tile,
)
from primus_turbo.flydsl.mega.fp8.quant import (
    _BLK as _QUANT_BLK,
    _quant_block_words,
    preshuffle_b_scale,
    quantize_rowwise_mxfp8_flydsl,
)
from primus_turbo.flydsl.mega.fp8.barrier import grid_sync
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
from primus_turbo.flydsl.mega.fp8.gemm_helper import (
    _PRESHUF_KT,
    _emit_lds_repack,
    ceildiv,
    make_value_attrs,
)
from primus_turbo.flydsl.mega.fp8.dispatch_prologue import dispatch_prologue
from primus_turbo.flydsl.mega.fp8.symm_buffer import get_symm_buffer_for_mega_moe

_FUSED_COMPILED: dict = {}  # (shape key) -> flyc.compile'd launch (eager; skip per-call @flyc.jit dispatch)
_BSP_CACHE: dict = {}  # (weight data_ptr, G, N, K) -> preshuffled weight scale b_sp (weights static)
_WEIGHTS_FLAT_CACHE: dict = {}  # (w1q data_ptr, G, N, K) -> int8 flat WEIGHTS view (weights static)
_L1_OUTPUT_SCRATCH: dict = {}  # (P, N, dtype, device) -> bf16 L1 output buffer (fixed training shape)
_XQ_SCRATCH: dict = {}  # (T, K, device) -> fp8 token scratch for fused setup quant
_XS_SCRATCH: dict = {}  # (T, K, device) -> raw E8M0 scale scratch for fused setup quant
_BSP_ALLOC: dict = {}  # (w1s data_ptr, G, N, K, pack) -> int32 b_sp buffer (filled by setup role)
_SETUP_GATE: dict = {}  # device -> int32[2] [counter, done] for setup->pipeline release


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
    num_setup_cu,
    num_comm,
    num_ranks,
    G,
    num_tokens,
    cbsz=0,
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
    _setup_cu = num_setup_cu
    _comm_cu = num_dispatch_cu
    _pipeline_base = _setup_cu
    _gemm_base = _pipeline_base + _comm_cu + num_preshuffle_cu  # gemm tiles after setup+comm+ps
    _grid_size = _gemm_base + worst_case_tiles * n_blocks
    pool_scale_bytes_raw = num_max_pool_tokens * (K // 32)  # raw E8M0 pool region bytes
    _ps_rounds = (worst_case_tiles + num_preshuffle_cu - 1) // num_preshuffle_cu
    _quant_n_blk = K // _QUANT_BLK
    _quant_k_fp8_i32 = K // 4
    _quant_blk_i32 = _QUANT_BLK // 4
    GN = G * N
    K128 = K // 128
    _b_ps_n_kt = ceildiv(K128, _PRESHUF_KT)
    _b_ps_ngrp = ((GN + 255) // 256) * 4
    _b_ps_blocks = _b_ps_ngrp * _b_ps_n_kt
    _b_raw_bytes = GN * K128 * 4
    _b_sp_bytes = _b_ps_ngrp * K128 * 256 * 4
    _x_bf16_bytes = num_tokens * K * 2
    _xq_bytes = num_tokens * K
    _xs_bytes = num_tokens * (K // 32)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def dispatch_grouped_gemm_mxfp8_kernel(
        X_BF16: fx.Tensor,  # bf16 activations [T, K] (dummy when pre-quantized)
        W1S_RAW: fx.Tensor,  # raw weight E8M0 int32 view [G*N, K128]
        XQ: fx.Tensor,  # fp8 tokens int32 view [T, K//4] flattened (setup writes, comm reads)
        XS_U8: fx.Tensor,  # raw E8M0 scales uint8 [T, K//32] flat (setup quant writes)
        XS: fx.Tensor,  # same storage int32 view [T, K//128] (comm reads)
        SETUP_GATE: fx.Tensor,  # i32[4]: [counter, done, needs_x, needs_b]
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
        setup_base = fx.Int32(_setup_cu)
        comm_block_count = fx.Int32(_comm_cu)
        pipeline_base = fx.Int32(_pipeline_base)
        gemm_base = fx.Int32(_gemm_base)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        setup_gate_res = create_buffer_resource(SETUP_GATE, max_size=True)

        # ---- SETUP role: x quant + weight B-scale preshuffle, then release pipeline ----
        if block_index < setup_base:
            setup_idx = block_index
            needs_x = buffer_load(setup_gate_res, fx.Int32(2), vec_width=1, dtype=fx.T.i32())
            needs_b = buffer_load(setup_gate_res, fx.Int32(3), vec_width=1, dtype=fx.T.i32())
            if (needs_x == fx.Int32(0)) and (needs_b == fx.Int32(0)):
                if block_index == fx.Int32(0) and thread_index == fx.Int32(0):
                    buffer_store(fx.Int32(1), setup_gate_res, fx.Int32(1))
                return
            if needs_x != fx.Int32(0):
                xr = create_buffer_resource(X_BF16, max_size=True)
                qr = create_buffer_resource(XQ, max_size=True)
                sr = create_buffer_resource(XS_U8, max_size=True)
                row = setup_idx
                while row < fx.Int32(num_tokens):
                    b = thread_index
                    while b < fx.Int32(_quant_n_blk):
                        base = row * fx.Int32(K) + b * fx.Int32(_QUANT_BLK)
                        words, biased = _quant_block_words(xr, base)
                        buffer_store(
                            fx.arith.ArithValue(biased).trunci(fx.T.i8()),
                            sr,
                            row * fx.Int32(_quant_n_blk) + b,
                        )
                        base_i32 = row * fx.Int32(_quant_k_fp8_i32) + b * fx.Int32(_quant_blk_i32)
                        for wi in range_constexpr(_quant_blk_i32):
                            buffer_store(words[wi], qr, base_i32 + fx.Int32(wi))
                        b = b + fx.Int32(_BLOCK_THREADS)
                    row = row + setup_base
                fx.rocdl.s_waitcnt(fx.Int32(0))
            fx.gpu.barrier()
            if needs_b != fx.Int32(0):
                b_raw_res = create_buffer_resource(
                    W1S_RAW, max_size=False, num_records_bytes=_b_raw_bytes
                )
                b_sp_res = create_buffer_resource(
                    WEIGHT_SCALE_PS, max_size=False, num_records_bytes=_b_sp_bytes
                )
                bb = setup_idx
                while bb < fx.Int32(_b_ps_blocks):
                    grp = bb // fx.Int32(_b_ps_n_kt)
                    k0 = (bb % fx.Int32(_b_ps_n_kt)) * fx.Int32(_PRESHUF_KT)
                    _emit_lds_repack(
                        False,
                        grp,
                        k0,
                        lds.ps_tile,
                        b_raw_res,
                        b_sp_res,
                        fx.Int32(GN),
                        K128,
                        _PRESHUF_KT,
                        thread_index,
                        _BLOCK_THREADS,
                        pack=1,
                    )
                    fx.gpu.barrier()
                    bb = bb + setup_base
            if _setup_cu > 1:
                grid_sync(sym_layout, thread_index, block_index, _setup_cu, -1, "mxfp8/setup")
            else:
                fx.gpu.barrier()
            l2_writeback()
            fx.gpu.barrier()
            if block_index == fx.Int32(0) and thread_index == fx.Int32(0):
                buffer_store(fx.Int32(1), setup_gate_res, fx.Int32(1))
            fx.gpu.barrier()
            return

        # Pipeline blocks wait until setup releases the comm|ps|gemm overlap region.
        if thread_index == fx.Int32(0):
            spin_start = read_clock()
            done = buffer_load(setup_gate_res, fx.Int32(1), vec_width=1, dtype=fx.T.i32())
            while done != fx.Int32(1):
                fx.rocdl.s_sleep(fx.Int32(2))
                if spin_timed_out(spin_start):
                    fx.printf("MEGA mxfp8 setup gate timeout: block={}\n", block_index)
                    spin_start = read_clock()
                done = buffer_load(setup_gate_res, fx.Int32(1), vec_width=1, dtype=fx.T.i32())
        fx.gpu.barrier()

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

        pipeline_idx = block_index - pipeline_base

        if pipeline_idx < comm_block_count:
            if not _gemm_only:
                # COMM: this block owns comm tasks {comm_idx, comm_idx+comm_cu, ...}.
                comm_idx = pipeline_idx
                local_task_count = (
                    fx.Int32(num_comm) - comm_idx + comm_block_count - fx.Int32(1)
                ) // comm_block_count
                for task_iteration in range(local_task_count):
                    dispatch_tile(comm_idx + task_iteration * comm_block_count, fx.Int32(0), 1)
        elif block_index < gemm_base:
            if not _push_only:
                # PRESHUFFLE role: each block waits for comm, transposes A-scale, signals gemm.
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
                        # The coalesced LDS transpose doubles as this role's fence (ported from the
                        # bwd fork R2/R3, -16.8% there): rd_cm=1 acquires the peer-pushed raw
                        # pool_scale with a glc coherent read instead of a whole-L2 buffer_inv, and
                        # st_cm=16 releases pool_scale_ps by writing through to the coherent point,
                        # so no device-wide l2_writeback handoff is needed -- the GEMM role's own
                        # l2_invalidate is what acquires it.
                        for _g in range(_n_ps_groups):
                            grp = block_m_ps * fx.Int32(_n_ps_groups) + fx.Int32(_g)
                            for _c in range(_n_ps_chunks):
                                _emit_lds_repack(
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
                    # ACQUIRE for the peer-pushed pool rows / preshuffled A-scale (~4096 workgroups
                    # per launch): `buffer_inv sc1` invalidates the issuing CU's vector L1 and the
                    # XCD's L2, both of which the whole workgroup shares, so one lane covers every
                    # wave. Wait for it to land, then release the others via the barrier. Issuing it
                    # per wave instead spends ~90% of the acquire's cost on the redundant copies
                    # (2.362 -> 2.169 ms here, against a 2.149 ms incoherent bound): the price is
                    # issue serialization on the L2 port, NOT the invalidate evicting the B weights
                    # this tile is about to stream -- that theory was tested by making the acquire
                    # rarer (fatter workgroups, fewer tiles each) and lost more to pipeline
                    # serialization than it saved.
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
                        cbsz=cbsz,
                        blgp=blgp,
                        out_fp16=out_fp16,
                        nt_vmcnt=nt_vmcnt,
                        scale_pack=1,
                    )

    @flyc.jit
    def launch(
        X_BF16,
        W1S_RAW,
        XQ,
        XS_U8,
        XS,
        SETUP_GATE,
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
            X_BF16,
            W1S_RAW,
            XQ,
            XS_U8,
            XS,
            SETUP_GATE,
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
    """Fused fp8 dispatch PUSH + grouped mxfp8 L1 GEMM (setup + comm/preshuffle/gemm pipeline).

    Pass a bf16 activation as ``xq`` with ``xs=None``: the kernel's SETUP role quantizes all T
    rows into scratch ``xq``/``xs`` before the comm|preshuffle|gemm overlap begins (no separate
    host quant launch). Pass pre-quantized ``xq`` [T, K] fp8 + ``xs`` [T, K//32] raw E8M0 to skip
    setup quant. Weight B-scale preshuffle (ScaleBComb) also runs in SETUP when the weight version
    is new; static weights are cached host-side after the first preshuffle.

    Self-resetting: the comm->preshuffle (``dispatch_flag``) and preshuffle->gemm
    (``preshuffle_flag``) gates are double-banked + device epoch-bumped, so no host scoreboard
    reset / rendezvous is needed (the epoch tensors ride on ``symm``)."""
    fuse_setup = int(os.environ.get("PT_DISPATCH_FUSE_SETUP", "0") == "1")
    needs_x_quant = xq.dtype == torch.bfloat16
    if needs_x_quant:
        assert xs is None, "bf16 activation path computes xs internally; pass xs=None"
    else:
        assert xs is not None, "pre-quantized fp8 path requires xs"
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
    if not needs_x_quant:
        assert xq.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), "fused kernel takes pre-quantized fp8 tokens"
    num_comm = int(expert_send_dst_rank.numel())
    num_ranks = int(sym_layout.num_ranks)
    num_max_pool_tokens = int(sym_layout.num_max_pool_tokens)
    cbsz = 1 if (not needs_x_quant and xq.dtype == torch.float8_e5m2) else 0
    blgp = 1 if w1q.dtype == torch.float8_e5m2 else 0
    out_fp16 = out_dtype == torch.float16
    c_n = N
    num_tokens = int(T)
    # Preshuffle-role fence opts (glc acquire + coalesced write-through release). Override via env.
    push_only = int(os.environ.get("PT_DISPATCH_PUSH_ONLY", "0") == "1")
    gemm_only = int(os.environ.get("PT_DISPATCH_GEMM_ONLY", "0") == "1")

    dev = xq.device
    if fuse_setup and needs_x_quant:
        x_bf16 = xq if xq.is_contiguous() else xq.contiguous()
        _xq_sk = (T, K, dev)
        xq_c = _XQ_SCRATCH.get(_xq_sk)
        if xq_c is None:
            xq_c = torch.empty((T, K), dtype=torch.float8_e4m3fn, device=dev)
            _XQ_SCRATCH[_xq_sk] = xq_c
        xs_c = _XS_SCRATCH.get(_xq_sk)
        if xs_c is None:
            xs_c = torch.empty((T, K // 32), dtype=torch.uint8, device=dev)
            _XS_SCRATCH[_xq_sk] = xs_c
    elif needs_x_quant:
        x_bf16 = torch.empty(1, dtype=torch.bfloat16, device=dev)
        xq, xs = quantize_rowwise_mxfp8_flydsl(xq)
        xs = xs.view(torch.float8_e8m0fnu)
        xq_c = xq if xq.is_contiguous() else xq.contiguous()
        xs_c = xs if xs.is_contiguous() else xs.contiguous()
        if xs_c.dtype != torch.uint8:
            xs_c = xs_c.view(torch.uint8)
        needs_x_quant = False  # host already quantized
    else:
        x_bf16 = torch.empty(1, dtype=torch.bfloat16, device=dev)
        xq_c = xq if xq.is_contiguous() else xq.contiguous()
        xs_c = xs if xs.is_contiguous() else xs.contiguous()
        if xs_c.dtype != torch.uint8:
            xs_c = xs_c.view(torch.uint8)
    XQ = xq_c.view(torch.int32)
    XS_U8 = xs_c
    XS = xs_c.view(torch.int32)
    W1S_RAW = w1s.contiguous().reshape(G * N, K // 32).view(torch.int32).reshape(-1)
    _wk = (w1q.data_ptr(), G, N, K)
    WEIGHTS = _WEIGHTS_FLAT_CACHE.get(_wk)
    if WEIGHTS is None:
        WEIGHTS = w1q.contiguous().reshape(G * N, K).view(torch.int8).reshape(-1)
        _WEIGHTS_FLAT_CACHE[_wk] = WEIGHTS
    _bk = (w1s.data_ptr(), G, N, K, 1)  # L1 GEMM reads ScaleBComb with pack=1
    needs_b_ps = fuse_setup and (_bk not in _BSP_CACHE)
    setup_env_cu = int(os.environ.get("PT_DISPATCH_SETUP_CU", "2"))
    if fuse_setup and (needs_x_quant or needs_b_ps):
        if needs_x_quant and not needs_b_ps:
            num_setup_cu = 1
        else:
            num_setup_cu = setup_env_cu if needs_b_ps else min(setup_env_cu, 2)
    else:
        num_setup_cu = 0
    if needs_b_ps:
        K128 = K // 128
        b_ngrp = ((G * N + 255) // 256) * 4
        weight_scale_ps = _BSP_ALLOC.get(_bk)
        if weight_scale_ps is None:
            weight_scale_ps = torch.zeros(b_ngrp * K128 * 256, dtype=torch.int32, device=dev)
            _BSP_ALLOC[_bk] = weight_scale_ps
    else:
        weight_scale_ps = _BSP_CACHE.get(_bk)
        if weight_scale_ps is None:
            weight_scale_ps = preshuffle_b_scale(w1s, G, N, K, pack=1)
            _BSP_CACHE[_bk] = weight_scale_ps
    pool_scale_ps = symm.pool_scale_ps  # local broadcast a_sp (preshuffle role writes it)

    setup_gate = _SETUP_GATE.get(dev)
    if setup_gate is None:
        setup_gate = torch.zeros(4, dtype=torch.int32, device=dev)
        _SETUP_GATE[dev] = setup_gate
    setup_gate.zero_()
    setup_gate[2] = int(fuse_setup and needs_x_quant)
    setup_gate[3] = int(fuse_setup and needs_b_ps)
    if num_setup_cu == 0 or not fuse_setup:
        setup_gate[1] = 1
    if num_setup_cu > 1:
        symm.grid_sync_count.zero_()

    num_tile_blocks = symm.meta_scalars[1:2]
    _out_sk = (num_max_pool_tokens, N, out_dtype, dev)
    output = _L1_OUTPUT_SCRATCH.get(_out_sk)
    if output is None:
        output = torch.empty((num_max_pool_tokens, N), dtype=out_dtype, device=dev)
        _L1_OUTPUT_SCRATCH[_out_sk] = output
    output_flat = output.view(-1)

    raw = _compile(
        N,
        K,
        num_max_pool_tokens,
        BM,
        BN,
        int(num_dispatch_cu),
        int(num_preshuffle_cu),
        int(num_setup_cu),
        int(num_comm),
        int(num_ranks),
        int(G),
        num_tokens,
        cbsz=cbsz,
        blgp=blgp,
        out_fp16=out_fp16,
        GROUP_M=int(GROUP_M),
        push_only=push_only,
        gemm_only=gemm_only,
    )
    args = (
        x_bf16,
        W1S_RAW,
        XQ,
        XS_U8,
        XS,
        setup_gate,
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
          int(num_setup_cu), int(num_comm), int(num_ranks), int(G), num_tokens, cbsz, blgp, out_fp16, int(GROUP_M),
          push_only, gemm_only)
    if torch.cuda.is_current_stream_capturing():
        raw(*args)
    else:
        compiled = _FUSED_COMPILED.get(ck)
        if compiled is None:
            compiled = flyc.compile(raw, *args)
            _FUSED_COMPILED[ck] = compiled
        compiled(*args)
    if needs_b_ps:
        _BSP_CACHE[_bk] = weight_scale_ps
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
    # Extra return views (LIVE symm pool, no clone):
    #  * dispatch_weights: per-pool-row routing weight (prologue-scattered into weight_recv_buf).
    #  * pool_x_fp8: dispatched fc1-input pool in rowwise fp8 — (pool_fp8 [P,H], pool_scale [P,H//32] E8M0).
    _Px, _Hx = symm.pool_fp8.shape
    return l1, handle, symm.weight_recv_buf, (symm.pool_fp8, symm.pool_scale.reshape(_Px, _Hx // 32))
