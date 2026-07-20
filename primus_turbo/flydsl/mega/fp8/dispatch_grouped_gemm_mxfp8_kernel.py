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
        EXPECTED: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        c_n: fx.Int32,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        comm_block_count = fx.Int32(_comm_cu)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

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
            scoreboard_base=sym_layout.scoreboard_ptr,
            scoreboard_offsets_resource=create_buffer_resource_from_addr(
                sym_layout.signal_offsets_ptr, num_records_bytes=num_ranks * 8
            ),
            block_m=BLOCK_M,
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
            sb_ps = sym_layout.scoreboard_ptr
            for _r in range(_ps_rounds):
                block_m_ps = ps_index + fx.Int32(_r * num_preshuffle_cu)
                if block_m_ps < real_tiles:
                    exp_ps = buffer_load(expected_resource, block_m_ps, vec_width=1, dtype=fx.T.i32())
                    if thread_index == fx.Int32(0):
                        spin_start = read_clock()
                        sig = ld(sb_ps, block_m_ps, scope="sys")
                        while sig < exp_ps:
                            fx.rocdl.s_sleep(fx.Int32(2))
                            if spin_timed_out(spin_start):
                                fx.printf(
                                    "MEGA mxfp8 preshuffle gate timeout: block={} sig={} exp={}\n",
                                    block_m_ps, sig, exp_ps,
                                )
                                spin_start = read_clock()
                            sig = ld(sb_ps, block_m_ps, scope="sys")
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
                        st(sb_ps, block_m_ps, fx.Int32(_PS_SENTINEL), scope="sys")
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
                sb_base = sym_layout.scoreboard_ptr
                # wait for the preshuffle role's SENTINEL (block_m's A-scale is in pool_scale_ps).
                if thread_index == fx.Int32(0):
                    spin_start = read_clock()
                    signal = ld(sb_base, block_m, scope="sys")
                    while signal < fx.Int32(_PS_SENTINEL):
                        fx.rocdl.s_sleep(fx.Int32(2))
                        if spin_timed_out(spin_start):
                            fx.printf(
                                "MEGA mxfp8 GEMM gate timeout: block={} signal={}\n", block_m, signal
                            )
                            spin_start = read_clock()
                        signal = ld(sb_base, block_m, scope="sys")
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
        c_n: int,
        stream: fx.Stream = fx.Stream(None),
    ):
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

    NOTE: the caller must zero ``symm.scoreboard`` (cross-rank, barrier-bracketed) before the
    launch so the per-block sentinel handoff starts clean."""
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
