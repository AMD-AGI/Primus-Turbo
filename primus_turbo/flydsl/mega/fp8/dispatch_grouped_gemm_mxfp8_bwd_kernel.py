###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""BACKWARD-ONLY fork of the fused fp8 dispatch PUSH + grouped MXFP8 GEMM (NT) kernel.

This is a dedicated backward-STEP1 implementation (dispatch(dy) + fc2 dgrad, NT-reuse).
It is a fork of ``dispatch_grouped_gemm_mxfp8_kernel.py`` so the FORWARD kernel (which
already wins ~1.4x) stays byte-identical and untouched. The fork is where we optimize the
L2-fence granularity of the gemm role without any risk to the forward path.

Baseline (round 0) is logically identical to the forward kernel: a comm -> preshuffle ->
gemm 3-stage pipeline gated per pool-block by the sys-scope scoreboard. See the forward
kernel's module docstring for the full role description.

DIAG (``PT_MXFP8_BWD_DIAG`` env, default 0 = normal, traces identically):
  1 = NO_GEMM   : comm + preshuffle run; gemm role early-exits (measure comm+preshuffle wall)
  2 = GEMM_ONLY : comm + preshuffle early-exit; gemm skips the sentinel gate (gemm-role wall)
  4 = NO_FENCE  : full pipeline minus the L2 invalidate/writeback fences (measure fence cost)
  8 = COMM_ONLY : only the comm role runs; preshuffle + gemm early-exit (measure comm wall)
 16 = DATA_ONLY : comm-only AND skip the raw-E8M0 scale push (data-only wall => scale-stream cost)

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
    gemm_mxfp8_nt_tile,
    make_mxfp8_shared_storage,
    make_mxfp8_shared_storage_lds_scale,
)
from primus_turbo.flydsl.mega.fp8.quant_flydsl import preshuffle_b_scale
from primus_turbo.flydsl.mega.prims import (
    l2_invalidate,
    l2_writeback,
    ld,
    read_clock,
    spin_timed_out,
    st,
)
from primus_turbo.flydsl.mega.sym_layout import SymLayout
from primus_turbo.flydsl.utils.gemm_helper import make_value_attrs

_FUSED_COMPILED: dict = {}  # (shape key) -> flyc.compile'd launch (eager; skip per-call @flyc.jit dispatch)
_BSP_CACHE: dict = {}  # (weight data_ptr, G, N, K) -> preshuffled weight scale b_sp (weights static)
_PS_SENTINEL = 1 << 20  # scoreboard value the preshuffle role stamps when a pool-block is ready


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
    diag=0,
    tok_unroll=1,
    two_stage=0,
    lds_kt=0,
    ps_read_cm=0,
):
    NO_GEMM = diag in (1, 5)  # 5 = comm+preshuffle, NO fences (isolate preshuffle write vs fence)
    GEMM_ONLY = diag == 2
    NO_FENCE = diag in (4, 5)
    COMM_ONLY = diag == 8
    DATA_ONLY = diag == 16  # comm-only AND skip the raw-E8M0 scale push (isolate scale-stream cost)
    _ONLY_COMM = COMM_ONLY or DATA_ONLY  # both run comm role only; gemm+preshuffle early-exit
    # 2-stage variants (bf16-style overlap): drop the separate preshuffle role + its L2 fences;
    # the gemm gates directly on the comm scoreboard (no SENTINEL). Two ways to feed the MMA:
    #   RAW_2STAGE (=1): read RAW pushed pool_scale + raw weight scale on-the-fly (ScaleS2RRaw).
    #                    MEASURED net-negative: the scattered raw loads make the gemm 2.6x slower.
    #   FUSED_PS   (=2): the gemm workgroup acquires (l2_invalidate) then preshuffles ITS tile's
    #                    A-scale raw -> POOL_SCALE_PS scratch with WRITE-THROUGH (sc1) stores, and
    #                    reads it back via the fast coalesced ScaleS2R (B stays host-preshuffled).
    #                    Removes the preshuffle stage + both L2 fences while keeping the fast read;
    #                    write-through keeps the scratch coherent vs concurrent device-wide inv.
    #                    MEASURED net-negative (round 5): 3.66 ms vs 2.15 ms 3-stage. The per-tile
    #                    preshuffle is 8x redundant (n_blocks tiles share a block_m's A-scale), the
    #                    224 KB broadcast scratch must be write-through (uncached, HBM-latency), and
    #                    it is a serial non-overlapped prologue -- all worse than the 3-stage's
    #                    once-per-block, cached, overlapped preshuffle role. -> not the default.
    RAW_2STAGE = two_stage == 1
    FUSED_PS = two_stage == 2
    # 3 = SKIP_PS_DIAG (correctness-INVALID timing probe): drop the preshuffle role + its L2
    # fences, gate the gemm on the comm scoreboard, but read POOL_SCALE_PS AS-IS (uninitialized
    # garbage -> wrong output) via the fast preshuffled ScaleS2R. Isolates whether the
    # comm-push ∥ gemm overlap is limited by the preshuffle stage itself.
    SKIP_PS_DIAG = two_stage == 3
    # 4 = LDS_STREAM (Plan A, correctness-VALID): drop the preshuffle role + its L2 fences; the
    # gemm gates on the comm scoreboard, reads the RAW pushed pool_scale, and stages it into LDS
    # (double-buffered KT-window) in its own prologue/K-loop via a_scale_lds. B stays host-
    # preshuffled (ScaleBComb). No separate preshuffle kernel, no HBM broadcast round-trip.
    LDS_STREAM = two_stage == 4
    SKIP_PS_ROLE = RAW_2STAGE or FUSED_PS or SKIP_PS_DIAG or LDS_STREAM  # none uses the preshuffle role
    K = hidden_size
    N = out_features
    assert num_max_pool_tokens % BLOCK_M == 0, "num_max_pool_tokens must be a multiple of BLOCK_M"
    assert N % BLOCK_N == 0, "out_features must be a multiple of BLOCK_N"
    assert K % 128 == 0 and K >= 256, f"mxfp8 needs K % 128 == 0 and K >= 256, got K={K}"
    assert K % 1024 == 0, f"clean fp8 push needs hidden % 1024 == 0, got K={K}"
    LDS_KT = lds_kt if lds_kt else 8  # streaming A-scale window (K-iters); double-buffered in LDS
    SharedStorage = (
        make_mxfp8_shared_storage_lds_scale(BLOCK_M, BLOCK_N, K, kt=LDS_KT)
        if LDS_STREAM else make_mxfp8_shared_storage(BLOCK_M, BLOCK_N)
    )
    n_blocks = N // BLOCK_N
    worst_case_tiles = num_max_pool_tokens // BLOCK_M
    _comm_cu = num_dispatch_cu
    # 2-stage variants have no preshuffle role -> gemm tiles start right after the comm blocks.
    _gemm_base = _comm_cu if SKIP_PS_ROLE else _comm_cu + num_preshuffle_cu
    _grid_size = _gemm_base + worst_case_tiles * n_blocks
    pool_scale_bytes_raw = num_max_pool_tokens * (K // 32)  # raw E8M0 pool region bytes
    K128 = K // 128
    _ps_rounds = (worst_case_tiles + num_preshuffle_cu - 1) // num_preshuffle_cu if num_preshuffle_cu else 0

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def dispatch_grouped_gemm_mxfp8_bwd_kernel(
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
            tok_unroll=tok_unroll,
            push_scale=not DATA_ONLY,
        )

        if block_index < comm_block_count:
            # COMM: this block owns comm tasks {block_index, block_index+comm_cu, ...}.
            if not GEMM_ONLY:
                local_task_count = (
                    fx.Int32(num_comm) - block_index + comm_block_count - fx.Int32(1)
                ) // comm_block_count
                for task_iteration in range(local_task_count):
                    dispatch_tile(block_index + task_iteration * comm_block_count, fx.Int32(0), 1)
        elif block_index < fx.Int32(_gemm_base):
            # PRESHUFFLE role: this block owns pool-blocks {ps_index, ps_index+ps_cu, ...}.
            # (Absent in 2-stage variants: _gemm_base == comm_block_count so this range is empty.)
            if not GEMM_ONLY and not _ONLY_COMM and not SKIP_PS_ROLE:
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
                                        "MEGA mxfp8 bwd preshuffle gate timeout: block={} sig={} exp={}\n",
                                        block_m_ps, sig, exp_ps,
                                    )
                                    spin_start = read_clock()
                                sig = ld(sb_ps, block_m_ps, scope="sys")
                        fx.gpu.barrier()
                        # acquire the peer-pushed raw pool_scale. Default: device-scope buffer_inv.
                        # ps_read_cm!=0 (sweep): skip the invalidate and instead read raw with that
                        # CPOL (e.g. 1=glc L1-bypass, 16=sc1, 17=glc+sc1) to test whether an uncached
                        # read can replace the invalidate (scoreboard-only coherence).
                        if not NO_FENCE and ps_read_cm == 0:
                            l2_invalidate()
                        fx.gpu.barrier()
                        preshuffle_a_scale_tile(
                            a_scale_raw_res, ps_res, block_m_ps * fx.Int32(BLOCK_M),
                            BLOCK_M, K128, thread_index, _BLOCK_THREADS,
                            read_cache_modifier=ps_read_cm,
                        )
                        fx.rocdl.s_waitcnt(fx.Int32(0))
                        if not NO_FENCE:
                            l2_writeback()  # release: make pool_scale_ps visible to the gemm role
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
                if not NO_GEMM and not _ONLY_COMM:
                    if SKIP_PS_ROLE:
                        # bf16-model gate: wait for the comm scoreboard to reach the pool-block's
                        # expected token count (no preshuffle role, no SENTINEL).
                        expected_count = buffer_load(expected_resource, block_m, vec_width=1, dtype=fx.T.i32())
                        if not GEMM_ONLY:
                            if thread_index == fx.Int32(0):
                                spin_start = read_clock()
                                signal = ld(sb_base, block_m, scope="sys")
                                while signal < expected_count:
                                    fx.rocdl.s_sleep(fx.Int32(2))
                                    if spin_timed_out(spin_start):
                                        fx.printf(
                                            "MEGA mxfp8 bwd 2stage gate timeout: block={} sig={} exp={}\n",
                                            block_m, signal, expected_count,
                                        )
                                        spin_start = read_clock()
                                    signal = ld(sb_base, block_m, scope="sys")
                        fx.gpu.barrier()
                        if FUSED_PS or LDS_STREAM:
                            # Acquire the peer-pushed pool_fp8 + RAW pool_scale (the single acquire
                            # invalidate the 3-stage gemm does; bf16 has this too). Plan A drops the
                            # preshuffle role's l2_writeback (no HBM broadcast handoff) but keeps this
                            # one acquire so the gemm sees the freshly pushed raw pool.
                            if not NO_FENCE:
                                l2_invalidate()
                            fx.gpu.barrier()
                        if FUSED_PS:
                            # FUSED_PS additionally preshuffles ITS tile's A-scale raw -> POOL_SCALE_PS
                            # scratch with write-through (sc1) so the scratch survives other WGs'
                            # device-wide invalidates (producer==consumer==this WG).
                            a_scale_raw_res = create_buffer_resource_from_addr(
                                sym_layout.pool_scale_ptr, num_records_bytes=pool_scale_bytes_raw
                            )
                            ps_res = create_buffer_resource(POOL_SCALE_PS, max_size=True)
                            preshuffle_a_scale_tile(
                                a_scale_raw_res, ps_res, block_m * fx.Int32(BLOCK_M),
                                BLOCK_M, K128, thread_index, _BLOCK_THREADS, cache_modifier=16,
                            )
                            fx.rocdl.s_waitcnt(fx.Int32(0))
                            fx.gpu.barrier()
                    else:
                        # 3-stage: wait for the preshuffle role's SENTINEL (A-scale in pool_scale_ps).
                        if not GEMM_ONLY:
                            if thread_index == fx.Int32(0):
                                spin_start = read_clock()
                                signal = ld(sb_base, block_m, scope="sys")
                                while signal < fx.Int32(_PS_SENTINEL):
                                    fx.rocdl.s_sleep(fx.Int32(2))
                                    if spin_timed_out(spin_start):
                                        fx.printf(
                                            "MEGA mxfp8 bwd GEMM gate timeout: block={} signal={}\n", block_m, signal
                                        )
                                        spin_start = read_clock()
                                    signal = ld(sb_base, block_m, scope="sys")
                        fx.gpu.barrier()
                        if not NO_FENCE:
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
                    if RAW_2STAGE:
                        # A-scale = RAW pushed pool_scale (E8M0 int32 [P, K//128]); B-scale = RAW
                        # weight scale (WEIGHT_SCALE_PS carries the raw weight scale in raw-2stage).
                        # ScaleS2RRaw broadcasts the E8M0 byte in-register during the MMA.
                        a_scale_res = create_buffer_resource_from_addr(
                            sym_layout.pool_scale_ptr, num_records_bytes=pool_scale_bytes_raw
                        )
                        b_scale_res = create_buffer_resource(WEIGHT_SCALE_PS, max_size=True)
                        gemm_mxfp8_nt_tile(
                            pool_fp8, a_scale_res, WEIGHTS, b_scale_res, OUTPUT, c_m_real, c_n, lds,
                            block_m, block_n, K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, G=G, group_idx=g_idx,
                            cbsz=cbsz, blgp=blgp, out_fp16=out_fp16, nt_vmcnt=nt_vmcnt, preshuffled=False,
                        )
                    elif LDS_STREAM:
                        # Plan A: A-scale = RAW pushed pool_scale, staged into LDS (double-buffered
                        # KT-window) inside the gemm; B-scale = host-preshuffled ScaleBComb b_sp.
                        a_scale_res = create_buffer_resource_from_addr(
                            sym_layout.pool_scale_ptr, num_records_bytes=pool_scale_bytes_raw
                        )
                        gemm_mxfp8_nt_tile(
                            pool_fp8, a_scale_res, WEIGHTS, WEIGHT_SCALE_PS, OUTPUT, c_m_real, c_n, lds,
                            block_m, block_n, K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, G=G, group_idx=g_idx,
                            cbsz=cbsz, blgp=blgp, out_fp16=out_fp16, nt_vmcnt=nt_vmcnt,
                            a_scale_lds=True, a_scale_lds_kt=LDS_KT,
                        )
                    else:
                        # 3-stage or FUSED_PS: A from the broadcast POOL_SCALE_PS (written by the
                        # preshuffle role in 3-stage, or by THIS workgroup in FUSED_PS), B from the
                        # host-preshuffled weight scale -> ScaleS2R / ScaleBComb (fast MMA load).
                        gemm_mxfp8_nt_tile(
                            pool_fp8, POOL_SCALE_PS, WEIGHTS, WEIGHT_SCALE_PS, OUTPUT, c_m_real, c_n, lds,
                            block_m, block_n, K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, G=G, group_idx=g_idx,
                            cbsz=cbsz, blgp=blgp, out_fp16=out_fp16, nt_vmcnt=nt_vmcnt, preshuffled=True,
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
        dispatch_grouped_gemm_mxfp8_bwd_kernel(
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


def dispatch_grouped_gemm_mxfp8_bwd(
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
    """Backward-STEP1 fork of ``dispatch_grouped_gemm_mxfp8`` (dispatch(dy) + fc2 dgrad, NT-reuse).

    Same 3-stage comm/preshuffle/gemm pipeline and same call signature as the forward function;
    forked so we can optimize the gemm-role L2-fence granularity without touching the forward.
    The caller must zero ``symm.scoreboard`` (cross-rank, barrier-bracketed) before launch."""
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
    diag = int(os.environ.get("PT_MXFP8_BWD_DIAG", "0"))  # 0=normal; 1=comm+ps; 2=gemm; 4=no-fence; 8=comm
    tok_unroll = int(os.environ.get("PT_MXFP8_BWD_TOK_UNROLL", "1"))  # comm token rows/warp-iter (MLP depth)
    # PT_MXFP8_BWD_2STAGE: 0 = 3-stage (default; comm->preshuffle-role->gemm, 2 L2 fences).
    #   1 = raw 2-stage: gemm reads RAW scales on-the-fly (ScaleS2RRaw). MEASURED NET-NEGATIVE
    #       (gemm 0.90 -> 2.37 ms, scattered loads) -> not the default.
    #   2 = FUSED_PS 2-stage: gemm gates on the comm scoreboard, acquires (1 invalidate), then
    #       preshuffles its A-scale into the POOL_SCALE_PS scratch (write-through sc1) and reads it
    #       back via the fast ScaleS2R. Drops the preshuffle role + both preshuffle-stage fences.
    two_stage = int(os.environ.get("PT_MXFP8_BWD_2STAGE", "0"))
    lds_kt = int(os.environ.get("PT_MXFP8_LDS_KT", "0"))  # LDS_STREAM (two_stage=4) A-scale window K-iters
    ps_read_cm = int(os.environ.get("PT_MXFP8_PS_READ_CM", "0"))  # !=0: uncached raw-scale read, skip acquire inv

    XQ = xq.contiguous().view(torch.uint8).view(torch.int32).reshape(-1)
    XS = xs.contiguous().view(torch.uint8).view(torch.int32).reshape(-1)
    WEIGHTS = w1q.contiguous().reshape(G * N, K).view(torch.int8).reshape(-1)
    # weights are static -> build the weight scale ONCE, cached. RAW-2stage (==1) wants the RAW
    # E8M0 weight scale (int32 [G*N, K//128], read by ScaleS2RRaw); 3-stage (0) and FUSED_PS (2)
    # want the preshuffled ScaleBComb b_sp. Both land in `weight_scale` (WEIGHT_SCALE_PS slot).
    _bk = (w1s.data_ptr(), G, N, K, two_stage)
    weight_scale = _BSP_CACHE.get(_bk)
    if weight_scale is None:
        if two_stage == 1:
            weight_scale = w1s.contiguous().reshape(G * N, K // 32).view(torch.uint8).view(torch.int32).reshape(-1)
        else:
            weight_scale = preshuffle_b_scale(w1s, G, N, K)
        _BSP_CACHE[_bk] = weight_scale
    pool_scale_ps = symm.pool_scale_ps  # local broadcast a_sp (3-stage preshuffle writes it; unused in 2-stage)

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
        diag=diag,
        tok_unroll=tok_unroll,
        two_stage=two_stage,
        lds_kt=lds_kt,
        ps_read_cm=ps_read_cm,
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
        weight_scale,
        pool_scale_ps,
        output_flat,
        tile_to_expert,
        expected_count,
        num_tile_blocks,
        c_n,
        torch.cuda.current_stream(),
    )
    ck = (N, K, num_max_pool_tokens, BM, BN, int(num_dispatch_cu), int(num_preshuffle_cu),
          int(num_comm), int(num_ranks), int(G), cbsz, blgp, out_fp16, int(GROUP_M), diag, tok_unroll,
          two_stage, lds_kt, ps_read_cm)
    if torch.cuda.is_current_stream_capturing():
        raw(*args)
    else:
        compiled = _FUSED_COMPILED.get(ck)
        if compiled is None:
            compiled = flyc.compile(raw, *args)
            _FUSED_COMPILED[ck] = compiled
        compiled(*args)
    return output
