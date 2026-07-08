###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused cross-rank dispatch PUSH (fp8) + grouped MXFP8 GEMM (NT), FlyDSL.

The mxfp8 analog of ``dispatch_grouped_gemm_bf16`` (NT path). One role-specialized
kernel launched on every rank:

  * ``block_index < num_dispatch_cu`` blocks quantize-push: each pushes a comm task's
    fp8 token rows to the peer ``pool_fp8`` region AND their raw E8M0 block scales to
    the peer ``pool_scale`` region over XGMI, then drains + signals the peer per-pool-block
    scoreboard (``dispatch_fp8_tile``). Half the dispatch bytes of the bf16 token push.
  * the remaining blocks each compute ONE NT output tile of the grouped L1 GEMM
    (``A = pool_fp8[M,K]`` + ``pool_scale`` E8M0, per-expert ``B = weight_fp8[G,N,K]`` +
    ``weight_scale`` E8M0 -> ``C = out[M,N]`` bf16) via ``gemm_mxfp8_nt_tile``, spinning
    on the scoreboard until its pool block is filled. The scoreboard sys-scope
    acquire/release carries cross-rank visibility of the peer-written pool (so, unlike
    the decoupled push, no host sync + L2 invalidate is needed); the comm latency hides
    under the MFMA-bound GEMM.

NT only. Constraints: hidden % 1024 == 0 (fp8 warp push), N % BLOCK_N == 0,
num_max_pool_tokens % BLOCK_M == 0, K % 128 == 0 and K >= 256 (mxfp8 MMA).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.buffer_ops import (
    buffer_load,
    create_buffer_resource,
    create_buffer_resource_from_addr,
)
from flydsl.expr.typing import AddressSpace, PointerType

from primus_turbo.flydsl.mega.fp8.ep_fp8 import (
    _BLOCK_THREADS,
    dispatch_fp8_copy_tile,
    dispatch_fp8_tile,
    preshuffle_a_scale_tile,
)
from primus_turbo.flydsl.mega.fp8.gemm_mxfp8_tile import (
    gemm_mxfp8_nt_tile,
    make_mxfp8_shared_storage,
)
from primus_turbo.flydsl.mega.prims import (
    atomic_add,
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


@functools.lru_cache(maxsize=64)
def _compile(
    out_features,
    hidden_size,
    num_max_pool_tokens,
    BLOCK_M,
    BLOCK_N,
    num_dispatch_cu,
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
    no_gate=False,  # DIAG: skip comm role + scoreboard gate (pool pre-filled), gemm-only
    no_fence=False,  # DIAG: skip the post-gate L2 invalidate (isolates fence cost; wrong answer)
    comm_only=False,  # DIAG: run ONLY the comm role (quant + push), no gemm (isolates comm cost)
):
    K = hidden_size
    N = out_features
    assert num_max_pool_tokens % BLOCK_M == 0, "num_max_pool_tokens must be a multiple of BLOCK_M"
    assert N % BLOCK_N == 0, "out_features must be a multiple of BLOCK_N"
    assert K % 128 == 0 and K >= 256, f"mxfp8 needs K % 128 == 0 and K >= 256, got K={K}"
    SharedStorage = make_mxfp8_shared_storage(BLOCK_M, BLOCK_N)
    n_blocks = N // BLOCK_N
    worst_case_tiles = num_max_pool_tokens // BLOCK_M
    _comm_cu = 0 if no_gate else num_dispatch_cu
    # comm_only: grid = just the comm CUs (no gemm tiles) -> isolates the quant+push cost.
    _grid_size = _comm_cu if comm_only else (_comm_cu + worst_case_tiles * n_blocks)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def dispatch_grouped_gemm_mxfp8_kernel(
        X: fx.Tensor,  # bf16 source tokens [T, K] (quantized in-push by the comm role)
        EXPERT_SEND_DST_RANK: fx.Tensor,
        EXPERT_SEND_DST_ROW: fx.Tensor,
        EXPERT_SEND_COUNT: fx.Tensor,
        EXPERT_SEND_OFFSET: fx.Tensor,
        DISPATCHED_TOKEN_IDX: fx.Tensor,
        sym_layout: SymLayout,
        WEIGHTS: fx.Tensor,  # fp8 weights viewed int8 [G*N*K] flattened
        WEIGHT_SCALE_PS: fx.Tensor,  # host-preshuffled weight E8M0 (ScaleBComb layout b_sp, int32)
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

        x_res = create_buffer_resource(X, max_size=True)
        esr = create_buffer_resource(EXPERT_SEND_DST_RANK, max_size=True)
        esrow = create_buffer_resource(EXPERT_SEND_DST_ROW, max_size=True)
        escnt = create_buffer_resource(EXPERT_SEND_COUNT, max_size=True)
        esoff = create_buffer_resource(EXPERT_SEND_OFFSET, max_size=True)
        dti = create_buffer_resource(DISPATCHED_TOKEN_IDX, max_size=True)
        group_resource = create_buffer_resource(TILE_TO_GROUP, max_size=True)
        expected_resource = create_buffer_resource(EXPECTED, max_size=True)
        num_tile_blocks_resource = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)

        dispatch_tile = dispatch_fp8_tile(
            thread_index=thread_index,
            hidden_size=hidden_size,
            num_max_pool_tokens=num_max_pool_tokens,
            x_resource=x_res,
            expert_send_dst_rank_resource=esr,
            expert_send_dst_row_resource=esrow,
            expert_send_count_resource=escnt,
            expert_send_offset_resource=esoff,
            dispatched_token_idx_resource=dti,
            pool_fp8_base=sym_layout.pool_fp8_ptr,
            pool_scale_base=sym_layout.pool_scale_ps_ptr,  # broadcast-layout scale region
            pool_offsets_resource=create_buffer_resource_from_addr(
                sym_layout.offsets_ptr, num_records_bytes=num_ranks * 8
            ),
            signal=True,
            scoreboard_base=sym_layout.scoreboard_ptr,
            scoreboard_offsets_resource=create_buffer_resource_from_addr(
                sym_layout.signal_offsets_ptr, num_records_bytes=num_ranks * 8
            ),
            block_m=BLOCK_M,
            fence=not no_fence,
        )

        if (not const_expr(no_gate)) and block_index < comm_block_count:
            local_task_count = (
                fx.Int32(num_comm) - block_index + comm_block_count - fx.Int32(1)
            ) // comm_block_count
            for task_iteration in range(local_task_count):
                dispatch_tile(block_index + task_iteration * comm_block_count, fx.Int32(0), 1)
        else:
            tile_index = block_index - comm_block_count
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
                expected_count = buffer_load(expected_resource, block_m, vec_width=1, dtype=fx.T.i32())
                if not const_expr(no_gate):
                    if thread_index == fx.Int32(0):
                        spin_start = read_clock()
                        signal = ld(sb_base, block_m, scope="sys")
                        while signal < expected_count:
                            fx.rocdl.s_sleep(fx.Int32(2))
                            if spin_timed_out(spin_start):
                                fx.printf(
                                    "MEGA mxfp8 dispatch GEMM gate timeout: block={} signal={} expected={}\n",
                                    block_m,
                                    signal,
                                    expected_count,
                                )
                                spin_start = read_clock()
                            signal = ld(sb_base, block_m, scope="sys")
                    fx.gpu.barrier()
                    # Device-scope acquire (post-gate, all waves): invalidate this CU/XCD's
                    # stale L2 view of the pool so the tile reads the comm-written
                    # (l2_writeback'd) tokens/scales. Bracketed by barriers so the invalidate
                    # happens after every wave has passed the gate and before any pool read.
                    if not const_expr(no_fence):
                        l2_invalidate()
                    fx.gpu.barrier()

                g_idx = buffer_load(group_resource, block_m, vec_width=1, dtype=fx.T.i32())
                pool_ptr_ty = PointerType.get(
                    elem_ty=fx.T.i8(), address_space=AddressSpace.Global, alignment=16
                )
                pool_fp8 = fx.make_view(
                    fx.inttoptr(pool_ptr_ty, sym_layout.pool_fp8_ptr),
                    fx.make_layout(num_max_pool_tokens * K, 1),
                )
                # preshuffled scale: A from the local broadcast pool_scale_ps (comm-written),
                # B from the host-preshuffled weight scale -> ScaleS2R / ScaleBComb (fast).
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
                if (not const_expr(no_gate)) and thread_index == fx.Int32(0):
                    prev = atomic_add(sb_base, block_m, fx.Int32(1), scope="sys")
                    if prev == expected_count + fx.Int32(n_blocks - 1):
                        st(sb_base, block_m, fx.Int32(0), scope="sys")

    @flyc.jit
    def launch(
        X,
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
            X,
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


_HANDLE_GROUP_OFFS = 10


def dispatch_grouped_gemm_mxfp8(
    x: torch.Tensor,
    w1q: torch.Tensor,
    w1s: torch.Tensor,
    handle,
    sym_layout,
    symm,
    *,
    num_dispatch_cu: int = 16,
    BM: int = 256,
    BN: int = 256,
    GROUP_M: int = 4,
    out_dtype: torch.dtype = torch.bfloat16,
    no_gate: bool = False,
    no_fence: bool = False,
    comm_only: bool = False,
):
    """Fused (quant + fp8 dispatch PUSH + grouped mxfp8 L1 GEMM). Returns L1 out
    [pool_cap, N] bf16.

    ``x`` [T, K] bf16 source tokens (quantized in-push; the comm role writes fp8 to the
    peer pool_fp8 and the E8M0 scale in the ScaleS2R broadcast layout to peer pool_scale_ps).
    ``w1q`` [G, N, K] fp8 weights + ``w1s`` [G, N, K//32] raw E8M0 (host-preshuffled to the
    ScaleBComb layout here). The gemm role reads pool_scale_ps + the preshuffled weight
    scale with ScaleS2R/ScaleBComb. ``handle``/``sym_layout``/``symm`` from the mxfp8
    prologue (use_mxfp8=True)."""
    from primus_turbo.flydsl.mega.fp8.quant_flydsl import preshuffle_b_scale

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
    assert x.dtype == torch.bfloat16, "fused quant-in-push takes bf16 source tokens"
    num_comm = int(expert_send_dst_rank.numel())
    num_ranks = int(sym_layout.num_ranks)
    num_max_pool_tokens = int(sym_layout.num_max_pool_tokens)
    cbsz = 1 if w1q.dtype == torch.float8_e5m2 else 0  # tokens are quantized to E4M3 in-push
    blgp = 1 if w1q.dtype == torch.float8_e5m2 else 0
    out_fp16 = out_dtype == torch.float16
    c_n = N

    X = x.contiguous()
    WEIGHTS = w1q.contiguous().reshape(G * N, K).view(torch.int8).reshape(-1)
    # weights are static -> preshuffle the weight scale to the ScaleBComb layout ONCE, cached
    # by the weight-scale tensor identity (kept out of the per-step / per-iter hot path).
    _bk = (w1s.data_ptr(), G, N, K)
    weight_scale_ps = _BSP_CACHE.get(_bk)
    if weight_scale_ps is None:
        weight_scale_ps = preshuffle_b_scale(w1s, G, N, K)
        _BSP_CACHE[_bk] = weight_scale_ps
    pool_scale_ps = symm.pool_scale_ps  # local broadcast a_sp (comm writes it cross-rank)

    num_tile_blocks = symm.meta_scalars[1:2]
    output = torch.empty((num_max_pool_tokens, N), dtype=out_dtype, device=x.device)
    output_flat = output.contiguous().view(-1)

    raw = _compile(
        N,
        K,
        num_max_pool_tokens,
        BM,
        BN,
        int(num_dispatch_cu),
        int(num_comm),
        int(num_ranks),
        int(G),
        cbsz=cbsz,
        blgp=blgp,
        out_fp16=out_fp16,
        GROUP_M=int(GROUP_M),
        no_gate=bool(no_gate),
        no_fence=bool(no_fence),
        comm_only=bool(comm_only),
    )
    args = (
        X,
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
    # eager: one-time flyc.compile'd object per shape (skips @flyc.jit per-call arg-hash /
    # drift-check dispatch, which otherwise pollutes timing); CUDA-graph capture uses raw.
    ck = (N, K, num_max_pool_tokens, BM, BN, int(num_dispatch_cu), int(num_comm), int(num_ranks),
          int(G), cbsz, blgp, out_fp16, int(GROUP_M), bool(no_gate), bool(no_fence), bool(comm_only))
    if torch.cuda.is_current_stream_capturing():
        raw(*args)
    else:
        compiled = _FUSED_COMPILED.get(ck)
        if compiled is None:
            compiled = flyc.compile(raw, *args)
            _FUSED_COMPILED[ck] = compiled
        compiled(*args)
    return output


# ─────────────────────────────────────────────────────────────────────────────
# CLEAN-PUSH variant: comm role pushes PRE-QUANTIZED fp8 + RAW E8M0 scale (coalesced,
# XGMI-saturating like dispatch_fp8_push) instead of quantizing in-push; the gemm role
# reads the raw pool_scale + raw weight scale on-the-fly (ScaleS2RRaw, preshuffled=False).
# Isolates whether decoupling the quant from the push recovers the comm bandwidth (the
# quant-in-push comm collapsed XGMI to ~78-148 GB/s); tokens are quantized once outside.
# ─────────────────────────────────────────────────────────────────────────────
_FUSED_CP_COMPILED: dict = {}


@functools.lru_cache(maxsize=64)
def _compile_cleanpush(
    out_features,
    hidden_size,
    num_max_pool_tokens,
    BLOCK_M,
    BLOCK_N,
    num_dispatch_cu,
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
    no_gate=False,
    no_fence=False,
    comm_only=False,
    gemm_preshuffled=False,  # False: comm pushes RAW scale, gemm ScaleS2RRaw. True: comm pushes
    #                          BROADCAST scale (pool_scale_ps), gemm ScaleS2R (fast, needs host-
    #                          preshuffled weight scale) -> "clean push + fast preshuffled gemm".
    local_preshuffle=False,  # with gemm_preshuffled=True: comm pushes RAW scale (coalesced/fast),
    #                          the gemm role locally preshuffles its tile's A-scale raw->broadcast
    #                          before the MMA -> fast comm AND fast preshuffled gemm (Option C').
    preshuffle_role=False,  # dedicated PRESHUFFLE role (3-stage pipeline): comm pushes RAW scale,
    #                         a role transposes raw->broadcast ONCE per block_m (non-redundant) and
    #                         hands off to the gemm via a scoreboard sentinel. Fast comm + fast gemm
    #                         + non-redundant preshuffle, all overlapped (targets ~2ms).
    num_preshuffle_cu=16,
):
    K = hidden_size
    N = out_features
    assert num_max_pool_tokens % BLOCK_M == 0, "num_max_pool_tokens must be a multiple of BLOCK_M"
    assert N % BLOCK_N == 0, "out_features must be a multiple of BLOCK_N"
    assert K % 128 == 0 and K >= 256, f"mxfp8 needs K % 128 == 0 and K >= 256, got K={K}"
    assert K % 1024 == 0, f"clean fp8 push needs hidden % 1024 == 0, got K={K}"
    SharedStorage = make_mxfp8_shared_storage(BLOCK_M, BLOCK_N)
    n_blocks = N // BLOCK_N
    worst_case_tiles = num_max_pool_tokens // BLOCK_M
    _comm_cu = 0 if no_gate else num_dispatch_cu
    _ps_cu = num_preshuffle_cu if (preshuffle_role and not no_gate and not comm_only) else 0
    _gemm_base = _comm_cu + _ps_cu
    _grid_size = _comm_cu if comm_only else (_gemm_base + worst_case_tiles * n_blocks)
    pool_scale_bytes_raw = num_max_pool_tokens * (K // 32)  # raw E8M0 pool region bytes
    K128 = K // 128
    # comm pushes broadcast only in pure-broadcast mode; in local_preshuffle / preshuffle_role mode
    # it pushes raw (coalesced) and the transpose happens on the dest (gemm per-tile / preshuffle role).
    _comm_broadcast = gemm_preshuffled and not local_preshuffle and not preshuffle_role
    _PS_SENTINEL = 1 << 20  # scoreboard value the preshuffle role writes when a block_m is ready
    _ps_rounds = (worst_case_tiles + num_preshuffle_cu - 1) // num_preshuffle_cu if _ps_cu else 0

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def dispatch_grouped_gemm_mxfp8_cleanpush_kernel(
        XQ: fx.Tensor,  # pre-quantized fp8 tokens int32 view [T, K//4] flattened
        XS: fx.Tensor,  # raw E8M0 scales int32 view [T, K//128] flattened
        EXPERT_SEND_DST_RANK: fx.Tensor,
        EXPERT_SEND_DST_ROW: fx.Tensor,
        EXPERT_SEND_COUNT: fx.Tensor,
        EXPERT_SEND_OFFSET: fx.Tensor,
        DISPATCHED_TOKEN_IDX: fx.Tensor,
        sym_layout: SymLayout,
        WEIGHTS: fx.Tensor,  # fp8 weights viewed int8 [G*N*K] flattened
        WEIGHT_SCALE: fx.Tensor,  # raw int32 [G*N*(K//128)] (raw mode) OR preshuffled b_sp (ps mode)
        POOL_SCALE_PS: fx.Tensor,  # broadcast A-scale tensor (ps mode); dummy [1] in raw mode
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
        wscale_res = create_buffer_resource(WEIGHT_SCALE, max_size=True)

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
            # broadcast ps region only when comm pushes broadcast; else the RAW region
            pool_scale_base=(sym_layout.pool_scale_ps_ptr if _comm_broadcast else sym_layout.pool_scale_ptr),
            pool_offsets_resource=create_buffer_resource_from_addr(
                sym_layout.offsets_ptr, num_records_bytes=num_ranks * 8
            ),
            signal=True,
            scoreboard_base=sym_layout.scoreboard_ptr,
            scoreboard_offsets_resource=create_buffer_resource_from_addr(
                sym_layout.signal_offsets_ptr, num_records_bytes=num_ranks * 8
            ),
            block_m=BLOCK_M,
            fence=not no_fence,
            preshuffle_scale=_comm_broadcast,
        )

        if (not const_expr(no_gate)) and block_index < comm_block_count:
            local_task_count = (
                fx.Int32(num_comm) - block_index + comm_block_count - fx.Int32(1)
            ) // comm_block_count
            for task_iteration in range(local_task_count):
                dispatch_tile(block_index + task_iteration * comm_block_count, fx.Int32(0), 1)
        elif const_expr(preshuffle_role) and block_index < fx.Int32(_gemm_base):
            # PRESHUFFLE role: this block owns block_m in {ps_index, ps_index+ps_cu, ...}.
            # For each: gate on the comm scoreboard (tokens arrived), invalidate to see the
            # peer-written raw pool_scale, transpose block_m's A-scale raw->broadcast into
            # pool_scale_ps (ONCE, non-redundant), write it back, then stamp a sentinel on the
            # scoreboard so that block_m's gemm tiles can proceed.
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
                    fx.gpu.barrier()
                    if not const_expr(no_fence):
                        l2_invalidate()
                    fx.gpu.barrier()
                    preshuffle_a_scale_tile(
                        a_scale_raw_res, ps_res, block_m_ps * fx.Int32(BLOCK_M),
                        BLOCK_M, K128, thread_index, _BLOCK_THREADS,
                    )
                    if not const_expr(no_fence):
                        fx.rocdl.s_waitcnt(fx.Int32(0))
                        l2_writeback()
                    fx.gpu.barrier()
                    if thread_index == fx.Int32(0):
                        st(sb_ps, block_m_ps, fx.Int32(_PS_SENTINEL), scope="sys")
        else:
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
                expected_count = buffer_load(expected_resource, block_m, vec_width=1, dtype=fx.T.i32())
                # preshuffle_role: wait for the preshuffle role's SENTINEL (block_m's A-scale
                # transposed to pool_scale_ps + written back). Else wait for the comm count.
                gate_target = fx.Int32(_PS_SENTINEL) if const_expr(preshuffle_role) else expected_count
                if not const_expr(no_gate):
                    if thread_index == fx.Int32(0):
                        spin_start = read_clock()
                        signal = ld(sb_base, block_m, scope="sys")
                        while signal < gate_target:
                            fx.rocdl.s_sleep(fx.Int32(2))
                            if spin_timed_out(spin_start):
                                fx.printf(
                                    "MEGA mxfp8 cleanpush GEMM gate timeout: block={} signal={} target={}\n",
                                    block_m,
                                    signal,
                                    gate_target,
                                )
                                spin_start = read_clock()
                            signal = ld(sb_base, block_m, scope="sys")
                    fx.gpu.barrier()
                    if not const_expr(no_fence):
                        l2_invalidate()
                    fx.gpu.barrier()

                g_idx = buffer_load(group_resource, block_m, vec_width=1, dtype=fx.T.i32())
                pool_ptr_ty = PointerType.get(
                    elem_ty=fx.T.i8(), address_space=AddressSpace.Global, alignment=16
                )
                pool_fp8 = fx.make_view(
                    fx.inttoptr(pool_ptr_ty, sym_layout.pool_fp8_ptr),
                    fx.make_layout(num_max_pool_tokens * K, 1),
                )
                if const_expr(gemm_preshuffled):
                    if const_expr(local_preshuffle):
                        # comm pushed the RAW scale (coalesced); preshuffle THIS tile's A-scale
                        # (block_m's rows) raw->broadcast into pool_scale_ps locally, then the
                        # MMA reads it with the fast ScaleS2R. Bracketed by barriers so the WG's
                        # broadcast writes are visible to its own scale reads.
                        a_scale_raw_res = create_buffer_resource_from_addr(
                            sym_layout.pool_scale_ptr, num_records_bytes=pool_scale_bytes_raw
                        )
                        ps_res = create_buffer_resource(POOL_SCALE_PS, max_size=True)
                        preshuffle_a_scale_tile(
                            a_scale_raw_res, ps_res, block_m * fx.Int32(BLOCK_M),
                            BLOCK_M, K128, thread_index, _BLOCK_THREADS,
                        )
                        fx.gpu.barrier()
                    # A from the local broadcast pool_scale_ps (comm-written), B from the
                    # host-preshuffled weight scale -> ScaleS2R / ScaleBComb (fast MMA load).
                    gemm_mxfp8_nt_tile(
                        pool_fp8,
                        POOL_SCALE_PS,
                        WEIGHTS,
                        WEIGHT_SCALE,
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
                else:
                    # RAW scale loader: A from the local pool_scale (raw region, comm-written),
                    # B from the raw weight scale -> ScaleS2RRaw (on-the-fly; no preshuffle pass).
                    a_scale_res = create_buffer_resource_from_addr(
                        sym_layout.pool_scale_ptr, num_records_bytes=pool_scale_bytes_raw
                    )
                    gemm_mxfp8_nt_tile(
                        pool_fp8,
                        a_scale_res,
                        WEIGHTS,
                        wscale_res,
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
                        preshuffled=False,
                    )
                # scoreboard self-reset (skip in preshuffle_role: the sentinel handoff is
                # cleared by the caller's per-iter reset, not by the gemm).
                if (not const_expr(no_gate)) and (not const_expr(preshuffle_role)) and thread_index == fx.Int32(0):
                    prev = atomic_add(sb_base, block_m, fx.Int32(1), scope="sys")
                    if prev == expected_count + fx.Int32(n_blocks - 1):
                        st(sb_base, block_m, fx.Int32(0), scope="sys")

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
        WEIGHT_SCALE,
        POOL_SCALE_PS,
        OUTPUT,
        TILE_TO_GROUP,
        EXPECTED,
        NUM_TILE_BLOCKS,
        c_n: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        dispatch_grouped_gemm_mxfp8_cleanpush_kernel(
            XQ,
            XS,
            EXPERT_SEND_DST_RANK,
            EXPERT_SEND_DST_ROW,
            EXPERT_SEND_COUNT,
            EXPERT_SEND_OFFSET,
            DISPATCHED_TOKEN_IDX,
            sym_layout,
            WEIGHTS,
            WEIGHT_SCALE,
            POOL_SCALE_PS,
            OUTPUT,
            TILE_TO_GROUP,
            EXPECTED,
            NUM_TILE_BLOCKS,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(_grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def dispatch_grouped_gemm_mxfp8_cleanpush(
    xq: torch.Tensor,
    xs: torch.Tensor,
    w1q: torch.Tensor,
    w1s: torch.Tensor,
    handle,
    sym_layout,
    symm,
    *,
    num_dispatch_cu: int = 16,
    BM: int = 256,
    BN: int = 256,
    GROUP_M: int = 4,
    out_dtype: torch.dtype = torch.bfloat16,
    no_gate: bool = False,
    no_fence: bool = False,
    comm_only: bool = False,
    gemm_preshuffled: bool = False,
    local_preshuffle: bool = False,
    preshuffle_role: bool = False,
    num_preshuffle_cu: int = 16,
):
    """Fused (CLEAN fp8 dispatch PUSH + grouped mxfp8 L1 GEMM), quant done OUTSIDE.

    ``xq`` [T, K] fp8 + ``xs`` [T, K//32] raw E8M0 (pre-quantized tokens); the comm role
    copies them coalesced to the peer ``pool_fp8`` (no in-push quant). Scale handling:
      * default (raw): comm pushes RAW scale to ``pool_scale``; gemm reads it + raw ``w1s``
        on-the-fly (``ScaleS2RRaw``, slower gemm).
      * ``gemm_preshuffled=True``: comm writes the E8M0 scale in the broadcast layout to
        ``pool_scale_ps`` (scattered XGMI); gemm reads it preshuffled (fast MMA).
      * ``gemm_preshuffled=True, local_preshuffle=True`` (Option C'): comm pushes the RAW scale
        (coalesced/fast); the gemm role preshuffles its tile's A-scale raw->broadcast locally
        before the MMA -> fast comm AND         fast preshuffled gemm. Returns L1 out [pool_cap, N] bf16.
      * ``preshuffle_role=True`` (3-stage pipeline): comm pushes RAW scale; a dedicated
        preshuffle role transposes raw->broadcast ONCE per block_m (non-redundant) and hands
        off to the gemm via a scoreboard sentinel -> fast comm + fast gemm + non-redundant
        preshuffle, all overlapped."""
    if preshuffle_role:
        gemm_preshuffled = True  # gemm reads the broadcast pool_scale_ps the role produced
        local_preshuffle = False  # the role does the transpose, not the gemm
    assert not (local_preshuffle and not gemm_preshuffled), "local_preshuffle requires gemm_preshuffled"
    from primus_turbo.flydsl.mega.fp8.quant_flydsl import preshuffle_b_scale

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
    assert xq.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), "cleanpush takes pre-quantized fp8 tokens"
    num_comm = int(expert_send_dst_rank.numel())
    num_ranks = int(sym_layout.num_ranks)
    num_max_pool_tokens = int(sym_layout.num_max_pool_tokens)
    cbsz = 1 if xq.dtype == torch.float8_e5m2 else 0
    blgp = 1 if w1q.dtype == torch.float8_e5m2 else 0
    out_fp16 = out_dtype == torch.float16
    c_n = N

    XQ = xq.contiguous().view(torch.uint8).view(torch.int32).reshape(-1)
    XS = xs.contiguous().view(torch.uint8).view(torch.int32).reshape(-1)
    WEIGHTS = w1q.contiguous().reshape(G * N, K).view(torch.int8).reshape(-1)
    if gemm_preshuffled:
        # weights are static -> preshuffle the weight scale ONCE (ScaleBComb b_sp), cached.
        _bk = (w1s.data_ptr(), G, N, K)
        weight_scale = _BSP_CACHE.get(_bk)
        if weight_scale is None:
            weight_scale = preshuffle_b_scale(w1s, G, N, K)
            _BSP_CACHE[_bk] = weight_scale
        pool_scale_ps = symm.pool_scale_ps  # local broadcast a_sp (comm writes it cross-rank)
    else:
        weight_scale = w1s.contiguous().reshape(G * N, K // 32).view(torch.uint8).view(torch.int32).reshape(-1)
        pool_scale_ps = XS[:1]  # dummy (unused in raw mode)

    num_tile_blocks = symm.meta_scalars[1:2]
    output = torch.empty((num_max_pool_tokens, N), dtype=out_dtype, device=xq.device)
    output_flat = output.contiguous().view(-1)

    raw = _compile_cleanpush(
        N,
        K,
        num_max_pool_tokens,
        BM,
        BN,
        int(num_dispatch_cu),
        int(num_comm),
        int(num_ranks),
        int(G),
        cbsz=cbsz,
        blgp=blgp,
        out_fp16=out_fp16,
        GROUP_M=int(GROUP_M),
        no_gate=bool(no_gate),
        no_fence=bool(no_fence),
        comm_only=bool(comm_only),
        gemm_preshuffled=bool(gemm_preshuffled),
        local_preshuffle=bool(local_preshuffle),
        preshuffle_role=bool(preshuffle_role),
        num_preshuffle_cu=int(num_preshuffle_cu),
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
    ck = (N, K, num_max_pool_tokens, BM, BN, int(num_dispatch_cu), int(num_comm), int(num_ranks),
          int(G), cbsz, blgp, out_fp16, int(GROUP_M), bool(no_gate), bool(no_fence), bool(comm_only),
          bool(gemm_preshuffled), bool(local_preshuffle), bool(preshuffle_role), int(num_preshuffle_cu))
    if torch.cuda.is_current_stream_capturing():
        raw(*args)
    else:
        compiled = _FUSED_CP_COMPILED.get(ck)
        if compiled is None:
            compiled = flyc.compile(raw, *args)
            _FUSED_CP_COMPILED[ck] = compiled
        compiled(*args)
    return output
