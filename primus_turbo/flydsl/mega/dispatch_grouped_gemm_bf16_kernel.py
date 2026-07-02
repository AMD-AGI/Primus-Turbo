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
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
)
from flydsl.expr.typing import AddressSpace, PointerType

from primus_turbo.flydsl.mega.dispatch_prologue_kernel import dispatch_prologue
from primus_turbo.flydsl.mega.ep_intranode import _BLOCK_THREADS, dispatch_bf16_tile
from primus_turbo.flydsl.mega.gemm_bf16_kernel import (
    GEMM_TILE,
    _make_shared_storage,
    gemm_bf16_tn_variable_k_tile,
    load_go_i64,
)
from primus_turbo.flydsl.mega.prims import (
    SPIN_TIMEOUT_CYCLES,
    atomic_add,
    ld,
    read_clock,
    st,
)
from primus_turbo.flydsl.mega.sym_layout import SymLayout
from primus_turbo.flydsl.mega.symm_buffer import get_symm_buffer_for_mega_moe
from primus_turbo.flydsl.utils.gemm_helper import make_value_attrs, xcd_remap_pid


@functools.lru_cache(maxsize=1)
def get_dummy_tensor():
    """Cached 1-elem i32 placeholder for kernel slots the active layout const-folds away."""
    return torch.empty(1, dtype=torch.int32)


_HANDLE_PROLOGUE_LEN = 11


@functools.lru_cache(maxsize=256)
def _compile(
    out_features,
    hidden_size,
    num_max_pool_tokens,
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
    trans_c=False,
    G=0,
    has_snapshots=False,
    num_xcd=8,
):
    K = hidden_size
    is_tn = layout == "tn"
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)
    assert num_max_pool_tokens % BLOCK_M == 0, "num_max_pool_tokens must be a multiple of BLOCK_M"
    if is_tn:
        OUT_M, OUT_N = hidden_size, out_features
        OUT_M_g, OUT_N_g = (OUT_N, OUT_M) if trans_c else (OUT_M, OUT_N)
        assert OUT_M_g % BLOCK_M == 0 and OUT_N_g % BLOCK_N == 0
        N_BLOCKS_M = OUT_M_g // BLOCK_M
        N_BLOCKS_N = OUT_N_g // BLOCK_N
        TILES_PER_GROUP = N_BLOCKS_M * N_BLOCKS_N
        TOTAL = G * TILES_PER_GROUP
    else:
        gemm_tile = GEMM_TILE[layout]
        assert out_features % BLOCK_N == 0, "out_features must be a multiple of BLOCK_N"
        n_blocks = out_features // BLOCK_N
        worst_case_tiles = num_max_pool_tokens // BLOCK_M

    _grid_size = num_dispatch_cu + (TOTAL if is_tn else worst_case_tiles * n_blocks)
    _copy_threads = _grid_size * _BLOCK_THREADS
    _copy_iters = (num_max_pool_tokens + _copy_threads - 1) // _copy_threads
    _meta_records = 8 * 4
    _pool_records = num_max_pool_tokens * 4

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
        GROUP_OFFS: fx.Tensor,
        DISPATCHED_SRC_RANK: fx.Tensor,
        DISPATCHED_SRC_ROW: fx.Tensor,
        DISPATCHED_META: fx.Tensor,
        c_n: fx.Int32,
        out_m_rt: fx.Int32,
        out_n_rt: fx.Int32,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        comm_block_count = fx.Int32(num_dispatch_cu)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        if const_expr(has_snapshots):
            src_rank_res = create_buffer_resource(DISPATCHED_SRC_RANK, max_size=True)
            src_row_res = create_buffer_resource(DISPATCHED_SRC_ROW, max_size=True)
            src_meta_res = create_buffer_resource(DISPATCHED_META, max_size=True)
            dst_rank_res = create_buffer_resource_from_addr(
                sym_layout.origin_rank_ptr, num_records_bytes=_pool_records
            )
            dst_slot_res = create_buffer_resource_from_addr(
                sym_layout.origin_slot_ptr, num_records_bytes=_pool_records
            )
            dst_meta_res = create_buffer_resource_from_addr(
                sym_layout.meta_scalars_ptr, num_records_bytes=_meta_records
            )
            gtid = block_index * fx.Int32(_BLOCK_THREADS) + thread_index
            for _c_it in range_constexpr(_copy_iters):
                row = gtid + fx.Int32(_c_it * _copy_threads)
                if row < fx.Int32(num_max_pool_tokens):
                    buffer_store(
                        buffer_load(src_rank_res, row, vec_width=1, dtype=fx.T.i32()), dst_rank_res, row
                    )
                    buffer_store(
                        buffer_load(src_row_res, row, vec_width=1, dtype=fx.T.i32()), dst_slot_res, row
                    )
            if gtid < fx.Int32(8):
                buffer_store(
                    buffer_load(src_meta_res, gtid, vec_width=1, dtype=fx.T.i32()), dst_meta_res, gtid
                )

        input_resource = create_buffer_resource(INPUT_TOKENS, max_size=True)
        expert_send_dst_rank_resource = create_buffer_resource(EXPERT_SEND_DST_RANK, max_size=True)
        expert_send_dst_row_resource = create_buffer_resource(EXPERT_SEND_DST_ROW, max_size=True)
        expert_send_count_resource = create_buffer_resource(EXPERT_SEND_COUNT, max_size=True)
        expert_send_offset_resource = create_buffer_resource(EXPERT_SEND_OFFSET, max_size=True)
        dispatched_token_idx_resource = create_buffer_resource(DISPATCHED_TOKEN_IDX, max_size=True)
        if const_expr(is_tn):
            go = fx.rocdl.make_buffer_tensor(GROUP_OFFS, max_size=False, num_records_bytes=(G + 1) * 8)
            go_div = fx.logical_divide(go, fx.make_layout(1, 1))
            expected_resource = create_buffer_resource(EXPECTED, max_size=True)
        else:
            group_resource = create_buffer_resource(TILE_TO_GROUP, max_size=True)
            expected_resource = create_buffer_resource(EXPECTED, max_size=True)
            num_tile_blocks_resource = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)

        dispatch_tile = dispatch_bf16_tile(
            thread_index=thread_index,
            hidden_size=hidden_size,
            num_max_pool_tokens=num_max_pool_tokens,
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
            pool_base=sym_layout.pool_ptr,
            pool_offsets_resource=create_buffer_resource_from_addr(
                sym_layout.offsets_ptr, num_records_bytes=int(sym_layout.num_ranks) * 8
            ),
            scoreboard_base=sym_layout.scoreboard_ptr,
            scoreboard_offsets_resource=create_buffer_resource_from_addr(
                sym_layout.signal_offsets_ptr, num_records_bytes=int(sym_layout.num_ranks) * 8
            ),
        )

        if block_index < comm_block_count:
            local_task_count = (
                fx.Int32(num_comm) - block_index + comm_block_count - fx.Int32(1)
            ) // comm_block_count
            for task_iteration in range(local_task_count):
                dispatch_tile(block_index + task_iteration * comm_block_count, fx.Int32(0), 1)
        elif const_expr(is_tn):
            tile_index = block_index - comm_block_count
            if tile_index < fx.Int32(TOTAL):
                group_idx = tile_index // fx.Int32(TILES_PER_GROUP)
                local_raw = tile_index % fx.Int32(TILES_PER_GROUP)
                local = xcd_remap_pid(local_raw, TILES_PER_GROUP, num_xcd)
                if const_expr(trans_c):
                    block_n = local // fx.Int32(N_BLOCKS_M)
                    block_m = local % fx.Int32(N_BLOCKS_M)
                else:
                    block_m = local // fx.Int32(N_BLOCKS_N)
                    block_n = local % fx.Int32(N_BLOCKS_N)
                m_start = load_go_i64(go_div, group_idx)
                m_end = load_go_i64(go_div, group_idx + fx.Int32(1))
                sb_base = sym_layout.scoreboard_ptr
                pb0 = m_start // fx.Int32(BLOCK_M)
                n_pb = m_end // fx.Int32(BLOCK_M) - pb0
                if thread_index == fx.Int32(0):
                    for j in range(n_pb):
                        pbj = pb0 + j
                        exp_c = buffer_load(expected_resource, pbj, vec_width=1, dtype=fx.T.i32())
                        spin_start = read_clock()
                        sig = ld(sb_base, pbj, scope="sys")
                        while sig < exp_c:
                            fx.rocdl.s_sleep(fx.Int32(2))
                            if (read_clock() - spin_start) > fx.Int64(SPIN_TIMEOUT_CYCLES):
                                fx.printf(
                                    "MEGA tn wgrad gate timeout: pb={} sig={} exp={}\n", pbj, sig, exp_c
                                )
                                spin_start = read_clock()
                            sig = ld(sb_base, pbj, scope="sys")
                fx.gpu.barrier()
                pool_ptr_ty = PointerType.get(
                    elem_ty=fx.BFloat16.ir_type, address_space=AddressSpace.Global, alignment=16
                )
                pool_tensor = fx.make_view(
                    fx.inttoptr(pool_ptr_ty, sym_layout.pool_ptr),
                    fx.make_layout(num_max_pool_tokens * OUT_M, 1),
                )
                if const_expr(trans_c):
                    gemm_a, gemm_b, rt_m, rt_n = WEIGHTS, pool_tensor, out_n_rt, out_m_rt
                else:
                    gemm_a, gemm_b, rt_m, rt_n = pool_tensor, WEIGHTS, out_m_rt, out_n_rt
                gemm_bf16_tn_variable_k_tile(
                    gemm_a,
                    gemm_b,
                    OUTPUT,
                    group_idx,
                    block_m,
                    block_n,
                    m_start,
                    m_end,
                    lds,
                    rt_m,
                    rt_n,
                    G=G,
                    OUT_M=OUT_M_g,
                    OUT_N=OUT_N_g,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    out_fp16=out_fp16,
                )
                if thread_index == fx.Int32(0):
                    exp_r = buffer_load(expected_resource, pb0, vec_width=1, dtype=fx.T.i32())
                    prev = atomic_add(sb_base, pb0, fx.Int32(1), scope="sys")
                    if prev == exp_r + fx.Int32(TILES_PER_GROUP - 1):
                        for j in range(n_pb):
                            st(sb_base, pb0 + j, fx.Int32(0), scope="sys")
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
                group_size_m = arith.select(remaining_m < fx.Int32(GROUP_M), remaining_m, fx.Int32(GROUP_M))
                block_m = first_pid_m + (pid_in_group % group_size_m)
                block_n = pid_in_group // group_size_m
                c_m_real = fx.Int32(num_max_pool_tokens)
                sb_base = sym_layout.scoreboard_ptr
                expected_count = buffer_load(expected_resource, block_m, vec_width=1, dtype=fx.T.i32())
                if thread_index == fx.Int32(0):
                    spin_start = read_clock()
                    signal = ld(sb_base, block_m, scope="sys")
                    while signal < expected_count:
                        fx.rocdl.s_sleep(fx.Int32(2))
                        if (read_clock() - spin_start) > fx.Int64(SPIN_TIMEOUT_CYCLES):
                            fx.printf(
                                "MEGA dispatch GEMM gate timeout: block={} signal={} expected={}\n",
                                block_m,
                                signal,
                                expected_count,
                            )
                            spin_start = read_clock()
                        signal = ld(sb_base, block_m, scope="sys")
                fx.gpu.barrier()

                g_idx = buffer_load(group_resource, block_m, vec_width=1, dtype=fx.T.i32())
                gbase = g_idx * fx.Int32(K) * c_n
                pool_ptr_ty = PointerType.get(
                    elem_ty=fx.BFloat16.ir_type, address_space=AddressSpace.Global, alignment=16
                )
                pool_tensor = fx.make_view(
                    fx.inttoptr(pool_ptr_ty, sym_layout.pool_ptr),
                    fx.make_layout(num_max_pool_tokens * K, 1),
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
        GROUP_OFFS,
        DISPATCHED_SRC_RANK,
        DISPATCHED_SRC_ROW,
        DISPATCHED_META,
        c_n: int,
        out_m_rt: int,
        out_n_rt: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_size = num_dispatch_cu + (TOTAL if is_tn else worst_case_tiles * n_blocks)
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
            GROUP_OFFS,
            DISPATCHED_SRC_RANK,
            DISPATCHED_SRC_ROW,
            DISPATCHED_META,
            c_n,
            out_m_rt,
            out_n_rt,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


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
    x: torch.Tensor,
    l1_weights: torch.Tensor,
    group: torch.distributed.group,
    handle: Optional[Tuple] = None,
    topk_idx: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    layout: str = "nt",
    num_dispatch_cu: int = 16,
    autotune=False,
    BM=256,
    BN=256,
    GROUP_M=4,
    pool_mult: int = 2,
    trans_c: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
):
    """Fused cross-rank dispatch PUSH + grouped BF16 GEMM (nt / nn / tn).

    ``x`` [num_src_tokens, K] bf16 source tokens dispatched into the pool. ``layout``
    selects the role-2 GEMM: NT (fwd) weight [G,N,K] / NN (dgrad) weight [G,K,N] ->
    output [pool_cap, N], per-block scoreboard gate; TN (variable-K wgrad) ``l1_weights``
    is the resident pool-order rhs [pool_cap, OUT_N] + ``group_offs`` -> output
    [G, OUT_M, OUT_N] ([G, OUT_N, OUT_M] if ``trans_c``), no gate (dispatch & wgrad are
    independent). The dispatch comm handle is flat
    (``expert_send_dst_rank/start/count/src_offset/src_tokens`` + ``num_comm``).

    The symmetric workspace (pool GEMM A operand + peer pool / peer scoreboard delta tables)
    is owned here: forward (``handle=None``) builds/creates it from ``group`` + tensor shapes
    via ``get_symm_buffer_for_mega_moe(group, ...)`` and appends this layer's origin/meta
    snapshots to the returned handle tail; reuse fetches the active buffer (no-group call).
    ``tile_to_expert`` / ``expected_count`` are read from the handle."""
    if handle is None:
        assert topk_idx is not None, "handle=None requires topk_idx to run the prologue"
        assert group is not None, "handle=None requires group to build the symm workspace"
        assert layout == "nt", "handle=None auto-prologue is forward-only (nt); pass handle for nn/tn"
        experts_per_rank = l1_weights.shape[0]
        num_tokens, hidden = x.shape
        num_topk = topk_idx.shape[-1]
        symm = get_symm_buffer_for_mega_moe(
            group,
            num_experts=experts_per_rank * group.size(),
            num_max_tokens_per_rank=num_tokens,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=l1_weights.shape[1] // 2,
            block_m=BM,
            block_n=BN,
            pool_mult=pool_mult,
        )
        sym_layout = symm.make_sym_layout()
        handle = tuple(
            dispatch_prologue(
                topk_idx,
                topk_weights,
                sym_layout=sym_layout,
                num_tokens=num_tokens,
                num_topk=num_topk,
                num_experts=symm.num_experts,
                world_size=symm.world,
                rank=symm.rank,
                experts_per_rank=experts_per_rank,
                block_m=BM,
                num_max_pool_tokens=symm.num_max_pool_tokens,
            )
        ) + (
            symm.origin_rank.clone(),
            symm.origin_slot.clone(),
            symm.meta_scalars.clone(),
        )
    else:
        symm = get_symm_buffer_for_mega_moe()
        sym_layout = symm.make_sym_layout()

    (
        expert_send_dst_rank,
        expert_send_dst_row,
        expert_send_count,
        expert_send_offset,
        dispatched_token_idx,
        _,
        _,
        tile_to_expert,
        expected_count,
        _,
        group_offs,
        *_snapshots,
    ) = handle

    num_comm = expert_send_dst_rank.numel()
    assert x.dtype == torch.bfloat16 and l1_weights.dtype == torch.bfloat16
    hidden_size = x.size(1)
    num_max_pool_tokens = int(sym_layout.num_max_pool_tokens)
    dummy_i32 = get_dummy_tensor()
    x_i32 = x.contiguous().view(torch.int32).view(-1)

    has_snapshots = len(_snapshots) >= 3
    if has_snapshots:
        dispatched_src_rank, dispatched_src_row, dispatched_meta_scalars = _snapshots[:3]
    else:
        dispatched_src_rank = dispatched_src_row = dispatched_meta_scalars = dummy_i32

    if layout == "tn":
        assert group_offs is not None, "tn layout requires group_offs"
        rhs = l1_weights
        OUT_M = hidden_size
        OUT_N = rhs.size(1)
        G = group_offs.numel() - 1
        out_fp16 = out_dtype == torch.float16
        out_shape = (G, OUT_N, OUT_M) if trans_c else (G, OUT_M, OUT_N)
        output = torch.empty(out_shape, device=x.device, dtype=out_dtype)
        assert group_offs.dtype == torch.int64, "tn group_offs must be int64"
        pos_args = (
            x_i32,
            expert_send_dst_rank,
            expert_send_dst_row,
            expert_send_count,
            expert_send_offset,
            dispatched_token_idx,
            sym_layout,
            rhs.contiguous().view(-1),
            output.view(-1),
            dummy_i32,
            expected_count,
            dummy_i32,
            group_offs,
            dispatched_src_rank,
            dispatched_src_row,
            dispatched_meta_scalars,
            0,
            int(OUT_M),
            int(OUT_N),
        )
        launch = _compile(
            OUT_N,
            OUT_M,
            num_max_pool_tokens,
            BM,
            BN,
            int(num_dispatch_cu),
            int(num_comm),
            out_fp16=out_fp16,
            layout="tn",
            trans_c=trans_c,
            G=G,
            has_snapshots=has_snapshots,
            num_xcd=2,  # fused TN optimum (post-CHUNK=4): balances L2 reuse vs dispatch-overlap scatter
        )
        launch(*pos_args, stream=torch.cuda.current_stream())
        return output, symm.pool, symm.weight_recv_buf, handle

    assert layout in ("nt", "nn"), f"unsupported layout {layout}"
    if layout == "nt":
        G, N, K = l1_weights.shape
        weight_flat = l1_weights.reshape(G * N, K).contiguous().view(-1)
    else:
        G, K, N = l1_weights.shape
        weight_flat = l1_weights.reshape(G * K, N).contiguous().view(-1)
    assert K == hidden_size, f"weight K={K} != activation K={hidden_size}"
    out_features = N
    c_n = out_features

    scoreboard = symm.scoreboard
    num_tile_blocks = dispatched_meta_scalars[1:2] if has_snapshots else symm.meta_scalars[1:2]
    nt_vmcnt = 3

    output = torch.empty((num_max_pool_tokens, out_features), dtype=x.dtype, device=x.device)

    output_flat = output.contiguous().view(-1)

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
        dummy_i32,
        dispatched_src_rank,
        dispatched_src_row,
        dispatched_meta_scalars,
        c_n,
        0,
        0,
    )

    if autotune:
        key = (out_features, hidden_size, num_max_pool_tokens, BM, BN, int(num_comm), layout)
        cached = _DISPATCH_AUTOTUNE_CACHE.get(key)
        if cached is None:
            cached = _autotune(
                pos_args,
                output_flat,
                out_features,
                hidden_size,
                num_max_pool_tokens,
                BM,
                BN,
                int(num_comm),
                scoreboard,
                None,
                layout,
            )
            _DISPATCH_AUTOTUNE_CACHE[key] = cached
        launch, _cfg = cached
    else:
        launch = _compile(
            out_features,
            hidden_size,
            num_max_pool_tokens,
            BM,
            BN,
            int(num_dispatch_cu),
            int(num_comm),
            int(nt_vmcnt),
            GROUP_M=int(GROUP_M),
            layout=layout,
            has_snapshots=has_snapshots,
        )
    launch(*pos_args, stream=torch.cuda.current_stream())
    return output, symm.pool, symm.weight_recv_buf, handle


def _autotune(
    pos_args,
    finite_view,
    out_features,
    hidden_size,
    num_max_pool_tokens,
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
                num_max_pool_tokens,
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
