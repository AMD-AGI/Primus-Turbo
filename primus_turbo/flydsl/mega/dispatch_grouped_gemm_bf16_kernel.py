###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import functools
from typing import Optional, Tuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl import Config, autotune
from flydsl.expr import arith, const_expr
from flydsl.expr.buffer_ops import (
    _unwrap_value,
    buffer_load,
    create_buffer_resource,
    extract_base_index,
)
from flydsl.expr.typing import AddressSpace, PointerType

from primus_turbo.flydsl.gemm.gemm_bf16_kernel import (
    GEMM_TILE,
    _make_shared_storage,
    gemm_bf16_tn_variable_k_tile,
    load_go_i64,
)
from primus_turbo.flydsl.mega.dispatch_prologue_kernel import dispatch_prologue
from primus_turbo.flydsl.mega.ep_intranode import _BLOCK_THREADS, dispatch_bf16_tile
from primus_turbo.flydsl.mega.prims import ld, read_clock, spin_timed_out
from primus_turbo.flydsl.mega.sym_layout import SymLayout
from primus_turbo.flydsl.mega.symm_buffer import get_symm_buffer_for_mega_moe
from primus_turbo.flydsl.mega.tune_utils import _suppress_stdout_stderr
from primus_turbo.flydsl.utils.gemm_helper import make_value_attrs, xcd_remap_pid


@functools.lru_cache(maxsize=1)
def get_dummy_tensor():
    return torch.empty(1, dtype=torch.int32)


@functools.lru_cache(maxsize=256)
def _make_kernel(
    out_features,
    hidden_size,
    num_max_pool_tokens,
    BLOCK_M,
    BLOCK_N,
    num_dispatch_cu,
    num_comm,
    nt_vmcnt=3,
    out_fp16=False,
    GROUP_M=1,
    layout="nt",
    trans_c=False,
    G=0,
    num_xcd=8,
    num_ranks=8,
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
    NPB = num_max_pool_tokens // BLOCK_M

    def _i64(v):
        return fx.arith.ArithValue(fx.arith.extsi(fx.T.i64(), _unwrap_value(v)), signed=True)

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
        NUM_TILE_BLOCKS: fx.Tensor,
        GROUP_OFFS: fx.Tensor,
        c_n: fx.Int32,
        out_m_rt: fx.Int32,
        out_n_rt: fx.Int32,
        disp_parity: fx.Int32,
        expected_dispatch: fx.Int32,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        comm_block_count = fx.Int32(num_dispatch_cu)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        bank_offset = disp_parity * fx.Int32(NPB)
        expected_dispatch_i64 = _i64(expected_dispatch)

        input_resource = create_buffer_resource(INPUT_TOKENS, max_size=True)
        expert_send_dst_rank_resource = create_buffer_resource(EXPERT_SEND_DST_RANK, max_size=True)
        expert_send_dst_row_resource = create_buffer_resource(EXPERT_SEND_DST_ROW, max_size=True)
        expert_send_count_resource = create_buffer_resource(EXPERT_SEND_COUNT, max_size=True)
        expert_send_offset_resource = create_buffer_resource(EXPERT_SEND_OFFSET, max_size=True)
        dispatched_token_idx_resource = create_buffer_resource(DISPATCHED_TOKEN_IDX, max_size=True)
        if const_expr(is_tn):
            go = fx.rocdl.make_buffer_tensor(GROUP_OFFS, max_size=False, num_records_bytes=(G + 1) * 8)
            go_div = fx.logical_divide(go, fx.make_layout(1, 1))
        else:
            group_resource = create_buffer_resource(TILE_TO_GROUP, max_size=True)
            num_tile_blocks_resource = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)

        if block_index < comm_block_count:
            local_task_count = (
                fx.Int32(num_comm) - block_index + comm_block_count - fx.Int32(1)
            ) // comm_block_count
            for task_iteration in range(local_task_count):
                dispatch_bf16_tile(
                    sym_layout,
                    thread_index=thread_index,
                    hidden_size=hidden_size,
                    num_max_pool_tokens=num_max_pool_tokens,
                    input_res=input_resource,
                    expert_send_dst_rank_res=expert_send_dst_rank_resource,
                    expert_send_dst_row_res=expert_send_dst_row_resource,
                    expert_send_count_res=expert_send_count_resource,
                    expert_send_offset_res=expert_send_offset_resource,
                    dispatched_token_idx_res=dispatched_token_idx_resource,
                    task_index=block_index + task_iteration * comm_block_count,
                    signal=True,
                    block_m=BLOCK_M,
                    disp_parity=disp_parity,
                    num_ranks=num_ranks,
                )
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
                sb_base = sym_layout.dispatch_flag_ptr
                ge_blk = bank_offset + group_idx
                if thread_index == fx.Int32(0):
                    spin_start = read_clock()
                    sig = ld(sb_base, ge_blk, scope="sys", dtype=fx.T.i64())
                    while sig != expected_dispatch_i64:
                        fx.rocdl.s_sleep(fx.Int32(2))
                        if spin_timed_out(spin_start):
                            fx.printf(
                                "MEGA tn wgrad gate timeout: expert={} sig={} exp={}\n",
                                group_idx,
                                sig,
                                expected_dispatch_i64,
                            )
                            spin_start = read_clock()
                        sig = ld(sb_base, ge_blk, scope="sys", dtype=fx.T.i64())
                fx.gpu.barrier()
                pool_ptr_ty = PointerType.get(
                    elem_ty=fx.BFloat16.ir_type, address_space=AddressSpace.Global, alignment=16
                )
                pool_tensor = fx.make_view(
                    fx.inttoptr(pool_ptr_ty, sym_layout.dispatch_token_pool_ptr),
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
                sb_base = sym_layout.dispatch_flag_ptr
                g_idx = buffer_load(group_resource, block_m, vec_width=1, dtype=fx.T.i32())
                blk = bank_offset + g_idx
                if thread_index == fx.Int32(0):
                    spin_start = read_clock()
                    signal = ld(sb_base, blk, scope="sys", dtype=fx.T.i64())
                    while signal != expected_dispatch_i64:
                        fx.rocdl.s_sleep(fx.Int32(2))
                        if spin_timed_out(spin_start):
                            fx.printf(
                                "MEGA dispatch GEMM gate timeout: expert={} signal={} expected={}\n",
                                g_idx,
                                signal,
                                expected_dispatch_i64,
                            )
                            spin_start = read_clock()
                        signal = ld(sb_base, blk, scope="sys", dtype=fx.T.i64())
                fx.gpu.barrier()

                gbase = g_idx * fx.Int32(K) * c_n
                pool_ptr_ty = PointerType.get(
                    elem_ty=fx.BFloat16.ir_type, address_space=AddressSpace.Global, alignment=16
                )

                a_byte_off = _i64(block_m) * fx.Int64(BLOCK_M * K * 2)
                c_byte_off = _i64(block_m) * fx.Int64(BLOCK_M * out_features * 2)
                pool_tensor = fx.make_view(
                    fx.inttoptr(pool_ptr_ty, sym_layout.dispatch_token_pool_ptr + a_byte_off),
                    fx.make_layout(BLOCK_M * K, 1),
                )
                out_base = fx.arith.ArithValue(
                    arith.index_cast(fx.T.i64(), extract_base_index(OUTPUT)), signed=True
                )
                out_tile = fx.make_view(
                    fx.inttoptr(pool_ptr_ty, out_base + c_byte_off),
                    fx.make_layout(BLOCK_M * out_features, 1),
                )
                gemm_tile(
                    pool_tensor,
                    WEIGHTS,
                    out_tile,
                    fx.Int32(BLOCK_M),
                    c_n,
                    lds,
                    fx.Int32(0),
                    block_n,
                    K=K,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    out_fp16=out_fp16,
                    nt_vmcnt=nt_vmcnt,
                    b_group_base=gbase,
                )

    grid_size = num_dispatch_cu + (TOTAL if is_tn else worst_case_tiles * n_blocks)
    return dispatch_grouped_gemm_kernel, grid_size


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
    num_xcd=8,
    num_ranks=8,
):
    kernel, grid_size = _make_kernel(
        out_features,
        hidden_size,
        num_max_pool_tokens,
        BLOCK_M,
        BLOCK_N,
        num_dispatch_cu,
        num_comm,
        nt_vmcnt=nt_vmcnt,
        out_fp16=out_fp16,
        GROUP_M=GROUP_M,
        layout=layout,
        trans_c=trans_c,
        G=G,
        num_xcd=num_xcd,
        num_ranks=num_ranks,
    )

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
        NUM_TILE_BLOCKS,
        GROUP_OFFS,
        c_n: int,
        out_m_rt: int,
        out_n_rt: int,
        disp_parity: int,
        expected_dispatch: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(
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
            NUM_TILE_BLOCKS,
            GROUP_OFFS,
            c_n,
            out_m_rt,
            out_n_rt,
            disp_parity,
            expected_dispatch,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def _rewind_dispatch_flag(kwargs):
    # tuning-only: rewind never-reset flag so each rerun matches baked expected
    symm = get_symm_buffer_for_mega_moe()
    p = int(kwargs["disp_parity"])
    base = int(kwargs["expected_dispatch"]) - int(symm.world)
    npb = int(symm.num_max_pool_tokens) // int(kwargs["BLOCK_M"])
    # reset local bank, make it visible, THEN rendezvous so no next-rep push
    # lands before every rank has rewound (else pushes get zeroed -> undercount).
    symm.dispatch_flag[p * npb : (p + 1) * npb].fill_(base)
    torch.cuda.synchronize()
    torch.distributed.barrier(symm.group)


@autotune(
    configs=[Config(num_dispatch_cu=cu, nt_vmcnt=v) for cu in (16, 32, 64) for v in (3, 4)],
    key=[
        "out_features",
        "hidden_size",
        "num_max_pool_tokens",
        "BLOCK_M",
        "BLOCK_N",
        "num_comm",
        "GROUP_M",
        "is_nn",
        "num_ranks",
    ],
    warmup=0,
    post_hook=_rewind_dispatch_flag,
)
@flyc.jit
def _compiled_dispatch_grouped_gemm(
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
    NUM_TILE_BLOCKS,
    GROUP_OFFS,
    c_n: int,
    out_m_rt: int,
    out_n_rt: int,
    disp_parity: int,
    expected_dispatch: int,
    out_features: fx.Constexpr[int],
    hidden_size: fx.Constexpr[int],
    num_max_pool_tokens: fx.Constexpr[int],
    BLOCK_M: fx.Constexpr[int],
    BLOCK_N: fx.Constexpr[int],
    num_comm: fx.Constexpr[int],
    GROUP_M: fx.Constexpr[int],
    is_nn: fx.Constexpr[bool],
    num_ranks: fx.Constexpr[int],
    num_dispatch_cu: fx.Constexpr[int],
    nt_vmcnt: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    layout = "nn" if is_nn else "nt"
    kernel, grid_size = _make_kernel(
        out_features,
        hidden_size,
        num_max_pool_tokens,
        BLOCK_M,
        BLOCK_N,
        int(num_dispatch_cu),
        int(num_comm),
        nt_vmcnt=int(nt_vmcnt),
        GROUP_M=int(GROUP_M),
        layout=layout,
        num_ranks=int(num_ranks),
    )
    kernel(
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
        NUM_TILE_BLOCKS,
        GROUP_OFFS,
        c_n,
        out_m_rt,
        out_n_rt,
        disp_parity,
        expected_dispatch,
        value_attrs=make_value_attrs(2, 0, "512,512"),
    ).launch(grid=(grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)


def dispatch_grouped_gemm_bf16(
    x: torch.Tensor,
    l1_weights: torch.Tensor,
    group: torch.distributed.group,
    handle: Optional[Tuple] = None,
    topk_idx: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    layout: str = "nt",
    num_dispatch_cu: int = 16,
    BM=256,
    BN=256,
    GROUP_M=4,
    pool_mult: int = 2,
    trans_c: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
):
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
                num_ranks=symm.world,
                rank=symm.rank,
                experts_per_rank=experts_per_rank,
                block_m=BM,
                num_max_pool_tokens=symm.num_max_pool_tokens,
            )
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
        tile_to_expert,
        _expected_count,
        _,
        group_offs,
    ) = handle

    num_comm = expert_send_dst_rank.numel()
    num_ranks = symm.world
    assert x.dtype == torch.bfloat16 and l1_weights.dtype == torch.bfloat16
    hidden_size = x.size(1)
    num_max_pool_tokens = int(sym_layout.num_max_pool_tokens)
    dummy_i32 = get_dummy_tensor()
    x_i32 = x.contiguous().view(torch.int32).view(-1)

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
        # tn goes through _compile (not autotuned), advance parity here.
        disp_parity, expected_dispatch = symm.next_dispatch()
        pos_args = (
            x_i32,
            expert_send_dst_rank,
            expert_send_dst_row,
            expert_send_count,
            expert_send_offset,
            dispatched_token_idx,
            sym_layout,
            rhs.contiguous(),
            output.view(-1),
            dummy_i32,
            dummy_i32,
            group_offs,
            0,
            int(OUT_M),
            int(OUT_N),
            disp_parity,
            int(expected_dispatch),
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
            num_xcd=2,
            num_ranks=int(num_ranks),
        )
        launch(*pos_args, stream=torch.cuda.current_stream())
        return output, symm.dispatch_token_pool, symm.weight_recv_buf, handle

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

    num_tile_blocks = symm.meta_scalars[1:2]

    output = torch.empty((num_max_pool_tokens, out_features), dtype=x.dtype, device=x.device)

    disp_parity, expected_dispatch = symm.next_dispatch()

    with _suppress_stdout_stderr():
        _compiled_dispatch_grouped_gemm(
            x_i32,
            expert_send_dst_rank,
            expert_send_dst_row,
            expert_send_count,
            expert_send_offset,
            dispatched_token_idx,
            sym_layout,
            weight_flat,
            output,
            tile_to_expert,
            num_tile_blocks,
            dummy_i32,
            c_n,
            0,
            0,
            disp_parity=disp_parity,
            expected_dispatch=int(expected_dispatch),
            out_features=int(out_features),
            hidden_size=int(hidden_size),
            num_max_pool_tokens=int(num_max_pool_tokens),
            BLOCK_M=int(BM),
            BLOCK_N=int(BN),
            num_comm=int(num_comm),
            GROUP_M=int(GROUP_M),
            is_nn=(layout == "nn"),
            num_ranks=int(num_ranks),
            stream=torch.cuda.current_stream(),
        )
    return output, symm.dispatch_token_pool, symm.weight_recv_buf, handle
