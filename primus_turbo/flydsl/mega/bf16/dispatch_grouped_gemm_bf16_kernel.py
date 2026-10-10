###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

import functools
from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    extract_base_index,
)
from flydsl.expr.typing import AddressSpace, PointerType

from primus_turbo.flydsl.mega.bf16.dispatch_prologue_kernel import (
    DispatchHandle,
    run_dispatch_prologue,
)
from primus_turbo.flydsl.mega.bf16.ep_intranode import (
    _BLOCK_THREADS,
    dispatch_bf16_block,
    spin_until_flag_reaches,
)
from primus_turbo.flydsl.mega.bf16.gemm_bf16_kernel import (
    _make_shared_storage,
    gemm_bf16_tile,
)
from primus_turbo.flydsl.mega.bf16.grouped_gemm_bf16_kernel import (
    NUM_LDS_ROW_IDX_ENTRIES,
    grouped_gemm_bf16_variable_k_tile,
)
from primus_turbo.flydsl.mega.bf16.symm_buffer import (
    TOKEN_DTYPE,
    SymBuffer,
    Workspace,
    get_symm_buffer_for_mega_moe,
)
from primus_turbo.flydsl.mega.tune_utils import (
    Config,
    autotune,
)
from primus_turbo.flydsl.utils.gemm_helper import (
    make_bf16_fp16_tile_tensor,
    make_value_attrs,
    xcd_remap_pid,
)
from primus_turbo.flydsl.utils.prims import _readfirstlane_i32, cast, ld


@ASTRewriter.transform
def _wait_for_expert(thread_index, dispatch_flag_base, flag_idx, dispatch_expected, rank):
    """Block until every sender has pushed the expert at ``flag_idx`` and, by push order, all lower ones."""
    if thread_index == fx.Int32(0):
        spin_until_flag_reaches(dispatch_flag_base, flag_idx, dispatch_expected, "sys", rank, "dispatch")
    fx.gpu.barrier()


def _i64_base(tensor):
    return fx.arith.ArithValue(arith.index_cast(fx.T.i64(), extract_base_index(tensor)), signed=True)


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
    rank=0,
    num_experts=0,
    num_max_tokens_per_rank=0,
    num_topk=0,
):
    K = hidden_size
    is_tn = layout == "tn"
    assert num_comm % num_ranks == 0 and num_dispatch_cu % num_ranks == 0, (
        "comm blocks split evenly per dst rank"
    )
    num_chunks = num_dispatch_cu // num_ranks
    # The variable-K tile reads a padded LDS frame; the dense tiles read the unpadded one.
    lds_chunk_stride = 1152 if is_tn else 1024
    SharedStorage = _make_shared_storage(
        BLOCK_M,
        BLOCK_N,
        chunk_stride=lds_chunk_stride,
        num_lds_row_idx_entries=NUM_LDS_ROW_IDX_ENTRIES if is_tn else 0,
    )
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
        gemm_tile = functools.partial(gemm_bf16_tile, layout, pair_n=not out_fp16)
        assert out_features % BLOCK_N == 0, "out_features must be a multiple of BLOCK_N"
        n_blocks = out_features // BLOCK_N
        worst_case_tiles = num_max_pool_tokens // BLOCK_M
    num_pool_blocks = num_max_pool_tokens // BLOCK_M
    row_idx_ptr_ty = PointerType.get(elem_ty=fx.T.i32(), address_space=AddressSpace.Global, alignment=4)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def dispatch_grouped_gemm_kernel(
        INPUT_TOKENS: fx.Tensor,
        EXPERT_SEND_DST_RANK: fx.Tensor,
        EXPERT_SEND_COUNT: fx.Tensor,
        EXPERT_SEND_OFFSET: fx.Tensor,
        DISPATCHED_TOKEN_IDX: fx.Tensor,
        sym_buffer: SymBuffer,
        WEIGHTS: fx.Tensor,
        OUTPUT: fx.Tensor,
        TILE_TO_EXPERT: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        NUM_TOKENS_PER_EXPERT_PREFIX: fx.Tensor,
        NUM_TOKENS_PER_EXPERT: fx.Tensor,
        POOL_ROW_TO_RECV_TOKEN: fx.Tensor,
        DISPATCH_CHUNK_COUNTER: fx.Tensor,
        c_n: fx.Int32,
        out_m_rt: fx.Int32,
        out_n_rt: fx.Int32,
        DISPATCH_PARITY: fx.Tensor,
        DISPATCH_EXPECTED: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        comm_block_count = fx.Int32(num_dispatch_cu)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # build the layout from explicit dims (bf16 path -> TOKEN_DTYPE), then hoist region
        # pointers before dynamic branches (rewriter can't carry SymBuffer/Workspace)
        workspace = Workspace(
            sym_buffer.get_base_ptr(),
            num_ranks,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden_size,
            token_dtype=TOKEN_DTYPE,
        )
        dispatch_flag_base = workspace.get_dispatch_flag_ptr()
        dispatch_token_buffer = make_bf16_fp16_tile_tensor(
            workspace.get_dispatch_token_buffer_ptr(), fx.Int64(0), workspace.num_max_recv_tokens * K
        )
        dispatch_chunk_counter_ptr = extract_base_index(DISPATCH_CHUNK_COUNTER, address_space=1)
        # read epoch (already bumped by the bump kernel): parity -> bank, expected -> spin target
        dispatch_parity_resource = create_buffer_resource(DISPATCH_PARITY, max_size=True)
        dispatch_expected_resource = create_buffer_resource(DISPATCH_EXPECTED, max_size=True)
        dispatch_parity = cast(
            buffer_load(dispatch_parity_resource, fx.Int32(0), vec_width=1, dtype=fx.T.i64()), fx.T.i32()
        )
        dispatch_bank_offset = dispatch_parity * fx.Int32(num_pool_blocks)
        dispatch_expected = buffer_load(
            dispatch_expected_resource, dispatch_parity, vec_width=1, dtype=fx.T.i64()
        )

        input_resource = create_buffer_resource(INPUT_TOKENS, max_size=True)
        expert_send_dst_rank_resource = create_buffer_resource(EXPERT_SEND_DST_RANK, max_size=True)
        expert_send_count_resource = create_buffer_resource(EXPERT_SEND_COUNT, max_size=True)
        expert_send_offset_resource = create_buffer_resource(EXPERT_SEND_OFFSET, max_size=True)
        dispatched_token_idx_resource = create_buffer_resource(DISPATCHED_TOKEN_IDX, max_size=True)
        if const_expr(is_tn):
            num_tokens_per_expert_prefix_ptr = _i64_base(NUM_TOKENS_PER_EXPERT_PREFIX)
            num_tokens_per_expert_ptr = _i64_base(NUM_TOKENS_PER_EXPERT)
        else:
            tile_to_expert_resource = create_buffer_resource(TILE_TO_EXPERT, max_size=True)
            num_tile_blocks_resource = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
            pool_row_to_recv_token_base = _i64_base(POOL_ROW_TO_RECV_TOKEN)

        if block_index < comm_block_count:
            dispatch_bf16_block(
                sym_buffer,
                workspace,
                thread_index=thread_index,
                block_index=block_index,
                hidden_size=hidden_size,
                input_res=input_resource,
                expert_send_dst_rank_res=expert_send_dst_rank_resource,
                expert_send_count_res=expert_send_count_resource,
                expert_send_offset_res=expert_send_offset_resource,
                dispatched_token_idx_res=dispatched_token_idx_resource,
                dispatch_parity=dispatch_parity,
                dispatch_chunk_counter_ptr=dispatch_chunk_counter_ptr,
                num_chunks=num_chunks,
                num_ranks=num_ranks,
                num_experts_per_rank=num_comm // num_ranks,
                rank=rank,
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
                m_start = cast(ld(num_tokens_per_expert_prefix_ptr, group_idx, dtype=fx.T.i64()), fx.T.i32())
                num_tokens = cast(ld(num_tokens_per_expert_ptr, group_idx, dtype=fx.T.i64()), fx.T.i32())
                # An empty expert has nothing to wait for; the tile itself stores zeros.
                if num_tokens > fx.Int32(0):
                    _wait_for_expert(
                        thread_index,
                        dispatch_flag_base,
                        dispatch_bank_offset + group_idx,
                        dispatch_expected,
                        rank,
                    )
                # the pool is the gathered operand: B for dW1 (trans_c), A otherwise
                if const_expr(trans_c):
                    gemm_a, gemm_b, gemm_out_m, gemm_out_n = (
                        WEIGHTS,
                        dispatch_token_buffer,
                        out_n_rt,
                        out_m_rt,
                    )
                    row_idx_kwargs = {"b_row_idx": POOL_ROW_TO_RECV_TOKEN}
                else:
                    gemm_a, gemm_b, gemm_out_m, gemm_out_n = (
                        dispatch_token_buffer,
                        WEIGHTS,
                        out_m_rt,
                        out_n_rt,
                    )
                    row_idx_kwargs = {"a_row_idx": POOL_ROW_TO_RECV_TOKEN}
                grouped_gemm_bf16_variable_k_tile(
                    gemm_a,
                    gemm_b,
                    OUTPUT,
                    group_idx,
                    block_m,
                    block_n,
                    m_start,
                    m_start + num_tokens,
                    lds,
                    gemm_out_m,
                    gemm_out_n,
                    G=G,
                    OUT_M=OUT_M_g,
                    OUT_N=OUT_N_g,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    out_fp16=out_fp16,
                    lds_chunk_stride=lds_chunk_stride,
                    num_row_idx_entries=fx.Int32(workspace.num_pool_row_to_recv_token_entries),
                    num_gathered_rows=fx.Int32(workspace.num_max_recv_tokens),
                    **row_idx_kwargs,
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
                local_expert_idx = buffer_load(
                    tile_to_expert_resource, block_m, vec_width=1, dtype=fx.T.i32()
                )
                _wait_for_expert(
                    thread_index,
                    dispatch_flag_base,
                    dispatch_bank_offset + local_expert_idx,
                    dispatch_expected,
                    rank,
                )

                # A = the whole dispatch token buffer, gathered by row; B/C base = WEIGHTS/OUTPUT tensors.
                out_base = _i64_base(OUTPUT)
                w_base = _i64_base(WEIGHTS)
                # Rebase per-tile B/C/row-index bases in int64; C bounds via HW num_records 0x40000000.
                b_off = cast(local_expert_idx, fx.T.i64()) * fx.Int64(K * out_features * 2)
                c_off = cast(block_m, fx.T.i64()) * fx.Int64(BLOCK_M * 2) * cast(c_n, fx.T.i64())
                B_tile = make_bf16_fp16_tile_tensor(w_base, b_off, K * out_features)
                C_tile = make_bf16_fp16_tile_tensor(out_base, c_off, 0x40000000)
                row_idx_byte_offset = cast(block_m, fx.T.i64()) * fx.Int64(BLOCK_M * 4)
                row_idx_ptr = fx.inttoptr(
                    row_idx_ptr_ty, _readfirstlane_i32(pool_row_to_recv_token_base + row_idx_byte_offset)
                )
                gemm_tile(
                    dispatch_token_buffer,
                    B_tile,
                    C_tile,
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
                    n_tail=out_features % BLOCK_N,
                    a_row_idx=fx.Tensor(fx.make_view(row_idx_ptr, fx.make_layout(BLOCK_M, 1))),
                )

    grid_size = num_dispatch_cu + (TOTAL if is_tn else worst_case_tiles * n_blocks)
    return dispatch_grouped_gemm_kernel, grid_size


@functools.lru_cache(maxsize=4)
def _make_epoch_bump(addend):
    """Single-block kernel: flip parity, bump the new bank's expected by addend."""

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def epoch_bump_kernel(PARITY: fx.Tensor, EXPECTED: fx.Tensor):
        if fx.thread_idx.x == fx.Int32(0):
            parity_res = create_buffer_resource(PARITY, max_size=True)
            expected_res = create_buffer_resource(EXPECTED, max_size=True)
            new_parity = buffer_load(parity_res, fx.Int32(0), vec_width=1, dtype=fx.T.i64()) ^ fx.Int64(1)
            buffer_store(new_parity, parity_res, fx.Int32(0))
            idx = cast(new_parity, fx.T.i32())
            new_exp = buffer_load(expected_res, idx, vec_width=1, dtype=fx.T.i64()) + fx.Int64(addend)
            buffer_store(new_exp, expected_res, idx)

    return epoch_bump_kernel


@autotune(
    configs=[Config(num_dispatch_cu=cu, nt_vmcnt=3) for cu in (16, 32, 64)],
    key=[
        "out_features",
        "hidden_size",
        "num_max_pool_tokens",
        "BLOCK_M",
        "BLOCK_N",
        "num_comm",
        "GROUP_M",
        "layout_code",
        "trans_c",
        "G",
        "num_ranks",
    ],
    rep=5,
)
@flyc.jit
def _compiled_dispatch_grouped_gemm(
    INPUT_TOKENS,
    EXPERT_SEND_DST_RANK,
    EXPERT_SEND_COUNT,
    EXPERT_SEND_OFFSET,
    DISPATCHED_TOKEN_IDX,
    sym_buffer,
    WEIGHTS,
    OUTPUT,
    TILE_TO_EXPERT,
    NUM_TILE_BLOCKS,
    NUM_TOKENS_PER_EXPERT_PREFIX,
    NUM_TOKENS_PER_EXPERT,
    POOL_ROW_TO_RECV_TOKEN,
    DISPATCH_CHUNK_COUNTER,
    c_n: int,
    out_m_rt: int,
    out_n_rt: int,
    DISPATCH_PARITY,
    DISPATCH_EXPECTED,
    out_features: fx.Constexpr[int],
    hidden_size: fx.Constexpr[int],
    num_max_pool_tokens: fx.Constexpr[int],
    BLOCK_M: fx.Constexpr[int],
    BLOCK_N: fx.Constexpr[int],
    num_comm: fx.Constexpr[int],
    GROUP_M: fx.Constexpr[int],
    layout_code: fx.Constexpr[int],
    trans_c: fx.Constexpr[bool],
    G: fx.Constexpr[int],
    out_fp16: fx.Constexpr[bool],
    num_ranks: fx.Constexpr[int],
    rank: fx.Constexpr[int],
    num_dispatch_cu: fx.Constexpr[int],
    nt_vmcnt: fx.Constexpr[int],
    num_experts: fx.Constexpr[int],
    num_max_tokens_per_rank: fx.Constexpr[int],
    num_topk: fx.Constexpr[int],
    stream: fx.Stream,
):
    # layout_code: 0=nt, 1=nn, 2=tn; tn uses 2 XCDs, nt/nn use 8
    layout = ("nt", "nn", "tn")[int(layout_code)]
    num_xcd = 2 if layout == "tn" else 8
    # bump epoch on device (expected += num_ranks) before the GEMM; same-stream makes it visible
    _make_epoch_bump(int(num_ranks))(DISPATCH_PARITY, DISPATCH_EXPECTED).launch(
        grid=(1, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream
    )
    kernel, grid_size = _make_kernel(
        out_features,
        hidden_size,
        num_max_pool_tokens,
        BLOCK_M,
        BLOCK_N,
        int(num_dispatch_cu),
        int(num_comm),
        nt_vmcnt=int(nt_vmcnt),
        out_fp16=bool(out_fp16),
        GROUP_M=int(GROUP_M),
        layout=layout,
        trans_c=bool(trans_c),
        G=int(G),
        num_xcd=num_xcd,
        num_ranks=int(num_ranks),
        rank=int(rank),
        num_experts=int(num_experts),
        num_max_tokens_per_rank=int(num_max_tokens_per_rank),
        num_topk=int(num_topk),
    )
    kernel(
        INPUT_TOKENS,
        EXPERT_SEND_DST_RANK,
        EXPERT_SEND_COUNT,
        EXPERT_SEND_OFFSET,
        DISPATCHED_TOKEN_IDX,
        sym_buffer,
        WEIGHTS,
        OUTPUT,
        TILE_TO_EXPERT,
        NUM_TILE_BLOCKS,
        NUM_TOKENS_PER_EXPERT_PREFIX,
        NUM_TOKENS_PER_EXPERT,
        POOL_ROW_TO_RECV_TOKEN,
        DISPATCH_CHUNK_COUNTER,
        c_n,
        out_m_rt,
        out_n_rt,
        DISPATCH_PARITY,
        DISPATCH_EXPECTED,
        value_attrs=make_value_attrs(2, 0, "512,512"),
    ).launch(grid=(grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)


def dispatch_grouped_gemm_bf16_flydsl_kernel(
    x: torch.Tensor,
    l1_weights: torch.Tensor,
    group: torch.distributed.group,
    handle: Optional[DispatchHandle] = None,
    topk_idx: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    layout: str = "nt",
    BM=256,
    BN=256,
    GROUP_M=4,
    trans_c: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
):
    """Return ``(output, dispatch_token_buffer, handle)``; the buffer feeds the gathered dW2."""
    if handle is None:
        assert layout == "nt" and topk_idx is not None, "handle=None runs the prologue, forward (nt) only"
        handle, _ = run_dispatch_prologue(x, l1_weights, group, topk_idx, topk_weights)
    symm = get_symm_buffer_for_mega_moe()
    num_ranks = symm.num_ranks
    assert x.dtype == torch.bfloat16 and l1_weights.dtype == torch.bfloat16
    hidden_size = x.size(1)
    assert hidden_size == int(symm.hidden), f"x hidden {hidden_size} != SymmBuffer hidden {int(symm.hidden)}"
    num_max_pool_tokens = int(symm.num_max_pool_tokens)
    # Gathered rows address the whole dispatch token buffer with one 32-bit offset.
    buffer_bytes = (symm.num_max_recv_tokens + 1) * hidden_size * 2
    assert buffer_bytes <= 2**31 - 1, "dispatch token buffer exceeds 2 GiB"
    x_i32 = x.contiguous().view(torch.int32)

    assert layout in ("nt", "nn", "tn"), f"unsupported layout {layout}"
    out_fp16 = out_dtype == torch.float16
    layout_code = {"nt": 0, "nn": 1, "tn": 2}[layout]

    # per-layout operand/output prep; all three share one launch below.
    if layout == "tn":
        # tn wgrad: WEIGHTS=rhs activation, contract against the token pool; C is per-group.
        rhs = l1_weights
        OUT_M, OUT_N = (
            hidden_size,
            rhs.size(1),
        )  # _make_kernel wants (out_features, hidden_size)=(OUT_N, OUT_M)
        G = handle.num_tokens_per_expert_prefix.numel() - 1
        out_shape = (G, OUT_N, OUT_M) if trans_c else (G, OUT_M, OUT_N)
        output = torch.empty(out_shape, device=x.device, dtype=out_dtype)
        weight_arg, output_arg = rhs.contiguous(), flyc.from_torch_tensor(output)
        c_n, out_m_rt, out_n_rt = 0, int(OUT_M), int(OUT_N)
        out_features_ce, hidden_size_ce = OUT_N, OUT_M
    else:
        if layout == "nt":
            G, N, K = l1_weights.shape
            weight_flat = l1_weights.reshape(G * N, K).contiguous()
        else:
            G, K, N = l1_weights.shape
            weight_flat = l1_weights.reshape(G * K, N).contiguous()
        assert K == hidden_size, f"weight K={K} != activation K={hidden_size}"
        output = torch.empty((num_max_pool_tokens, N), dtype=x.dtype, device=x.device)
        weight_arg, output_arg = flyc.from_torch_tensor(weight_flat), output
        c_n, out_m_rt, out_n_rt = N, 0, 0
        out_features_ce, hidden_size_ce = N, hidden_size
        G, trans_c = 0, False  # nt/nn grid uses worst_case_tiles; C never transposed

    # epoch tensors are bumped by _make_epoch_bump inside _compiled; just pass them through
    _compiled_dispatch_grouped_gemm(
        x_i32,
        handle.expert_send_dst_rank,
        handle.expert_send_count,
        handle.expert_send_offset,
        handle.dispatched_token_idx,
        symm.get_sym_buffer(),
        weight_arg,
        output_arg,
        handle.tile_to_expert,
        handle.num_tile_blocks,
        handle.num_tokens_per_expert_prefix,
        handle.num_tokens_per_expert,
        handle.pool_row_to_recv_token,
        symm._dispatch_chunk_counter,
        c_n,
        out_m_rt,
        out_n_rt,
        DISPATCH_PARITY=symm._dispatch_parity,
        DISPATCH_EXPECTED=symm._dispatch_expected,
        out_features=int(out_features_ce),
        hidden_size=int(hidden_size_ce),
        num_max_pool_tokens=int(num_max_pool_tokens),
        BLOCK_M=int(BM),
        BLOCK_N=int(BN),
        num_comm=int(handle.expert_send_dst_rank.numel()),
        GROUP_M=int(GROUP_M),
        layout_code=int(layout_code),
        trans_c=bool(trans_c),
        G=int(G),
        out_fp16=bool(out_fp16),
        num_ranks=int(num_ranks),
        rank=int(symm.rank),
        num_experts=int(symm.num_experts),
        num_max_tokens_per_rank=int(symm.num_max_tokens_per_rank),
        num_topk=int(symm.num_topk),
        stream=torch.cuda.current_stream(),
    )
    return output, symm.dispatch_token_buffer, handle
