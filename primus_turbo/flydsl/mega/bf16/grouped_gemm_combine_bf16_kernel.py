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

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
    extract_base_index,
)

from primus_turbo.flydsl.mega.bf16.ep_intranode import (
    combine_bf16_tile,
    spin_until_flag_reaches,
    topk_reduce_bf16_tile,
)
from primus_turbo.flydsl.mega.bf16.gemm_bf16_kernel import (
    _make_shared_storage,
    gemm_bf16_tile,
)
from primus_turbo.flydsl.mega.bf16.symm_buffer import (
    TOKEN_DTYPE,
    SymBuffer,
    Workspace,
    get_symm_buffer_for_mega_moe,
)
from primus_turbo.flydsl.utils.gemm_helper import (
    make_bf16_fp16_tile_tensor,
    make_value_attrs,
)
from primus_turbo.flydsl.utils.prims import (
    atomic_add,
    cast,
)

_WARP = 64
_BLOCK_THREADS = 512


_PVEC = 8
_NUM_WARPS = _BLOCK_THREADS // _WARP

_LAYOUTS = ("nt", "nn")
_LAYOUT_CODES = {name: code for code, name in enumerate(_LAYOUTS)}
# a GEMM tile with K <= _SHORT_GEMM_MAX_K finishes too fast for the push to keep up on 32 combine CUs
_SHORT_GEMM_MAX_K = 2048
_NUM_COMBINE_CU_SHORT_GEMM = 64
_NUM_COMBINE_CU_LONG_GEMM = 32
# parts of one segment go to consecutive blocks, spreading its push over more CUs
_PARTS_PER_SEGMENT = 2
# share of the real GEMM tiles dispatched before the combine blocks, so those do not idle on CUs from t=0
_LEAD_GEMM_PERCENT = 25
# reduce blocks; more warps shorten the reduce that trails the GEMM
_NUM_REDUCE_CU = 512


def _block_role(block_idx, num_lead_gemm_blocks, num_combine_cu, num_gemm_blocks):
    """Map a workgroup to (is_combine, is_reduce, gemm_block_idx, combine_block_idx, reduce_block_idx)."""
    combine_start = num_lead_gemm_blocks
    reduce_start = fx.Int32(num_combine_cu + num_gemm_blocks)
    is_combine = (block_idx >= combine_start) & (block_idx < combine_start + fx.Int32(num_combine_cu))
    is_reduce = block_idx >= reduce_start
    gemm_block_idx = arith.select(block_idx < combine_start, block_idx, block_idx - fx.Int32(num_combine_cu))
    return is_combine, is_reduce, gemm_block_idx, block_idx - combine_start, block_idx - reduce_start


def _make_grouped_gemm_combine(
    out_features,
    hidden_size,
    num_max_pool_tokens,
    BLOCK_M,
    BLOCK_N,
    num_combine_cu,
    num_max_routes,
    num_topk,
    num_experts,
    rank,
    num_ranks=0,
    num_max_tokens_per_rank=0,
    nt_vmcnt=3,
    out_fp16=False,
    layout="nt",
    has_grad_gate=False,
):
    K = hidden_size
    gemm_tile = functools.partial(gemm_bf16_tile, layout, pair_n=not out_fp16)
    assert out_features % BLOCK_N == 0, "out_features must be a multiple of BLOCK_N"
    assert num_max_pool_tokens % BLOCK_M == 0, "num_max_pool_tokens must be a multiple of BLOCK_M"
    assert out_features % _PVEC == 0, "out_features must be a multiple of 8 (bf16 vec)"
    assert num_topk >= 1, "num_topk must be >= 1"
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)
    num_tiles_n = out_features // BLOCK_N
    worst_case_tiles = num_max_pool_tokens // BLOCK_M
    num_gemm_blocks = worst_case_tiles * num_tiles_n
    combine_token_bytes = num_max_routes * out_features * 2

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def grouped_gemm_combine_kernel(
        ACT: fx.Tensor,
        WEIGHTS: fx.Tensor,
        TILE_TO_EXPERT: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        RECV_DST_RANK: fx.Tensor,
        RECV_START_ROW: fx.Tensor,
        RECV_FOLD_COUNT: fx.Tensor,
        POOL_ROW_TO_ROUTE: fx.Tensor,
        POOL_ROW_TO_RECV_TOKEN: fx.Tensor,
        RECV_TOKEN_TO_POOL_ROWS: fx.Tensor,
        OUTPUT: fx.Tensor,
        TOPK_INDICES: fx.Tensor,
        NUM_TOKENS_PER_RANK: fx.Tensor,
        GRAD_GATE: fx.Tensor,
        GRAD_TOPK_WEIGHTS: fx.Tensor,
        sym_buffer: SymBuffer,
        c_n: fx.Int32,
        COMBINE_PARITY: fx.Tensor,
        COMBINE_EXPECTED: fx.Tensor,
        REDUCE_EXPECTED: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # token pools are out_features-wide (the model hidden), not the down-proj K (hidden_size)
        workspace = Workspace(
            sym_buffer.get_base_ptr(),
            num_ranks,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            out_features,
            token_dtype=TOKEN_DTYPE,
        )
        # read epoch (already bumped by the bump kernel): parity -> bank, expected -> spin target
        combine_parity_res = create_buffer_resource(COMBINE_PARITY, max_size=True)
        combine_expected_res = create_buffer_resource(COMBINE_EXPECTED, max_size=True)
        reduce_expected_res = create_buffer_resource(REDUCE_EXPECTED, max_size=True)
        combine_parity = cast(
            buffer_load(combine_parity_res, fx.Int32(0), vec_width=1, dtype=fx.T.i64()), fx.T.i32()
        )
        combine_bank = combine_parity * fx.Int32(worst_case_tiles)
        reduce_bank = combine_parity * fx.Int32(num_max_routes)
        expected_combine = buffer_load(combine_expected_res, combine_parity, vec_width=1, dtype=fx.T.i64())
        expected_reduce = buffer_load(reduce_expected_res, combine_parity, vec_width=1, dtype=fx.T.i64())

        combine_flag_base = workspace.get_combine_flag_ptr()
        reduce_flag_base = workspace.get_reduce_flag_ptr()
        l2_token_buffer_base = workspace.get_l2_token_buffer_ptr()
        combine_token_res = create_buffer_resource_from_addr(
            workspace.get_combine_token_buffer_ptr(), num_records_bytes=combine_token_bytes
        )
        combine_gate_res = (
            create_buffer_resource_from_addr(
                workspace.get_combine_gate_ptr(), num_records_bytes=num_max_routes * 4
            )
            if has_grad_gate
            else None
        )
        # recv tables ride the handle (per-forward), NOT shared symm -> else bwd reads stale
        recv_dst_rank_res = create_buffer_resource(RECV_DST_RANK, max_size=True)
        recv_start_row_res = create_buffer_resource(RECV_START_ROW, max_size=True)
        recv_fold_count_res = create_buffer_resource(RECV_FOLD_COUNT, max_size=True)
        pool_row_to_route_res = create_buffer_resource(POOL_ROW_TO_ROUTE, max_size=True)
        pool_row_to_recv_token_res = create_buffer_resource(POOL_ROW_TO_RECV_TOKEN, max_size=True)
        recv_token_to_pool_rows_res = create_buffer_resource(RECV_TOKEN_TO_POOL_ROWS, max_size=True)

        tile_to_expert_res = create_buffer_resource(TILE_TO_EXPERT, max_size=True)
        num_tile_blocks_res = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
        output_res = create_buffer_resource(OUTPUT, max_size=True)
        topk_indices_res = create_buffer_resource(TOPK_INDICES, max_size=True)
        num_tokens_res = create_buffer_resource(NUM_TOKENS_PER_RANK, max_size=True)
        num_tiles_m = buffer_load(num_tile_blocks_res, fx.Int32(0), vec_width=1, dtype=fx.T.i32())
        grad_gate_res = create_buffer_resource(GRAD_GATE, max_size=True) if has_grad_gate else None
        grad_topk_weights_res = (
            create_buffer_resource(GRAD_TOPK_WEIGHTS, max_size=True) if has_grad_gate else None
        )

        num_lead_gemm_blocks = num_tiles_m * fx.Int32(num_tiles_n * _LEAD_GEMM_PERCENT // 100)
        is_combine, is_reduce, gemm_block_idx, combine_block_idx, reduce_block_idx = _block_role(
            block_index, num_lead_gemm_blocks, num_combine_cu, num_gemm_blocks
        )
        if is_combine:
            # combine role: push the fold rows of every num_combine_cu-th segment part to their source ranks
            num_segment_parts = num_experts * _PARTS_PER_SEGMENT
            num_local_segment_parts = (
                fx.Int32(num_segment_parts) - combine_block_idx + fx.Int32(num_combine_cu - 1)
            ) // fx.Int32(num_combine_cu)
            # every tile below next_tile_m is already confirmed, so each tile is polled once
            next_tile_m = fx.Int32(0)
            for segment_part_iter in range(num_local_segment_parts):
                segment_part_idx = combine_block_idx + segment_part_iter * fx.Int32(num_combine_cu)
                segment_idx = segment_part_idx // fx.Int32(_PARTS_PER_SEGMENT)
                part_idx = segment_part_idx % fx.Int32(_PARTS_PER_SEGMENT)
                start_row = buffer_load(recv_start_row_res, segment_idx, vec_width=1, dtype=fx.T.i32())
                fold_count = buffer_load(recv_fold_count_res, segment_idx, vec_width=1, dtype=fx.T.i32())
                rows_per_part = (fold_count + fx.Int32(_PARTS_PER_SEGMENT - 1)) // fx.Int32(
                    _PARTS_PER_SEGMENT
                )
                part_offset = part_idx * rows_per_part
                num_part_rows = arith.select(
                    fold_count - part_offset < rows_per_part, fold_count - part_offset, rows_per_part
                )
                if num_part_rows > fx.Int32(0):
                    # a fold row's other pool rows lie in lower tiles, so wait for the whole prefix
                    last_tile_m = (start_row + part_offset + num_part_rows - fx.Int32(1)) // fx.Int32(BLOCK_M)
                    if thread_index == fx.Int32(0):
                        polled_tile_m = next_tile_m
                        while polled_tile_m <= last_tile_m:
                            spin_until_flag_reaches(
                                combine_flag_base,
                                combine_bank + polled_tile_m,
                                expected_combine,
                                "agent",
                                rank,
                                "combine",
                            )
                            polled_tile_m = polled_tile_m + fx.Int32(1)
                    next_tile_m = arith.select(
                        last_tile_m < next_tile_m, next_tile_m, last_tile_m + fx.Int32(1)
                    )
                    fx.rocdl.s_waitcnt(0)
                    fx.gpu.barrier()
                    combine_bf16_tile(
                        sym_buffer,
                        workspace,
                        thread_index=thread_index,
                        segment_idx=segment_idx,
                        first_fold_row=start_row + part_offset,
                        num_fold_rows=num_part_rows,
                        recv_dst_rank_res=recv_dst_rank_res,
                        pool_row_to_route_res=pool_row_to_route_res,
                        pool_row_to_recv_token_res=pool_row_to_recv_token_res,
                        recv_token_to_pool_rows_res=recv_token_to_pool_rows_res,
                        epoch=expected_reduce,
                        reduce_bank=reduce_bank,
                        grad_gate_res=grad_gate_res,
                        has_grad_gate=has_grad_gate,
                    )
        elif is_reduce:
            # reduce role: align empty tile_m flags to the never-reset expected, then sum primary routes
            num_empty_tiles_m = fx.Int32(worst_case_tiles) - num_tiles_m
            num_local_empty_tiles_m = (
                num_empty_tiles_m - reduce_block_idx + fx.Int32(_NUM_REDUCE_CU - 1)
            ) // fx.Int32(_NUM_REDUCE_CU)
            for empty_iter in range(num_local_empty_tiles_m):
                empty_tile_m = num_tiles_m + reduce_block_idx + empty_iter * fx.Int32(_NUM_REDUCE_CU)
                if thread_index == fx.Int32(0):
                    atomic_add(combine_flag_base, combine_bank + empty_tile_m, fx.Int64(num_tiles_n))
            topk_reduce_bf16_tile(
                thread_index=thread_index,
                reduce_block_idx=reduce_block_idx,
                num_reduce_warps=_NUM_REDUCE_CU * _NUM_WARPS,
                num_topk=num_topk,
                hidden=out_features,
                num_experts=num_experts,
                num_experts_per_rank=num_experts // num_ranks,
                num_max_routes=num_max_routes,
                rank=rank,
                combine_token_res=combine_token_res,
                output_res=output_res,
                topk_indices_res=topk_indices_res,
                num_tokens_res=num_tokens_res,
                reduce_flag_base=reduce_flag_base,
                reduce_bank=reduce_bank,
                epoch=expected_reduce,
                combine_gate_res=combine_gate_res,
                grad_topk_weights_res=grad_topk_weights_res,
                has_grad_gate=has_grad_gate,
            )
        else:
            # GEMM role: one (tile_m, tile_n) tile per block; blocks past the real tiles exit
            tile_m = gemm_block_idx // fx.Int32(num_tiles_n)
            tile_n = gemm_block_idx % fx.Int32(num_tiles_n)
            if tile_m < num_tiles_m:
                expert_idx = buffer_load(tile_to_expert_res, tile_m, vec_width=1, dtype=fx.T.i32())
                # A/B base = ACT/WEIGHTS tensors; C base = l2_token_buffer (int64 symm addr).
                act_base = fx.arith.ArithValue(
                    fx.arith.index_cast(fx.T.i64(), extract_base_index(ACT)), signed=True
                )
                w_base = fx.arith.ArithValue(
                    fx.arith.index_cast(fx.T.i64(), extract_base_index(WEIGHTS)), signed=True
                )
                # Rebase per-tile base in int64 (pool >4GB), voffset stays int32; C sized via 0x40000000.
                a_off = cast(tile_m, fx.T.i64()) * fx.Int64(BLOCK_M * K * 2)
                b_off = cast(expert_idx, fx.T.i64()) * fx.Int64(K * out_features * 2)
                c_off = cast(tile_m, fx.T.i64()) * fx.Int64(BLOCK_M * 2) * cast(c_n, fx.T.i64())
                A_tile = make_bf16_fp16_tile_tensor(act_base, a_off, BLOCK_M * K)
                B_tile = make_bf16_fp16_tile_tensor(w_base, b_off, K * out_features)
                C_tile = make_bf16_fp16_tile_tensor(l2_token_buffer_base, c_off, 0x40000000)
                gemm_tile(
                    A_tile,
                    B_tile,
                    C_tile,
                    fx.Int32(BLOCK_M),
                    c_n,
                    lds,
                    fx.Int32(0),
                    tile_n,
                    K=K,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    out_fp16=out_fp16,
                    nt_vmcnt=nt_vmcnt,
                    n_tail=out_features % BLOCK_N,
                    c_cache_modifier=18,  # sc1|nt: agent-visible non-temporal local stage.
                )
                fx.rocdl.s_waitcnt(0)
                fx.gpu.barrier()
                # Keep a separator: LLVM folds adjacent barriers, but two rendezvous are required.
                fx.rocdl.s_waitcnt(0)
                fx.gpu.barrier()
                if thread_index == fx.Int32(0):
                    atomic_add(combine_flag_base, combine_bank + tile_m, fx.Int64(1))

    return grouped_gemm_combine_kernel


@functools.lru_cache(maxsize=4)
def _make_epoch_bump(add_combine, add_reduce):
    """Single-block kernel: flip parity, bump combine and reduce expected."""

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def epoch_bump_kernel(PARITY: fx.Tensor, COMBINE_EXP: fx.Tensor, REDUCE_EXP: fx.Tensor):
        if fx.thread_idx.x == fx.Int32(0):
            parity_res = create_buffer_resource(PARITY, max_size=True)
            combine_res = create_buffer_resource(COMBINE_EXP, max_size=True)
            reduce_res = create_buffer_resource(REDUCE_EXP, max_size=True)
            new_parity = buffer_load(parity_res, fx.Int32(0), vec_width=1, dtype=fx.T.i64()) ^ fx.Int64(1)
            buffer_store(new_parity, parity_res, fx.Int32(0))
            idx = cast(new_parity, fx.T.i32())
            new_combine = buffer_load(combine_res, idx, vec_width=1, dtype=fx.T.i64()) + fx.Int64(add_combine)
            buffer_store(new_combine, combine_res, idx)
            new_reduce = buffer_load(reduce_res, idx, vec_width=1, dtype=fx.T.i64()) + fx.Int64(add_reduce)
            buffer_store(new_reduce, reduce_res, idx)

    return epoch_bump_kernel


@flyc.jit
def _compiled_grouped_gemm_combine(
    ACT,
    WEIGHTS,
    TILE_TO_EXPERT,
    NUM_TILE_BLOCKS,
    RECV_DST_RANK,
    RECV_START_ROW,
    RECV_FOLD_COUNT,
    POOL_ROW_TO_ROUTE,
    POOL_ROW_TO_RECV_TOKEN,
    RECV_TOKEN_TO_POOL_ROWS,
    OUTPUT,
    TOPK_INDICES,
    NUM_TOKENS_PER_RANK,
    GRAD_GATE,
    GRAD_TOPK_WEIGHTS,
    sym_buffer,
    c_n,
    COMBINE_PARITY,
    COMBINE_EXPECTED,
    REDUCE_EXPECTED,
    out_features: fx.Constexpr[int],
    hidden_size: fx.Constexpr[int],
    num_max_pool_tokens: fx.Constexpr[int],
    BLOCK_M: fx.Constexpr[int],
    BLOCK_N: fx.Constexpr[int],
    num_max_routes: fx.Constexpr[int],
    num_topk: fx.Constexpr[int],
    num_experts: fx.Constexpr[int],
    rank: fx.Constexpr[int],
    num_ranks: fx.Constexpr[int],
    num_max_tokens_per_rank: fx.Constexpr[int],
    layout_code: fx.Constexpr[int],
    has_grad_gate: fx.Constexpr[bool],
    out_fp16: fx.Constexpr[bool],
    stream: fx.Stream,
    nt_vmcnt: fx.Constexpr[int] = 3,
    agpr_alloc: fx.Constexpr[int] = 0,
    waves: fx.Constexpr[int] = 2,
):
    num_tiles_n = out_features // BLOCK_N
    num_gemm_blocks = (num_max_pool_tokens // BLOCK_M) * num_tiles_n
    num_combine_cu = (
        _NUM_COMBINE_CU_SHORT_GEMM if hidden_size <= _SHORT_GEMM_MAX_K else _NUM_COMBINE_CU_LONG_GEMM
    )
    kernel = _make_grouped_gemm_combine(
        out_features,
        hidden_size,
        num_max_pool_tokens,
        BLOCK_M,
        BLOCK_N,
        num_combine_cu,
        num_max_routes,
        num_topk,
        num_experts,
        rank,
        num_ranks,
        num_max_tokens_per_rank,
        nt_vmcnt,
        out_fp16,
        _LAYOUTS[layout_code],
        has_grad_gate,
    )
    # reduce blocks come last, so they are dispatched only after every GEMM and combine block
    grid_size = num_gemm_blocks + num_combine_cu + _NUM_REDUCE_CU
    # bump epoch on device (combine += num_tiles_n, reduce += 1) before the GEMM; same-stream visible
    _make_epoch_bump(int(num_tiles_n), 1)(COMBINE_PARITY, COMBINE_EXPECTED, REDUCE_EXPECTED).launch(
        grid=(1, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream
    )
    kernel(
        ACT,
        WEIGHTS,
        TILE_TO_EXPERT,
        NUM_TILE_BLOCKS,
        RECV_DST_RANK,
        RECV_START_ROW,
        RECV_FOLD_COUNT,
        POOL_ROW_TO_ROUTE,
        POOL_ROW_TO_RECV_TOKEN,
        RECV_TOKEN_TO_POOL_ROWS,
        OUTPUT,
        TOPK_INDICES,
        NUM_TOKENS_PER_RANK,
        GRAD_GATE,
        GRAD_TOPK_WEIGHTS,
        sym_buffer,
        c_n,
        COMBINE_PARITY,
        COMBINE_EXPECTED,
        REDUCE_EXPECTED,
        value_attrs=make_value_attrs(waves, agpr_alloc, "512,512"),
    ).launch(grid=(grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)


def grouped_gemm_combine_bf16_flydsl_kernel(
    x,
    l2_weights,
    handle,
    *,
    topk_indices,
    grad_gate=None,
    layout="nt",
    BM=256,
    BN=256,
):
    assert layout in ("nt", "nn"), f"unsupported combine layout {layout}"
    assert x.dtype == torch.bfloat16 and l2_weights.dtype == torch.bfloat16
    symm = get_symm_buffer_for_mega_moe()
    sym_buffer = symm.get_sym_buffer()
    num_max_pool_tokens, hidden_size = x.shape
    if layout == "nt":
        G, N, K = l2_weights.shape
    else:
        G, K, N = l2_weights.shape
    assert K == hidden_size, f"weight K={K} != activation K={hidden_size}"
    out_features = N
    assert out_features == int(symm.hidden), (
        f"out_features {out_features} != SymmBuffer hidden {int(symm.hidden)}"
    )
    assert num_max_pool_tokens == int(symm.num_max_pool_tokens), "x rows must match SymmBuffer pool capacity"
    num_topk = int(symm.num_topk)
    num_experts = int(symm.num_experts)
    assert num_topk >= 1 and num_experts > 0, "topk reduce needs num_topk>=1 and num_experts>0"
    assert topk_indices.dtype == torch.int64, "the reduce reads topk_indices as i64"

    # combine blocks spin on GEMM tiles, so they must leave CUs for the GEMM to finish
    assert _NUM_COMBINE_CU_SHORT_GEMM < torch.cuda.get_device_properties(x.device).multi_processor_count
    has_grad_gate = grad_gate is not None
    # Pass 2D: kernel advances ACT base per-tile in int64 (flat MxK overflows int32 ABI).
    if layout == "nt":
        weight_flat = l2_weights.reshape(G * N, K).contiguous()
    else:
        weight_flat = l2_weights.reshape(G * K, N).contiguous()
    output = torch.empty(
        int(symm.num_max_tokens_per_rank), out_features, dtype=torch.bfloat16, device=x.device
    )
    grad_topk_weights = (
        torch.empty(int(symm.num_max_routes), dtype=torch.float32, device=x.device) if has_grad_gate else None
    )

    _compiled_grouped_gemm_combine(
        x.contiguous(),
        flyc.from_torch_tensor(weight_flat),
        handle.tile_to_expert,
        handle.num_tile_blocks,
        handle.combine_recv_dst_rank,
        handle.combine_recv_start_row,
        handle.combine_recv_fold_count,
        handle.pool_row_to_route,
        handle.pool_row_to_recv_token,
        handle.recv_token_to_pool_rows,
        output,
        topk_indices.contiguous(),
        symm.num_tokens_per_rank,
        # without a grad gate both gate arguments are traced out; any device i32 tensor fills them
        grad_gate.contiguous() if has_grad_gate else handle.num_tile_blocks,
        grad_topk_weights if has_grad_gate else handle.num_tile_blocks,
        sym_buffer,
        out_features,
        COMBINE_PARITY=symm._combine_parity,
        COMBINE_EXPECTED=symm._combine_expected,
        REDUCE_EXPECTED=symm._reduce_expected,
        out_features=out_features,
        hidden_size=hidden_size,
        num_max_pool_tokens=num_max_pool_tokens,
        BLOCK_M=BM,
        BLOCK_N=BN,
        num_max_routes=int(symm.num_max_routes),
        num_topk=num_topk,
        num_experts=num_experts,
        rank=int(symm.rank),
        num_ranks=int(symm.num_ranks),
        num_max_tokens_per_rank=int(symm.num_max_tokens_per_rank),
        layout_code=_LAYOUT_CODES[layout],
        has_grad_gate=has_grad_gate,
        out_fp16=False,
        stream=torch.cuda.current_stream(),
    )
    return output, grad_topk_weights
