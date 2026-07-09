###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl import Config, autotune
from flydsl.expr.buffer_ops import (
    buffer_load,
    create_buffer_resource,
    create_buffer_resource_from_addr,
    extract_base_index,
)

from primus_turbo.flydsl.gemm.gemm_bf16_kernel import (
    _make_shared_storage,
    gemm_bf16_tile,
)
from primus_turbo.flydsl.mega.ep_intranode import (
    combine_bf16_tile,
    topk_reduce_bf16_tile,
)
from primus_turbo.flydsl.mega.prims import (
    atomic_add,
    cast,
    l2_invalidate,
    ld,
    read_clock,
    spin_timed_out,
)
from primus_turbo.flydsl.mega.symm_buffer import SymLayout, get_symm_buffer_for_mega_moe
from primus_turbo.flydsl.mega.tune_utils import _suppress_stdout_stderr
from primus_turbo.flydsl.utils.gemm_helper import (
    make_bf16_fp16_tile_tensor,
    make_value_attrs,
)

_WARP = 64
_BLOCK_THREADS = 512


_PVEC = 8
_NUM_WARPS = _BLOCK_THREADS // _WARP

_LAYOUTS = ("nt", "nn", "tn")
_LAYOUT_CODES = {name: code for code, name in enumerate(_LAYOUTS)}


def _make_grouped_gemm_combine(
    out_features,
    hidden_size,
    num_max_pool_tokens,
    BLOCK_M,
    BLOCK_N,
    num_combine_cu,
    num_reduce_cu,
    num_combine_slots,
    topk,
    num_experts,
    rank,
    nt_vmcnt=3,
    out_fp16=False,
    layout="nt",
    apply_weights=False,
    with_gate=False,
):
    K = hidden_size
    gemm_tile = functools.partial(gemm_bf16_tile, layout)
    assert out_features % BLOCK_N == 0, "out_features must be a multiple of BLOCK_N"
    assert num_max_pool_tokens % BLOCK_M == 0, "num_max_pool_tokens must be a multiple of BLOCK_M"
    assert out_features % _PVEC == 0, "out_features must be a multiple of 8 (bf16 vec)"
    assert topk >= 1, "topk must be >= 1"
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)
    n_blocks = out_features // BLOCK_N
    worst_case_tiles = num_max_pool_tokens // BLOCK_M
    comb_records = num_combine_slots * out_features * 2
    gate_records = num_combine_slots * 4
    gemm_base = num_combine_cu

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def grouped_gemm_combine_kernel(
        ACT: fx.Tensor,
        WEIGHTS: fx.Tensor,
        TILE_TO_GROUP: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        OUTPUT: fx.Tensor,
        TOPK_INDICES: fx.Tensor,
        NUM_TOKENS_PER_RANK: fx.Tensor,
        TOPK_WEIGHTS: fx.Tensor,
        GRAD_GATE: fx.Tensor,
        D_TOPK_W: fx.Tensor,
        sym_layout: SymLayout,
        c_n: fx.Int32,
        combine_parity: fx.Int32,
        expected_combine: fx.Int32,
        expected_reduce: fx.Int32,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        combine_cu = fx.Int32(num_combine_cu)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        combine_bank = combine_parity * fx.Int32(worst_case_tiles)
        reduce_bank = combine_parity * fx.Int32(num_combine_slots)
        expected_combine_i64 = cast(expected_combine, fx.T.i64())
        expected_reduce_i64 = cast(expected_reduce, fx.T.i64())

        combine_flag_base = sym_layout.combine_flag
        comb_base = sym_layout.combine_token_buffer
        reduce_flag_base = sym_layout.reduce_flag
        comb_local_res = create_buffer_resource_from_addr(comb_base, num_records_bytes=comb_records)
        # recv-segment table for task-based (sustained-peer) combine push
        seg_bytes = num_experts * 4
        recv_dst_rank_res = create_buffer_resource_from_addr(
            sym_layout.combine_recv_dst_rank, num_records_bytes=seg_bytes
        )
        recv_start_row_res = create_buffer_resource_from_addr(
            sym_layout.combine_recv_start_row, num_records_bytes=seg_bytes
        )
        recv_count_res = create_buffer_resource_from_addr(
            sym_layout.combine_recv_count, num_records_bytes=seg_bytes
        )
        origin_slot_res = create_buffer_resource_from_addr(
            sym_layout.pool_src_slot, num_records_bytes=num_max_pool_tokens * 4
        )

        group_resource = create_buffer_resource(TILE_TO_GROUP, max_size=True)
        num_tile_blocks_res = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
        output_res = create_buffer_resource(OUTPUT, max_size=True)
        topk_indices_res = create_buffer_resource(TOPK_INDICES, max_size=True)
        num_tokens_res = create_buffer_resource(NUM_TOKENS_PER_RANK, max_size=True)
        topk_weights_res = create_buffer_resource(TOPK_WEIGHTS, max_size=True)
        real_tiles = buffer_load(num_tile_blocks_res, fx.Int32(0), vec_width=1, dtype=fx.T.i32())
        grad_gate_res = create_buffer_resource(GRAD_GATE, max_size=True) if with_gate else None
        gate_base = sym_layout.combine_gate if with_gate else None
        gate_local_res = (
            create_buffer_resource_from_addr(gate_base, num_records_bytes=gate_records) if with_gate else None
        )
        d_topk_w_res = create_buffer_resource(D_TOPK_W, max_size=True) if with_gate else None

        if block_index < combine_cu:
            # Task-based combine: one warp per recv-segment, gated on its spanned GEMM tiles.
            seg_local = (fx.Int32(num_experts) - block_index + combine_cu - fx.Int32(1)) // combine_cu
            for seg_iter in range(seg_local):
                task_index = block_index + seg_iter * combine_cu
                seg_start = buffer_load(recv_start_row_res, task_index, vec_width=1, dtype=fx.T.i32())
                seg_count = buffer_load(recv_count_res, task_index, vec_width=1, dtype=fx.T.i32())
                if seg_count > fx.Int32(0):
                    t0 = seg_start // fx.Int32(BLOCK_M)
                    t1 = (seg_start + seg_count - fx.Int32(1)) // fx.Int32(BLOCK_M)
                    if thread_index == fx.Int32(0):
                        tile_cursor = t0
                        while tile_cursor <= t1:
                            spin_start = read_clock()
                            signal_count = ld(
                                combine_flag_base, combine_bank + tile_cursor, scope="agent", dtype=fx.T.i64()
                            )
                            while signal_count != expected_combine_i64:
                                fx.rocdl.s_sleep(fx.Int32(2))
                                if spin_timed_out(spin_start):
                                    fx.printf(
                                        "MEGA combine(task) gate timeout: tile={} signal={} thr={}\n",
                                        tile_cursor,
                                        signal_count,
                                        expected_combine_i64,
                                    )
                                    spin_start = read_clock()
                                signal_count = ld(
                                    combine_flag_base,
                                    combine_bank + tile_cursor,
                                    scope="agent",
                                    dtype=fx.T.i64(),
                                )
                            tile_cursor = tile_cursor + fx.Int32(1)
                    fx.gpu.barrier()
                    l2_invalidate()
                    combine_bf16_tile(
                        sym_layout,
                        thread_index=thread_index,
                        task_index=task_index,
                        recv_dst_rank_res=recv_dst_rank_res,
                        recv_start_row_res=recv_start_row_res,
                        recv_count_res=recv_count_res,
                        origin_slot_res=origin_slot_res,
                        grad_gate_res=grad_gate_res,
                        signal=True,
                        epoch=expected_reduce_i64,
                        bank_offset=reduce_bank,
                        with_gate=with_gate,
                    )
            fx.rocdl.s_waitcnt(0)
        else:
            gemm_tile_index = block_index - fx.Int32(gemm_base)
            block_m = gemm_tile_index // fx.Int32(n_blocks)
            block_n = gemm_tile_index % fx.Int32(n_blocks)
            if block_m < real_tiles:
                # GEMM role: one real tile (block_m, block_n) per block (unchanged).
                group_index = buffer_load(group_resource, block_m, vec_width=1, dtype=fx.T.i32())
                group_base = group_index * fx.Int32(K) * c_n
                # A base = ACT tensor; C base = l2_token_buffer (int64 symm addr).
                act_base = fx.arith.ArithValue(
                    fx.arith.index_cast(fx.T.i64(), extract_base_index(ACT)), signed=True
                )
                # Fold each per-tile base in int64 (worst-case pool >4GB) so voffset stays
                # int32, then address tile-local (block_m=0). A: precise bound;
                # C: HW num_records bounds via 0x40000000 flat layout.
                a_off = cast(block_m, fx.T.i64()) * fx.Int64(BLOCK_M * K * 2)
                c_off = cast(block_m, fx.T.i64()) * fx.Int64(BLOCK_M * 2) * cast(c_n, fx.T.i64())
                A_tile = make_bf16_fp16_tile_tensor(act_base, a_off, BLOCK_M * K)
                C_tile = make_bf16_fp16_tile_tensor(sym_layout.l2_token_buffer, c_off, 0x40000000)
                gemm_tile(
                    A_tile,
                    WEIGHTS,
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
                    b_group_base=group_base,
                    c_cache_modifier=16,
                )
                fx.rocdl.s_waitcnt(0)
                fx.gpu.barrier()
                if thread_index == fx.Int32(0):
                    atomic_add(combine_flag_base, combine_bank + block_m, fx.Int64(1), scope="agent")
            else:
                # Empty region: first num_reduce_cu blocks do topk reduce, rest early-exit.
                empty_ordinal = gemm_tile_index - real_tiles * fx.Int32(n_blocks)
                if empty_ordinal < fx.Int32(num_reduce_cu):
                    # Never-reset alignment: reduce blocks bump empty block_m's combine_flag to cumulative expected.
                    n_empty = fx.Int32(worst_case_tiles) - real_tiles
                    reduce_stride = fx.Int32(num_reduce_cu)
                    align_count = (n_empty - empty_ordinal + reduce_stride - fx.Int32(1)) // reduce_stride
                    for align_iter in range(align_count):
                        empty_block_m = real_tiles + empty_ordinal + align_iter * reduce_stride
                        if thread_index == fx.Int32(0):
                            atomic_add(
                                combine_flag_base,
                                combine_bank + empty_block_m,
                                fx.Int64(n_blocks),
                                scope="agent",
                            )

                    n_reduce_tiles = n_empty * fx.Int32(n_blocks)
                    active_reduce_blocks = fx.arith.select(
                        n_reduce_tiles < fx.Int32(num_reduce_cu), n_reduce_tiles, fx.Int32(num_reduce_cu)
                    )
                    topk_reduce_bf16_tile(
                        True,
                        apply_weights,
                        with_gate,
                        thread_index,
                        empty_ordinal,
                        active_reduce_blocks * fx.Int32(_NUM_WARPS),
                        topk,
                        out_features,
                        num_experts,
                        rank,
                        comb_local_res,
                        output_res,
                        topk_indices_res,
                        num_tokens_res,
                        reduce_flag_base,
                        reduce_bank,
                        topk_weights_res,
                        gate_local_res,
                        d_topk_w_res,
                        expected_reduce_i64,
                    )

    return grouped_gemm_combine_kernel


def _rewind_combine_flags(kwargs):
    # tuning-only: rewind the two never-reset flags so each rerun matches the
    # baked expected. combine_flag is local (GEMM writes, combine reads); reduce_flag
    # is cross-rank (combine pushes to peers). fill -> sync -> barrier so no next-rep
    # write lands before every rank has rewound (else counts get zeroed).
    symm = get_symm_buffer_for_mega_moe()
    p = int(kwargs["combine_parity"])
    n_blocks = int(kwargs["out_features"]) // int(kwargs["BLOCK_N"])
    wct = int(symm.num_max_pool_tokens) // int(kwargs["BLOCK_M"])
    ncs = int(kwargs["num_combine_slots"])
    comb_base = int(kwargs["expected_combine"]) - n_blocks
    red_base = int(kwargs["expected_reduce"]) - 1
    symm.combine_flag[p * wct : (p + 1) * wct].fill_(comb_base)
    symm.reduce_flag[p * ncs : (p + 1) * ncs].fill_(red_base)
    torch.cuda.synchronize()
    torch.distributed.barrier(symm.group)


@autotune(
    configs=[Config(num_combine_cu=cc, num_reduce_cu=rc) for cc in (16, 32, 64) for rc in (256, 512)],
    # layout_code MUST be a key: nt/nn have OPPOSITE combine_cu optima (see wrapper note).
    key=[
        "out_features",
        "hidden_size",
        "num_max_pool_tokens",
        "BLOCK_M",
        "BLOCK_N",
        "num_combine_slots",
        "topk",
        "num_experts",
        "rank",
        "layout_code",
        "apply_weights",
        "with_gate",
        "out_fp16",
    ],
    warmup=0,
    rep=5,
    post_hook=_rewind_combine_flags,
)
@flyc.jit
def _compiled_grouped_gemm_combine(
    ACT,
    WEIGHTS,
    TILE_TO_GROUP,
    NUM_TILE_BLOCKS,
    OUTPUT,
    TOPK_INDICES,
    NUM_TOKENS_PER_RANK,
    TOPK_WEIGHTS,
    GRAD_GATE,
    D_TOPK_W,
    sym_layout,
    c_n,
    combine_parity,
    expected_combine,
    expected_reduce,
    out_features: fx.Constexpr[int],
    hidden_size: fx.Constexpr[int],
    num_max_pool_tokens: fx.Constexpr[int],
    BLOCK_M: fx.Constexpr[int],
    BLOCK_N: fx.Constexpr[int],
    num_combine_slots: fx.Constexpr[int],
    topk: fx.Constexpr[int],
    num_experts: fx.Constexpr[int],
    rank: fx.Constexpr[int],
    layout_code: fx.Constexpr[int],
    apply_weights: fx.Constexpr[bool],
    with_gate: fx.Constexpr[bool],
    out_fp16: fx.Constexpr[bool],
    num_combine_cu: fx.Constexpr[int] = 64,
    num_reduce_cu: fx.Constexpr[int] = 256,
    nt_vmcnt: fx.Constexpr[int] = 3,
    agpr_alloc: fx.Constexpr[int] = 0,
    waves: fx.Constexpr[int] = 2,
    stream: fx.Stream = fx.Stream(None),
):
    kernel = _make_grouped_gemm_combine(
        out_features,
        hidden_size,
        num_max_pool_tokens,
        BLOCK_M,
        BLOCK_N,
        num_combine_cu,
        num_reduce_cu,
        num_combine_slots,
        topk,
        num_experts,
        rank,
        nt_vmcnt,
        out_fp16,
        _LAYOUTS[layout_code],
        apply_weights,
        with_gate,
    )
    n_blocks = out_features // BLOCK_N
    worst_case_tiles = num_max_pool_tokens // BLOCK_M
    grid_size = num_combine_cu + worst_case_tiles * n_blocks
    kernel(
        ACT,
        WEIGHTS,
        TILE_TO_GROUP,
        NUM_TILE_BLOCKS,
        OUTPUT,
        TOPK_INDICES,
        NUM_TOKENS_PER_RANK,
        TOPK_WEIGHTS,
        GRAD_GATE,
        D_TOPK_W,
        sym_layout,
        c_n,
        combine_parity,
        expected_combine,
        expected_reduce,
        value_attrs=make_value_attrs(waves, agpr_alloc, "512,512"),
    ).launch(grid=(grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)


def grouped_gemm_combine_bf16(
    x,
    l2_weights,
    handle,
    *,
    topk_indices,
    topk_weights=None,
    grad_gate=None,
    layout="nt",
    BM=256,
    BN=256,
):
    assert layout in ("nt", "nn", "tn"), f"unknown layout {layout}"
    assert x.dtype == torch.bfloat16 and l2_weights.dtype == torch.bfloat16
    assert topk_indices is not None, "topk reduce needs topk_indices"
    tile_to_expert = handle[5]
    symm = get_symm_buffer_for_mega_moe()
    sym_layout = symm.get_sym_layout()
    if layout == "tn":
        hidden_size, num_max_pool_tokens = x.shape
    else:
        num_max_pool_tokens, hidden_size = x.shape
    if layout == "nt":
        G, N, K = l2_weights.shape
    else:
        G, K, N = l2_weights.shape
    assert K == hidden_size, f"weight K={K} != activation K={hidden_size}"
    out_features = N
    c_n = out_features
    assert out_features == int(sym_layout.hidden), (
        f"out_features {out_features} != SymLayout hidden {int(sym_layout.hidden)}"
    )
    assert num_max_pool_tokens == int(sym_layout.num_max_pool_tokens), (
        "x rows must match SymLayout pool capacity"
    )

    device = x.device
    num_combine_slots = int(sym_layout.num_combine_slots)
    rank = int(sym_layout.rank_idx)
    topk = int(sym_layout.num_topk)
    num_experts = int(sym_layout.num_experts)
    assert topk >= 1 and num_experts > 0, "topk reduce needs topk>=1 and num_experts>0"
    num_tile_blocks = symm.meta_scalars[1:2]
    dummy = num_tile_blocks

    apply_weights = topk_weights is not None
    with_gate = grad_gate is not None

    # Pass 2D: kernel advances ACT base per-tile in int64 (flat MxK overflows int32 ABI).
    act_2d = x.contiguous()
    if layout == "nt":
        weight_flat = l2_weights.reshape(G * N, K).contiguous().view(-1)
    else:
        weight_flat = l2_weights.reshape(G * K, N).contiguous().view(-1)
    num_tokens = int(symm.num_tokens)
    output = torch.empty(num_tokens, out_features, dtype=torch.bfloat16, device=device)
    output_d = output.view(-1)
    topk_indices_d = topk_indices.contiguous().view(-1)
    num_tokens_d = symm.num_tokens_per_rank
    topk_weights_d = topk_weights.contiguous().view(-1) if apply_weights else dummy
    grad_gate_d = grad_gate.contiguous().view(-1) if with_gate else dummy
    d_topk_w = torch.empty(num_combine_slots, dtype=torch.float32, device=device) if with_gate else None
    d_topk_w_d = d_topk_w if with_gate else dummy

    n_blocks = out_features // BN
    combine_parity, expected_combine, expected_reduce = symm.next_combine(n_blocks)

    # num_combine_cu / num_reduce_cu are tunable per shape+layout (nt/nn optima differ).
    with _suppress_stdout_stderr():
        _compiled_grouped_gemm_combine(
            act_2d,
            weight_flat,
            tile_to_expert,
            num_tile_blocks,
            output_d,
            topk_indices_d,
            num_tokens_d,
            topk_weights_d,
            grad_gate_d,
            d_topk_w_d,
            sym_layout,
            c_n,
            combine_parity=combine_parity,
            expected_combine=expected_combine,
            expected_reduce=expected_reduce,
            out_features=out_features,
            hidden_size=hidden_size,
            num_max_pool_tokens=num_max_pool_tokens,
            BLOCK_M=BM,
            BLOCK_N=BN,
            num_combine_slots=int(num_combine_slots),
            topk=int(topk),
            num_experts=int(num_experts),
            rank=int(rank),
            layout_code=_LAYOUT_CODES[layout],
            apply_weights=bool(apply_weights),
            with_gate=bool(with_gate),
            out_fp16=False,
            stream=torch.cuda.current_stream(),
        )
    return output, d_topk_w
