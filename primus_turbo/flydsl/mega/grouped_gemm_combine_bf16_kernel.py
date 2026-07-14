###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
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
from primus_turbo.flydsl.mega.symm_buffer import (
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
    num_ranks=0,
    num_max_tokens_per_rank=0,
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
        RECV_DST_RANK: fx.Tensor,
        RECV_START_ROW: fx.Tensor,
        RECV_COUNT: fx.Tensor,
        POOL_SRC_SLOT: fx.Tensor,
        OUTPUT: fx.Tensor,
        TOPK_INDICES: fx.Tensor,
        NUM_TOKENS_PER_RANK: fx.Tensor,
        TOPK_WEIGHTS: fx.Tensor,
        GRAD_GATE: fx.Tensor,
        D_TOPK_W: fx.Tensor,
        sym_buffer: SymBuffer,
        c_n: fx.Int32,
        COMBINE_PARITY: fx.Tensor,
        COMBINE_EXPECTED: fx.Tensor,
        REDUCE_EXPECTED: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        combine_cu = fx.Int32(num_combine_cu)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # build the layout from explicit dims (bf16 path -> TOKEN_DTYPE); the token pools are
        # out_features-wide (the model hidden), not the down-proj K (hidden_size)
        workspace = Workspace(
            sym_buffer.get_base_ptr(),
            num_ranks,
            num_experts,
            num_max_tokens_per_rank,
            topk,
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
        reduce_bank = combine_parity * fx.Int32(num_combine_slots)
        expected_combine_i64 = buffer_load(
            combine_expected_res, combine_parity, vec_width=1, dtype=fx.T.i64()
        )
        expected_reduce_i64 = buffer_load(reduce_expected_res, combine_parity, vec_width=1, dtype=fx.T.i64())

        combine_flag_base = workspace.get_combine_flag_ptr()
        comb_base = workspace.get_combine_token_buffer_ptr()
        reduce_flag_base = workspace.get_reduce_flag_ptr()
        l2_token_buffer_base = workspace.get_l2_token_buffer_ptr()
        comb_local_res = create_buffer_resource_from_addr(comb_base, num_records_bytes=comb_records)
        # recv-segment table + origin slots ride the handle (per-forward), NOT shared symm -> else bwd reads stale.
        recv_dst_rank_res = create_buffer_resource(RECV_DST_RANK, max_size=True)
        recv_start_row_res = create_buffer_resource(RECV_START_ROW, max_size=True)
        recv_count_res = create_buffer_resource(RECV_COUNT, max_size=True)
        origin_slot_res = create_buffer_resource(POOL_SRC_SLOT, max_size=True)

        group_resource = create_buffer_resource(TILE_TO_GROUP, max_size=True)
        num_tile_blocks_res = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
        output_res = create_buffer_resource(OUTPUT, max_size=True)
        topk_indices_res = create_buffer_resource(TOPK_INDICES, max_size=True)
        num_tokens_res = create_buffer_resource(NUM_TOKENS_PER_RANK, max_size=True)
        topk_weights_res = create_buffer_resource(TOPK_WEIGHTS, max_size=True)
        real_tiles = buffer_load(num_tile_blocks_res, fx.Int32(0), vec_width=1, dtype=fx.T.i32())
        grad_gate_res = create_buffer_resource(GRAD_GATE, max_size=True) if with_gate else None
        gate_base = workspace.get_combine_gate_ptr() if with_gate else None
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
                        sym_buffer,
                        workspace,
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
                # Fold per-tile base in int64 (pool >4GB), voffset stays int32. A: precise bound; C: HW num_records via 0x40000000.
                a_off = cast(block_m, fx.T.i64()) * fx.Int64(BLOCK_M * K * 2)
                c_off = cast(block_m, fx.T.i64()) * fx.Int64(BLOCK_M * 2) * cast(c_n, fx.T.i64())
                A_tile = make_bf16_fp16_tile_tensor(act_base, a_off, BLOCK_M * K)
                C_tile = make_bf16_fp16_tile_tensor(l2_token_buffer_base, c_off, 0x40000000)
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
                    c_cache_modifier=19,
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


@autotune(
    configs=[Config(num_combine_cu=cc, num_reduce_cu=rc) for cc in (16, 32, 64) for rc in (256,)],
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
    rep=5,
)
@flyc.jit
def _compiled_grouped_gemm_combine(
    ACT,
    WEIGHTS,
    TILE_TO_GROUP,
    NUM_TILE_BLOCKS,
    RECV_DST_RANK,
    RECV_START_ROW,
    RECV_COUNT,
    POOL_SRC_SLOT,
    OUTPUT,
    TOPK_INDICES,
    NUM_TOKENS_PER_RANK,
    TOPK_WEIGHTS,
    GRAD_GATE,
    D_TOPK_W,
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
    num_combine_slots: fx.Constexpr[int],
    topk: fx.Constexpr[int],
    num_experts: fx.Constexpr[int],
    rank: fx.Constexpr[int],
    num_ranks: fx.Constexpr[int],
    num_max_tokens_per_rank: fx.Constexpr[int],
    layout_code: fx.Constexpr[int],
    apply_weights: fx.Constexpr[bool],
    with_gate: fx.Constexpr[bool],
    out_fp16: fx.Constexpr[bool],
    stream: fx.Stream,
    num_combine_cu: fx.Constexpr[int] = 64,
    num_reduce_cu: fx.Constexpr[int] = 256,
    nt_vmcnt: fx.Constexpr[int] = 3,
    agpr_alloc: fx.Constexpr[int] = 0,
    waves: fx.Constexpr[int] = 2,
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
        num_ranks,
        num_max_tokens_per_rank,
        nt_vmcnt,
        out_fp16,
        _LAYOUTS[layout_code],
        apply_weights,
        with_gate,
    )
    n_blocks = out_features // BLOCK_N
    worst_case_tiles = num_max_pool_tokens // BLOCK_M
    grid_size = num_combine_cu + worst_case_tiles * n_blocks
    # bump epoch on device (combine += n_blocks, reduce += 1) before the GEMM; same-stream visible
    _make_epoch_bump(int(n_blocks), 1)(COMBINE_PARITY, COMBINE_EXPECTED, REDUCE_EXPECTED).launch(
        grid=(1, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream
    )
    kernel(
        ACT,
        WEIGHTS,
        TILE_TO_GROUP,
        NUM_TILE_BLOCKS,
        RECV_DST_RANK,
        RECV_START_ROW,
        RECV_COUNT,
        POOL_SRC_SLOT,
        OUTPUT,
        TOPK_INDICES,
        NUM_TOKENS_PER_RANK,
        TOPK_WEIGHTS,
        GRAD_GATE,
        D_TOPK_W,
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
    num_tile_blocks = handle[8]
    recv_dst_rank = handle[9]
    recv_start_row = handle[10]
    recv_count = handle[11]
    pool_src_slot = handle[12]
    symm = get_symm_buffer_for_mega_moe()
    sym_buffer = symm.get_sym_buffer()
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
    assert out_features == int(symm.hidden), (
        f"out_features {out_features} != SymmBuffer hidden {int(symm.hidden)}"
    )
    assert num_max_pool_tokens == int(symm.num_max_pool_tokens), "x rows must match SymmBuffer pool capacity"

    device = x.device
    num_combine_slots = int(symm.num_combine_slots)
    rank = int(symm.rank)
    topk = int(symm.num_topk)
    num_experts = int(symm.num_experts)
    assert topk >= 1 and num_experts > 0, "topk reduce needs topk>=1 and num_experts>0"

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

    # epoch advance moved inside _compiled_grouped_gemm_combine (autotune-safe, no rewind)
    # num_combine_cu / num_reduce_cu are tunable per shape+layout (nt/nn optima differ).
    _compiled_grouped_gemm_combine(
        act_2d,
        weight_flat,
        tile_to_expert,
        num_tile_blocks,
        recv_dst_rank,
        recv_start_row,
        recv_count,
        pool_src_slot,
        output_d,
        topk_indices_d,
        num_tokens_d,
        topk_weights_d,
        grad_gate_d,
        d_topk_w_d,
        sym_buffer,
        c_n,
        COMBINE_PARITY=symm._combine_parity,
        COMBINE_EXPECTED=symm._combine_expected,
        REDUCE_EXPECTED=symm._reduce_expected,
        out_features=out_features,
        hidden_size=hidden_size,
        num_max_pool_tokens=num_max_pool_tokens,
        BLOCK_M=BM,
        BLOCK_N=BN,
        num_combine_slots=int(num_combine_slots),
        topk=int(topk),
        num_experts=int(num_experts),
        rank=int(rank),
        num_ranks=int(symm.world),
        num_max_tokens_per_rank=int(symm.num_max_tokens_per_rank),
        layout_code=_LAYOUT_CODES[layout],
        apply_weights=bool(apply_weights),
        with_gate=bool(with_gate),
        out_fp16=False,
        stream=torch.cuda.current_stream(),
    )
    return output, d_topk_w
