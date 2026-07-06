###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr.buffer_ops import (
    _unwrap_value,
    buffer_load,
    create_buffer_resource,
    create_buffer_resource_from_addr,
)
from flydsl.expr.typing import AddressSpace, PointerType

from primus_turbo.flydsl.gemm.gemm_bf16_kernel import GEMM_TILE, _make_shared_storage
from primus_turbo.flydsl.mega.ep_intranode import (
    combine_bf16_tile,
    topk_reduce_bf16_tile,
)
from primus_turbo.flydsl.mega.prims import (
    atomic_add,
    l2_invalidate,
    ld,
    read_clock,
    spin_timed_out,
)
from primus_turbo.flydsl.mega.sym_layout import SymLayout
from primus_turbo.flydsl.mega.symm_buffer import get_symm_buffer_for_mega_moe
from primus_turbo.flydsl.utils.gemm_helper import _emit_if_then, make_value_attrs

_WARP = 64
_BLOCK_THREADS = 512


def _i64(v):
    return fx.arith.ArithValue(fx.arith.extsi(fx.T.i64(), _unwrap_value(v)), signed=True)


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
    combine_slots,
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
    gemm_tile = GEMM_TILE[layout]
    assert out_features % BLOCK_N == 0, "out_features must be a multiple of BLOCK_N"
    assert num_max_pool_tokens % BLOCK_M == 0, "num_max_pool_tokens must be a multiple of BLOCK_M"
    assert out_features % _PVEC == 0, "out_features must be a multiple of 8 (bf16 vec)"
    assert topk >= 1, "topk must be >= 1"
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)
    n_blocks = out_features // BLOCK_N
    worst_case_tiles = num_max_pool_tokens // BLOCK_M
    gemm_grid_blocks = worst_case_tiles * n_blocks
    comb_records = combine_slots * out_features * 2
    gate_records = combine_slots * 4
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
        reduce_bank = combine_parity * fx.Int32(combine_slots)
        expected_combine_i64 = _i64(expected_combine)
        expected_reduce_i64 = _i64(expected_reduce)

        combine_flag_base = sym_layout.combine_flag_ptr
        comb_base = sym_layout.comb_ptr
        reduce_flag_base = sym_layout.reduce_flag_ptr
        l2_token_buffer_ptr_ty = PointerType.get(
            elem_ty=fx.BFloat16.ir_type, address_space=AddressSpace.Global, alignment=16
        )
        l2_token_buffer_tensor = fx.make_view(
            fx.inttoptr(l2_token_buffer_ptr_ty, sym_layout.l2_token_buffer_ptr),
            fx.make_layout(num_max_pool_tokens * out_features, 1),
        )
        comb_local_res = create_buffer_resource_from_addr(comb_base, num_records_bytes=comb_records)

        group_resource = create_buffer_resource(TILE_TO_GROUP, max_size=True)
        num_tile_blocks_res = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
        output_res = create_buffer_resource(OUTPUT, max_size=True)
        topk_indices_res = create_buffer_resource(TOPK_INDICES, max_size=True)
        num_tokens_res = create_buffer_resource(NUM_TOKENS_PER_RANK, max_size=True)
        topk_weights_res = create_buffer_resource(TOPK_WEIGHTS, max_size=True)
        real_tiles = buffer_load(num_tile_blocks_res, fx.Int32(0), vec_width=1, dtype=fx.T.i32())
        grad_gate_res = create_buffer_resource(GRAD_GATE, max_size=True) if with_gate else None
        gate_base = sym_layout.combine_gate_ptr if with_gate else None
        gate_local_res = (
            create_buffer_resource_from_addr(gate_base, num_records_bytes=gate_records) if with_gate else None
        )
        d_topk_w_res = create_buffer_resource(D_TOPK_W, max_size=True) if with_gate else None

        if block_index < combine_cu:
            local_count = (real_tiles - block_index + combine_cu - fx.Int32(1)) // combine_cu
            for tile_iter in range(local_count):
                block_m = block_index + tile_iter * combine_cu
                if thread_index == fx.Int32(0):
                    spin_start = read_clock()
                    signal_count = ld(
                        combine_flag_base, combine_bank + block_m, scope="agent", dtype=fx.T.i64()
                    )
                    while signal_count != expected_combine_i64:
                        fx.rocdl.s_sleep(fx.Int32(2))
                        if spin_timed_out(spin_start):
                            fx.printf(
                                "MEGA combine L2 gate timeout: block={} signal={} thr={}\n",
                                block_m,
                                signal_count,
                                expected_combine_i64,
                            )
                            spin_start = read_clock()
                        signal_count = ld(
                            combine_flag_base, combine_bank + block_m, scope="agent", dtype=fx.T.i64()
                        )
                fx.gpu.barrier()
                l2_invalidate()
                combine_bf16_tile(
                    sym_layout,
                    thread_index=thread_index,
                    block_m=block_m,
                    block_m_size=BLOCK_M,
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
                c_m_const = fx.Int32(num_max_pool_tokens)
                group_index = buffer_load(group_resource, block_m, vec_width=1, dtype=fx.T.i32())
                group_base = group_index * fx.Int32(K) * c_n
                gemm_tile(
                    ACT,
                    WEIGHTS,
                    l2_token_buffer_tensor,
                    c_m_const,
                    c_n,
                    lds,
                    block_m,
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
                _emit_if_then(
                    thread_index == fx.Int32(0),
                    lambda: atomic_add(combine_flag_base, combine_bank + block_m, fx.Int64(1), scope="agent"),
                )
            else:
                empty_ordinal = gemm_tile_index - real_tiles * fx.Int32(n_blocks)
                total_empty_warps = (fx.Int32(gemm_grid_blocks) - real_tiles * fx.Int32(n_blocks)) * fx.Int32(
                    _NUM_WARPS
                )
                _emit_if_then(
                    thread_index == fx.Int32(0),
                    lambda: atomic_add(combine_flag_base, combine_bank + block_m, fx.Int64(1), scope="agent"),
                )
                topk_reduce_bf16_tile(
                    True,
                    apply_weights,
                    with_gate,
                    thread_index,
                    empty_ordinal,
                    total_empty_warps,
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
    combine_slots: fx.Constexpr[int],
    topk: fx.Constexpr[int],
    num_experts: fx.Constexpr[int],
    rank: fx.Constexpr[int],
    layout_code: fx.Constexpr[int],
    apply_weights: fx.Constexpr[bool],
    with_gate: fx.Constexpr[bool],
    out_fp16: fx.Constexpr[bool],
    num_combine_cu: fx.Constexpr[int] = 64,
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
        combine_slots,
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
    sym_layout = symm.make_sym_layout()
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
    assert out_features == int(
        sym_layout.hidden
    ), f"out_features {out_features} != SymLayout hidden {int(sym_layout.hidden)}"
    assert num_max_pool_tokens == int(
        sym_layout.num_max_pool_tokens
    ), "x rows must match SymLayout pool capacity"

    device = x.device
    combine_slots = int(sym_layout.combine_slots)
    rank = int(sym_layout.rank_idx)
    topk = int(sym_layout.num_topk)
    num_experts = int(sym_layout.num_experts)
    assert topk >= 1 and num_experts > 0, "topk reduce needs topk>=1 and num_experts>0"
    num_tile_blocks = symm.meta_scalars[1:2]
    dummy = num_tile_blocks

    apply_weights = topk_weights is not None
    with_gate = grad_gate is not None

    act_flat = x.contiguous().view(-1)
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
    d_topk_w = torch.empty(combine_slots, dtype=torch.float32, device=device) if with_gate else None
    d_topk_w_d = d_topk_w if with_gate else dummy

    n_blocks = out_features // BN
    combine_parity, expected_combine, expected_reduce = symm.next_combine(n_blocks)

    _compiled_grouped_gemm_combine(
        act_flat,
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
        combine_parity,
        expected_combine,
        expected_reduce,
        out_features=out_features,
        hidden_size=hidden_size,
        num_max_pool_tokens=num_max_pool_tokens,
        BLOCK_M=BM,
        BLOCK_N=BN,
        combine_slots=int(combine_slots),
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
