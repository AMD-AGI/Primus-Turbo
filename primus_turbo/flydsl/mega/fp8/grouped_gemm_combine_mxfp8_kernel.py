###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused grouped MXFP8 L2 GEMM + cross-rank combine PUSH + intra-node topk reduce (FlyDSL).

The mxfp8 analog of ``grouped_gemm_combine_bf16`` — the L2 (down-projection) mirror of the
L1 fused dispatch+GEMM: **compute -> comm** (L1 was comm -> compute). One 3-role kernel:

  * role 1 COMBINE PUSH ``[0, num_combine_cu)``: spin on the local L2 scoreboard, then push each
    finished ``L2Y`` row to the origin rank's ``comb[slot]`` + raise the per-slot flag.
  * role 2 TOPK REDUCE (on the empty grouped-GEMM blocks, + optional dedicated region): sum each
    token's ``topk`` combine rows (weighted) into ``output`` [num_tokens, N].
  * role 3 GROUPED MXFP8 GEMM ``[gemm_base, ...)``: one NT tile ``A=act[M,I] @ B=w2[G,H,I]^T ->
    L2Y[M,H]`` bf16 via ``gemm_mxfp8_nt_tile`` (preshuffled ScaleS2R/ScaleBComb), then
    ``l2_writeback`` + bump the per-pool-block scoreboard so the combine role can stream-push.

The combine + reduce roles are bf16 (they operate on the bf16 ``L2Y`` / ``comb``) and are reused
verbatim from ``grouped_gemm_combine_bf16_kernel``; only the GEMM role is mxfp8. The A (act) +
B (w2) E8M0 scales are preshuffled raw->broadcast ONCE by a fused pre-pass on the same stream
(``build_preshuffle_ab_kernel``), so the GEMM role reads them with the fast coalesced loader.

Tokens/act quantized outside. NT only. K(=I) % 128 == 0; N(=H) % BLOCK_N == 0; out_features % 8.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr.buffer_ops import (
    buffer_load,
    create_buffer_resource,
    create_buffer_resource_from_addr,
)
from flydsl.expr.typing import AddressSpace, PointerType

# combine PUSH tile + topk reduce role are bf16 (operate on bf16 L2Y / comb) -> reuse verbatim.
from primus_turbo.flydsl.mega.grouped_gemm_combine_bf16_kernel import (
    _BLOCK_THREADS,
    _NUM_WARPS,
    _get_topk_reduce,
    combine_bf16_tile,
)
from primus_turbo.flydsl.mega.fp8.gemm_mxfp8_tile import (
    gemm_mxfp8_nt_tile,
    make_mxfp8_shared_storage,
)
from primus_turbo.flydsl.mega.fp8.grouped_gemm_mxfp8_kernel import _get_grouped_mx_workspace
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
from primus_turbo.flydsl.mega.symm_buffer import get_symm_buffer_for_mega_moe
from primus_turbo.flydsl.utils.gemm_helper import build_preshuffle_ab_kernel, make_value_attrs

_PRESHUF_BLK = 256

_MXFP8_COMBINE_COMPILED: dict = {}
_BSP2_CACHE: dict = {}  # (w2 data_ptr, G, H, I) -> preshuffled w2 scale b_sp (weights static)


@functools.lru_cache(maxsize=64)
def _compile(
    out_features,  # N = H (down-proj output; == SymLayout hidden, == L2Y cols)
    hidden_size,  # K = I (contraction; act/weight inner dim)
    num_max_pool_tokens,
    BLOCK_M,
    BLOCK_N,
    num_combine_cu,
    num_reduce_cu,
    combine_slots,
    topk,
    num_experts,
    rank,
    num_ranks,
    G,
    cbsz=0,
    blgp=0,
    nt_vmcnt=3,
    waves_per_eu=2,
    agpr_alloc=0,
    out_fp16=False,
    apply_weights=True,
):
    K = hidden_size
    K128 = K // 128
    assert out_features % BLOCK_N == 0, "out_features must be a multiple of BLOCK_N"
    assert num_max_pool_tokens % BLOCK_M == 0, "num_max_pool_tokens must be a multiple of BLOCK_M"
    assert out_features % 8 == 0, "out_features must be a multiple of 8 (bf16 vec)"
    assert K % 128 == 0 and K >= 256, f"mxfp8 needs K % 128 == 0 and K >= 256, got K={K}"
    SharedStorage = make_mxfp8_shared_storage(BLOCK_M, BLOCK_N)
    n_blocks = out_features // BLOCK_N
    worst_case_tiles = num_max_pool_tokens // BLOCK_M
    gemm_grid_blocks = worst_case_tiles * n_blocks
    comb_records = combine_slots * out_features * 2  # bf16 peer combine-buffer bound (bytes)
    delta_records = num_ranks * 8
    pool_records = num_max_pool_tokens * 4
    dedicated_reduce_warps = num_reduce_cu * _NUM_WARPS
    gemm_base = num_combine_cu + num_reduce_cu
    GN = G * out_features  # flattened B rows [G*H]
    topk_reduce = _get_topk_reduce(True, apply_weights, False)
    pre_kern, n_kt = build_preshuffle_ab_kernel(K128)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def grouped_gemm_combine_mxfp8_kernel(
        ACT: fx.Tensor,  # fp8 act viewed int8 [M*K] flattened
        A_SP: fx.Tensor,  # preshuffled act E8M0 (ScaleS2R broadcast a_sp, int32)
        WEIGHTS: fx.Tensor,  # fp8 w2 viewed int8 [G*H*I] flattened
        B_SP: fx.Tensor,  # preshuffled w2 E8M0 (ScaleBComb b_sp, int32)
        TILE_TO_GROUP: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        OUTPUT: fx.Tensor,  # bf16 [num_tokens, H] reduce result
        TOPK_INDICES: fx.Tensor,
        NUM_TOKENS_PER_RANK: fx.Tensor,
        TOPK_WEIGHTS: fx.Tensor,
        sym_layout: SymLayout,
        c_n: fx.Int32,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        combine_cu = fx.Int32(num_combine_cu)
        reduce_cu = fx.Int32(num_reduce_cu)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        sb_l2_base = sym_layout.sb_l2_ptr
        comb_base = sym_layout.comb_ptr
        barrier_base = sym_layout.barrier_local_ptr
        l2y_ptr_ty = PointerType.get(
            elem_ty=fx.BFloat16.ir_type, address_space=AddressSpace.Global, alignment=16
        )
        l2y_tensor = fx.make_view(
            fx.inttoptr(l2y_ptr_ty, sym_layout.l2_token_buffer_ptr),
            fx.make_layout(num_max_pool_tokens * out_features, 1),
        )
        l2y_resource = create_buffer_resource_from_addr(
            sym_layout.l2_token_buffer_ptr, num_records_bytes=num_max_pool_tokens * out_features * 2
        )
        signal_delta_res = create_buffer_resource_from_addr(
            sym_layout.signal_offsets_ptr, num_records_bytes=delta_records
        )
        origin_rank_res = create_buffer_resource_from_addr(
            sym_layout.origin_rank_ptr, num_records_bytes=pool_records
        )
        origin_slot_res = create_buffer_resource_from_addr(
            sym_layout.origin_slot_ptr, num_records_bytes=pool_records
        )
        comb_local_res = create_buffer_resource_from_addr(comb_base, num_records_bytes=comb_records)

        group_resource = create_buffer_resource(TILE_TO_GROUP, max_size=True)
        num_tile_blocks_res = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
        output_res = create_buffer_resource(OUTPUT, max_size=True)
        topk_indices_res = create_buffer_resource(TOPK_INDICES, max_size=True)
        num_tokens_res = create_buffer_resource(NUM_TOKENS_PER_RANK, max_size=True)
        topk_weights_res = create_buffer_resource(TOPK_WEIGHTS, max_size=True)
        real_tiles = buffer_load(num_tile_blocks_res, fx.Int32(0), vec_width=1, dtype=fx.T.i32())

        push_block = combine_bf16_tile(
            thread_index=thread_index,
            block_m_size=BLOCK_M,
            out_features=out_features,
            comb_records=comb_records,
            n_slots=combine_slots,
            l2y_resource=l2y_resource,
            origin_rank_res=origin_rank_res,
            origin_slot_res=origin_slot_res,
            comb_base=comb_base,
            signal_delta_res=signal_delta_res,
            barrier_base=barrier_base,
            enable_barrier=True,
            with_gate=False,
        )

        if block_index < combine_cu:
            # ── role 1: COMBINE PUSH (grid-stride pool blocks) ──
            local_count = (real_tiles - block_index + combine_cu - fx.Int32(1)) // combine_cu
            for tile_iter in range(local_count):
                block_m = block_index + tile_iter * combine_cu
                if thread_index == fx.Int32(0):
                    spin_start = read_clock()
                    signal_count = ld(sb_l2_base, block_m, scope="agent")
                    while signal_count < fx.Int32(n_blocks):
                        fx.rocdl.s_sleep(fx.Int32(2))
                        if spin_timed_out(spin_start):
                            fx.printf(
                                "MEGA mxfp8 combine L2 gate timeout: block={} signal={} n_blocks={}\n",
                                block_m, signal_count, fx.Int32(n_blocks),
                            )
                            spin_start = read_clock()
                        signal_count = ld(sb_l2_base, block_m, scope="agent")
                    st(sb_l2_base, block_m, fx.Int32(0), scope="agent")  # single consumer -> reset
                fx.gpu.barrier()
                l2_invalidate()  # re-fetch L2Y from HBM (GEMM l2_writeback'd it)
                push_block(block_m)
            fx.rocdl.s_waitcnt(0)
        else:
            if block_index < combine_cu + reduce_cu:
                # ── role 2a: dedicated topk reduce (no-op when num_reduce_cu == 0) ──
                topk_reduce(
                    thread_index, block_index - combine_cu, fx.Int32(dedicated_reduce_warps),
                    topk, out_features, num_experts, rank, comb_local_res, output_res,
                    topk_indices_res, num_tokens_res, barrier_base, topk_weights_res, None, None,
                )
            else:
                # ── role 3: GROUPED MXFP8 GEMM (one tile -> L2Y, signal scoreboard) ──
                # naive M-major order (block_m contiguous) so the combine role can stream-push.
                gemm_tile_index = block_index - fx.Int32(gemm_base)
                block_m = gemm_tile_index // fx.Int32(n_blocks)
                block_n = gemm_tile_index % fx.Int32(n_blocks)
                if block_m < real_tiles:
                    c_m_const = fx.Int32(num_max_pool_tokens)
                    group_index = buffer_load(group_resource, block_m, vec_width=1, dtype=fx.T.i32())
                    gemm_mxfp8_nt_tile(
                        ACT,  # fp8 act int8 tensor (per-tile i64 SRD re-base reads only the base ptr)
                        A_SP,
                        WEIGHTS,
                        B_SP,
                        l2y_tensor,
                        c_m_const,
                        c_n,
                        lds,
                        block_m,
                        block_n,
                        K=K,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        G=G,
                        group_idx=group_index,
                        cbsz=cbsz,
                        blgp=blgp,
                        out_fp16=out_fp16,
                        nt_vmcnt=nt_vmcnt,
                        preshuffled=True,
                    )
                    fx.rocdl.s_waitcnt(0)
                    l2_writeback()  # flush L2Y to HBM so the role-1 push (l2_invalidate) reads fresh
                    fx.gpu.barrier()
                    if thread_index == fx.Int32(0):
                        atomic_add(sb_l2_base, block_m, fx.Int32(1), scope="agent")
                else:
                    # role 2b: empty GEMM tiles reduce on freed CUs (overlaps the push tail)
                    empty_ordinal = gemm_tile_index - real_tiles * fx.Int32(n_blocks)
                    total_empty_warps = (
                        fx.Int32(gemm_grid_blocks) - real_tiles * fx.Int32(n_blocks)
                    ) * fx.Int32(_NUM_WARPS)
                    topk_reduce(
                        thread_index, empty_ordinal, total_empty_warps, topk, out_features,
                        num_experts, rank, comb_local_res, output_res, topk_indices_res,
                        num_tokens_res, barrier_base, topk_weights_res, None, None,
                    )

    @flyc.jit
    def launch(
        ACT,
        A_RAW,
        A_SP,
        WEIGHTS,
        B_RAW,
        B_SP,
        TILE_TO_GROUP,
        NUM_TILE_BLOCKS,
        OUTPUT,
        TOPK_INDICES,
        NUM_TOKENS_PER_RANK,
        TOPK_WEIGHTS,
        sym_layout,
        c_n: int,
        a_blocks: int,
        a_ngrp: int,
        b_ngrp: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        # 1) scale preshuffle (raw E8M0 -> broadcast int32 in A_SP/B_SP), same stream
        pre_kern(A_RAW, B_RAW, A_SP, B_SP, fx.Int32(num_max_pool_tokens), fx.Int32(GN),
                 a_blocks, a_ngrp, b_ngrp).launch(
            grid=(a_blocks + b_ngrp * n_kt, 1, 1), block=(_PRESHUF_BLK, 1, 1), stream=stream
        )
        # 2) fused L2 GEMM + combine + reduce (reads the just-written A_SP/B_SP)
        grid_size = gemm_base + worst_case_tiles * n_blocks
        grouped_gemm_combine_mxfp8_kernel(
            ACT, A_SP, WEIGHTS, B_SP, TILE_TO_GROUP, NUM_TILE_BLOCKS, OUTPUT, TOPK_INDICES,
            NUM_TOKENS_PER_RANK, TOPK_WEIGHTS, sym_layout, c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def grouped_gemm_combine_mxfp8(
    aq: torch.Tensor,
    as_: torch.Tensor,
    w2q: torch.Tensor,
    w2s: torch.Tensor,
    handle,
    group,
    *,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    BM: int = 256,
    BN: int = 256,
    num_combine_cu: int = 64,
    num_reduce_cu: int = 0,
):
    """Fused grouped MXFP8 L2 GEMM + combine PUSH + topk reduce (forward).

    ``aq`` [M, I] fp8 act + ``as_`` [M, I//32] raw E8M0 (M = num_max_pool_tokens, pool-grouped);
    ``w2q`` [G, H, I] fp8 + ``w2s`` [G, H, I//32] raw E8M0. Preshuffles both scales raw->broadcast
    ONCE, runs the L2 GEMM -> ``L2Y`` [M, H] -> cross-rank combine -> weighted top-k reduce.
    Returns ``y`` [num_tokens, H] bf16. The caller must init ``symm.barrier_local``=-1 and
    ``symm.sb_l2``=0 (cross-rank barrier'd) before calling."""
    tile_to_expert = handle[7]
    symm = get_symm_buffer_for_mega_moe()
    sym_layout = symm.make_sym_layout()

    M, I = aq.shape
    G, H, Iw = w2q.shape
    assert Iw == I, f"w2 K={Iw} != act K={I}"
    assert M == int(sym_layout.num_max_pool_tokens), "aq rows must match pool capacity"
    assert H == int(sym_layout.hidden), f"w2 N={H} != SymLayout hidden {int(sym_layout.hidden)}"
    assert aq.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), "L2 fused takes pre-quantized fp8 act"
    K128 = I // 128
    GN = G * H
    out_features = H
    cbsz = 1 if aq.dtype == torch.float8_e5m2 else 0
    blgp = 1 if w2q.dtype == torch.float8_e5m2 else 0

    combine_slots = int(sym_layout.combine_slots)
    num_ranks = int(sym_layout.num_ranks)
    rank = int(sym_layout.rank_idx)
    topk = int(sym_layout.num_topk)
    num_experts = int(sym_layout.num_experts)
    num_tile_blocks = symm.meta_scalars[1:2]

    a8 = aq.contiguous().view(torch.int8)
    b8 = w2q.contiguous().reshape(GN, I).view(torch.int8)
    a_raw = as_.contiguous().view(torch.int32).reshape(-1)
    b_raw = w2s.contiguous().reshape(GN, I // 32).view(torch.int32).reshape(-1)
    a_sp, b_sp, a_blocks, a_ngrp, b_ngrp = _get_grouped_mx_workspace(M, GN, K128, aq.device)

    num_tokens = int(symm.num_tokens)
    output = torch.empty(num_tokens, out_features, dtype=torch.bfloat16, device=aq.device)
    topk_weights_d = topk_weights.contiguous().view(-1)
    topk_indices_d = topk_indices.contiguous().view(-1)

    launch = _compile(
        out_features, I, M, BM, BN, int(num_combine_cu), int(num_reduce_cu), int(combine_slots),
        int(topk), int(num_experts), int(rank), int(num_ranks), int(G),
        cbsz=cbsz, blgp=blgp,
    )
    args = (
        a8, a_raw, a_sp, b8, b_raw, b_sp, tile_to_expert, num_tile_blocks, output.view(-1),
        topk_indices_d, symm.num_tokens_per_rank, topk_weights_d, sym_layout, out_features,
        a_blocks, a_ngrp, b_ngrp, torch.cuda.current_stream(),
    )
    ck = (out_features, I, M, BM, BN, int(num_combine_cu), int(num_reduce_cu), int(combine_slots),
          int(topk), int(num_experts), int(rank), int(num_ranks), int(G), cbsz, blgp)
    if torch.cuda.is_current_stream_capturing():
        launch(*args)
    else:
        compiled = _MXFP8_COMBINE_COMPILED.get(ck)
        if compiled is None:
            compiled = flyc.compile(launch, *args)
            _MXFP8_COMBINE_COMPILED[ck] = compiled
        compiled(*args)
    return output
