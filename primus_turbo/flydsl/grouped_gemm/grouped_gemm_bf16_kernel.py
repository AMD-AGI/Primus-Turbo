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

"""FlyDSL bf16 variable-K GROUPED GEMM — the MoE wgrad operator.

Computes ``out[g] = a[rows_g].T @ b[rows_g]`` for G groups, where A is
[M_total, OUT_M] and B is [M_total, OUT_N] (groups concatenated along the
reduction dim), ``group_k_offsets`` [G+1] int64 gives each group's row start,
and ``masked_k`` [G] int64 gives the per-group VALID row count so the padded
tail is never read.

Grid is exactly ``G * (OUT_M/BLOCK_M) * ceil(OUT_N/BLOCK_N)`` tiles; each WG
maps its pid -> (group_idx, block_m, block_n) and reads m_start/m_end from the
two index tables on-device (no CPU sync). A/B are rebased per group with an
int64 base offset and span, so a worst-case pool cannot wrap int32 before
``make_bf16_buffer_tensor_rebased`` clamps the span into the 32-bit HW
num_records field.

Shares the dense kernel's LDS layout and primitives; see gemm_bf16_kernel.py
for the 4-buffer pipeline / barrier rationale (identical here, except the K
loop is chunked because K is a runtime value).

Two variable-K wgrad operators live here:

- ``grouped_gemm_bf16_variable_k_flydsl_kernel`` — dense, K walks a contiguous
  row range, LDS frame pads its chunk stride.
- ``grouped_gemm_variable_k_bf16`` — slot-indexed, K walks a slot table because
  the Mega-MoE dispatch pool is deduplicated (duplicate route rows are never
  materialized). Its LDS frame permutes the tr16 granule via ``swz=True``, so it
  keeps its own ``_make_slot_wgrad_shared_storage`` rather than sharing the
  dense one.
"""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _std_arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.buffer_ops import (
    _create_i64_constant,
    _unwrap_value,
    create_llvm_ptr,
    get_element_ptr,
)
from flydsl.expr.primitive import get_iter as _get_iter
from flydsl.expr.primitive import ptrtoint as _ptrtoint
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue

from primus_turbo.flydsl.gemm.gemm_bf16_kernel import (
    _make_shared_storage,
    gemm_bf16_nn_tile,
    gemm_bf16_nt_tile,
)
from primus_turbo.flydsl.utils.gemm_helper import (
    BLOCK_K,
    G2SLoader,
    GatherVarKG2SLoaderBf16,
    Mfma16x16x32,
    S2RLoaderTr16x32Bf16,
    S2RLoaderTr16x32Bf16Wide,
    StoreCBf16,
    _readfirstlane_i32,
    compute_global_swizzle_nn_bf16,
    compute_global_swizzle_nn_bf16_rc,
    compute_global_swizzle_nn_bf16_wide,
    emit_for,
    emit_if_then,
    group_m_tile_decode,
    make_bf16_buffer_tensor_rebased,
    make_bf16_fp16_tile_tensor,
    make_value_attrs,
    wait_barrier,
    wave_lane_with_rank,
    wave_rank_desc_stable,
    xcd_band_remap_pid,
    xcd_remap_pid,
)
from primus_turbo.flydsl.utils.prims import _i64


def _load_i32(base, offset):
    """Scalar i32 table read at element `offset` off an int64 base pointer."""
    ptr = create_llvm_ptr(base + _i64(offset) * _create_i64_constant(4))
    return ArithValue(_unwrap_value(_llvm.load(ir.IntegerType.get_signless(32), ptr)))


def _load_i64_as_i32(base, offset):
    # load global i64 at base[offset] and truncate to i32
    ptr = create_llvm_ptr(_unwrap_value(base), 1)  # global address space
    idx = _unwrap_value(offset)
    if isinstance(idx.type, ir.IndexType):
        idx = _unwrap_value(_std_arith.IndexCastOp(fx.T.i64(), idx).result)
    elif isinstance(idx.type, ir.IntegerType) and idx.type.width < 64:
        idx = _unwrap_value(_std_arith.ExtSIOp(fx.T.i64(), idx).result)
    byte_off = _unwrap_value(_std_arith.MulIOp(idx, _create_i64_constant(8)).result)
    elem = get_element_ptr(ptr, byte_offset=byte_off, elem_type=fx.T.i8())
    val = _llvm.LoadOp(fx.T.i64(), elem, ordering=_llvm.AtomicOrdering.monotonic, alignment=8)
    trunc = _std_arith.TruncIOp(fx.T.i32(), val.result)
    return ArithValue(trunc.result, signed=True)


def _tail_quad_conds(q_row, q_col, out_m, out_n, half_m, half_n, mask_m, mask_n):
    """Liveness of each (A half, B half) quadrant, or None on an axis that tiles exactly.
    A run starting past the output extent is masked away at store time, so skip its MFMA.
    Module level: inside the kernel body a plain ``if`` is rewritten into device control flow."""
    a = (q_row < out_m, q_row + half_m < out_m) if mask_m else (None, None)
    b = (q_col < out_n, q_col + half_n < out_n) if mask_n else (None, None)
    conds = {}
    for i in range(2):
        for j in range(2):
            parts = [p for p in (a[i], b[j]) if p is not None]
            conds[i, j] = arith.andi(*parts) if len(parts) == 2 else (parts[0] if parts else None)
    return conds


# bf16 and fp16 run the same pipeline; only the mfma operand format differs.
_F16 = (torch.bfloat16, torch.float16)


def _ab_ty(dtype: torch.dtype):
    """Operand format for the mfma atom. Cf. the fp8 kernels' _ea/_eb: the format is a type
    the tile carries, not a boolean it has to re-derive."""
    return fx.Float16 if dtype == torch.float16 else fx.BFloat16


def _grid_x(total_tiles: int, cap_cu: int) -> int:
    """WGs to launch: one per tile unless a CU budget caps it, in which case a fixed grid of
    that many strides the tile space. Both operands are host ints, so this is a plain min --
    the fp8 kernel needs arith.select only because its tile count is a runtime value."""
    if cap_cu <= 0:
        return total_tiles
    ncus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return min(total_tiles, int(cap_cu), ncus)


@ASTRewriter.transform
def grouped_gemm_bf16_variable_k_tile(
    A,
    B,
    C,
    group_idx,
    block_m,
    block_n,
    m_start,
    m_end,
    lds,
    out_m_rt,
    out_n_rt,
    *,
    G,
    OUT_M,
    OUT_N,
    BLOCK_M,
    BLOCK_N,
    persistent=False,
    ab_ty=fx.BFloat16,
    out_fp16=False,
    c_cache_modifier=0,
    trans_c=False,
    lds_chunk_stride=1152,
    mask_m=None,
):
    CHUNK = 4
    WGRAD_WAVES = 8  # fixed 8 waves per block
    assert BLOCK_M >= 128 and BLOCK_N >= 64 and BLOCK_M % 128 == 0 and BLOCK_N % 64 == 0
    N_TILES_A = BLOCK_M // 128
    # A ragged OUT_M over-launches the last M block; a partitioned launch passes mask_m itself.
    MASK_M = (OUT_M % BLOCK_M != 0) if mask_m is None else mask_m
    MASK_N = OUT_N % BLOCK_N != 0
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = (BLOCK_M // 16) // WGRAD_WAVES
    N_LDS_STEPS_B = (BLOCK_N // 16) // WGRAD_WAVES
    N_WAVE_N = WGRAD_WAVES // 2

    lane_id = fx.thread_idx.x % 64
    wave_id = fx.thread_idx.x // 64
    wave_m = wave_id // N_WAVE_N
    wave_n = wave_id % N_WAVE_N

    group_tokens = m_end - m_start
    bf16_ir = fx.BFloat16.ir_type
    # base offset and per-group span (group_tokens * OUT * 2 bytes) can both exceed
    # int32 for a worst-case pool; compute in int64 so the span does not wrap before
    # make_bf16_buffer_tensor_rebased clamps it to the 32-bit HW num_records field.
    a_base_off = _i64(m_start) * fx.Int64(OUT_M * 2)
    b_base_off = _i64(m_start) * fx.Int64(OUT_N * 2)
    a_span = _i64(group_tokens) * _i64(out_m_rt) * fx.Int64(2)
    b_span = _i64(group_tokens) * _i64(out_n_rt) * fx.Int64(2)
    gA = make_bf16_buffer_tensor_rebased(A, bf16_ir, a_base_off, a_span)
    gB = make_bf16_buffer_tensor_rebased(B, bf16_ir, b_base_off, b_span)
    a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
    b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

    gl_off_a = compute_global_swizzle_nn_bf16_wide(lane_id, wave_id, OUT_M, N_LDS_STEPS_A)
    gl_off_b = compute_global_swizzle_nn_bf16_wide(lane_id, wave_id, OUT_N, N_LDS_STEPS_B)

    a0_off = block_m * BLOCK_M
    a1_off = a0_off + LDS_BLOCK_M
    b0_off = block_n * BLOCK_N
    b1_off = b0_off + LDS_BLOCK_N
    a_k_step = fx.Int32(BLOCK_K) * out_m_rt
    b_k_step = fx.Int32(BLOCK_K) * out_n_rt

    NTA16 = N_TILES_A * 2
    NTB16 = (BLOCK_N // 16) // (2 * N_WAVE_N)
    N_ACCUMS16 = NTA16 * NTB16
    mfma = Mfma16x16x32(NTA16, NTB16, ab_ty)
    a_s2r = S2RLoaderTr16x32Bf16Wide(wave_m, NTA16, chunk_stride=lds_chunk_stride)
    b_s2r = S2RLoaderTr16x32Bf16Wide(wave_n, NTB16, chunk_stride=lds_chunk_stride)
    ACC_VEC_N = 4
    N_ACCUMS_EFF = N_ACCUMS16
    a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, bf16_ir, wave_id, chunk_stride=lds_chunk_stride)
    b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, bf16_ir, wave_id, chunk_stride=lds_chunk_stride)
    out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    if const_expr(trans_c):
        store_c = StoreCBf16(C, G * OUT_N, OUT_M, out_ty, cache_modifier=c_cache_modifier)
    else:
        store_c = StoreCBf16(C, G * OUT_M, OUT_N, out_ty, cache_modifier=c_cache_modifier)

    acc00 = [fx.make_rmem_tensor(fx.make_layout(ACC_VEC_N, 1), fx.Float32) for _ in range(N_ACCUMS_EFF)]
    acc01 = [fx.make_rmem_tensor(fx.make_layout(ACC_VEC_N, 1), fx.Float32) for _ in range(N_ACCUMS_EFF)]
    acc10 = [fx.make_rmem_tensor(fx.make_layout(ACC_VEC_N, 1), fx.Float32) for _ in range(N_ACCUMS_EFF)]
    acc11 = [fx.make_rmem_tensor(fx.make_layout(ACC_VEC_N, 1), fx.Float32) for _ in range(N_ACCUMS_EFF)]
    for quad in (acc00, acc01, acc10, acc11):
        for reg in quad:
            fx.memref_store_vec(mfma.zero_value, reg)

    # Predicate the over-launched quadrants behind one wave-uniform branch, not a second tile class.
    quad_live = _tail_quad_conds(
        block_m * BLOCK_M + wave_m * (NTA16 * 16),
        block_n * BLOCK_N + wave_n * (NTB16 * 16),
        OUT_M,
        OUT_N,
        LDS_BLOCK_M,
        LDS_BLOCK_N,
        MASK_M,
        MASK_N,
    )

    def _mma_quad(acc, a, b, cond):
        """One accumulator quadrant, skipped whole when its output rows/columns are masked."""

        def _do():
            c = [Vec(fx.memref_load_vec(r)) for r in acc]
            c = mfma.call(a, b, c)
            for idx in range_constexpr(len(acc)):
                fx.memref_store_vec(c[idx], acc[idx])

        if const_expr(cond is None):
            _do()
        else:
            emit_if_then(cond, _do)

    # An empty expert only has to store zero accumulators, so the whole fetch pipeline is skipped.
    def _prologue():
        wait_barrier(0)
        b_g2s.load(lds.B_lds_cur_0, b0_off + 0 * b_k_step)
        a_g2s.load(lds.A_lds_cur_0, a0_off + 0 * a_k_step)
        b_g2s.load(lds.B_lds_cur_1, b1_off + 0 * b_k_step)
        a_g2s.load(lds.A_lds_cur_1, a1_off + 0 * a_k_step)
        # Divergent only holds for one tile per WG; a persistent loop needs every wave to
        # stop before the next tile's g2s reuses this LDS. Cf. dense_mma_pipeline_bf16.
        if const_expr(persistent):
            rocdl.s_barrier()
        elif wave_m == 1:
            rocdl.s_barrier()
        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)
        b_g2s.load(lds.B_lds_next_0, b0_off + 1 * b_k_step)
        a_g2s.load(lds.A_lds_next_0, a0_off + 1 * a_k_step)
        b_g2s.load(lds.B_lds_next_1, b1_off + 1 * b_k_step)
        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

    emit_if_then(group_tokens > 0, _prologue)

    k_iters = (group_tokens + (BLOCK_K - 1)) // BLOCK_K
    n_chunks = (k_iters + (CHUNK - 1)) // CHUNK

    # nested to isolate Python-level buffer rotation from the runtime chunk loop
    def _chunk(chunk_iv, live):
        chunk_idx = ArithValue(chunk_iv)
        a_cur0, a_cur1 = lds.A_lds_cur_0, lds.A_lds_cur_1
        a_next0, a_next1 = lds.A_lds_next_0, lds.A_lds_next_1
        b_cur0, b_cur1 = lds.B_lds_cur_0, lds.B_lds_cur_1
        b_next0, b_next1 = lds.B_lds_next_0, lds.B_lds_next_1
        for j in range_constexpr(CHUNK):
            k = chunk_idx * CHUNK + j
            # 4-buffer pipelined body: interleave s2r/g2s with the 4 mfma quadrants
            b0 = b_s2r.load(b_cur0)
            a0 = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, a1_off + (k + 1) * a_k_step)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            _mma_quad(acc00, a0, b0, live[0, 0])
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            b1 = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, b0_off + (k + 2) * b_k_step)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            _mma_quad(acc01, a0, b1, live[0, 1])
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            a1 = a_s2r.load(a_cur1)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            _mma_quad(acc10, a1, b0, live[1, 0])
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            # Both k+2 refills sit in the last phase, most-urgent first; issuing earlier only ages the line.
            a_g2s.load(a_cur0, a0_off + (k + 2) * a_k_step)
            b_g2s.load(b_cur1, b1_off + (k + 2) * b_k_step)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)
            rocdl.sched_barrier(0)
            _mma_quad(acc11, a1, b1, live[1, 1])
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

    # Only ragged boundary tiles branch, and on a workgroup-uniform test so barriers stay matched.
    all_live = {key: None for key in quad_live}
    interior, boundary = None, None
    if const_expr(MASK_M):
        interior = (block_m + 1) * BLOCK_M <= fx.Int32(OUT_M)
        boundary = (block_m + 1) * BLOCK_M > fx.Int32(OUT_M)
    if const_expr(MASK_N):
        n_in = (block_n + 1) * BLOCK_N <= fx.Int32(OUT_N)
        n_bd = (block_n + 1) * BLOCK_N > fx.Int32(OUT_N)
        interior = n_in if interior is None else arith.andi(interior, n_in)
        boundary = n_bd if boundary is None else arith.ori(boundary, n_bd)

    def _loop(live):
        emit_for(n_chunks, lambda iv: _chunk(iv, live))

    if const_expr(interior is None):
        _loop(all_live)
    else:
        emit_if_then(interior, lambda: _loop(all_live))
        emit_if_then(boundary, lambda: _loop(quad_live))

    c00 = [Vec(fx.memref_load_vec(reg)) for reg in acc00]
    c01 = [Vec(fx.memref_load_vec(reg)) for reg in acc01]
    c10 = [Vec(fx.memref_load_vec(reg)) for reg in acc10]
    c11 = [Vec(fx.memref_load_vec(reg)) for reg in acc11]

    if const_expr(trans_c):
        local_m = block_m * BLOCK_M + wave_m * (NTA16 * 16)
        local_n = block_n * BLOCK_N + wave_n * (NTB16 * 16)
        for cfrag, q_row, q_col in (
            (c00, local_m, local_n),
            (c01, local_m, local_n + LDS_BLOCK_N),
            (c10, local_m + LDS_BLOCK_M, local_n),
            (c11, local_m + LDS_BLOCK_M, local_n + LDS_BLOCK_N),
        ):
            for i in range_constexpr(NTA16):
                for j in range_constexpr(NTB16):
                    store_c.store_trans16(
                        [cfrag[i * NTB16 + j]],
                        group_idx,
                        q_row + i * 16,
                        q_col + j * 16,
                        OUT_M,
                        OUT_N,
                        mask_m=MASK_M,
                    )
    else:
        base_row = group_idx * OUT_M + block_m * BLOCK_M + wave_m * (NTA16 * 16)
        row_bound = (group_idx + 1) * OUT_M
        base_col = block_n * BLOCK_N + wave_n * (NTB16 * 16)
        # Both column halves share the band SRD and every row address; only the store immediate differs.
        for cfrags, q_row in (((c00, c01), base_row), ((c10, c11), base_row + LDS_BLOCK_M)):
            store_c.store_band16(cfrags, q_row, base_col, LDS_BLOCK_N, NTA16, NTB16, row_bound, mask_n=MASK_N)


@functools.lru_cache(maxsize=64)
def _compile_grouped_bf16_wgrad(
    OUT_M,
    OUT_N,
    G,
    BLOCK_M=256,
    BLOCK_N=256,
    num_xcd=8,
    waves_per_eu=2,
    agpr_alloc=0,
    ab_ty=fx.BFloat16,
    out_fp16=False,
    trans_c=False,
    # One padded chunk splits the tr16 reader's four lane groups off a single bank half: 128 mod 256.
    lds_chunk_stride=1152,
    group_m=1,
    xcd_band=24,
    cap_cu=0,
):
    # See _compile_grouped_bf16_nt: cap_cu = 0 keeps the tuned one-tile-per-WG launch.
    persistent = cap_cu > 0
    N_BLOCKS_M = (OUT_M + BLOCK_M - 1) // BLOCK_M
    N_BLOCKS_N = (OUT_N + BLOCK_N - 1) // BLOCK_N
    TILES_PER_GROUP = N_BLOCKS_M * N_BLOCKS_N
    TOTAL = G * TILES_PER_GROUP
    # A ragged OUT_M over-launches the last M block; its tail rows are dropped at store time.
    MASK_M = N_BLOCKS_M * BLOCK_M > OUT_M
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N, chunk_stride=lds_chunk_stride)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_grouped_variable_k(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        group_k_offsets: fx.Tensor,
        masked_k: fx.Tensor,
        out_m_rt: fx.Int32,
        out_n_rt: fx.Int32,
    ):
        _ = str(fx.thread_idx.x)
        go_base = fx.Int64(_ptrtoint(_get_iter(group_k_offsets)))
        gk_base = fx.Int64(_ptrtoint(_get_iter(masked_k)))
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # Dispatch order ranked in the prologue by a lane-resident descending rank, not a host argsort.
        # Tile-independent, so it is hoisted above the persistent loop and ranked once.
        lane = fx.Int32(fx.thread_idx.x) % fx.Int32(64)
        in_g = lane < fx.Int32(G)
        k_lane = _load_i32(gk_base, arith.select(in_g, lane, fx.Int32(0)) * fx.Int32(2))
        order_rank = wave_rank_desc_stable(arith.select(in_g, k_lane, fx.Int32(-1)), lane, G)

        # Free function for the same ast-rewriter reason as the NT kernel.
        def _do_tile(pid):
            # Band-cyclic XCD assignment: runs short enough to stay inside one expert, so skew spreads.
            tile = xcd_band_remap_pid(pid, TOTAL, num_xcd, xcd_band)
            group_idx = wave_lane_with_rank(order_rank, tile // TILES_PER_GROUP)
            local_tile = tile % TILES_PER_GROUP
            if const_expr(trans_c):
                block_n, block_m = group_m_tile_decode(local_tile, N_BLOCKS_N, N_BLOCKS_M, group_m)
            else:
                block_m, block_n = group_m_tile_decode(local_tile, N_BLOCKS_M, N_BLOCKS_N, group_m)
            m_start = _load_i64_as_i32(go_base, group_idx)
            m_end = m_start + _load_i64_as_i32(gk_base, group_idx)
            grouped_gemm_bf16_variable_k_tile(
                A,
                B,
                C,
                group_idx,
                block_m,
                block_n,
                m_start,
                m_end,
                lds,
                out_m_rt,
                out_n_rt,
                G=G,
                OUT_M=OUT_M,
                OUT_N=OUT_N,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                ab_ty=ab_ty,
                out_fp16=out_fp16,
                trans_c=trans_c,
                lds_chunk_stride=lds_chunk_stride,
                mask_m=MASK_M,
                persistent=persistent,
            )

        if const_expr(persistent):
            for t in range(fx.block_idx.x, TOTAL, fx.grid_dim.x):
                _do_tile(t)
        else:
            _do_tile(fx.block_idx.x)

    @flyc.jit
    def launch_grouped_variable_k(
        A,
        B,
        C,
        group_k_offsets,
        masked_k,
        out_m_rt: fx.Int32,
        out_n_rt: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = fx.Int32(_grid_x(TOTAL, cap_cu))
        kernel_grouped_variable_k(
            A,
            B,
            C,
            group_k_offsets,
            masked_k,
            out_m_rt,
            out_n_rt,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_grouped_variable_k


_COMPILED_GROUPED_GEMM_CACHE = {}


@functools.lru_cache(maxsize=8)
def _row_starts(total_m: int, block_m: int, device) -> torch.Tensor:
    """Row index of every M block; depends on nothing but the shape, so it is built once."""
    return torch.arange(0, total_m, block_m, device=device, dtype=torch.int64)


def m_tile_upper_bound(total_m: int, block_m: int, groups: int) -> int:
    """Tiles the M axis can need, from the shape alone: cutting at expert boundaries costs at
    most one short tile per expert, so this holds for any group lengths."""
    return (total_m + block_m - 1) // block_m + groups


def build_m_tile_table(group_lens, group_offs, block_m: int, upper: int) -> torch.Tensor:
    """(expert, first row, row count) per M tile, one expert per tile, built on device."""
    lens = group_lens.to(torch.int64)
    offs = group_offs.to(torch.int64)
    blocks = (lens + (block_m - 1)) // block_m
    first = torch.cumsum(blocks, 0) - blocks
    total = first[-1] + blocks[-1]
    idx = torch.arange(upper, device=lens.device, dtype=torch.int64)
    g = (torch.searchsorted(first, idx, right=True) - 1).clamp_(min=0, max=lens.numel() - 1)
    local = idx - first[g]
    live = idx < total
    rows = torch.where(live, (lens[g] - local * block_m).clamp_(min=0, max=block_m), torch.zeros_like(idx))
    start = torch.where(live, offs[g] + local * block_m, torch.zeros_like(idx))
    return torch.stack([g, start, rows], dim=1).to(torch.int32).contiguous()


def _ptr_only_view(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def grouped_gemm_bf16_variable_k_flydsl_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    group_k_offsets: torch.Tensor,
    masked_k: torch.Tensor = None,
    out_dtype: torch.dtype = torch.bfloat16,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    # An XCD remap reorders tiles across groups, so a skewed token count skews the per-XCD cost.
    num_xcd: int = 8,
    # Super-tile width in M blocks: co-resident workgroups share an A column slice.  Retune with the tile.
    group_m: int = 4,
    # Band-cyclic run length in tiles; it tracks co-residency rather than grid divisibility.
    xcd_band: int = 32,
    # CUs this launch may occupy; None/0 = the whole device, which keeps the tuned
    # one-tile-per-WG launch. A real budget switches to a capped persistent grid.
    cap_cu: int = 0,
    trans_c: bool = False,
) -> torch.Tensor:
    """Variable-K grouped wgrad: out[g]=a[g_rows].T@b[g_rows], K=[offsets[g],offsets[g]+masked_k[g])."""
    assert a.dim() == 2 and b.dim() == 2 and a.shape[0] == b.shape[0]
    assert a.dtype in _F16 and b.dtype == a.dtype, f"16-bit float operands only, got {a.dtype}/{b.dtype}"
    OUT_M = a.shape[1]
    OUT_N = b.shape[1]
    G = group_k_offsets.numel() - 1
    out_fp16 = out_dtype == torch.float16
    out_shape = (G, OUT_N, OUT_M) if trans_c else (G, OUT_M, OUT_N)
    out = torch.empty(out_shape, device=a.device, dtype=out_dtype)
    # index tables loaded as i64 in-kernel
    offsets_i64 = group_k_offsets if group_k_offsets.dtype == torch.int64 else group_k_offsets.to(torch.int64)
    # per-expert valid K length; default = padded span
    if masked_k is None:
        masked_k_i64 = (offsets_i64[1:] - offsets_i64[:-1]).contiguous()
    else:
        assert masked_k.numel() == G, f"masked_k len {masked_k.numel()} != G {G}"
        masked_k_i64 = (masked_k if masked_k.dtype == torch.int64 else masked_k.to(torch.int64)).contiguous()
    args = (
        _ptr_only_view(a),
        _ptr_only_view(b),
        flyc.from_torch_tensor(out),
        offsets_i64,
        masked_k_i64,
        OUT_M,
        OUT_N,
        torch.cuda.current_stream(),
    )
    key = (OUT_M, OUT_N, G, BLOCK_M, BLOCK_N, num_xcd, group_m, xcd_band, a.dtype, out_fp16, trans_c, cap_cu)
    compiled = _COMPILED_GROUPED_GEMM_CACHE.get(key)
    if compiled is None:
        launch = _compile_grouped_bf16_wgrad(
            OUT_M,
            OUT_N,
            G,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_xcd=num_xcd,
            ab_ty=_ab_ty(a.dtype),
            out_fp16=out_fp16,
            trans_c=trans_c,
            group_m=group_m,
            xcd_band=xcd_band,
            cap_cu=cap_cu,
        )
        compiled = flyc.compile(launch, *args)
        _COMPILED_GROUPED_GEMM_CACHE[key] = compiled
    compiled(*args)
    return out


_COMPILED_GROUPED_NT_CACHE = {}


def _compile_grouped_bf16_nt(
    M_TILES,
    N,
    K,
    G,
    BLOCK_M=256,
    BLOCK_N=256,
    GROUP_M=1,
    num_xcd=8,
    xcd_band=32,
    waves_per_eu=2,
    agpr_alloc=0,
    nt_vmcnt=3,
    ab_ty=fx.BFloat16,
    out_fp16=False,
    cap_cu=0,
):
    # cap_cu > 0 reserves CUs for a co-running kernel: a fixed grid of cap_cu WGs strides the
    # tile space instead of one WG per tile. cap_cu = 0 keeps the one-tile-per-WG launch the
    # full-device path was tuned on, byte for byte.
    persistent = cap_cu > 0
    # Tiles are cut per expert (build_m_tile_table), so none straddles two whatever the lengths.
    N_BLOCKS_M = M_TILES
    N_BLOCKS_N = (N + BLOCK_N - 1) // BLOCK_N
    TOTAL_TILES = N_BLOCKS_M * N_BLOCKS_N
    B_GRP = N * K  # elements of one expert's weight slab
    assert GROUP_M >= 1 and GROUP_M & (GROUP_M - 1) == 0, "GROUP_M must be a power of two"
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_grouped_nt(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        m_tiles: fx.Tensor,
        c_n: fx.Int32,
    ):
        _ = str(fx.thread_idx.x)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        mt_base = fx.Int64(_ptrtoint(_get_iter(m_tiles)))

        # Free function, not inlined at the call: the ast-rewriter would otherwise collect the
        # loaders built inside it as scf.for iter_args. Same reason as the fp8 kernel's _do_tile.
        def _do_tile(pid):
            tile = xcd_band_remap_pid(pid, TOTAL_TILES, num_xcd, xcd_band)
            block_m, block_n = group_m_tile_decode(tile, N_BLOCKS_M, N_BLOCKS_N, GROUP_M)
            _e = block_m * fx.Int32(3)
            g_idx = _load_i32(mt_base, _e)
            m_row = _load_i32(mt_base, _e + fx.Int32(1))
            # Wave-uniform: these three feed SRD bases and record counts, and a buffer descriptor
            # built from a per-lane value is garbage -- which is how the row count first showed up,
            # as a non-deterministic wrong answer rather than a fault.
            m_rows = _readfirstlane_i32(_load_i32(mt_base, _e + fx.Int32(2)))

            a_base = fx.Int64(_ptrtoint(_get_iter(A)))
            b_base = fx.Int64(_ptrtoint(_get_iter(B)))
            c_base = fx.Int64(_ptrtoint(_get_iter(C)))
            # Sized by the tile's own rows: the last tile of a run is short, and a fixed BLOCK_M
            # window would read past the end of A for the final expert.
            a_tile = make_bf16_fp16_tile_tensor(a_base, _i64(m_row) * fx.Int64(K * 2), m_rows * fx.Int32(K))
            b_tile = make_bf16_fp16_tile_tensor(b_base, _i64(g_idx) * fx.Int64(B_GRP * 2), B_GRP)
            c_tile = make_bf16_fp16_tile_tensor(
                c_base, _i64(m_row) * fx.Int64(2) * _i64(c_n), m_rows * fx.Int32(N)
            )

            gemm_bf16_nt_tile(
                a_tile,
                b_tile,
                c_tile,
                m_rows,  # a short run ends in a tile that stores fewer than BLOCK_M rows
                c_n,
                lds,
                fx.Int32(0),  # A/C are already rebased onto this tile's rows
                block_n,
                K=K,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                n_blocks=N_BLOCKS_N,
                GROUP_M=GROUP_M,
                num_xcd=num_xcd,
                ab_ty=ab_ty,
                out_fp16=out_fp16,
                nt_vmcnt=nt_vmcnt,
                pair_n=N % 2 == 0 and not out_fp16,
                n_tail=N % BLOCK_N,
                persistent=persistent,
            )

        if const_expr(persistent):
            for t in range(fx.block_idx.x, TOTAL_TILES, fx.grid_dim.x):
                _do_tile(t)
        else:
            _do_tile(fx.block_idx.x)

    @flyc.jit
    def launch_grouped_nt(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        m_tiles: fx.Tensor,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_grouped_nt(
            A,
            B,
            C,
            m_tiles,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(_grid_x(TOTAL_TILES, cap_cu), 1, 1), block=(512, 1, 1), stream=stream)

    return launch_grouped_nt


def grouped_gemm_bf16_nt_flydsl_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    group_offs: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    # Super-tile width in row blocks, a power of two: co-resident workgroups share B column blocks.
    GROUP_M: int = 4,
    # Band-cyclic XCD partition: a compact patch of one expert's B slab, still sampling the token range.
    num_xcd: int = 8,
    xcd_band: int = 32,
    # CUs this launch may occupy; None/0 = the whole device, which keeps the tuned
    # one-tile-per-WG launch. A real budget switches to a capped persistent grid.
    cap_cu: int = 0,
) -> torch.Tensor:
    """Grouped NT forward: out[rows] = a[rows] @ b[g]^T for the expert g owning each row run."""
    assert a.dim() == 2 and b.dim() == 3 and a.dtype == b.dtype and a.dtype in _F16
    TOTAL_M, K = a.shape
    G, N, Kb = b.shape
    assert Kb == K, f"b K={Kb} != a K={K}"
    out = torch.empty(TOTAL_M, N, device=a.device, dtype=out_dtype)
    offs = group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)
    m_upper = m_tile_upper_bound(TOTAL_M, BLOCK_M, G)
    m_tiles = build_m_tile_table(offs[1:] - offs[:-1], offs, BLOCK_M, m_upper)
    args = (
        _ptr_only_view(a),
        flyc.from_torch_tensor(b.reshape(-1)),
        flyc.from_torch_tensor(out),
        m_tiles.reshape(-1),
        N,
        torch.cuda.current_stream(),
    )
    key = (m_upper, N, K, G, BLOCK_M, BLOCK_N, GROUP_M, num_xcd, xcd_band, a.dtype, out_dtype, cap_cu)
    compiled = _COMPILED_GROUPED_NT_CACHE.get(key)
    if compiled is None:
        launch = _compile_grouped_bf16_nt(
            m_upper,
            N,
            K,
            G,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            GROUP_M=GROUP_M,
            num_xcd=num_xcd,
            xcd_band=xcd_band,
            cap_cu=cap_cu,
            ab_ty=_ab_ty(a.dtype),
            out_fp16=out_dtype == torch.float16,
        )
        compiled = flyc.compile(launch, *args)
        _COMPILED_GROUPED_NT_CACHE[key] = compiled
    compiled(*args)
    return out


_COMPILED_GROUPED_NN_CACHE = {}


def _compile_grouped_bf16_nn(
    M_TILES,
    N,
    K,
    G,
    BLOCK_M=256,
    BLOCK_N=256,
    GROUP_M=1,
    num_xcd=8,
    xcd_band=32,
    waves_per_eu=2,
    agpr_alloc=0,
    nt_vmcnt=3,
    ab_ty=fx.BFloat16,
    out_fp16=False,
    cap_cu=0,
):
    # See _compile_grouped_bf16_nt: cap_cu = 0 keeps the tuned one-tile-per-WG launch.
    persistent = cap_cu > 0
    N_BLOCKS_M = M_TILES
    N_BLOCKS_N = (N + BLOCK_N - 1) // BLOCK_N
    TOTAL_TILES = N_BLOCKS_M * N_BLOCKS_N
    B_GRP = N * K  # elements of one expert's weight slab
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_grouped_nn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        m_tiles: fx.Tensor,
        c_n: fx.Int32,
    ):
        _ = str(fx.thread_idx.x)
        mt_base = fx.Int64(_ptrtoint(_get_iter(m_tiles)))
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        # Free function for the same ast-rewriter reason as the NT kernel.
        def _do_tile(pid):
            tile = xcd_band_remap_pid(pid, TOTAL_TILES, num_xcd, xcd_band)
            block_m, block_n = group_m_tile_decode(tile, N_BLOCKS_M, N_BLOCKS_N, GROUP_M)
            _e = block_m * fx.Int32(3)
            g_idx = _load_i32(mt_base, _e)
            m_row = _load_i32(mt_base, _e + fx.Int32(1))
            # Wave-uniform: these three feed SRD bases and record counts, and a buffer descriptor
            # built from a per-lane value is garbage -- which is how the row count first showed up,
            # as a non-deterministic wrong answer rather than a fault.
            m_rows = _readfirstlane_i32(_load_i32(mt_base, _e + fx.Int32(2)))

            a_base = fx.Int64(_ptrtoint(_get_iter(A)))
            b_base = fx.Int64(_ptrtoint(_get_iter(B)))
            c_base = fx.Int64(_ptrtoint(_get_iter(C)))
            # Sized by the tile's own rows: the last tile of a run is short, and a fixed BLOCK_M
            # window would read past the end of A for the final expert.
            a_tile = make_bf16_fp16_tile_tensor(a_base, _i64(m_row) * fx.Int64(K * 2), m_rows * fx.Int32(K))
            b_tile = make_bf16_fp16_tile_tensor(b_base, _i64(g_idx) * fx.Int64(B_GRP * 2), B_GRP)
            c_tile = make_bf16_fp16_tile_tensor(
                c_base, _i64(m_row) * fx.Int64(2) * _i64(c_n), m_rows * fx.Int32(N)
            )

            gemm_bf16_nn_tile(
                a_tile,
                b_tile,
                c_tile,
                m_rows,  # a short run ends in a tile that stores fewer than BLOCK_M rows
                c_n,
                lds,
                fx.Int32(0),  # A/C are already rebased onto this tile's rows
                block_n,
                K=K,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                n_blocks=N_BLOCKS_N,
                GROUP_M=GROUP_M,
                num_xcd=num_xcd,
                ab_ty=ab_ty,
                out_fp16=out_fp16,
                nt_vmcnt=nt_vmcnt,
                n_tail=N % BLOCK_N,
                persistent=persistent,
            )

        if const_expr(persistent):
            for t in range(fx.block_idx.x, TOTAL_TILES, fx.grid_dim.x):
                _do_tile(t)
        else:
            _do_tile(fx.block_idx.x)

    @flyc.jit
    def launch_grouped_nn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        m_tiles: fx.Tensor,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_grouped_nn(
            A,
            B,
            C,
            m_tiles,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(_grid_x(TOTAL_TILES, cap_cu), 1, 1), block=(512, 1, 1), stream=stream)

    return launch_grouped_nn


def grouped_gemm_bf16_nn_flydsl_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    group_offs: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    # Same band-cyclic mapping as the NT path; 0 picks the super-tile height from the weight slab.
    GROUP_M: int = 0,
    num_xcd: int = 8,
    xcd_band: int = 32,
    # CUs this launch may occupy; None/0 = the whole device, which keeps the tuned
    # one-tile-per-WG launch. A real budget switches to a capped persistent grid.
    cap_cu: int = 0,
) -> torch.Tensor:
    """Grouped NN: out[rows] = a[rows] @ b[g] for the expert g owning each row run."""
    assert a.dim() == 2 and b.dim() == 3 and a.dtype == b.dtype and a.dtype in _F16
    TOTAL_M, K = a.shape
    G, Kb, N = b.shape
    assert Kb == K, f"b K={Kb} != a K={K}"
    # A super-tile trades A-slab against B-slab reuse and the balance moves with the weight slab.
    if GROUP_M == 0:
        GROUP_M = 8 if K * N * a.element_size() > 24 << 20 else 4
    out = torch.empty(TOTAL_M, N, device=a.device, dtype=out_dtype)
    offs = group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)
    m_upper = m_tile_upper_bound(TOTAL_M, BLOCK_M, G)
    m_tiles = build_m_tile_table(offs[1:] - offs[:-1], offs, BLOCK_M, m_upper)
    args = (
        _ptr_only_view(a),
        flyc.from_torch_tensor(b.reshape(-1)),
        flyc.from_torch_tensor(out),
        m_tiles.reshape(-1),
        N,
        torch.cuda.current_stream(),
    )
    key = (m_upper, N, K, G, BLOCK_M, BLOCK_N, GROUP_M, num_xcd, xcd_band, a.dtype, out_dtype, cap_cu)
    compiled = _COMPILED_GROUPED_NN_CACHE.get(key)
    if compiled is None:
        launch = _compile_grouped_bf16_nn(
            m_upper,
            N,
            K,
            G,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            GROUP_M=GROUP_M,
            num_xcd=num_xcd,
            xcd_band=xcd_band,
            cap_cu=cap_cu,
            ab_ty=_ab_ty(a.dtype),
            out_fp16=out_dtype == torch.float16,
        )
        compiled = flyc.compile(launch, *args)
        _COMPILED_GROUPED_NN_CACHE[key] = compiled
    compiled(*args)
    return out


# Staged slot ids per block, in entries. The K loop reads slots from here instead
# of issuing a buffer load per chunk. One direct-to-LDS pass stages SLOT_LDS_PASS
# entries (4 dwords per lane, WGRAD_WAVES waves); only as many passes as the group
# needs are issued, so a small group does not pay for the whole array.
SLOT_LDS_PASS = 8 * 64 * 4
SLOT_LDS_CAP = 2 * SLOT_LDS_PASS


def _make_slot_wgrad_shared_storage(BLOCK_M, BLOCK_N, slot_lds=False):
    """LDS frame for this wgrad tile: 512-elem tr16 blocks, matching ``swz=True``.
    Deliberately not the padded chunk_stride frame the dense bf16 tiles use."""
    a_lds_size = (BLOCK_M // 2) * BLOCK_K
    b_lds_size = (BLOCK_N // 2) * BLOCK_K

    if slot_lds:

        @fx.struct
        class SharedStorage:
            A_lds_cur_0: fx.Array[fx.BFloat16, a_lds_size, 16]
            A_lds_cur_1: fx.Array[fx.BFloat16, a_lds_size, 16]
            A_lds_next_0: fx.Array[fx.BFloat16, a_lds_size, 16]
            A_lds_next_1: fx.Array[fx.BFloat16, a_lds_size, 16]
            B_lds_cur_0: fx.Array[fx.BFloat16, b_lds_size, 16]
            B_lds_cur_1: fx.Array[fx.BFloat16, b_lds_size, 16]
            B_lds_next_0: fx.Array[fx.BFloat16, b_lds_size, 16]
            B_lds_next_1: fx.Array[fx.BFloat16, b_lds_size, 16]
            SLOT_lds: fx.Array[fx.Int32, SLOT_LDS_CAP, 16]

        return SharedStorage

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.BFloat16, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.BFloat16, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.BFloat16, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.BFloat16, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.BFloat16, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.BFloat16, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.BFloat16, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.BFloat16, b_lds_size, 16]

    return SharedStorage


@ASTRewriter.transform
def gemm_bf16_variable_k_tile(
    A,
    B,
    C,
    group_idx,
    block_m,
    block_n,
    m_start,
    m_end,
    lds,
    out_m_rt,
    out_n_rt,
    *,
    G,
    OUT_M,
    OUT_N,
    BLOCK_M,
    BLOCK_N,
    out_fp16=False,
    c_cache_modifier=0,
    trans_c=False,
    a_slot_ids=None,
    b_slot_ids=None,
    slot_len=None,
    slot_x4=False,
    slot_unroll=1,
    slot_lds=False,
    slot_alu=False,
    slot_u16=False,
):
    CHUNK = 4
    # slot_x4 folds a chunk's slot loads into one dwordx4; without it a wider window
    # still costs one dwordx1 per k, but it is the number of drain points that matters.
    # the staged table is read linearly from LDS, so the interleave buys nothing
    assert not (slot_lds and slot_x4), "slot_lds and slot_x4 are alternatives"
    # slot_lds removes the drain slot_unroll amortizes, and windowing assumes 1-chunk lookahead
    assert not (slot_lds and slot_unroll > 1), "slot_lds implies slot_unroll == 1"
    # probe only: slot_alu ignores the table, so it cannot combine with either scheme
    assert not (slot_alu and (slot_lds or slot_x4)), "slot_alu excludes slot_lds/slot_x4"
    # slot_u16 is the same interleaved table at half width; the others read i32
    assert not (slot_u16 and (slot_x4 or slot_lds or slot_alu)), "slot_u16 excludes the other modes"
    WGRAD_WAVES = 8  # fixed 8 waves per block
    assert BLOCK_M >= 128 and BLOCK_N >= 64 and BLOCK_M % 128 == 0 and BLOCK_N % 64 == 0
    N_TILES_A = BLOCK_M // 128
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = (BLOCK_M // 16) // WGRAD_WAVES
    N_LDS_STEPS_B = (BLOCK_N // 16) // WGRAD_WAVES
    N_WAVE_N = WGRAD_WAVES // 2

    lane_id = fx.thread_idx.x % 64
    wave_id = fx.thread_idx.x // 64
    wave_m = wave_id // N_WAVE_N
    wave_n = wave_id % N_WAVE_N

    group_tokens = m_end - m_start
    bf16_ir = fx.BFloat16.ir_type
    # base offset and per-group span (group_tokens * OUT * 2 bytes) can both exceed
    # int32 for a worst-case pool; compute in int64 so the span does not wrap before
    # make_bf16_buffer_tensor_rebased clamps it to the 32-bit HW num_records field.
    a_base_off = _i64(m_start) * fx.Int64(OUT_M * 2)
    b_base_off = _i64(m_start) * fx.Int64(OUT_N * 2)
    a_span = _i64(group_tokens) * _i64(out_m_rt) * fx.Int64(2)
    b_span = _i64(group_tokens) * _i64(out_n_rt) * fx.Int64(2)

    a0_off = block_m * BLOCK_M
    a1_off = a0_off + LDS_BLOCK_M
    b0_off = block_n * BLOCK_N
    b1_off = b0_off + LDS_BLOCK_N
    a_k_step = fx.Int32(BLOCK_K) * out_m_rt
    b_k_step = fx.Int32(BLOCK_K) * out_n_rt

    NTA16 = N_TILES_A * 2
    NTB16 = (BLOCK_N // 16) // (2 * N_WAVE_N)
    N_ACCUMS16 = NTA16 * NTB16
    mfma = Mfma16x16x32(NTA16, NTB16)
    a_s2r = S2RLoaderTr16x32Bf16(wave_m, NTA16, swz=True)
    b_s2r = S2RLoaderTr16x32Bf16(wave_n, NTB16, swz=True)
    ACC_VEC_N = 4
    N_ACCUMS_EFF = N_ACCUMS16

    a_offs = [a0_off, a1_off]
    b_offs = [b0_off, b1_off]

    def _make_g2s(operand, slot_ids, row_stride, n_steps, base_off, span, cols):
        """Dense rebased loader, or one gather loader serving both LDS halves."""
        if const_expr(slot_ids is None):
            g = make_bf16_buffer_tensor_rebased(operand, bf16_ir, base_off, span)
            gl_off = compute_global_swizzle_nn_bf16(lane_id, wave_id, row_stride, n_steps, swz=True)
            return G2SLoader(fx.logical_divide(g, fx.make_layout(1, 1)), gl_off, n_steps, bf16_ir, wave_id)
        gl_rc = compute_global_swizzle_nn_bf16_rc(lane_id, wave_id, n_steps, WGRAD_WAVES, swz=True)
        x4 = (CHUNK * BLOCK_K, CHUNK) if (slot_x4 or slot_u16) else None
        return GatherVarKG2SLoaderBf16(
            operand,
            gl_rc,
            n_steps,
            wave_id,
            slot_ids,
            slot_len,
            row_stride,
            m_start,
            cols,
            slot_x4=x4,
            slot_u16=slot_u16,
        )

    a_g2s = _make_g2s(A, a_slot_ids, OUT_M, N_LDS_STEPS_A, a_base_off, a_span, a_offs)
    b_g2s = _make_g2s(B, b_slot_ids, OUT_N, N_LDS_STEPS_B, b_base_off, b_span, b_offs)

    # Dense loaders take the fused (column + k * row_stride) offset; gather loaders
    # hold the column base and take voffsets resolved ahead of the MFMA quadrants.
    def _load_a(dst, half, k, voffs):
        if const_expr(a_slot_ids is None):
            a_g2s.load(dst, a_offs[half] + k * a_k_step)
        elif const_expr(slot_u16):
            # the window hands back a thunk: unpack here, not at the window head
            a_g2s.load(dst, half, voffs())
        else:
            a_g2s.load(dst, half, voffs)

    def _load_b(dst, half, k, voffs):
        if const_expr(b_slot_ids is None):
            b_g2s.load(dst, b_offs[half] + k * b_k_step)
        elif const_expr(slot_u16):
            b_g2s.load(dst, half, voffs())
        else:
            b_g2s.load(dst, half, voffs)

    def _voffs(loader, slot_ids, k, wbase=None, wlim=None):
        if const_expr(slot_ids is None):
            return None
        if const_expr(slot_lds):
            return loader.voffsets_lds(lds.SLOT_lds, k * fx.Int32(BLOCK_K) - wbase, wlim)
        if const_expr(slot_alu):
            return loader.voffsets_alu(k * fx.Int32(BLOCK_K))
        return loader.voffsets(k * fx.Int32(BLOCK_K))

    def _voffs_x4(loader, slot_ids, chunk_idx):
        if const_expr(slot_ids is None):
            return None
        return loader.voffsets_x4(chunk_idx)

    def _packs_u16(loader, slot_ids, chunk_idx):
        if const_expr(slot_ids is None):
            return None
        return loader.slot_pack_u16(chunk_idx)

    def _lazy_u16(loader, packs, j):
        """Thunk unpacking k step j at the use site, keeping voffsets short-lived."""
        return lambda: loader.unpack_u16(packs, j)

    out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    if const_expr(trans_c):
        store_c = StoreCBf16(C, G * OUT_N, OUT_M, out_ty, cache_modifier=c_cache_modifier)
    else:
        store_c = StoreCBf16(C, G * OUT_M, OUT_N, out_ty, cache_modifier=c_cache_modifier)

    acc00 = [fx.make_rmem_tensor(fx.make_layout(ACC_VEC_N, 1), fx.Float32) for _ in range(N_ACCUMS_EFF)]
    acc01 = [fx.make_rmem_tensor(fx.make_layout(ACC_VEC_N, 1), fx.Float32) for _ in range(N_ACCUMS_EFF)]
    acc10 = [fx.make_rmem_tensor(fx.make_layout(ACC_VEC_N, 1), fx.Float32) for _ in range(N_ACCUMS_EFF)]
    acc11 = [fx.make_rmem_tensor(fx.make_layout(ACC_VEC_N, 1), fx.Float32) for _ in range(N_ACCUMS_EFF)]
    for quad in (acc00, acc01, acc10, acc11):
        for reg in quad:
            fx.memref_store_vec(mfma.zero_value, reg)

    # A window stages SLOT_LDS_CAP entries but advances by one chunk less, because
    # the K pipeline looks a chunk ahead and must still find those slots staged.
    WIN_CHUNKS = SLOT_LDS_CAP // (CHUNK * BLOCK_K) - 1
    WIN_TOKENS = WIN_CHUNKS * CHUNK * BLOCK_K

    def _win_fills(w):
        """Passes needed for window w: enough to cover it, never more."""
        rem = group_tokens - ArithValue(w) * fx.Int32(WIN_TOKENS)
        n = (rem + fx.Int32(SLOT_LDS_PASS - 1)) // fx.Int32(SLOT_LDS_PASS)
        return ArithValue(
            arith.minsi(arith._to_raw(n), arith._to_raw(fx.Int32(SLOT_LDS_CAP // SLOT_LDS_PASS))),
            signed=True,
        )

    def _win_limit(w):
        return _win_fills(w) * fx.Int32(SLOT_LDS_PASS) - fx.Int32(1)

    if const_expr(slot_lds):
        # Stage before any pool prefetch is in flight: this fill is the tile's only
        # slot-side VMEM traffic, so the drain it costs is paid once, not per chunk.
        slot_g2s = a_g2s if const_expr(a_slot_ids is not None) else b_g2s

        # plain call so the dynamic for does not make slot_g2s/lds loop-carried
        def _fill_pass(base, p):
            slot_g2s.fill_lds(lds.SLOT_lds, base, ArithValue(p), WGRAD_WAVES)

        def _fill_win(w):
            base = ArithValue(w) * fx.Int32(WIN_TOKENS)
            for fill_iv in range(_win_fills(w)):
                _fill_pass(base, fill_iv)
            wait_barrier(0)
            rocdl.s_barrier()

        _fill_win(fx.Int32(0))
        w0_base = fx.Int32(0)
        w0_lim = _win_limit(fx.Int32(0))
    else:
        w0_base = None
        w0_lim = None

    wait_barrier(0)
    av1 = None
    bv1 = None
    if const_expr(slot_x4):
        # one dwordx4 already covers the preamble's k = 0 and k = 1
        aw = _voffs_x4(a_g2s, a_slot_ids, fx.Int32(0))
        bw = _voffs_x4(b_g2s, b_slot_ids, fx.Int32(0))
        av0 = aw[0] if const_expr(a_slot_ids is not None) else None
        bv0 = bw[0] if const_expr(b_slot_ids is not None) else None
        if const_expr(a_slot_ids is not None):
            av1 = aw[1]
        if const_expr(b_slot_ids is not None):
            bv1 = bw[1]
    elif const_expr(slot_u16):
        # one dwordx2 already covers the preamble's k = 0 and k = 1
        ap = _packs_u16(a_g2s, a_slot_ids, fx.Int32(0))
        bp = _packs_u16(b_g2s, b_slot_ids, fx.Int32(0))
        av0 = _lazy_u16(a_g2s, ap, 0) if const_expr(a_slot_ids is not None) else None
        bv0 = _lazy_u16(b_g2s, bp, 0) if const_expr(b_slot_ids is not None) else None
        if const_expr(a_slot_ids is not None):
            av1 = _lazy_u16(a_g2s, ap, 1)
        if const_expr(b_slot_ids is not None):
            bv1 = _lazy_u16(b_g2s, bp, 1)
    else:
        av0 = _voffs(a_g2s, a_slot_ids, fx.Int32(0), w0_base, w0_lim)
        bv0 = _voffs(b_g2s, b_slot_ids, fx.Int32(0), w0_base, w0_lim)
    _load_b(lds.B_lds_cur_0, 0, 0, bv0)
    _load_a(lds.A_lds_cur_0, 0, 0, av0)
    _load_b(lds.B_lds_cur_1, 1, 0, bv0)
    _load_a(lds.A_lds_cur_1, 1, 0, av0)
    if wave_m == 1:
        rocdl.s_barrier()
    wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)
    if const_expr(not slot_x4 and not slot_u16):
        av1 = _voffs(a_g2s, a_slot_ids, fx.Int32(1), w0_base, w0_lim)
        bv1 = _voffs(b_g2s, b_slot_ids, fx.Int32(1), w0_base, w0_lim)
    _load_b(lds.B_lds_next_0, 0, 1, bv1)
    _load_a(lds.A_lds_next_0, 0, 1, av1)
    _load_b(lds.B_lds_next_1, 1, 1, bv1)
    wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

    k_iters = (group_tokens + (BLOCK_K - 1)) // BLOCK_K
    n_chunks = (k_iters + (CHUNK - 1)) // CHUNK

    # nested to isolate Python-level buffer rotation from the runtime chunk loop
    def _window(chunk_idx, n_chunk, wbase=None, wlim=None):
        """Slot voffsets for n_chunk consecutive chunks, indexed by k offset.

        Batched here because the body only reads k+1 and k+2: offsets
        1..n_chunk*CHUNK+1 cover the whole span and the VMEM latency is paid once
        per window instead of once per MFMA quadrant. Both halves reuse them.
        """
        av = [None] * (n_chunk * CHUNK + 2)
        bv = [None] * (n_chunk * CHUNK + 2)
        if const_expr(slot_x4):
            # n_chunk+1 dwordx4 span k = c*CHUNK .. c*CHUNK+(n_chunk+1)*CHUNK-1,
            # which contains the window; the last one runs off the table on the
            # final chunk and the SRD clamp turns it into pool row 0, same as the
            # linear path.
            aw = [_voffs_x4(a_g2s, a_slot_ids, chunk_idx + c) for c in range_constexpr(n_chunk + 1)]
            bw = [_voffs_x4(b_g2s, b_slot_ids, chunk_idx + c) for c in range_constexpr(n_chunk + 1)]
            for ko in range_constexpr(1, n_chunk * CHUNK + 2):
                if const_expr(a_slot_ids is not None):
                    av[ko] = aw[ko // CHUNK][ko % CHUNK]
                if const_expr(b_slot_ids is not None):
                    bv[ko] = bw[ko // CHUNK][ko % CHUNK]
        elif const_expr(slot_u16):
            # same span as slot_x4, but each chunk costs CHUNK/2 dwords instead of
            # CHUNK, so a window this wide fits where the i32 one spills
            ap = [_packs_u16(a_g2s, a_slot_ids, chunk_idx + c) for c in range_constexpr(n_chunk + 1)]
            bp = [_packs_u16(b_g2s, b_slot_ids, chunk_idx + c) for c in range_constexpr(n_chunk + 1)]
            for ko in range_constexpr(1, n_chunk * CHUNK + 2):
                if const_expr(a_slot_ids is not None):
                    av[ko] = _lazy_u16(a_g2s, ap[ko // CHUNK], ko % CHUNK)
                if const_expr(b_slot_ids is not None):
                    bv[ko] = _lazy_u16(b_g2s, bp[ko // CHUNK], ko % CHUNK)
        else:
            for ko in range_constexpr(1, n_chunk * CHUNK + 2):
                av[ko] = _voffs(a_g2s, a_slot_ids, chunk_idx * CHUNK + ko, wbase, wlim)
                bv[ko] = _voffs(b_g2s, b_slot_ids, chunk_idx * CHUNK + ko, wbase, wlim)
        return av, bv

    def _steps(k_base, av, bv, off):
        """CHUNK pipelined k steps reading the window at off+j.

        Starts from the LDS buffers in their declared roles and swaps them CHUNK
        (even) times, so the rotation is parity-neutral and no state crosses calls.
        """
        a_cur0, a_cur1 = lds.A_lds_cur_0, lds.A_lds_cur_1
        a_next0, a_next1 = lds.A_lds_next_0, lds.A_lds_next_1
        b_cur0, b_cur1 = lds.B_lds_cur_0, lds.B_lds_cur_1
        b_next0, b_next1 = lds.B_lds_next_0, lds.B_lds_next_1
        for j in range_constexpr(CHUNK):
            k = k_base + j
            jw = off + j
            # 4-buffer pipelined body: interleave s2r/g2s with the 4 mfma quadrants
            b0 = b_s2r.load(b_cur0)
            a0 = a_s2r.load(a_cur0)
            _load_a(a_next1, 1, k + 1, av[jw + 1])
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c = [Vec(fx.memref_load_vec(r)) for r in acc00]
            c = mfma.call(a0, b0, c)
            for idx in range_constexpr(len(acc00)):
                fx.memref_store_vec(c[idx], acc00[idx])
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            b1 = b_s2r.load(b_cur1)
            _load_b(b_cur0, 0, k + 2, bv[jw + 2])
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c = [Vec(fx.memref_load_vec(r)) for r in acc01]
            c = mfma.call(a0, b1, c)
            for idx in range_constexpr(len(acc01)):
                fx.memref_store_vec(c[idx], acc01[idx])
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            a1 = a_s2r.load(a_cur1)
            _load_a(a_cur0, 0, k + 2, av[jw + 2])
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c = [Vec(fx.memref_load_vec(r)) for r in acc10]
            c = mfma.call(a1, b0, c)
            for idx in range_constexpr(len(acc10)):
                fx.memref_store_vec(c[idx], acc10[idx])
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            _load_b(b_cur1, 1, k + 2, bv[jw + 2])
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)
            rocdl.s_setprio(1)
            c = [Vec(fx.memref_load_vec(r)) for r in acc11]
            c = mfma.call(a1, b1, c)
            for idx in range_constexpr(len(acc11)):
                fx.memref_store_vec(c[idx], acc11[idx])
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

    def _chunk(chunk_iv):
        chunk_idx = ArithValue(chunk_iv)
        av, bv = _window(chunk_idx, 1)
        _steps(chunk_idx * CHUNK, av, bv, 0)

    def _chunk_w(chunk_iv, win_iv):
        """One chunk resolving slots from the LDS window that holds them."""
        chunk_idx = ArithValue(chunk_iv)
        wbase = ArithValue(win_iv) * fx.Int32(WIN_TOKENS)
        av, bv = _window(chunk_idx, 1, wbase, _win_limit(win_iv))
        _steps(chunk_idx * CHUNK, av, bv, 0)

    def _win_lo(w):
        return ArithValue(w) * fx.Int32(WIN_CHUNKS)

    def _win_hi(w):
        hi = (ArithValue(w) + fx.Int32(1)) * fx.Int32(WIN_CHUNKS)
        return ArithValue(arith.minsi(arith._to_raw(hi), arith._to_raw(n_chunks)), signed=True)

    def _n_win(nc):
        return (ArithValue(nc) + fx.Int32(WIN_CHUNKS - 1)) // fx.Int32(WIN_CHUNKS)

    def _chunk_n(chunk_iv):
        # One window per slot_unroll chunks. vmcnt retires in order, so a slot load
        # drains the previous chunk's outstanding pool prefetches; sharing a window
        # divides the number of those drains by slot_unroll.
        chunk_idx = ArithValue(chunk_iv)
        av, bv = _window(chunk_idx, slot_unroll)
        for c in range_constexpr(slot_unroll):
            _steps((chunk_idx + fx.Int32(c)) * CHUNK, av, bv, c * CHUNK)

    if const_expr(slot_lds):
        # Window 0 is peeled: its fill has to precede the preamble, which already
        # resolves k = 0 and k = 1. Later windows refill the same LDS array in place.
        for chunk_iv in range(_win_hi(fx.Int32(0))):
            _chunk_w(chunk_iv, fx.Int32(0))
        for win_iv in range(fx.Int32(1), _n_win(n_chunks)):
            _fill_win(win_iv)
            for chunk_iv in range(_win_lo(win_iv), _win_hi(win_iv)):
                _chunk_w(chunk_iv, win_iv)
    elif const_expr(slot_unroll > 1):
        n_grouped = (n_chunks // slot_unroll) * slot_unroll
        for chunk_iv in range(0, n_grouped, slot_unroll):
            _chunk_n(chunk_iv)
        for chunk_iv in range(n_grouped, n_chunks):
            _chunk(chunk_iv)
    else:
        for chunk_iv in range(n_chunks):
            _chunk(chunk_iv)

    c00 = [Vec(fx.memref_load_vec(reg)) for reg in acc00]
    c01 = [Vec(fx.memref_load_vec(reg)) for reg in acc01]
    c10 = [Vec(fx.memref_load_vec(reg)) for reg in acc10]
    c11 = [Vec(fx.memref_load_vec(reg)) for reg in acc11]

    # Static facts the transposed epilogue needs to pack its stores: every q_row
    # term is a multiple of 16, and the N range is tiled exactly so the per-lane
    # column mask is statically true.
    _trans_m_align = math.gcd(math.gcd(int(BLOCK_M), int(LDS_BLOCK_M)), 16)
    _trans_n_exact = int(OUT_N) % int(BLOCK_N) == 0

    def _emit_q(cfrag, q_row, q_col):
        for i in range_constexpr(NTA16):
            for j in range_constexpr(NTB16):
                blk = [cfrag[i * NTB16 + j]]
                if const_expr(trans_c):
                    store_c.store_trans16(
                        blk,
                        group_idx,
                        q_row + i * 16,
                        q_col + j * 16,
                        OUT_M,
                        OUT_N,
                        m_align=_trans_m_align,
                        n_exact=_trans_n_exact,
                    )
                else:
                    store_c.store16(blk, q_row + i * 16, q_col + j * 16)

    if const_expr(trans_c):
        local_m = block_m * BLOCK_M + wave_m * (NTA16 * 16)
        local_n = block_n * BLOCK_N + wave_n * (NTB16 * 16)
        _emit_q(c00, local_m + 0, local_n + 0)
        _emit_q(c01, local_m + 0, local_n + LDS_BLOCK_N)
        _emit_q(c10, local_m + LDS_BLOCK_M, local_n + 0)
        _emit_q(c11, local_m + LDS_BLOCK_M, local_n + LDS_BLOCK_N)
    else:
        base_row = group_idx * OUT_M + block_m * BLOCK_M + wave_m * (NTA16 * 16)
        base_col = block_n * BLOCK_N + wave_n * (NTB16 * 16)
        _emit_q(c00, base_row + 0, base_col + 0)
        _emit_q(c01, base_row + 0, base_col + LDS_BLOCK_N)
        _emit_q(c10, base_row + LDS_BLOCK_M, base_col + 0)
        _emit_q(c11, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)


@functools.lru_cache(maxsize=64)
def _compile_grouped_variable_k_bf16(
    OUT_M,
    OUT_N,
    G,
    BLOCK_M=256,
    BLOCK_N=256,
    num_xcd=8,
    waves_per_eu=2,
    agpr_alloc=0,
    out_fp16=False,
    trans_c=False,
    gather_a=False,
    gather_b=False,
    slot_x4=False,
    slot_unroll=1,
    slot_lds=False,
    slot_alu=False,
    slot_u16=False,
):
    assert OUT_M % BLOCK_M == 0, "OUT_M (unclamped store dim) must divide BLOCK_M"
    N_BLOCKS_M = OUT_M // BLOCK_M
    N_BLOCKS_N = (OUT_N + BLOCK_N - 1) // BLOCK_N
    TILES_PER_GROUP = N_BLOCKS_M * N_BLOCKS_N
    TOTAL = G * TILES_PER_GROUP
    SharedStorage = _make_slot_wgrad_shared_storage(BLOCK_M, BLOCK_N, slot_lds=slot_lds)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_grouped_variable_k(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        group_k_offsets: fx.Tensor,
        masked_k: fx.Tensor,
        A_SLOT_IDS: fx.Tensor,
        slot_len: fx.Int32,
        out_m_rt: fx.Int32,
        out_n_rt: fx.Int32,
    ):
        _ = str(fx.thread_idx.x)
        go_base = fx.Int64(_ptrtoint(_get_iter(group_k_offsets)))
        gk_base = fx.Int64(_ptrtoint(_get_iter(masked_k)))
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        pid = fx.block_idx.x

        def _do_tile(tile_idx):
            tile = xcd_remap_pid(tile_idx, TOTAL, num_xcd)
            group_idx = tile // TILES_PER_GROUP
            local_tile = tile % TILES_PER_GROUP
            if const_expr(trans_c):
                block_n = local_tile // N_BLOCKS_M
                block_m = local_tile % N_BLOCKS_M
            else:
                block_m = local_tile // N_BLOCKS_N
                block_n = local_tile % N_BLOCKS_N
            m_start = _load_i64_as_i32(go_base, group_idx)
            # bound K to valid rows; padding tail never read
            m_end = m_start + _load_i64_as_i32(gk_base, group_idx)
            gemm_bf16_variable_k_tile(
                A,
                B,
                C,
                group_idx,
                block_m,
                block_n,
                m_start,
                m_end,
                lds,
                out_m_rt,
                out_n_rt,
                G=G,
                OUT_M=OUT_M,
                OUT_N=OUT_N,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                out_fp16=out_fp16,
                trans_c=trans_c,
                a_slot_ids=A_SLOT_IDS if const_expr(gather_a) else None,
                b_slot_ids=A_SLOT_IDS if const_expr(gather_b) else None,
                slot_len=slot_len,
                slot_x4=slot_x4,
                slot_unroll=slot_unroll,
                slot_lds=slot_lds,
                slot_alu=slot_alu,
                slot_u16=slot_u16,
            )

        _do_tile(pid)

    @flyc.jit
    def launch_grouped_variable_k(
        A,
        B,
        C,
        group_k_offsets,
        masked_k,
        a_slot_ids,
        slot_len: fx.Int32,
        out_m_rt: fx.Int32,
        out_n_rt: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = fx.Int32(TOTAL)
        kernel_grouped_variable_k(
            A,
            B,
            C,
            group_k_offsets,
            masked_k,
            a_slot_ids,
            slot_len,
            out_m_rt,
            out_n_rt,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_grouped_variable_k


_COMPILED_SLOT_WGRAD_CACHE = {}


_COMPILED_DENSE_CACHE: dict = {}


def _get_compiled_dense(launch, args):
    """Compile cache keyed on shape/dtype. Not the fp8 kernel's same-named helper: that
    one races a scratch-out beta=1 build, which this launcher has no epilogue for."""
    key_parts = [id(launch)]
    for a in args:
        if isinstance(a, torch.Tensor):
            key_parts.append((tuple(a.shape), a.dtype))
        elif isinstance(a, int):
            key_parts.append(a)
        else:
            # static-memref JitArgs bake shape into the IR, so shape must be in the key
            shape = getattr(a, "shape", None)
            key_parts.append((type(a).__name__, tuple(shape) if shape is not None else None))
    key = tuple(key_parts)
    cached = _COMPILED_DENSE_CACHE.get(key)
    if cached is None:
        cached = flyc.compile(launch, *args)
        _COMPILED_DENSE_CACHE[key] = cached
    return cached


def grouped_gemm_variable_k_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    group_k_offsets: torch.Tensor,
    masked_k: torch.Tensor = None,
    out_dtype: torch.dtype = torch.bfloat16,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    num_xcd: int = 8,
    trans_c: bool = False,
    a_slot_ids: torch.Tensor = None,
    b_slot_ids: torch.Tensor = None,
    slot_x4: bool = False,
    slot_unroll: int = 1,
    slot_lds: bool = False,
    slot_alu: bool = False,
    slot_u16: bool = False,
) -> torch.Tensor:
    """Variable-K grouped wgrad: out[g]=a[g_rows].T@b[g_rows], K=[offsets[g],offsets[g]+masked_k[g]).

    ``a_slot_ids`` makes the A rows indirect: row r reads a[a_slot_ids[r]]. Used
    when the dispatch pool is deduplicated, so a[] holds unique slots while the
    K axis still walks logical route rows."""
    assert a.dim() == 2 and b.dim() == 2
    assert a_slot_ids is None or b_slot_ids is None, "only one operand may be gathered"
    assert a_slot_ids is not None or b_slot_ids is not None or a.shape[0] == b.shape[0]
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    OUT_M = a.shape[1]
    OUT_N = b.shape[1]
    G = group_k_offsets.numel() - 1
    out_fp16 = out_dtype == torch.float16
    out_shape = (G, OUT_N, OUT_M) if trans_c else (G, OUT_M, OUT_N)
    out = torch.empty(out_shape, device=a.device, dtype=out_dtype)
    # index tables loaded as i64 in-kernel
    offsets_i64 = group_k_offsets if group_k_offsets.dtype == torch.int64 else group_k_offsets.to(torch.int64)
    # per-expert valid K length; default = padded span
    if masked_k is None:
        masked_k_i64 = (offsets_i64[1:] - offsets_i64[:-1]).contiguous()
    else:
        assert masked_k.numel() == G, f"masked_k len {masked_k.numel()} != G {G}"
        masked_k_i64 = (masked_k if masked_k.dtype == torch.int64 else masked_k.to(torch.int64)).contiguous()
    launch = _compile_grouped_variable_k_bf16(
        OUT_M,
        OUT_N,
        G,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_xcd=num_xcd,
        out_fp16=out_fp16,
        trans_c=trans_c,
        gather_a=a_slot_ids is not None,
        gather_b=b_slot_ids is not None,
        slot_x4=slot_x4,
        slot_unroll=slot_unroll,
        slot_lds=slot_lds,
        slot_alu=slot_alu,
        slot_u16=slot_u16,
    )
    # static memref: create_buffer_resource needs a real memref, not a raw ptr arg
    slots = a_slot_ids if a_slot_ids is not None else b_slot_ids
    slot_src = slots.contiguous() if slots is not None else masked_k_i64.view(torch.int32)
    slot_arg = flyc.from_torch_tensor(slot_src)
    args = (
        _ptr_only_view(a),
        _ptr_only_view(b),
        flyc.from_torch_tensor(out),
        offsets_i64,
        masked_k_i64,
        slot_arg,
        slot_src.numel(),
        OUT_M,
        OUT_N,
        torch.cuda.current_stream(),
    )
    key = (
        OUT_M,
        OUT_N,
        G,
        BLOCK_M,
        BLOCK_N,
        out_fp16,
        trans_c,
        a_slot_ids is not None,
        b_slot_ids is not None,
        slot_x4,
        slot_unroll,
        slot_lds,
        slot_alu,
        slot_u16,
    )
    compiled = _COMPILED_SLOT_WGRAD_CACHE.get(key)
    if compiled is None:
        compiled = flyc.compile(launch, *args)
        _COMPILED_SLOT_WGRAD_CACHE[key] = compiled
    compiled(*args)
    return out
