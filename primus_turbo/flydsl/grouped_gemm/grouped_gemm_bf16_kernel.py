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
"""

import functools

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
    Mfma16x16x32,
    S2RLoaderTr16x32Bf16Wide,
    StoreCBf16,
    _i64,
    compute_global_swizzle_nn_bf16_wide,
    emit_for,
    emit_if_then,
    expert_group_tile_decode,
    group_m_tile_decode,
    make_bf16_buffer_tensor_rebased,
    make_bf16_fp16_tile_tensor,
    make_value_attrs,
    wait_barrier,
    wave_lane_with_rank,
    wave_rank_desc_stable,
    xcd_band_remap_pid,
)


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
    mfma = Mfma16x16x32(NTA16, NTB16)
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
        if wave_m == 1:
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
    out_fp16=False,
    trans_c=False,
    # One padded chunk splits the tr16 reader's four lane groups off a single bank half: 128 mod 256.
    lds_chunk_stride=1152,
    group_m=1,
    xcd_band=24,
):
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
        pid = fx.block_idx.x
        # Dispatch order ranked in the prologue by a lane-resident descending rank, not a host argsort.
        lane = fx.Int32(fx.thread_idx.x) % fx.Int32(64)
        in_g = lane < fx.Int32(G)
        k_lane = _load_i32(gk_base, arith.select(in_g, lane, fx.Int32(0)) * fx.Int32(2))
        order_rank = wave_rank_desc_stable(arith.select(in_g, k_lane, fx.Int32(-1)), lane, G)

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
            out_fp16=out_fp16,
            trans_c=trans_c,
            lds_chunk_stride=lds_chunk_stride,
            mask_m=MASK_M,
        )

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
        grid_x = fx.Int32(TOTAL)
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
    trans_c: bool = False,
) -> torch.Tensor:
    """Variable-K grouped wgrad: out[g]=a[g_rows].T@b[g_rows], K=[offsets[g],offsets[g]+masked_k[g])."""
    assert a.dim() == 2 and b.dim() == 2 and a.shape[0] == b.shape[0]
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
    key = (OUT_M, OUT_N, G, BLOCK_M, BLOCK_N, num_xcd, group_m, xcd_band, out_fp16, trans_c)
    compiled = _COMPILED_GROUPED_GEMM_CACHE.get(key)
    if compiled is None:
        launch = _compile_grouped_bf16_wgrad(
            OUT_M,
            OUT_N,
            G,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_xcd=num_xcd,
            out_fp16=out_fp16,
            trans_c=trans_c,
            group_m=group_m,
            xcd_band=xcd_band,
        )
        compiled = flyc.compile(launch, *args)
        _COMPILED_GROUPED_GEMM_CACHE[key] = compiled
    compiled(*args)
    return out


_COMPILED_GROUPED_NT_CACHE = {}


def _compile_grouped_bf16_nt(
    TOTAL_M,
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
    out_fp16=False,
):
    # A tile never straddles two experts, so its group follows from its row block and the grid is static.
    assert TOTAL_M % BLOCK_M == 0, "TOTAL_M must be a multiple of BLOCK_M (padded token runs)"
    N_BLOCKS_M = TOTAL_M // BLOCK_M
    N_BLOCKS_N = (N + BLOCK_N - 1) // BLOCK_N
    TOTAL_TILES = N_BLOCKS_M * N_BLOCKS_N
    B_GRP = N * K  # elements of one expert's weight slab
    assert GROUP_M >= 1 and GROUP_M & (GROUP_M - 1) == 0, "GROUP_M must be a power of two"
    assert G < 64, f"expert_group_tile_decode holds the G+1 offsets in one wave (got G={G})"
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_grouped_nt(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        group_offs: fx.Tensor,
        c_n: fx.Int32,
    ):
        _ = str(fx.thread_idx.x)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        tile = xcd_band_remap_pid(fx.block_idx.x, TOTAL_TILES, num_xcd, xcd_band)
        # Restarting the walk per expert keeps each super-tile on one weight slab and yields the owner.
        block_m, block_n, g_idx = expert_group_tile_decode(
            group_offs, fx.thread_idx.x % fx.Int32(64), tile, G, N_BLOCKS_N, BLOCK_M, GROUP_M
        )
        m_row = block_m * BLOCK_M

        a_base = fx.Int64(_ptrtoint(_get_iter(A)))
        b_base = fx.Int64(_ptrtoint(_get_iter(B)))
        c_base = fx.Int64(_ptrtoint(_get_iter(C)))
        a_tile = make_bf16_fp16_tile_tensor(a_base, _i64(m_row) * fx.Int64(K * 2), BLOCK_M * K)
        b_tile = make_bf16_fp16_tile_tensor(b_base, _i64(g_idx) * fx.Int64(B_GRP * 2), B_GRP)
        c_tile = make_bf16_fp16_tile_tensor(c_base, _i64(m_row) * fx.Int64(2) * _i64(c_n), BLOCK_M * N)

        gemm_bf16_nt_tile(
            a_tile,
            b_tile,
            c_tile,
            fx.Int32(BLOCK_M),  # the run is block-aligned, so the tile owns a full BLOCK_M
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
            out_fp16=out_fp16,
            nt_vmcnt=nt_vmcnt,
            pair_n=N % 2 == 0 and not out_fp16,
            n_tail=N % BLOCK_N,
        )

    @flyc.jit
    def launch_grouped_nt(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        group_offs: fx.Tensor,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_grouped_nt(
            A,
            B,
            C,
            group_offs,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(TOTAL_TILES, 1, 1), block=(512, 1, 1), stream=stream)

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
) -> torch.Tensor:
    """Grouped NT forward: out[rows] = a[rows] @ b[g]^T for the expert g owning each row run."""
    assert a.dim() == 2 and b.dim() == 3 and a.dtype == b.dtype == torch.bfloat16
    TOTAL_M, K = a.shape
    G, N, Kb = b.shape
    assert Kb == K, f"b K={Kb} != a K={K}"
    out = torch.empty(TOTAL_M, N, device=a.device, dtype=out_dtype)
    offs = group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)
    # The tile resolves its row block and expert itself; a host-built table is one more launch.
    args = (
        _ptr_only_view(a),
        flyc.from_torch_tensor(b.reshape(-1)),
        flyc.from_torch_tensor(out),
        offs,
        N,
        torch.cuda.current_stream(),
    )
    key = (TOTAL_M, N, K, G, BLOCK_M, BLOCK_N, GROUP_M, num_xcd, xcd_band, out_dtype)
    compiled = _COMPILED_GROUPED_NT_CACHE.get(key)
    if compiled is None:
        launch = _compile_grouped_bf16_nt(
            TOTAL_M,
            N,
            K,
            G,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            GROUP_M=GROUP_M,
            num_xcd=num_xcd,
            xcd_band=xcd_band,
            out_fp16=out_dtype == torch.float16,
        )
        compiled = flyc.compile(launch, *args)
        _COMPILED_GROUPED_NT_CACHE[key] = compiled
    compiled(*args)
    return out


_COMPILED_GROUPED_NN_CACHE = {}


def _compile_grouped_bf16_nn(
    TOTAL_M,
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
    out_fp16=False,
):
    assert TOTAL_M % BLOCK_M == 0, "TOTAL_M must be a multiple of BLOCK_M (padded token runs)"
    N_BLOCKS_M = TOTAL_M // BLOCK_M
    N_BLOCKS_N = (N + BLOCK_N - 1) // BLOCK_N
    TOTAL_TILES = N_BLOCKS_M * N_BLOCKS_N
    B_GRP = N * K  # elements of one expert's weight slab
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_grouped_nn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        block_expert: fx.Tensor,
        c_n: fx.Int32,
    ):
        _ = str(fx.thread_idx.x)
        be_base = fx.Int64(_ptrtoint(_get_iter(block_expert)))
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        tile = xcd_band_remap_pid(fx.block_idx.x, TOTAL_TILES, num_xcd, xcd_band)
        block_m, block_n = group_m_tile_decode(tile, N_BLOCKS_M, N_BLOCKS_N, GROUP_M)
        m_row = block_m * BLOCK_M

        # Owning expert precomputed once per call: a per-workgroup rescan of G costs more than the K loop.
        g_idx = _load_i32(be_base, block_m)

        a_base = fx.Int64(_ptrtoint(_get_iter(A)))
        b_base = fx.Int64(_ptrtoint(_get_iter(B)))
        c_base = fx.Int64(_ptrtoint(_get_iter(C)))
        a_tile = make_bf16_fp16_tile_tensor(a_base, _i64(m_row) * fx.Int64(K * 2), BLOCK_M * K)
        b_tile = make_bf16_fp16_tile_tensor(b_base, _i64(g_idx) * fx.Int64(B_GRP * 2), B_GRP)
        c_tile = make_bf16_fp16_tile_tensor(c_base, _i64(m_row) * fx.Int64(2) * _i64(c_n), BLOCK_M * N)

        gemm_bf16_nn_tile(
            a_tile,
            b_tile,
            c_tile,
            fx.Int32(BLOCK_M),  # the run is block-aligned, so the tile owns a full BLOCK_M
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
            out_fp16=out_fp16,
            nt_vmcnt=nt_vmcnt,
            n_tail=N % BLOCK_N,
        )

    @flyc.jit
    def launch_grouped_nn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        block_expert: fx.Tensor,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_grouped_nn(
            A,
            B,
            C,
            block_expert,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(TOTAL_TILES, 1, 1), block=(512, 1, 1), stream=stream)

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
) -> torch.Tensor:
    """Grouped NN: out[rows] = a[rows] @ b[g] for the expert g owning each row run."""
    assert a.dim() == 2 and b.dim() == 3 and a.dtype == b.dtype == torch.bfloat16
    TOTAL_M, K = a.shape
    G, Kb, N = b.shape
    assert Kb == K, f"b K={Kb} != a K={K}"
    # A super-tile trades A-slab against B-slab reuse and the balance moves with the weight slab.
    if GROUP_M == 0:
        GROUP_M = 8 if K * N * a.element_size() > 24 << 20 else 4
    out = torch.empty(TOTAL_M, N, device=a.device, dtype=out_dtype)
    offs = group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)
    block_expert = torch.searchsorted(
        offs[1:], _row_starts(TOTAL_M, BLOCK_M, a.device), right=True, out_int32=True
    )
    args = (
        _ptr_only_view(a),
        flyc.from_torch_tensor(b.reshape(-1)),
        flyc.from_torch_tensor(out),
        block_expert,
        N,
        torch.cuda.current_stream(),
    )
    key = (TOTAL_M, N, K, G, BLOCK_M, BLOCK_N, GROUP_M, num_xcd, xcd_band, out_dtype)
    compiled = _COMPILED_GROUPED_NN_CACHE.get(key)
    if compiled is None:
        launch = _compile_grouped_bf16_nn(
            TOTAL_M,
            N,
            K,
            G,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            GROUP_M=GROUP_M,
            num_xcd=num_xcd,
            xcd_band=xcd_band,
            out_fp16=out_dtype == torch.float16,
        )
        compiled = flyc.compile(launch, *args)
        _COMPILED_GROUPED_NN_CACHE[key] = compiled
    compiled(*args)
    return out
