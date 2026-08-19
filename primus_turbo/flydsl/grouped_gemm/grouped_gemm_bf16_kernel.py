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
from flydsl.expr import const_expr, range_constexpr, rocdl
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

from primus_turbo.flydsl.gemm.gemm_bf16_kernel import _make_shared_storage
from primus_turbo.flydsl.utils.gemm_helper import (
    BLOCK_K,
    G2SLoader,
    Mfma16x16x32,
    S2RLoaderTr16x32Bf16,
    StoreCBf16,
    _i64,
    compute_global_swizzle_nn_bf16,
    make_bf16_buffer_tensor_rebased,
    make_value_attrs,
    wait_barrier,
    xcd_remap_pid,
)


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
):
    CHUNK = 4
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
    gA = make_bf16_buffer_tensor_rebased(A, bf16_ir, a_base_off, a_span)
    gB = make_bf16_buffer_tensor_rebased(B, bf16_ir, b_base_off, b_span)
    a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
    b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

    gl_off_a = compute_global_swizzle_nn_bf16(lane_id, wave_id, OUT_M, N_LDS_STEPS_A)
    gl_off_b = compute_global_swizzle_nn_bf16(lane_id, wave_id, OUT_N, N_LDS_STEPS_B)

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
    a_s2r = S2RLoaderTr16x32Bf16(wave_m, NTA16)
    b_s2r = S2RLoaderTr16x32Bf16(wave_n, NTB16)
    ACC_VEC_N = 4
    N_ACCUMS_EFF = N_ACCUMS16
    a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, bf16_ir, wave_id)
    b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, bf16_ir, wave_id)
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

    k_iters = (group_tokens + (BLOCK_K - 1)) // BLOCK_K
    n_chunks = (k_iters + (CHUNK - 1)) // CHUNK

    # nested to isolate Python-level buffer rotation from the runtime chunk loop
    def _chunk(chunk_iv):
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
            rocdl.s_setprio(1)
            c = [Vec(fx.memref_load_vec(r)) for r in acc00]
            c = mfma.call(a0, b0, c)
            for idx in range_constexpr(len(acc00)):
                fx.memref_store_vec(c[idx], acc00[idx])
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            b1 = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, b0_off + (k + 2) * b_k_step)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c = [Vec(fx.memref_load_vec(r)) for r in acc01]
            c = mfma.call(a0, b1, c)
            for idx in range_constexpr(len(acc01)):
                fx.memref_store_vec(c[idx], acc01[idx])
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            a1 = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, a0_off + (k + 2) * a_k_step)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c = [Vec(fx.memref_load_vec(r)) for r in acc10]
            c = mfma.call(a1, b0, c)
            for idx in range_constexpr(len(acc10)):
                fx.memref_store_vec(c[idx], acc10[idx])
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            b_g2s.load(b_cur1, b1_off + (k + 2) * b_k_step)
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

    for chunk_iv in range(n_chunks):
        _chunk(chunk_iv)

    c00 = [Vec(fx.memref_load_vec(reg)) for reg in acc00]
    c01 = [Vec(fx.memref_load_vec(reg)) for reg in acc01]
    c10 = [Vec(fx.memref_load_vec(reg)) for reg in acc10]
    c11 = [Vec(fx.memref_load_vec(reg)) for reg in acc11]

    def _emit_q(cfrag, q_row, q_col):
        for i in range_constexpr(NTA16):
            for j in range_constexpr(NTB16):
                blk = [cfrag[i * NTB16 + j]]
                if const_expr(trans_c):
                    store_c.store_trans16(blk, group_idx, q_row + i * 16, q_col + j * 16, OUT_M, OUT_N)
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
):
    assert OUT_M % BLOCK_M == 0, "OUT_M (unclamped store dim) must be divisible by BLOCK_M"
    N_BLOCKS_M = OUT_M // BLOCK_M
    N_BLOCKS_N = (OUT_N + BLOCK_N - 1) // BLOCK_N
    TILES_PER_GROUP = N_BLOCKS_M * N_BLOCKS_N
    TOTAL = G * TILES_PER_GROUP
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)

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
            )

        _do_tile(pid)

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
    num_xcd: int = 8,
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
    launch = _compile_grouped_bf16_wgrad(
        OUT_M,
        OUT_N,
        G,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_xcd=num_xcd,
        out_fp16=out_fp16,
        trans_c=trans_c,
    )
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
    key = (OUT_M, OUT_N, G, BLOCK_M, BLOCK_N, num_xcd, out_fp16, trans_c)
    compiled = _COMPILED_GROUPED_GEMM_CACHE.get(key)
    if compiled is None:
        compiled = flyc.compile(launch, *args)
        _COMPILED_GROUPED_GEMM_CACHE[key] = compiled
    compiled(*args)
    return out
