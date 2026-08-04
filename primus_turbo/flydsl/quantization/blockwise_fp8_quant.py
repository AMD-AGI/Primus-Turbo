###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL); see LICENSE-APACHE for the Apache-2.0 terms.
#
# See LICENSE for license information.
###############################################################################

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import buffer_ops as bo
from flydsl.expr import math as fm
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

_compiled_cache = {}


def _run_cached(key, build, args):
    entry = _compiled_cache.get(key)
    if entry is None:
        raw = build()
        entry = raw, flyc.compile(raw, *args)
        _compiled_cache[key] = entry
    raw, compiled = entry
    if torch.cuda.is_current_stream_capturing():
        raw(*args)
    else:
        compiled(*args)


def compile_blockwise_fp8_weight_quant(M: int, N: int):
    """Compile a BF16 -> E4M3 2D-block quantizer for static ``[M,N]`` input."""

    TILE = 128
    THREADS = 1024
    ELEMS_PER_THREAD = TILE * TILE // THREADS
    VEC = 8
    WAVES = THREADS // 64
    M_BLOCKS = (M + TILE - 1) // TILE
    N_BLOCKS = (N + TILE - 1) // TILE
    FP8_MAX = 448.0
    assert ELEMS_PER_THREAD == 16
    assert N % VEC == 0

    @fx.struct
    class SharedStorage:
        tile: fx.Array[fx.BFloat16, TILE * TILE, 16]
        wave_max: fx.Array[fx.Float32, WAVES, 16]
        quant_scale: fx.Array[fx.Float32, 1, 4]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def quant_kernel(x: fx.Tensor, q: fx.Tensor, scale: fx.Tensor):
        I32 = fx.Int32
        F32 = fx.Float32
        tid = fx.thread_idx.x
        lane = tid % I32(64)
        wave = tid // I32(64)
        block = fx.block_idx.x
        block_m = block // I32(N_BLOCKS)
        block_n = block - block_m * I32(N_BLOCKS)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        tile = lds.tile
        wave_max = lds.wave_max.view(fx.make_layout(WAVES, 1))
        quant_scale = lds.quant_scale.view(fx.make_layout(1, 1))

        x_rsrc = bo.create_buffer_resource(x, max_size=False, num_records_bytes=M * N * 2)
        q_rsrc = bo.create_buffer_resource(q, max_size=False, num_records_bytes=M * N)
        scale_rsrc = bo.create_buffer_resource(
            scale,
            max_size=False,
            num_records_bytes=M_BLOCKS * N_BLOCKS * 4,
        )

        thread_max = F32(0.0)
        for part in range_constexpr(ELEMS_PER_THREAD // VEC):
            local_linear = tid * I32(ELEMS_PER_THREAD) + I32(part * VEC)
            local_row = local_linear // I32(TILE)
            local_col = local_linear - local_row * I32(TILE)
            global_row = block_m * I32(TILE) + local_row
            global_col = block_n * I32(TILE) + local_col
            valid = (global_row < I32(M)) & ((global_col + I32(VEC)) <= I32(N))
            values = Vec(
                bo.buffer_load(
                    x_rsrc,
                    global_row * I32(N) + global_col,
                    vec_width=VEC,
                    dtype=fx.BFloat16.ir_type,
                    mask=valid,
                )
            )
            tile_ptr = fx.add_offset(tile.ptr, fx.make_int_tuple(local_linear))
            fx.make_view(tile_ptr, fx.make_layout(VEC, 1)).store(values)
            values_f32 = values.to(F32)
            local_max = fm.absf(values_f32).reduce("max")
            thread_max = thread_max.maximumf(valid.select(local_max, F32(0.0)))

        for offset in (32, 16, 8, 4, 2, 1):
            thread_max = thread_max.maximumf(thread_max.shuffle_xor(offset, 64))
        if lane == I32(0):
            fx.memref_store(thread_max, wave_max, wave)
        _llvm.inline_asm(
            res=None,
            operands_=[],
            asm_string="s_waitcnt vmcnt(0) lgkmcnt(0)",
            constraints="",
            has_side_effects=True,
        )
        rocdl.s_barrier()

        if wave == I32(0):
            in_range = lane < I32(WAVES)
            lane_safe = in_range.select(lane, I32(0))
            block_max = fx.memref_load(wave_max, lane_safe)
            block_max = in_range.select(block_max, F32(0.0))
            for offset in (32, 16, 8, 4, 2, 1):
                block_max = block_max.maximumf(block_max.shuffle_xor(offset, 64))
            if lane == I32(0):
                amax = block_max.maximumf(F32(1e-4))
                qscale = F32(FP8_MAX) / amax
                fx.memref_store(qscale, quant_scale, 0)
                bo.buffer_store(
                    F32(1.0) / qscale,
                    scale_rsrc,
                    block_m * I32(N_BLOCKS) + block_n,
                )
        _llvm.inline_asm(
            res=None,
            operands_=[],
            asm_string="s_waitcnt lgkmcnt(0)",
            constraints="",
            has_side_effects=True,
        )
        rocdl.s_barrier()
        qscale = fx.memref_load(quant_scale, 0)

        for part in range_constexpr(ELEMS_PER_THREAD // VEC):
            local_linear = tid * I32(ELEMS_PER_THREAD) + I32(part * VEC)
            local_row = local_linear // I32(TILE)
            local_col = local_linear - local_row * I32(TILE)
            global_row = block_m * I32(TILE) + local_row
            global_col = block_n * I32(TILE) + local_col
            valid = (global_row < I32(M)) & ((global_col + I32(VEC)) <= I32(N))
            tile_ptr = fx.add_offset(tile.ptr, fx.make_int_tuple(local_linear))
            values = Vec(fx.make_view(tile_ptr, fx.make_layout(VEC, 1)).load()).to(F32)
            values = values * qscale
            packed_words = []
            for word_index in range_constexpr(VEC // 4):
                base = word_index * 4
                packed = I32(
                    rocdl.cvt_pk_fp8_f32(
                        I32.ir_type,
                        fm.clampf(values[base], F32(-FP8_MAX), F32(FP8_MAX)),
                        fm.clampf(values[base + 1], F32(-FP8_MAX), F32(FP8_MAX)),
                        I32(0),
                        0,
                    )
                )
                packed = I32(
                    rocdl.cvt_pk_fp8_f32(
                        I32.ir_type,
                        fm.clampf(values[base + 2], F32(-FP8_MAX), F32(FP8_MAX)),
                        fm.clampf(values[base + 3], F32(-FP8_MAX), F32(FP8_MAX)),
                        packed,
                        1,
                    )
                )
                packed_words.append(packed)
            bo.buffer_store(
                Vec.from_elements(packed_words, fx.Int32).ir_value(),
                q_rsrc,
                global_row * I32(N) + global_col,
                mask=valid,
                offset_is_bytes=True,
            )

    @flyc.jit
    def launch(x: fx.Tensor, q: fx.Tensor, scale: fx.Tensor, stream: fx.Stream):
        quant_kernel(x, q, scale).launch(
            grid=(M_BLOCKS * N_BLOCKS, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    return launch


def compile_blockwise_fp8_dual_quant(M: int, N: int, *, row_scale_transposed: bool = True):
    """Compile aligned BF16 row+column blockwise E4M3 quantization."""

    TILE = 128
    THREADS = 1024
    VEC = 8
    ELEMS_PER_THREAD = TILE * TILE // THREADS
    M_BLOCKS = M // TILE
    N_BLOCKS = N // TILE
    FP8_MAX = 448.0
    assert M % TILE == 0 and N % TILE == 0
    assert ELEMS_PER_THREAD == 16

    @fx.struct
    class SharedStorage:
        tile: fx.Array[fx.BFloat16, TILE * TILE, 16]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def quant_kernel(
        x: fx.Tensor,
        q_row: fx.Tensor,
        scale_row: fx.Tensor,
        q_col: fx.Tensor,
        scale_col: fx.Tensor,
    ):
        I32 = fx.Int32
        F32 = fx.Float32
        tid = fx.thread_idx.x
        block = fx.block_idx.x
        block_m = block // I32(N_BLOCKS)
        block_n = block - block_m * I32(N_BLOCKS)
        group = tid // I32(8)
        lane = tid - group * I32(8)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        tile = lds.tile

        x_rsrc = bo.create_buffer_resource(x, max_size=False, num_records_bytes=M * N * 2)
        q_row_rsrc = bo.create_buffer_resource(q_row, max_size=False, num_records_bytes=M * N)
        q_col_rsrc = bo.create_buffer_resource(q_col, max_size=False, num_records_bytes=M * N)
        scale_row_rsrc = bo.create_buffer_resource(
            scale_row,
            max_size=False,
            num_records_bytes=M * N_BLOCKS * 4,
        )
        scale_col_rsrc = bo.create_buffer_resource(
            scale_col,
            max_size=False,
            num_records_bytes=M_BLOCKS * N * 4,
        )

        for part in range_constexpr(ELEMS_PER_THREAD // VEC):
            local_linear = tid * I32(ELEMS_PER_THREAD) + I32(part * VEC)
            local_row = local_linear // I32(TILE)
            local_col = local_linear - local_row * I32(TILE)
            global_row = block_m * I32(TILE) + local_row
            global_col = block_n * I32(TILE) + local_col
            values = Vec(
                bo.buffer_load(
                    x_rsrc,
                    global_row * I32(N) + global_col,
                    vec_width=VEC,
                    dtype=fx.BFloat16.ir_type,
                )
            )
            tile_ptr = fx.add_offset(tile.ptr, fx.make_int_tuple(local_linear))
            fx.make_view(tile_ptr, fx.make_layout(VEC, 1)).store(values)
        _llvm.inline_asm(
            res=None,
            operands_=[],
            asm_string="s_waitcnt vmcnt(0) lgkmcnt(0)",
            constraints="",
            has_side_effects=True,
        )
        rocdl.s_barrier()

        def reduce_8(value):
            result = value
            for offset in (4, 2, 1):
                result = result.maximumf(result.shuffle_xor(offset, 8))
            return result

        def pack_and_store(values, rsrc, byte_offset):
            for word_index in range_constexpr(4):
                base = word_index * 4
                packed = I32(
                    rocdl.cvt_pk_fp8_f32(
                        I32.ir_type,
                        fm.clampf(values[base], F32(-FP8_MAX), F32(FP8_MAX)),
                        fm.clampf(values[base + 1], F32(-FP8_MAX), F32(FP8_MAX)),
                        I32(0),
                        0,
                    )
                )
                packed = I32(
                    rocdl.cvt_pk_fp8_f32(
                        I32.ir_type,
                        fm.clampf(values[base + 2], F32(-FP8_MAX), F32(FP8_MAX)),
                        fm.clampf(values[base + 3], F32(-FP8_MAX), F32(FP8_MAX)),
                        packed,
                        1,
                    )
                )
                bo.buffer_store(
                    packed,
                    rsrc,
                    byte_offset + I32(base),
                    offset_is_bytes=True,
                )

        local_row = group
        local_col = lane * I32(ELEMS_PER_THREAD)
        row_ptr = fx.add_offset(tile.ptr, fx.make_int_tuple(local_row * I32(TILE) + local_col))
        row_values = Vec(fx.make_view(row_ptr, fx.make_layout(ELEMS_PER_THREAD, 1)).load()).to(F32)
        row_max = reduce_8(fm.absf(row_values).reduce("max")).maximumf(F32(1e-4))
        row_qscale = F32(FP8_MAX) / row_max
        global_row = block_m * I32(TILE) + local_row
        global_col = block_n * I32(TILE) + local_col
        pack_and_store(
            row_values * row_qscale,
            q_row_rsrc,
            global_row * I32(N) + global_col,
        )
        row_scale_idx = (
            block_n * I32(M) + global_row if row_scale_transposed else global_row * I32(N_BLOCKS) + block_n
        )
        if lane == I32(0):
            bo.buffer_store(F32(1.0) / row_qscale, scale_row_rsrc, row_scale_idx)

        local_col = group
        local_row = lane * I32(ELEMS_PER_THREAD)
        col_ptr = fx.add_offset(tile.ptr, fx.make_int_tuple(local_row * I32(TILE) + local_col))
        col_values = Vec(fx.make_view(col_ptr, fx.make_layout(ELEMS_PER_THREAD, TILE)).load()).to(F32)
        col_max = reduce_8(fm.absf(col_values).reduce("max")).maximumf(F32(1e-4))
        col_qscale = F32(FP8_MAX) / col_max
        global_col = block_n * I32(TILE) + local_col
        global_row = block_m * I32(TILE) + local_row
        pack_and_store(
            col_values * col_qscale,
            q_col_rsrc,
            global_col * I32(M) + global_row,
        )
        if lane == I32(0):
            bo.buffer_store(
                F32(1.0) / col_qscale,
                scale_col_rsrc,
                block_m * I32(N) + global_col,
            )

    @flyc.jit
    def launch(
        x: fx.Tensor,
        q_row: fx.Tensor,
        scale_row: fx.Tensor,
        q_col: fx.Tensor,
        scale_col: fx.Tensor,
        stream: fx.Stream,
    ):
        quant_kernel(x, q_row, scale_row, q_col, scale_col).launch(
            grid=(M_BLOCKS * N_BLOCKS, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    return launch


def quantize_blockwise_fp8_weight(x: torch.Tensor):
    if not str(get_rocm_arch()).startswith("gfx95"):
        raise ValueError("FlyDSL blockwise FP8 quantization is gfx950-only")
    if x.dtype != torch.bfloat16 or x.dim() != 2 or not x.is_contiguous():
        raise ValueError("FlyDSL weight quant requires contiguous 2D BF16 input")
    M, N = x.shape
    if N % 16 or M * N * x.element_size() > 0xFFFFFFFF:
        raise ValueError("FlyDSL weight quant requires N%16==0 and input no larger than 4 GiB")
    q = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty(((M + 127) // 128, (N + 127) // 128), dtype=torch.float32, device=x.device)
    stream = torch.cuda.current_stream(x.device)
    args = (x, q, scale, stream)
    key = ("weight", str(get_rocm_arch()), M, N)
    _run_cached(key, lambda: compile_blockwise_fp8_weight_quant(M, N), args)
    return q, scale


def quantize_blockwise_fp8_dual(x: torch.Tensor, *, row_scale_transposed: bool):
    if not str(get_rocm_arch()).startswith("gfx95"):
        raise ValueError("FlyDSL blockwise FP8 quantization is gfx950-only")
    if x.dtype != torch.bfloat16 or x.dim() != 2 or not x.is_contiguous():
        raise ValueError("FlyDSL dual quant requires contiguous 2D BF16 input")
    M, N = x.shape
    if M % 128 or N % 128 or M * N * x.element_size() > 0xFFFFFFFF:
        raise ValueError("FlyDSL dual quant requires 128-aligned input no larger than 4 GiB")
    q_row = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale_row_shape = (N // 128, M) if row_scale_transposed else (M, N // 128)
    scale_row = torch.empty(scale_row_shape, dtype=torch.float32, device=x.device)
    q_col = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=x.device)
    scale_col = torch.empty((M // 128, N), dtype=torch.float32, device=x.device)
    stream = torch.cuda.current_stream(x.device)
    args = (x, q_row, scale_row, q_col, scale_col, stream)
    key = ("dual", str(get_rocm_arch()), M, N, row_scale_transposed)
    _run_cached(
        key,
        lambda: compile_blockwise_fp8_dual_quant(M, N, row_scale_transposed=row_scale_transposed),
        args,
    )
    return q_row, scale_row, q_col, scale_col


__all__ = [
    "compile_blockwise_fp8_dual_quant",
    "compile_blockwise_fp8_weight_quant",
    "quantize_blockwise_fp8_dual",
    "quantize_blockwise_fp8_weight",
]
