###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL); see LICENSE-APACHE for the Apache-2.0 terms.
#
# See LICENSE for license information.
###############################################################################

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr import buffer_ops as _buffer_ops
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.utils.gemm_helper import (
    swizzle_128,
)


def divmod(a, b):
    """Integer divmod that works on DSL values (e.g. ``Int32``).

    The builtin ``divmod`` rejects DSL scalar types, so this uses the overloaded
    ``//`` / ``%`` operators to emit the corresponding ops.
    """
    return (a // b, a % b)


def make_fp8_buffer_tensor(
    arg_i8,
    fp8_ir_t,
    *,
    num_records_bytes=None,
    base_byte_offset=None,
):
    # max_size=False with no num_records_bytes: cosize(layout) becomes a
    # runtime expression because TensorAdaptor defaults to layout-dynamic
    # memref (post #554), so the descriptor adapts to the actual tensor
    # extent and no longer bakes the first-call's shape into IR.
    if base_byte_offset is not None:
        original_iter = fx.get_iter(arg_i8)
        shifted_iter = fx.inttoptr(
            original_iter.type,
            fx.Int64(fx.ptrtoint(original_iter)) + fx.Int64(base_byte_offset),
        )
        arg_i8 = fx.Tensor(fx.make_view(shifted_iter, fx.make_layout(1, 1)))
    t_i8 = fx.rocdl.make_buffer_tensor(
        arg_i8,
        max_size=False,
        num_records_bytes=num_records_bytes,
    )
    iter_i8 = fx.get_iter(t_i8)
    f8_buf_ptr_ty = fx.PointerType.get(
        elem_ty=fp8_ir_t,
        address_space=TargetAddressSpace.BufferDesc,
        alignment=fx.PointerType(iter_i8.type).alignment,
    )
    iter_f8 = fx.recast_iter(f8_buf_ptr_ty, iter_i8)
    return fx.Tensor(fx.make_view(iter_f8, fx.get_layout(t_i8)))


class Mfma16x16x128:
    def __init__(self, n_tiles_a, n_tiles_b):
        self.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _make_operand_frag(self, value):
        frag = fx.make_rmem_tensor(8, fx.Int32)
        frag.store(Vec(value))
        return frag

    def _make_accum_frag(self, value):
        frag = fx.make_rmem_tensor(4, fx.Float32)
        frag.store(Vec(value))
        return frag

    def _do_mma(self, a, b, c):
        a_frag = self._make_operand_frag(a)
        b_frag = self._make_operand_frag(b)
        c_frag = self._make_accum_frag(c)
        fx.gemm(self.atom, c_frag, a_frag, b_frag, c_frag)
        return c_frag.load().ir_value()


def _min(a, b):
    return arith.select(a < b, a, b)


def _xcd_swizzle(num_pid_m, num_pid_n, workgroups_per_group_m):
    num_xcds = 8
    num_cus = 32 * num_xcds
    swizzle_threshold = 4 * num_cus

    wgid = fx.block_idx.x
    num_wg = num_pid_m * num_pid_n

    simple_m, simple_n = divmod(wgid, num_pid_n)

    intra_xcd, xcd = divmod(wgid, num_xcds)
    remapped_wgid = xcd * (num_wg // num_xcds) + intra_xcd
    group_size = workgroups_per_group_m * num_pid_n
    group_id, intra_group = divmod(remapped_wgid, group_size)
    first_pid_m = group_id * workgroups_per_group_m
    group_m = _min(num_pid_m - first_pid_m, workgroups_per_group_m)
    pid_n, intra_group_m = divmod(intra_group, group_m)
    pid_m = first_pid_m + intra_group_m

    use_simple = (num_wg < swizzle_threshold) | (num_wg % num_xcds != 0)
    return arith.select(use_simple, simple_m, pid_m), arith.select(use_simple, simple_n, pid_n)


def _flat_frag(frag):
    out = []
    for value in frag:
        if isinstance(value, (list, tuple)):
            out.extend(_flat_frag(value))
        else:
            out.append(arith._to_raw(value))
    return out


def _flat_accs(accs):
    out = []
    for frag in accs:
        out.extend(arith._to_raw(value) for value in frag)
    return out


def _g2s_thunks(g2s, dst, global_offset, n_steps):
    return [lambda step=step: g2s.load_one(dst, global_offset, step) for step in range(n_steps)]


def _s2r_thunks(s2r, src, holder, n_tiles):
    thunks = []
    for tile in range(n_tiles):
        row = s2r.wave_idx * (n_tiles * 16) + tile * 16 + s2r.lane_id % 16
        for half in range(2):
            col = (s2r.lane_id // 16) * 16 + half * 64
            row_swizzled, col_swizzled = swizzle_128(row, col)
            offset = row_swizzled * 128 + col_swizzled

            def load(tile=tile, half=half, offset=offset):
                holder[tile][half] = s2r.load_one(src, offset)

            thunks.append(load)
    return thunks


class _BlockScaleMfma(Mfma16x16x128):
    """Fold one K128 MFMA partial immediately into a VGPR-readable accumulator."""

    def __init__(
        self,
        n_tiles_a,
        n_tiles_b,
        fold_group_size,
        use_setprio,
        mfma_priority,
        interleave_width,
        priority_includes_fold,
        defer_col_scale,
    ):
        super().__init__(n_tiles_a, n_tiles_b)
        self.fold_group_size = fold_group_size
        self.use_setprio = use_setprio
        self.mfma_priority = mfma_priority
        self.interleave_width = interleave_width
        self.priority_includes_fold = priority_includes_fold
        self.defer_col_scale = defer_col_scale

    def _fold_partial(self, partial, scale, global_acc):
        from flydsl._mlir.dialects import math as math_dialect

        if const_expr(self.defer_col_scale):
            scale_a, scale_b = scale
            partial = fx.Vector(partial) * fx.Float32(scale_b)
            scale = scale_a
        return math_dialect.fma(
            arith._to_raw(partial),
            arith._to_raw(scale),
            arith._to_raw(global_acc),
        )

    def fold(self, a, b, global_acc, combined_scales, interleave=None):
        thunks = list(interleave) if interleave else []
        next_thunk = 0
        partials = []
        if const_expr(self.use_setprio):
            rocdl.s_setprio(self.mfma_priority)
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                idx = self.idx(i, j)
                partial = self._do_mma(a[i], b[j], self.zero_value)
                for _ in range_constexpr(self.interleave_width):
                    if next_thunk < len(thunks):
                        thunks[next_thunk]()
                        next_thunk += 1
                partials.append((i, j, idx, partial))
                if len(partials) == self.fold_group_size:
                    if const_expr(self.use_setprio and not self.priority_includes_fold):
                        rocdl.s_setprio(0)
                    for scale_i, scale_j, scale_idx, scale_partial in partials:
                        folded = self._fold_partial(
                            scale_partial,
                            combined_scales[scale_i][scale_j],
                            global_acc[scale_idx],
                        )
                        global_acc[scale_idx] = arith._to_raw(folded)
                    partials = []
                    if const_expr(self.use_setprio and not self.priority_includes_fold):
                        rocdl.s_setprio(self.mfma_priority)
        if const_expr(self.use_setprio):
            rocdl.s_setprio(0)
        for i, j, idx, partial in partials:
            folded = self._fold_partial(partial, combined_scales[i][j], global_acc[idx])
            global_acc[idx] = arith._to_raw(folded)
        while next_thunk < len(thunks):
            thunks[next_thunk]()
            next_thunk += 1
        return global_acc


def compile_fp8_transpose_32x32(rows: int, cols: int):
    """Return a launcher computing ``dst[cols, rows] = src[rows, cols].T``.

    Rows and columns must be multiples of 32. Global traffic uses one packed
    dword per thread. LDS uses a padded 32x36 byte tile to avoid column-read
    bank aliasing.
    """

    TILE = 32
    PACKED = 4
    LDS_STRIDE = 36
    THREADS = TILE * TILE // PACKED

    if rows <= 0 or cols <= 0 or rows % TILE != 0 or cols % TILE != 0:
        raise ValueError(f"rows and cols must be positive multiples of {TILE}, got ({rows}, {cols})")

    @fx.struct
    class SharedStorage:
        data: fx.Array[fx.Uint8, TILE * LDS_STRIDE, 16]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def transpose_kernel(src: fx.Tensor, dst: fx.Tensor):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek().data
        src_rsrc = _buffer_ops.create_buffer_resource(
            src,
            max_size=False,
            num_records_bytes=rows * cols,
        )
        dst_rsrc = _buffer_ops.create_buffer_resource(
            dst,
            max_size=False,
            num_records_bytes=rows * cols,
        )

        tid = fx.Int32(fx.thread_idx.x)
        input_row = tid // (TILE // PACKED)
        input_col = (tid % (TILE // PACKED)) * PACKED
        global_row = fx.Int32(fx.block_idx.y) * TILE + input_row
        global_col = fx.Int32(fx.block_idx.x) * TILE + input_col
        input_byte_offset = global_row * cols + global_col
        packed_input = _buffer_ops.buffer_load(
            src_rsrc,
            input_byte_offset // PACKED,
            vec_width=1,
            dtype=_T.i32,
        )

        lds_byte_offset = input_row * LDS_STRIDE + input_col
        lds_i32_type = fx.PointerType.get(fx.Int32.ir_type, lds.ptr.memspace, 4)
        lds_i32 = fx.inttoptr(
            lds_i32_type,
            fx.Int32(fx.ptrtoint(lds.ptr)) + lds_byte_offset,
        )
        fx.ptr_store(packed_input, lds_i32)
        fx.gpu.barrier()

        output_row = tid // (TILE // PACKED)
        output_col = (tid % (TILE // PACKED)) * PACKED
        output_bytes = []
        for element in range_constexpr(PACKED):
            lds_offset = (output_col + element) * LDS_STRIDE + output_row
            output_bytes.append(fx.ptr_load(lds.ptr + lds_offset))
        packed_output = fx.Vector.from_elements(output_bytes, fx.Uint8).bitcast(fx.Int32)[0]

        dst_row = fx.Int32(fx.block_idx.x) * TILE + output_row
        dst_col = fx.Int32(fx.block_idx.y) * TILE + output_col
        output_byte_offset = dst_row * rows + dst_col
        _buffer_ops.buffer_store(
            packed_output,
            dst_rsrc,
            output_byte_offset,
            offset_is_bytes=True,
        )

    @flyc.jit
    def launch_transpose(
        src: fx.Tensor,
        dst: fx.Tensor,
        stream: fx.Stream,
    ):
        transpose_kernel(src, dst).launch(
            grid=(cols // TILE, rows // TILE, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    return launch_transpose
