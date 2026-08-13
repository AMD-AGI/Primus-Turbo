###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL); see LICENSE-APACHE for the Apache-2.0 terms.
#
# See LICENSE for license information.
###############################################################################

"""gfx950 blockwise FP8 GEMM kernels and runtime dispatch."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr import buffer_ops as _buffer_ops
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from primus_turbo.flydsl.utils.gemm_helper import (
    G2SLoader,
    S2RLoader,
    S2RLoaderTr,
    _robust_time,
    ceildiv,
    compute_global_swizzle,
    compute_global_swizzle_nn,
    make_value_attrs,
    pack_i32x4_i32x8,
    swizzle_128,
    wait_barrier,
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


class _WideS2RLoader(S2RLoader):
    """Load 32x64 FP8 operands from the row-major XOR-swizzled LDS tile."""

    def _offset(self, tile, k_half, load_half):
        row = self.wave_idx * (self.n_tiles * 32) + tile * 32 + self.lane_id % 32
        col = k_half * 64 + (self.lane_id // 32) * 32 + load_half * 16
        row_swizzled, col_swizzled = swizzle_128(row, col)
        return row_swizzled * 128 + col_swizzled

    def load(self, lds_src):
        frag = []
        for tile in range_constexpr(self.n_tiles):
            k_operands = []
            for k_half in range_constexpr(2):
                lo = self.load_one(lds_src, self._offset(tile, k_half, 0))
                hi = self.load_one(lds_src, self._offset(tile, k_half, 1))
                k_operands.append(pack_i32x4_i32x8(lo, hi))
            frag.append(k_operands)
        return frag


class _WideBlockScaleMfma:
    """Fold two 32x32x64 MFMA partials into one block-scaled accumulator."""

    def __init__(self, n_tiles_a, n_tiles_b):
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.zero_value = fx.Vector.filled(16, 0.0, fx.Float32)
        self.result_type = _T.vec(16, _T.f32)
        self.unit_scale = fx.Int32(0x7F7F7F7F)

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def fold(self, a, b, global_acc, combined_scales, interleave=None):
        from flydsl._mlir.dialects import math as math_dialect

        thunks = list(interleave) if interleave else []
        next_thunk = 0
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                scale = combined_scales[i][j]
                idx = self.idx(i, j)
                partial = self.zero_value.ir_value()
                for k_half in range_constexpr(2):
                    partial = rocdl.mfma_scale_f32_32x32x64_f8f6f4(
                        self.result_type,
                        arith._to_raw(a[i][k_half]),
                        arith._to_raw(b[j][k_half]),
                        arith._to_raw(partial),
                        0,
                        0,
                        0,
                        arith._to_raw(self.unit_scale),
                        0,
                        arith._to_raw(self.unit_scale),
                    ).result
                    for _ in range_constexpr(3):
                        if next_thunk < len(thunks):
                            thunks[next_thunk]()
                            next_thunk += 1
                folded = math_dialect.fma(
                    arith._to_raw(partial),
                    arith._to_raw(scale),
                    arith._to_raw(global_acc[idx]),
                )
                global_acc[idx] = arith._to_raw(folded)
        while next_thunk < len(thunks):
            thunks[next_thunk]()
            next_thunk += 1
        return global_acc


def compile_blockscale_fp8_gemm_4w(
    *,
    K: int,
    M: int,
    N: int,
    BLOCK_M: int = 128,
    mfma_shape: str = "16x16x128",
    out_dtype: str = "bf16",
    use_xcd_remap: bool = True,
    waves_per_eu: int = 2,
    use_partial_waits: bool = True,
    scalar_b_scale: bool = True,
    scalar_a_scale: bool = False,
    fold_group_size: int = 4,
    vmcnt_slack: int = 0,
    use_setprio: bool = True,
    mfma_priority: int = 1,
    interleave_width: int = 1,
    s2r_first: bool = False,
    alternate_interleave: bool = False,
    group_m: int = 4,
    split_scale_halves: bool = False,
    k_loop_unroll: int = 2,
    priority_includes_fold: bool = False,
    scale_b_mode: str = "block2d",
    defer_col_scale: bool = True,
    scale_a_k_major: bool = True,
    scale_b_k_major: bool = False,
):
    """Return a fixed-shape gfx950 4-wave block-scaled FP8 GEMM launcher.

    The returned launcher has arguments
    ``(A, B, C, A_scale, B_scale, stream)``.  A and B are contiguous
    row-major FP8 tensors with logical shapes ``[M, K]`` and ``[N, K]``.
    ``scale_b_mode="block2d"`` uses one B scale per 128 output columns;
    ``"col1d"`` uses one B scale per output column for normalized TN/wgrad.
    ``scale_a_k_major`` and ``scale_b_k_major`` select whether each scale
    tensor stores the K128-block dimension first.
    C is BF16 or FP16 according to ``out_dtype``. ``BLOCK_M`` supports
    128 and 192; BLOCK_N and BLOCK_K are fixed at 128. ``mfma_shape``
    selects the narrow or wide gfx950 MFMA form.
    """

    block_m = BLOCK_M
    block_n = 128
    block_k = 128
    lds_half_m = block_m // 2
    lds_half_n = block_n // 2
    use_wide_mfma = mfma_shape == "32x32x64"
    effective_partial_waits = use_partial_waits
    mfma_mn = 32 if use_wide_mfma else 16
    accumulator_width = 16 if use_wide_mfma else 4
    n_tiles_a = block_m // 4 // mfma_mn
    n_tiles_b = block_n // 4 // mfma_mn
    n_accums = n_tiles_a * n_tiles_b
    n_lds_rounds_a = lds_half_m // 32
    n_lds_rounds_b = lds_half_n // 32
    main_wait_count = 2 * n_lds_rounds_a + 2 * n_lds_rounds_b
    tail_wait_count = n_lds_rounds_a + n_lds_rounds_b
    prologue_a_wait_count = 3 * n_lds_rounds_a + 4 * n_lds_rounds_b
    prologue_b_wait_count = 3 * n_lds_rounds_a + 3 * n_lds_rounds_b

    if BLOCK_M not in (128, 192):
        raise ValueError(f"BLOCK_M must be 128 or 192, got {BLOCK_M!r}")
    if mfma_shape not in ("16x16x128", "32x32x64"):
        raise ValueError(f"unsupported mfma_shape {mfma_shape!r}")
    if use_wide_mfma and BLOCK_M != 128:
        raise ValueError("32x32x64 currently requires BLOCK_M=128")
    if not isinstance(K, int) or K < block_k or K % block_k != 0:
        raise ValueError(f"K must be a positive multiple of {block_k}, got {K!r}")
    if out_dtype not in ("bf16", "fp16"):
        raise ValueError(f"out_dtype must be 'bf16' or 'fp16', got {out_dtype!r}")
    if scale_b_mode not in ("block2d", "col1d"):
        raise ValueError(f"scale_b_mode must be 'block2d' or 'col1d', got {scale_b_mode!r}")
    if scale_b_mode == "col1d" and use_wide_mfma:
        raise ValueError("scale_b_mode='col1d' currently requires 16x16x128 MFMA")
    if defer_col_scale and scale_b_mode != "col1d":
        defer_col_scale = False
    if scalar_a_scale and not scale_a_k_major:
        raise ValueError("scalar_a_scale requires scale_a_k_major=True")
    if waves_per_eu not in (1, 2):
        raise ValueError(f"waves_per_eu must be 1 or 2, got {waves_per_eu!r}")
    if fold_group_size not in (1, 2, 3, 4, 5, 6):
        raise ValueError(f"fold_group_size must be in [1, 6], got {fold_group_size!r}")
    if vmcnt_slack not in (0, 2, 4):
        raise ValueError(f"vmcnt_slack must be 0, 2, or 4, got {vmcnt_slack!r}")
    if interleave_width not in (1, 2):
        raise ValueError(f"interleave_width must be 1 or 2, got {interleave_width!r}")
    if mfma_priority not in (1, 2, 3):
        raise ValueError(f"mfma_priority must be 1, 2, or 3, got {mfma_priority!r}")
    if not isinstance(group_m, int) or not 1 <= group_m <= 16:
        raise ValueError(f"group_m must be an integer in [1, 16], got {group_m!r}")
    if split_scale_halves and (use_wide_mfma or scalar_a_scale or scale_b_mode != "block2d"):
        raise ValueError("split_scale_halves currently requires narrow MFMA and vector A-scale loads")
    if k_loop_unroll not in (2, 4, 6):
        raise ValueError(f"k_loop_unroll must be 2, 4, or 6, got {k_loop_unroll!r}")

    if use_partial_waits:
        main_wait_count += vmcnt_slack
        tail_wait_count += vmcnt_slack
        prologue_a_wait_count += vmcnt_slack
        prologue_b_wait_count += vmcnt_slack

    k_iters = K // block_k
    main_steps = max(k_iters - 2, 0)
    loop_steps = (main_steps // k_loop_unroll) * k_loop_unroll
    remainder_steps = main_steps - loop_steps
    out_type = fx.BFloat16 if out_dtype == "bf16" else fx.Float16
    mn_aligned = M % block_m == 0 and N % block_n == 0
    large_a = M * K > 0xFFFFFFFF
    large_b = N * K > 0xFFFFFFFF
    large_output = M * N * 2 > 0xFFFFFFFF

    a_lds_size = lds_half_m * block_k
    b_lds_size = lds_half_n * block_k

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
    ):
        f8_ir_type = fx.Float8E4M3FN.ir_type
        m_extent = fx.Int32(M)
        n_extent = fx.Int32(N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_i = wave_id // 2
        wave_j = wave_id % 2

        num_pid_m = ceildiv(m_extent, block_m)
        num_pid_n = ceildiv(n_extent, block_n)
        if const_expr(use_xcd_remap):
            tile_i, tile_j = _xcd_swizzle(num_pid_m, num_pid_n, group_m)
        else:
            tile_i, tile_j = divmod(fx.block_idx.x, num_pid_n)

        if const_expr(large_a):
            a0_global_offset = fx.Int32(0)
            a1_global_offset = fx.Int32(lds_half_m * K)
        else:
            a0_global_offset = (tile_i * block_m) * K
            a1_global_offset = (tile_i * block_m + lds_half_m) * K
        if const_expr(large_b):
            b0_global_offset = fx.Int32(0)
            b1_global_offset = fx.Int32(lds_half_n * K)
        else:
            b0_global_offset = (tile_j * block_n) * K
            b1_global_offset = (tile_j * block_n + lds_half_n) * K

        # Force one-dimensional logical views so callers may pass either flat
        # storage or ordinary contiguous [M,K]/[N,K] tensors.
        if const_expr(large_a):
            a_total_bytes = fx.Index(M) * K
            a_base_off_bytes = fx.Index(tile_i) * block_m * K
            a_records_bytes = fx.Index(
                arith.minsi(
                    (a_total_bytes - a_base_off_bytes).ir_value(),
                    fx.Index(0xFFFFFFFF).ir_value(),
                )
            )
            gA_base = make_fp8_buffer_tensor(
                A,
                f8_ir_type,
                num_records_bytes=a_records_bytes,
                base_byte_offset=a_base_off_bytes,
            )
            gA = fx.make_view(fx.get_iter(gA_base), fx.make_layout(block_m * K, 1))
        else:
            gA_base = make_fp8_buffer_tensor(A, f8_ir_type)
            gA = fx.make_view(fx.get_iter(gA_base), fx.make_layout(m_extent * K, 1))
        if const_expr(large_b):
            b_total_bytes = fx.Index(N) * K
            b_base_off_bytes = fx.Index(tile_j) * block_n * K
            b_records_bytes = fx.Index(
                arith.minsi(
                    (b_total_bytes - b_base_off_bytes).ir_value(),
                    fx.Index(0xFFFFFFFF).ir_value(),
                )
            )
            gB_base = make_fp8_buffer_tensor(
                B,
                f8_ir_type,
                num_records_bytes=b_records_bytes,
                base_byte_offset=b_base_off_bytes,
            )
            gB = fx.make_view(fx.get_iter(gB_base), fx.make_layout(block_n * K, 1))
        else:
            gB_base = make_fp8_buffer_tensor(B, f8_ir_type)
            gB = fx.make_view(fx.get_iter(gB_base), fx.make_layout(n_extent * K, 1))
        ga_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        gb_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, n_lds_rounds_a)
        gl_off_b = compute_global_swizzle(lane_id, wave_id, K, n_lds_rounds_b)
        a_g2s = G2SLoader(ga_div, gl_off_a, n_lds_rounds_a, f8_ir_type, wave_id)
        b_g2s = G2SLoader(gb_div, gl_off_b, n_lds_rounds_b, f8_ir_type, wave_id)
        if const_expr(use_wide_mfma):
            a_s2r = _WideS2RLoader(wave_i, n_tiles_a)
            b_s2r = _WideS2RLoader(wave_j, n_tiles_b)
            mfma = _WideBlockScaleMfma(n_tiles_a, n_tiles_b)
        else:
            a_s2r = S2RLoader(wave_i, n_tiles_a)
            b_s2r = S2RLoader(wave_j, n_tiles_b)
            mfma = _BlockScaleMfma(
                n_tiles_a,
                n_tiles_b,
                fold_group_size,
                use_setprio,
                mfma_priority,
                interleave_width,
                priority_includes_fold,
                defer_col_scale,
            )

        scale_a_bytes = m_extent * k_iters * 4
        scale_b_elements = n_extent * k_iters if scale_b_mode == "col1d" else num_pid_n * k_iters
        scale_b_bytes = scale_b_elements * 4
        scale_a_rsrc = _buffer_ops.create_buffer_resource(
            A_scale,
            max_size=False,
            num_records_bytes=scale_a_bytes,
        )
        scale_b_rsrc = _buffer_ops.create_buffer_resource(
            B_scale,
            max_size=False,
            num_records_bytes=scale_b_bytes,
        )
        if const_expr(large_output):
            c_total_bytes = fx.Index(M) * fx.Index(N) * 2
            c_base_off_bytes = fx.Index(tile_i) * block_m * fx.Index(N) * 2
            c_records_bytes = fx.Index(
                arith.minsi(
                    (c_total_bytes - c_base_off_bytes).ir_value(),
                    fx.Index(0xFFFFFFFF).ir_value(),
                )
            )
            c_rsrc = _buffer_ops.create_buffer_resource(
                C,
                max_size=False,
                num_records_bytes=c_records_bytes,
                base_byte_offset=c_base_off_bytes,
            )
        else:
            c_bytes = m_extent * n_extent * 2
            c_rsrc = _buffer_ops.create_buffer_resource(C, max_size=False, num_records_bytes=c_bytes)

        base_row = tile_i * block_m + wave_i * (n_tiles_a * mfma_mn)
        base_col = tile_j * block_n + wave_j * (n_tiles_b * mfma_mn)
        lane_row = (lane_id // mfma_mn) * 4

        def _load_combined_scales(k_block):
            k_block_i32 = fx.Int32(k_block)
            scale_a_base = k_block_i32 * m_extent

            def _load_b_scales(upper_half):
                values = []
                if const_expr(scale_b_mode == "block2d"):
                    if const_expr(scale_b_k_major):
                        scale_b_idx = k_block_i32 * fx.Int32(num_pid_n) + fx.Int32(tile_j)
                    else:
                        scale_b_idx = fx.Int32(tile_j) * fx.Int32(k_iters) + k_block_i32
                    if const_expr(scalar_b_scale):
                        scale_b = fx.Int32(
                            _buffer_ops.buffer_load(
                                scale_b_rsrc,
                                scale_b_idx,
                                vec_width=1,
                                is_scalar=True,
                            )
                        ).bitcast(fx.Float32)
                    else:
                        scale_b = fx.Float32(
                            _buffer_ops.buffer_load(
                                scale_b_rsrc,
                                scale_b_idx,
                                vec_width=1,
                                dtype=_T.f32,
                            )
                        )
                    for _ in range_constexpr(n_tiles_b):
                        values.append(scale_b)
                else:
                    half_offset = lds_half_n if upper_half else 0
                    for tile in range_constexpr(n_tiles_b):
                        output_col = base_col + half_offset + tile * mfma_mn + lane_id % mfma_mn
                        if const_expr(scale_b_k_major):
                            scale_b_idx = k_block_i32 * n_extent + fx.Int32(output_col)
                        else:
                            scale_b_idx = fx.Int32(output_col) * fx.Int32(k_iters) + k_block_i32
                        values.append(
                            fx.Float32(
                                _buffer_ops.buffer_load(
                                    scale_b_rsrc,
                                    scale_b_idx,
                                    vec_width=1,
                                    dtype=_T.f32,
                                )
                            )
                        )
                return values

            scale_a_r0 = []
            scale_a_r1 = []
            for tile in range_constexpr(n_tiles_a):
                row_offset = tile * mfma_mn + lane_row
                if const_expr(use_wide_mfma):
                    scale_values0 = []
                    scale_values1 = []
                    for row_group in range_constexpr(4):
                        group_offset = row_offset + row_group * 8
                        row0 = fx.Int32(base_row + group_offset)
                        row1 = fx.Int32(base_row + lds_half_m + group_offset)
                        if const_expr(scale_a_k_major):
                            scale_a_idx0 = scale_a_base + row0
                            scale_a_idx1 = scale_a_base + row1
                            scale_a0 = fx.Vector(
                                _buffer_ops.buffer_load(
                                    scale_a_rsrc,
                                    scale_a_idx0,
                                    vec_width=4,
                                    dtype=_T.f32,
                                )
                            )
                            scale_a1 = fx.Vector(
                                _buffer_ops.buffer_load(
                                    scale_a_rsrc,
                                    scale_a_idx1,
                                    vec_width=4,
                                    dtype=_T.f32,
                                )
                            )
                        else:
                            scale_a0_values = []
                            scale_a1_values = []
                            for element in range_constexpr(4):
                                scale_a0_values.append(
                                    fx.Float32(
                                        _buffer_ops.buffer_load(
                                            scale_a_rsrc,
                                            (row0 + element) * fx.Int32(k_iters) + k_block_i32,
                                            vec_width=1,
                                            dtype=_T.f32,
                                        )
                                    )
                                )
                                scale_a1_values.append(
                                    fx.Float32(
                                        _buffer_ops.buffer_load(
                                            scale_a_rsrc,
                                            (row1 + element) * fx.Int32(k_iters) + k_block_i32,
                                            vec_width=1,
                                            dtype=_T.f32,
                                        )
                                    )
                                )
                            scale_a0 = fx.Vector.from_elements(scale_a0_values, fx.Float32)
                            scale_a1 = fx.Vector.from_elements(scale_a1_values, fx.Float32)
                        for element in range_constexpr(4):
                            scale_values0.append(scale_a0[element])
                            scale_values1.append(scale_a1[element])
                    scale_a_r0.append(fx.Vector.from_elements(scale_values0, fx.Float32))
                    scale_a_r1.append(fx.Vector.from_elements(scale_values1, fx.Float32))
                else:
                    if const_expr(scalar_a_scale):
                        scalar_groups0 = []
                        scalar_groups1 = []
                        for row_group in range_constexpr(4):
                            group_offset = row_offset - lane_row + row_group * 4
                            row0 = fx.Int32(base_row + group_offset)
                            row1 = fx.Int32(base_row + lds_half_m + group_offset)
                            if const_expr(scale_a_k_major):
                                scale_a_idx0 = scale_a_base + row0
                                scale_a_idx1 = scale_a_base + row1
                            else:
                                scale_a_idx0 = row0 * fx.Int32(k_iters) + k_block_i32
                                scale_a_idx1 = row1 * fx.Int32(k_iters) + k_block_i32
                            scalar_groups0.append(
                                fx.Vector(
                                    _buffer_ops.buffer_load(
                                        scale_a_rsrc,
                                        scale_a_idx0,
                                        vec_width=4,
                                        is_scalar=True,
                                    )
                                ).bitcast(fx.Float32)
                            )
                            scalar_groups1.append(
                                fx.Vector(
                                    _buffer_ops.buffer_load(
                                        scale_a_rsrc,
                                        scale_a_idx1,
                                        vec_width=4,
                                        is_scalar=True,
                                    )
                                ).bitcast(fx.Float32)
                            )
                        selected0 = scalar_groups0[0]
                        selected1 = scalar_groups1[0]
                        lane_group = lane_id // 16
                        for row_group in range_constexpr(1, 4):
                            condition = lane_group == row_group
                            selected0 = fx.Vector(
                                arith.select(
                                    condition,
                                    arith._to_raw(scalar_groups0[row_group]),
                                    arith._to_raw(selected0),
                                )
                            )
                            selected1 = fx.Vector(
                                arith.select(
                                    condition,
                                    arith._to_raw(scalar_groups1[row_group]),
                                    arith._to_raw(selected1),
                                )
                            )
                        scale_a_r0.append(selected0)
                        scale_a_r1.append(selected1)
                    else:
                        row0 = fx.Int32(base_row + row_offset)
                        row1 = fx.Int32(base_row + lds_half_m + row_offset)
                        if const_expr(scale_a_k_major):
                            scale_a_idx0 = scale_a_base + row0
                            scale_a_idx1 = scale_a_base + row1
                            scale_a0 = fx.Vector(
                                _buffer_ops.buffer_load(
                                    scale_a_rsrc,
                                    scale_a_idx0,
                                    vec_width=4,
                                    dtype=_T.f32,
                                )
                            )
                            scale_a1 = fx.Vector(
                                _buffer_ops.buffer_load(
                                    scale_a_rsrc,
                                    scale_a_idx1,
                                    vec_width=4,
                                    dtype=_T.f32,
                                )
                            )
                        else:
                            scale_a0_values = []
                            scale_a1_values = []
                            for element in range_constexpr(4):
                                scale_a0_values.append(
                                    fx.Float32(
                                        _buffer_ops.buffer_load(
                                            scale_a_rsrc,
                                            (row0 + element) * fx.Int32(k_iters) + k_block_i32,
                                            vec_width=1,
                                            dtype=_T.f32,
                                        )
                                    )
                                )
                                scale_a1_values.append(
                                    fx.Float32(
                                        _buffer_ops.buffer_load(
                                            scale_a_rsrc,
                                            (row1 + element) * fx.Int32(k_iters) + k_block_i32,
                                            vec_width=1,
                                            dtype=_T.f32,
                                        )
                                    )
                                )
                            scale_a0 = fx.Vector.from_elements(scale_a0_values, fx.Float32)
                            scale_a1 = fx.Vector.from_elements(scale_a1_values, fx.Float32)
                        scale_a_r0.append(scale_a0)
                        scale_a_r1.append(scale_a1)

            def _combine(scale_a_values, scale_b_values):
                combined = []
                for scale_a in scale_a_values:
                    row = []
                    for scale_b in scale_b_values:
                        if const_expr(defer_col_scale):
                            row.append((scale_a, scale_b))
                        else:
                            scale_b_vec = fx.Vector.filled(
                                accumulator_width,
                                fx.Float32(scale_b),
                                fx.Float32,
                            )
                            row.append(scale_a * scale_b_vec)
                    combined.append(row)
                return combined

            scale_b_r0 = _load_b_scales(False)
            scale_b_r1 = _load_b_scales(True)
            return (
                _combine(scale_a_r0, scale_b_r0),
                _combine(scale_a_r0, scale_b_r1),
                _combine(scale_a_r1, scale_b_r0),
                _combine(scale_a_r1, scale_b_r1),
            )

        def _load_narrow_scale_half(k_block, upper_half):
            k_block_i32 = fx.Int32(k_block)
            scale_a_base = k_block_i32 * m_extent
            if const_expr(scale_b_k_major):
                scale_b_idx = k_block_i32 * fx.Int32(num_pid_n) + fx.Int32(tile_j)
            else:
                scale_b_idx = fx.Int32(tile_j) * fx.Int32(k_iters) + k_block_i32
            if const_expr(scalar_b_scale):
                scale_b = fx.Int32(
                    _buffer_ops.buffer_load(
                        scale_b_rsrc,
                        scale_b_idx,
                        vec_width=1,
                        is_scalar=True,
                    )
                ).bitcast(fx.Float32)
            else:
                scale_b = _buffer_ops.buffer_load(
                    scale_b_rsrc,
                    scale_b_idx,
                    vec_width=1,
                    dtype=_T.f32,
                )
            scale_b_vec = fx.Vector.filled(4, fx.Float32(scale_b), fx.Float32)
            row_half_offset = lds_half_m if upper_half else 0
            combined = []
            for tile in range_constexpr(n_tiles_a):
                row_offset = tile * 16 + lane_row
                row = fx.Int32(base_row + row_half_offset + row_offset)
                if const_expr(scale_a_k_major):
                    scale_a_idx = scale_a_base + row
                    scale_a = fx.Vector(
                        _buffer_ops.buffer_load(
                            scale_a_rsrc,
                            scale_a_idx,
                            vec_width=4,
                            dtype=_T.f32,
                        )
                    )
                else:
                    scale_a_values = []
                    for element in range_constexpr(4):
                        scale_a_values.append(
                            fx.Float32(
                                _buffer_ops.buffer_load(
                                    scale_a_rsrc,
                                    (row + element) * fx.Int32(k_iters) + k_block_i32,
                                    vec_width=1,
                                    dtype=_T.f32,
                                )
                            )
                        )
                    scale_a = fx.Vector.from_elements(scale_a_values, fx.Float32)
                value = scale_a * scale_b_vec
                combined.append([value for _ in range_constexpr(n_tiles_b)])
            return combined

        def _store_fragment(c_frag, row_base, col_base):
            if const_expr(mn_aligned):
                for ti in range_constexpr(n_tiles_a):
                    row = row_base + ti * mfma_mn + lane_row
                    for tj in range_constexpr(n_tiles_b):
                        col = col_base + tj * mfma_mn + lane_id % mfma_mn
                        vec_f32 = fx.Vector(c_frag[mfma.idx(ti, tj)])
                        for elem in range_constexpr(accumulator_width):
                            row_offset = (elem // 4) * 8 + elem % 4 if use_wide_mfma else elem
                            out_row = row + row_offset
                            if const_expr(large_output):
                                row_local = out_row - fx.Int32(tile_i * block_m)
                                c_idx = row_local * n_extent + col
                            else:
                                c_idx = out_row * n_extent + col
                            _buffer_ops.buffer_store(vec_f32[elem].to(out_type), c_rsrc, c_idx)
            else:
                for ti in range_constexpr(n_tiles_a):
                    row = row_base + ti * mfma_mn + lane_row
                    for tj in range_constexpr(n_tiles_b):
                        col = col_base + tj * mfma_mn + lane_id % mfma_mn
                        vec_f32 = fx.Vector(c_frag[mfma.idx(ti, tj)])
                        col_valid = col < n_extent
                        for elem in range_constexpr(accumulator_width):
                            row_offset = (elem // 4) * 8 + elem % 4 if use_wide_mfma else elem
                            out_row = row + row_offset
                            valid = (out_row < m_extent) & col_valid
                            if const_expr(large_output):
                                row_local = out_row - fx.Int32(tile_i * block_m)
                                c_idx = row_local * n_extent + col
                                _buffer_ops.buffer_store(
                                    vec_f32[elem].to(out_type),
                                    c_rsrc,
                                    c_idx,
                                    mask=valid,
                                )
                            else:
                                c_idx = out_row * n_extent + col
                                oob = m_extent * n_extent
                                safe_idx = arith.select(valid, c_idx, oob)
                                _buffer_ops.buffer_store(vec_f32[elem].to(out_type), c_rsrc, safe_idx)

        def _store_output(accs):
            _store_fragment(accs[0], base_row, base_col)
            _store_fragment(accs[1], base_row, base_col + lds_half_n)
            _store_fragment(accs[2], base_row + lds_half_m, base_col)
            _store_fragment(accs[3], base_row + lds_half_m, base_col + lds_half_n)

        def _compute_stage(
            lds_dst,
            g2s,
            g2s_steps,
            global_offset,
            s2r,
            s2r_tiles,
            lds_src,
            a_frag,
            b_frag,
            c_frag,
            combined_scales,
        ):
            if const_expr(use_wide_mfma):
                g2s.load(lds_dst, global_offset)
                next_frag = s2r.load(lds_src)
                c_frag = mfma.fold(a_frag, b_frag, c_frag, combined_scales)
                return c_frag, next_frag

            next_halves = [[None, None] for _ in range(s2r_tiles)]
            g2s_thunks = _g2s_thunks(g2s, lds_dst, global_offset, g2s_steps)
            s2r_thunks = _s2r_thunks(s2r, lds_src, next_halves, s2r_tiles)
            if const_expr(alternate_interleave):
                interleave = []
                for index in range_constexpr(max(len(g2s_thunks), len(s2r_thunks))):
                    if index < len(g2s_thunks):
                        interleave.append(g2s_thunks[index])
                    if index < len(s2r_thunks):
                        interleave.append(s2r_thunks[index])
            else:
                interleave = s2r_thunks + g2s_thunks if s2r_first else g2s_thunks + s2r_thunks
            c_frag = mfma.fold(
                a_frag,
                b_frag,
                c_frag,
                combined_scales,
                interleave=interleave,
            )
            next_frag = [pack_i32x4_i32x8(halves[0], halves[1]) for halves in next_halves]
            return c_frag, next_frag

        def _one_step(k_block, a0_frag, b0_frag, accs, bufs):
            ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
            c00, c01, c10, c11 = accs

            k_i32 = fx.Int32(k_block)
            next_k = (k_i32 + fx.Int32(2)) * fx.Int32(block_k)
            a0_offset = fx.Int32(a0_global_offset) + next_k
            a1_offset = fx.Int32(a1_global_offset) + next_k
            b0_offset = fx.Int32(b0_global_offset) + next_k
            b1_offset = fx.Int32(b1_global_offset) + next_k

            wait_barrier(main_wait_count if effective_partial_waits else 0)
            if const_expr(split_scale_halves):
                scale_r00 = _load_narrow_scale_half(k_i32, False)
                scale_r01 = scale_r00
            else:
                scale_r00, scale_r01, scale_r10, scale_r11 = _load_combined_scales(k_i32)

            c00, b1_frag = _compute_stage(
                ac0,
                a_g2s,
                n_tiles_a,
                a0_offset,
                b_s2r,
                n_tiles_b,
                bc1,
                a0_frag,
                b0_frag,
                c00,
                scale_r00,
            )
            c01, a1_frag = _compute_stage(
                bc0,
                b_g2s,
                n_tiles_b,
                b0_offset,
                a_s2r,
                n_tiles_a,
                ac1,
                a0_frag,
                b1_frag,
                c01,
                scale_r01,
            )

            if const_expr(split_scale_halves):
                scale_r10 = _load_narrow_scale_half(k_i32, True)
                scale_r11 = scale_r10
            wait_barrier(main_wait_count if effective_partial_waits else 0)

            c10, a0_next_frag = _compute_stage(
                bc1,
                b_g2s,
                n_tiles_b,
                b1_offset,
                a_s2r,
                n_tiles_a,
                an0,
                a1_frag,
                b0_frag,
                c10,
                scale_r10,
            )
            c11, b0_next_frag = _compute_stage(
                ac1,
                a_g2s,
                n_tiles_a,
                a1_offset,
                b_s2r,
                n_tiles_b,
                bn0,
                a1_frag,
                b1_frag,
                c11,
                scale_r11,
            )

            next_bufs = an0, an1, ac0, ac1, bn0, bn1, bc0, bc1
            return a0_next_frag, b0_next_frag, (c00, c01, c10, c11), next_bufs

        def _finish_two_tiles(k0, a0_frag, b0_frag, accs, bufs):
            ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
            c00, c01, c10, c11 = accs

            wait_barrier(main_wait_count if effective_partial_waits else 0)
            if const_expr(split_scale_halves):
                scale_r00 = _load_narrow_scale_half(k0, False)
                scale_r01 = scale_r00
            else:
                scale_r00, scale_r01, scale_r10, scale_r11 = _load_combined_scales(k0)

            b1_frag = b_s2r.load(bc1)
            c00 = mfma.fold(a0_frag, b0_frag, c00, scale_r00)
            a1_frag = a_s2r.load(ac1)
            c01 = mfma.fold(a0_frag, b1_frag, c01, scale_r01)
            if const_expr(split_scale_halves):
                scale_r10 = _load_narrow_scale_half(k0, True)
                scale_r11 = scale_r10
            wait_barrier(tail_wait_count if effective_partial_waits else 0)
            a0_last = a_s2r.load(an0)
            c10 = mfma.fold(a1_frag, b0_frag, c10, scale_r10)
            b0_last = b_s2r.load(bn0)
            c11 = mfma.fold(a1_frag, b1_frag, c11, scale_r11)

            wait_barrier(0)
            scale_r00, scale_r01, scale_r10, scale_r11 = _load_combined_scales(k0 + 1)

            b1_last = b_s2r.load(bn1)
            a1_last = a_s2r.load(an1)
            c00 = mfma.fold(a0_last, b0_last, c00, scale_r00)
            c01 = mfma.fold(a0_last, b1_last, c01, scale_r01)
            c10 = mfma.fold(a1_last, b0_last, c10, scale_r10)
            c11 = mfma.fold(a1_last, b1_last, c11, scale_r11)

            _store_output((c00, c01, c10, c11))

        c00_frag = [mfma.zero_value] * n_accums
        c01_frag = [mfma.zero_value] * n_accums
        c10_frag = [mfma.zero_value] * n_accums
        c11_frag = [mfma.zero_value] * n_accums

        # Current K128 tile.
        a_g2s.load(a_cur0, a0_global_offset)
        b_g2s.load(b_cur0, b0_global_offset)
        b_g2s.load(b_cur1, b1_global_offset)
        a_g2s.load(a_cur1, a1_global_offset)

        if const_expr(k_iters > 1):
            # Next K128 tile.  These four buffers become current after a swap.
            a_g2s.load(a_next0, a0_global_offset + block_k)
            b_g2s.load(b_next0, b0_global_offset + block_k)
            b_g2s.load(b_next1, b1_global_offset + block_k)
            a_g2s.load(a_next1, a1_global_offset + block_k)

        if const_expr(k_iters > 1 and effective_partial_waits):
            wait_barrier(prologue_a_wait_count)
            a0_frag = a_s2r.load(a_cur0)
            wait_barrier(prologue_b_wait_count)
            b0_frag = b_s2r.load(b_cur0)
        else:
            wait_barrier(0)
            a0_frag = a_s2r.load(a_cur0)
            b0_frag = b_s2r.load(b_cur0)

        if const_expr(k_iters == 1):
            if const_expr(split_scale_halves):
                scale_r00 = _load_narrow_scale_half(0, False)
                scale_r01 = scale_r00
            else:
                scale_r00, scale_r01, scale_r10, scale_r11 = _load_combined_scales(0)
            b1_frag = b_s2r.load(b_cur1)
            a1_frag = a_s2r.load(a_cur1)
            c00_frag = mfma.fold(a0_frag, b0_frag, c00_frag, scale_r00)
            c01_frag = mfma.fold(a0_frag, b1_frag, c01_frag, scale_r01)
            if const_expr(split_scale_halves):
                scale_r10 = _load_narrow_scale_half(0, True)
                scale_r11 = scale_r10
            c10_frag = mfma.fold(a1_frag, b0_frag, c10_frag, scale_r10)
            c11_frag = mfma.fold(a1_frag, b1_frag, c11_frag, scale_r11)
            _store_output((c00_frag, c01_frag, c10_frag, c11_frag))
        else:
            bufs0 = (
                a_cur0,
                a_cur1,
                a_next0,
                a_next1,
                b_cur0,
                b_cur1,
                b_next0,
                b_next1,
            )

            init_state = (
                _flat_frag(a0_frag)
                + _flat_frag(b0_frag)
                + _flat_accs((c00_frag, c01_frag, c10_frag, c11_frag))
            )
            operand_parts = 2 if use_wide_mfma else 1
            n_a_state = n_tiles_a * operand_parts
            n_b_state = n_tiles_b * operand_parts

            def _unflat_operand(flat, n_tiles):
                if const_expr(use_wide_mfma):
                    out = []
                    for tile in range_constexpr(n_tiles):
                        out.append([flat[tile * 2], flat[tile * 2 + 1]])
                    return out
                return list(flat)

            # Two K128 stages per SCF body keep the physical LDS buffer mapping
            # unchanged at the loop boundary.  Deep K therefore does not expand
            # the kernel text with one body per scale block.
            if const_expr(loop_steps > 0):
                for kk, state in range(0, loop_steps, k_loop_unroll, init=init_state):
                    offset = 0
                    a0_loop = _unflat_operand(state[offset : offset + n_a_state], n_tiles_a)
                    offset += n_a_state
                    b0_loop = _unflat_operand(state[offset : offset + n_b_state], n_tiles_b)
                    offset += n_b_state
                    c00_loop = list(state[offset : offset + n_accums])
                    offset += n_accums
                    c01_loop = list(state[offset : offset + n_accums])
                    offset += n_accums
                    c10_loop = list(state[offset : offset + n_accums])
                    offset += n_accums
                    c11_loop = list(state[offset : offset + n_accums])

                    loop_accs = c00_loop, c01_loop, c10_loop, c11_loop
                    step_bufs = bufs0
                    for step in range_constexpr(k_loop_unroll):
                        a0_loop, b0_loop, loop_accs, step_bufs = _one_step(
                            kk + step,
                            a0_loop,
                            b0_loop,
                            loop_accs,
                            step_bufs,
                        )

                    loop_results = yield (_flat_frag(a0_loop) + _flat_frag(b0_loop) + _flat_accs(loop_accs))

                offset = 0
                a0_frag = _unflat_operand(loop_results[offset : offset + n_a_state], n_tiles_a)
                offset += n_a_state
                b0_frag = _unflat_operand(loop_results[offset : offset + n_b_state], n_tiles_b)
                offset += n_b_state
                c00_frag = list(loop_results[offset : offset + n_accums])
                offset += n_accums
                c01_frag = list(loop_results[offset : offset + n_accums])
                offset += n_accums
                c10_frag = list(loop_results[offset : offset + n_accums])
                offset += n_accums
                c11_frag = list(loop_results[offset : offset + n_accums])
                accs = c00_frag, c01_frag, c10_frag, c11_frag
            else:
                accs = c00_frag, c01_frag, c10_frag, c11_frag

            tail_bufs = bufs0
            for tail_step in range_constexpr(remainder_steps):
                a0_frag, b0_frag, accs, tail_bufs = _one_step(
                    loop_steps + tail_step,
                    a0_frag,
                    b0_frag,
                    accs,
                    tail_bufs,
                )
            _finish_two_tiles(k_iters - 2, a0_frag, b0_frag, accs, tail_bufs)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(M, block_m) * ceildiv(N, block_n)
        kernel_gemm(
            A,
            B,
            C,
            A_scale,
            B_scale,
            value_attrs=make_value_attrs(waves_per_eu, 0, "256,256"),
        ).launch(
            grid=(grid_x, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_gemm


def compile_blockscale_fp8_gemm_nn_4w(**kwargs):
    """Compile normalized NN/dgrad block-scale GEMM.

    The physical NN right operand ``B[K, N]`` must be supplied as a contiguous
    ``B.T[N, K]`` tensor. Its 2D scales must likewise be supplied as
    ``[ceil(N / 128), K / 128]``. This normalized representation can be emitted
    directly by a dual-layout quantizer and avoids a conversion in the GEMM.
    """

    kwargs.pop("scale_b_mode", None)
    return compile_blockscale_fp8_gemm_4w(scale_b_mode="block2d", **kwargs)


def compile_blockscale_fp8_gemm_tn_4w(**kwargs):
    """Compile normalized TN/wgrad block-scale GEMM.

    Both physical operands are transposed into contiguous row-major tensors:
    ``A.T[M, K]`` and ``B.T[N, K]``. A scales use ``[K / 128, M]`` and B scales
    use the per-output-column layout ``[N, K / 128]``.
    """

    kwargs.pop("scale_b_mode", None)
    return compile_blockscale_fp8_gemm_4w(scale_b_mode="col1d", **kwargs)


def compile_blockscale_fp8_gemm_nn_physical_4w(*, M: int, N: int, K: int, **kwargs):
    """Compile physical NN ``A[M,K] @ B[K,N]`` plus B transpose.

    The returned launcher takes
    ``(A, B, C, A_scale, B_scale, B_workspace, stream)``. ``B_workspace`` is
    contiguous FP8 storage of shape ``[N, K]``. Physical scales are consumed
    directly as ``A_scale[M, K/128]`` and ``B_scale[K/128, ceil(N/128)]``.
    """

    if "BLOCK_M" not in kwargs:
        kwargs["BLOCK_M"] = 192 if M % 192 == 0 else 128
    transpose_b = compile_fp8_transpose_32x32(K, N)
    gemm = compile_blockscale_fp8_gemm_nn_4w(
        K=K,
        M=M,
        N=N,
        scale_a_k_major=False,
        scale_b_k_major=True,
        **kwargs,
    )

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        B_workspace: fx.Tensor,
        stream: fx.Stream,
    ):
        transpose_b(B, B_workspace, stream)
        gemm(A, B_workspace, C, A_scale, B_scale, stream)

    return launch


def compile_blockscale_fp8_gemm_tn_physical_4w(*, M: int, N: int, K: int, **kwargs):
    """Compile physical TN ``A[K,M].T @ B[K,N]`` plus both transposes.

    The returned launcher takes
    ``(A, B, C, A_scale, B_scale, A_workspace, B_workspace, stream)``.
    Workspaces are contiguous FP8 tensors with shapes ``[M,K]`` and ``[N,K]``.
    Scales remain in their physical ``[K/128,M]`` and ``[K/128,N]`` layouts.
    """

    if "BLOCK_M" not in kwargs:
        kwargs["BLOCK_M"] = 192 if M % 192 == 0 else 128
    transpose_a = compile_fp8_transpose_32x32(K, M)
    transpose_b = compile_fp8_transpose_32x32(K, N)
    gemm = compile_blockscale_fp8_gemm_tn_4w(
        K=K,
        M=M,
        N=N,
        scale_a_k_major=True,
        scale_b_k_major=True,
        **kwargs,
    )

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        A_workspace: fx.Tensor,
        B_workspace: fx.Tensor,
        stream: fx.Stream,
    ):
        transpose_a(A, A_workspace, stream)
        transpose_b(B, B_workspace, stream)
        gemm(A_workspace, B_workspace, C, A_scale, B_scale, stream)

    return launch


def compile_blockscale_fp8_gemm_8w_3stage(
    *,
    K: int,
    M: int,
    N: int,
    out_dtype: str = "bf16",
    fold_group_size: int = 5,
    use_setprio: bool = True,
    mfma_priority: int = 1,
    interleave_width: int = 2,
    group_m: int = 4,
    waves_per_eu: int = 2,
    scale_a_k_major: bool = False,
    prefetch_before_wait: bool = False,
    interleave_mode: str = "g2s_wait6_s2r",
    priority_includes_fold: bool = False,
    pipeline_wait_count: int = 6,
    wait_delay_thunks: int = 2,
    physical_b_nn: bool = False,
):
    """Return a fixed-shape gfx950 block-scale GEMM launcher.

    The launcher signature is
    ``(A, B, C, A_scale, B_scale, stream)``.  ``A`` and ``B`` are
    contiguous row-major FP8 tensors with logical shapes ``[M, K]`` and
    ``[N, K]``.  ``C`` is BF16 or FP16 according to ``out_dtype``.
    """

    block_m = 256
    block_n = 128
    block_k = 128
    n_tiles_a = 4
    n_tiles_b = 4
    n_accums = n_tiles_a * n_tiles_b
    n_lds_rounds_a = 4
    n_lds_rounds_b = 2

    if not isinstance(K, int) or K < block_k or K % block_k != 0:
        raise ValueError(f"K must be a positive multiple of {block_k}, got {K!r}")
    if not isinstance(M, int) or M <= 0:
        raise ValueError(f"M must be a positive integer, got {M!r}")
    if not isinstance(N, int) or N <= 0:
        raise ValueError(f"N must be a positive integer, got {N!r}")
    if out_dtype not in ("bf16", "fp16"):
        raise ValueError(f"out_dtype must be 'bf16' or 'fp16', got {out_dtype!r}")
    if not isinstance(fold_group_size, int) or not 1 <= fold_group_size <= n_accums:
        raise ValueError(f"fold_group_size must be in [1, {n_accums}], got {fold_group_size!r}")
    if mfma_priority not in (1, 2, 3):
        raise ValueError(f"mfma_priority must be 1, 2, or 3, got {mfma_priority!r}")
    if interleave_width not in (1, 2):
        raise ValueError(f"interleave_width must be 1 or 2, got {interleave_width!r}")
    if interleave_mode not in ("alternate", "g2s_first", "g2s_wait6_s2r", "s2r_first"):
        raise ValueError(
            "interleave_mode must be 'alternate', 'g2s_first', "
            f"'g2s_wait6_s2r', or 's2r_first', got {interleave_mode!r}"
        )
    if pipeline_wait_count not in (0, 2, 4, 6):
        raise ValueError(f"pipeline_wait_count must be 0, 2, 4, or 6, got {pipeline_wait_count!r}")
    if wait_delay_thunks not in (0, 2, 4, 6, 8):
        raise ValueError(f"wait_delay_thunks must be 0, 2, 4, 6, or 8, got {wait_delay_thunks!r}")
    if not isinstance(group_m, int) or not 1 <= group_m <= 16:
        raise ValueError(f"group_m must be an integer in [1, 16], got {group_m!r}")
    if waves_per_eu not in (1, 2):
        raise ValueError(f"waves_per_eu must be 1 or 2, got {waves_per_eu!r}")

    k_iters = K // block_k
    main_steps = max(k_iters - 2, 0)
    loop_steps = (main_steps // 3) * 3
    remainder_steps = main_steps - loop_steps
    out_type = fx.BFloat16 if out_dtype == "bf16" else fx.Float16
    large_a = M * K > 0xFFFFFFFF
    large_b = N * K > 0xFFFFFFFF
    large_output = M * N * 2 > 0xFFFFFFFF
    if physical_b_nn and large_b:
        raise ValueError("physical_b_nn does not yet support B tensors larger than 4 GiB")

    a_lds_size = block_m * block_k
    b_lds_size = block_n * block_k

    @fx.struct
    class SharedStorage:
        A0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A2: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B2: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
    ):
        f8_ir_type = fx.Float8E4M3FN.ir_type
        m_extent = fx.Int32(M)
        n_extent = fx.Int32(N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a0, a1, a2 = lds.A0, lds.A1, lds.A2
        b0, b1, b2 = lds.B0, lds.B1, lds.B2
        a_stages = (a0, a1, a2)
        b_stages = (b0, b1, b2)

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 2
        wave_n = wave_id % 2

        num_pid_m = ceildiv(m_extent, block_m)
        num_pid_n = ceildiv(n_extent, block_n)
        tile_m, tile_n = _xcd_swizzle(num_pid_m, num_pid_n, group_m)

        if const_expr(large_a):
            a_global_offset = fx.Int32(0)
            a_total_bytes = fx.Index(M) * K
            a_base_off_bytes = fx.Index(tile_m) * block_m * K
            a_records_bytes = fx.Index(
                arith.minsi(
                    (a_total_bytes - a_base_off_bytes).ir_value(),
                    fx.Index(0xFFFFFFFF).ir_value(),
                )
            )
            gA_base = make_fp8_buffer_tensor(
                A,
                f8_ir_type,
                num_records_bytes=a_records_bytes,
                base_byte_offset=a_base_off_bytes,
            )
            gA = fx.make_view(fx.get_iter(gA_base), fx.make_layout(block_m * K, 1))
        else:
            a_global_offset = (tile_m * block_m) * K
            gA_base = make_fp8_buffer_tensor(A, f8_ir_type)
            gA = fx.make_view(fx.get_iter(gA_base), fx.make_layout(m_extent * K, 1))
        if const_expr(physical_b_nn):
            b_global_offset = tile_n * block_n
            gB_base = make_fp8_buffer_tensor(B, f8_ir_type)
            gB = fx.make_view(fx.get_iter(gB_base), fx.make_layout(n_extent * K, 1))
        elif const_expr(large_b):
            b_global_offset = fx.Int32(0)
            b_total_bytes = fx.Index(N) * K
            b_base_off_bytes = fx.Index(tile_n) * block_n * K
            b_records_bytes = fx.Index(
                arith.minsi(
                    (b_total_bytes - b_base_off_bytes).ir_value(),
                    fx.Index(0xFFFFFFFF).ir_value(),
                )
            )
            gB_base = make_fp8_buffer_tensor(
                B,
                f8_ir_type,
                num_records_bytes=b_records_bytes,
                base_byte_offset=b_base_off_bytes,
            )
            gB = fx.make_view(fx.get_iter(gB_base), fx.make_layout(block_n * K, 1))
        else:
            b_global_offset = (tile_n * block_n) * K
            gB_base = make_fp8_buffer_tensor(B, f8_ir_type)
            gB = fx.make_view(fx.get_iter(gB_base), fx.make_layout(n_extent * K, 1))
        ga_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        gb_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle(
            lane_id,
            wave_id,
            K,
            n_lds_rounds_a,
        )
        if const_expr(physical_b_nn):
            gl_off_b = compute_global_swizzle_nn(
                lane_id,
                wave_id,
                N,
                n_lds_rounds_b,
                width=block_n,
                wswz=True,
            )
        else:
            gl_off_b = compute_global_swizzle(
                lane_id,
                wave_id,
                K,
                n_lds_rounds_b,
            )
        a_g2s = G2SLoader(ga_div, gl_off_a, n_lds_rounds_a, f8_ir_type, wave_id)
        b_g2s = G2SLoader(gb_div, gl_off_b, n_lds_rounds_b, f8_ir_type, wave_id)
        a_s2r = S2RLoader(wave_m, n_tiles_a)
        if const_expr(physical_b_nn):
            b_s2r = S2RLoaderTr(
                wave_n,
                n_tiles_b,
                n_tiles_b * 16,
                inline_asm=True,
                vmcnt_hint=-1,
                n_waves=8,
                width=block_n,
                wswz=True,
            )
        else:
            b_s2r = S2RLoader(wave_n, n_tiles_b)
        mfma = _BlockScaleMfma(
            n_tiles_a,
            n_tiles_b,
            fold_group_size,
            use_setprio,
            mfma_priority,
            interleave_width,
            priority_includes_fold,
            False,
        )

        scale_a_rsrc = _buffer_ops.create_buffer_resource(
            A_scale,
            max_size=False,
            num_records_bytes=M * k_iters * 4,
        )
        scale_b_rsrc = _buffer_ops.create_buffer_resource(
            B_scale,
            max_size=False,
            num_records_bytes=ceildiv(N, block_n) * k_iters * 4,
        )

        if const_expr(large_output):
            c_total_bytes = fx.Index(M) * fx.Index(N) * 2
            c_base_off_bytes = fx.Index(tile_m) * block_m * fx.Index(N) * 2
            c_records_bytes = fx.Index(
                arith.minsi(
                    (c_total_bytes - c_base_off_bytes).ir_value(),
                    fx.Index(0xFFFFFFFF).ir_value(),
                )
            )
            c_rsrc = _buffer_ops.create_buffer_resource(
                C,
                max_size=False,
                num_records_bytes=c_records_bytes,
                base_byte_offset=c_base_off_bytes,
            )
        else:
            c_rsrc = _buffer_ops.create_buffer_resource(
                C,
                max_size=False,
                num_records_bytes=M * N * 2,
            )

        wave_tile_m = n_tiles_a * 16
        wave_tile_n = n_tiles_b * 16
        base_row = tile_m * block_m + wave_m * wave_tile_m
        base_col = tile_n * block_n + wave_n * wave_tile_n
        lane_row = (lane_id // 16) * 4

        def _load_combined_scales(k_block):
            k_block_i32 = fx.Int32(k_block)
            scale_b_idx = fx.Int32(tile_n) * fx.Int32(k_iters) + k_block_i32
            scale_b = fx.Int32(
                _buffer_ops.buffer_load(
                    scale_b_rsrc,
                    scale_b_idx,
                    vec_width=1,
                    is_scalar=True,
                )
            ).bitcast(fx.Float32)
            scale_b_vec = fx.Vector.filled(4, scale_b, fx.Float32)

            combined = []
            for tile in range_constexpr(n_tiles_a):
                row = fx.Int32(base_row + tile * 16 + lane_row)
                if const_expr(scale_a_k_major):
                    scale_a_idx = k_block_i32 * fx.Int32(M) + row
                    scale_a = fx.Vector(
                        _buffer_ops.buffer_load(
                            scale_a_rsrc,
                            scale_a_idx,
                            vec_width=4,
                            dtype=_T.f32,
                        )
                    )
                else:
                    scale_a_values = []
                    for element in range_constexpr(4):
                        scale_a_values.append(
                            fx.Float32(
                                _buffer_ops.buffer_load(
                                    scale_a_rsrc,
                                    (row + element) * fx.Int32(k_iters) + k_block_i32,
                                    vec_width=1,
                                    dtype=_T.f32,
                                )
                            )
                        )
                    scale_a = fx.Vector.from_elements(scale_a_values, fx.Float32)
                combined_scale = scale_a * scale_b_vec
                combined.append([combined_scale for _ in range_constexpr(n_tiles_b)])
            return combined

        def _store_output(c_frag):
            for tile_i in range_constexpr(n_tiles_a):
                row = fx.Int32(base_row + tile_i * 16 + lane_row)
                for tile_j in range_constexpr(n_tiles_b):
                    col = fx.Int32(base_col + tile_j * 16 + lane_id % 16)
                    vec_f32 = fx.Vector(c_frag[mfma.idx(tile_i, tile_j)])
                    col_valid = col < n_extent
                    for element in range_constexpr(4):
                        out_row = row + element
                        valid = (out_row < m_extent) & col_valid
                        if const_expr(large_output):
                            row_local = out_row - fx.Int32(tile_m * block_m)
                            c_idx = row_local * n_extent + col
                            _buffer_ops.buffer_store(
                                vec_f32[element].to(out_type),
                                c_rsrc,
                                c_idx,
                                mask=valid,
                            )
                        else:
                            c_idx = out_row * n_extent + col
                            oob = m_extent * n_extent
                            safe_idx = arith.select(valid, c_idx, oob)
                            _buffer_ops.buffer_store(vec_f32[element].to(out_type), c_rsrc, safe_idx)

        def _one_step(
            k_block,
            a_frag,
            b_frag,
            c_frag,
            future_a,
            future_b,
            next_a,
            next_b,
        ):
            future_k = (fx.Int32(k_block) + fx.Int32(2)) * fx.Int32(block_k)
            g2s_thunks = _g2s_thunks(
                a_g2s,
                future_a,
                fx.Int32(a_global_offset) + future_k,
                n_lds_rounds_a,
            )
            g2s_thunks += _g2s_thunks(
                b_g2s,
                future_b,
                fx.Int32(b_global_offset) + future_k * fx.Int32(N if physical_b_nn else 1),
                n_lds_rounds_b,
            )
            next_a_halves = [[None, None] for _ in range(n_tiles_a)]
            s2r_thunks = _s2r_thunks(a_s2r, next_a, next_a_halves, n_tiles_a)
            if const_expr(physical_b_nn):
                next_b_calls = [None for _ in range(n_tiles_b)]
                for tile in range_constexpr(n_tiles_b):

                    def load(tile=tile):
                        next_b_calls[tile] = b_s2r._issue_one(next_b, tile)

                    s2r_thunks.append(load)
            else:
                next_b_halves = [[None, None] for _ in range(n_tiles_b)]
                s2r_thunks += _s2r_thunks(b_s2r, next_b, next_b_halves, n_tiles_b)

            if const_expr(prefetch_before_wait):
                # Six new G2S requests join the six requests from the previous
                # step. vmcnt(6) drains the older stage while leaving this
                # step's future stage in flight during the current MFMA group.
                for thunk in g2s_thunks:
                    thunk()
                wait_barrier(len(g2s_thunks))
                interleave = s2r_thunks
            else:
                if const_expr(interleave_mode == "g2s_wait6_s2r"):

                    def wait_for_read_stage():
                        rocdl.sched_barrier(0)
                        wait_barrier(pipeline_wait_count)
                        rocdl.sched_barrier(0)

                    def no_op():
                        pass

                    interleave = (
                        g2s_thunks
                        + [no_op for _ in range_constexpr(wait_delay_thunks)]
                        + [wait_for_read_stage]
                        + s2r_thunks
                    )
                elif const_expr(interleave_mode == "g2s_first"):
                    wait_barrier(0)
                    interleave = g2s_thunks + s2r_thunks
                elif const_expr(interleave_mode == "s2r_first"):
                    wait_barrier(0)
                    interleave = s2r_thunks + g2s_thunks
                else:
                    wait_barrier(0)
                    interleave = []
                    for index in range_constexpr(max(len(g2s_thunks), len(s2r_thunks))):
                        if index < len(g2s_thunks):
                            interleave.append(g2s_thunks[index])
                        if index < len(s2r_thunks):
                            interleave.append(s2r_thunks[index])

            c_frag = mfma.fold(
                a_frag,
                b_frag,
                c_frag,
                _load_combined_scales(k_block),
                interleave=interleave,
            )
            next_a_frag = [pack_i32x4_i32x8(halves[0], halves[1]) for halves in next_a_halves]
            if const_expr(physical_b_nn):
                S2RLoaderTr._wait_lgkmcnt(0)
                next_b_frag = [S2RLoaderTr._assemble(calls) for calls in next_b_calls]
            else:
                next_b_frag = [pack_i32x4_i32x8(halves[0], halves[1]) for halves in next_b_halves]
            return next_a_frag, next_b_frag, c_frag

        def _finish_two_tiles(
            k_block,
            a_frag,
            b_frag,
            c_frag,
            last_a_stage,
            last_b_stage,
        ):
            wait_barrier(0)
            last_a_frag = a_s2r.load(last_a_stage)
            last_b_frag = b_s2r.load(last_b_stage)
            c_frag = mfma.fold(
                a_frag,
                b_frag,
                c_frag,
                _load_combined_scales(k_block),
            )
            c_frag = mfma.fold(
                last_a_frag,
                last_b_frag,
                c_frag,
                _load_combined_scales(k_block + 1),
            )
            _store_output(c_frag)

        c_frag = [mfma.zero_value] * n_accums

        # Prologue: stage 0 holds tile 0; stage 1 holds tile 1 when present.
        a_g2s.load(a0, a_global_offset)
        b_g2s.load(b0, b_global_offset)
        if const_expr(k_iters > 1):
            a_g2s.load(a1, a_global_offset + block_k)
            b_g2s.load(b1, b_global_offset + block_k * (N if physical_b_nn else 1))
        wait_barrier(0)
        a_frag = a_s2r.load(a0)
        b_frag = b_s2r.load(b0)

        if const_expr(k_iters == 1):
            c_frag = mfma.fold(
                a_frag,
                b_frag,
                c_frag,
                _load_combined_scales(0),
            )
            _store_output(c_frag)
        else:
            init_state = _flat_frag(a_frag) + _flat_frag(b_frag) + _flat_accs((c_frag,))

            # Three static rotations restore the physical stage mapping at each
            # SCF boundary: (0,1,2), (1,2,0), then (2,0,1).
            if const_expr(loop_steps > 0):
                for kk, state in range(
                    fx.Int64(0),
                    fx.Int64(loop_steps),
                    fx.Int64(3),
                    init=init_state,
                ):
                    offset = 0
                    a_loop = list(state[offset : offset + n_tiles_a])
                    offset += n_tiles_a
                    b_loop = list(state[offset : offset + n_tiles_b])
                    offset += n_tiles_b
                    c_loop = list(state[offset : offset + n_accums])

                    a_loop, b_loop, c_loop = _one_step(
                        kk,
                        a_loop,
                        b_loop,
                        c_loop,
                        a2,
                        b2,
                        a1,
                        b1,
                    )
                    a_loop, b_loop, c_loop = _one_step(
                        kk + 1,
                        a_loop,
                        b_loop,
                        c_loop,
                        a0,
                        b0,
                        a2,
                        b2,
                    )
                    a_loop, b_loop, c_loop = _one_step(
                        kk + 2,
                        a_loop,
                        b_loop,
                        c_loop,
                        a1,
                        b1,
                        a0,
                        b0,
                    )
                    loop_results = yield (_flat_frag(a_loop) + _flat_frag(b_loop) + _flat_accs((c_loop,)))

                offset = 0
                a_frag = list(loop_results[offset : offset + n_tiles_a])
                offset += n_tiles_a
                b_frag = list(loop_results[offset : offset + n_tiles_b])
                offset += n_tiles_b
                c_frag = list(loop_results[offset : offset + n_accums])

            for tail_step in range_constexpr(remainder_steps):
                stage = tail_step % 3
                a_frag, b_frag, c_frag = _one_step(
                    loop_steps + tail_step,
                    a_frag,
                    b_frag,
                    c_frag,
                    a_stages[(stage + 2) % 3],
                    b_stages[(stage + 2) % 3],
                    a_stages[(stage + 1) % 3],
                    b_stages[(stage + 1) % 3],
                )

            last_stage = (remainder_steps + 1) % 3
            _finish_two_tiles(
                k_iters - 2,
                a_frag,
                b_frag,
                c_frag,
                a_stages[last_stage],
                b_stages[last_stage],
            )

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(M, block_m) * ceildiv(N, block_n)
        kernel_gemm(
            A,
            B,
            C,
            A_scale,
            B_scale,
            value_attrs=make_value_attrs(waves_per_eu, 0, "512,512"),
        ).launch(
            grid=(grid_x, 1, 1),
            block=(512, 1, 1),
            stream=stream,
        )

    return launch_gemm


def compile_blockscale_fp8_gemm_nn_fused_8w(*, M: int, N: int, K: int, **kwargs):
    """Compile physical NN GEMM with the B transpose fused into its LDS load."""

    return compile_blockscale_fp8_gemm_8w_3stage(
        K=K,
        M=M,
        N=N,
        scale_a_k_major=False,
        physical_b_nn=True,
        **kwargs,
    )


_compiled_cache = {}
_autotune_cache = {}
_MAX_AUTOTUNE_CONFIGS = 8
_MAX_BUFFER_BYTES = 0xFFFFFFFF
_DEEP_K_BLOCKS = 256


def _shape_supported(M: int, N: int, K: int) -> bool:
    return (
        str(get_rocm_arch()).startswith("gfx95")
        and M >= 128
        and N >= 128
        and K >= 128
        and M % 128 == 0
        and N % 128 == 0
        and K % 128 == 0
    )


def _buffers_fit(*byte_counts: int) -> bool:
    return all(0 < count <= _MAX_BUFFER_BYTES for count in byte_counts)


def flydsl_blockwise_4wave_dgrad_supported(M: int, N: int, K: int) -> bool:
    """Check ``grad[M,N] @ weight[N,K] -> dA[M,K]`` support."""

    return _shape_supported(M, K, N) and _buffers_fit(N * K)


def flydsl_blockwise_4wave_forward_supported(M: int, N: int, K: int) -> bool:
    """Check ``activation[M,K] @ weight[N,K].T`` support."""

    return (
        str(get_rocm_arch()).startswith("gfx95")
        and M >= 128
        and N >= 16
        and K >= 128
        and M % 128 == 0
        and N % 16 == 0
        and K % 128 == 0
    )


def flydsl_blockwise_4wave_wgrad_supported(
    M: int,
    N: int,
    K: int,
    *,
    require_workspace: bool = True,
) -> bool:
    """Check ``grad[M,N].T @ activation[M,K] -> dW[N,K]`` support."""

    return (
        str(get_rocm_arch()).startswith("gfx95")
        and M >= 128
        and N >= 16
        and K >= 128
        and M % 128 == 0
        and N % 16 == 0
        and K % 128 == 0
        and (not require_workspace or _buffers_fit(M * N, M * K))
    )


def flydsl_blockwise_forward_scale_a_k_major(K: int) -> bool:
    """Use the K-major scale producer until the contraction enters the deep-K regime."""

    return K < _DEEP_K_BLOCKS * 128


def _autotune_configs(M: int, N: int, K: int) -> list[dict]:
    """Build a small schedule pool from kernel constraints, without model-shape rules."""

    configs = [
        {
            "family": "4wave",
            "block_m": 128,
            "fold_group_size": 4,
            "k_loop_unroll": 2,
        }
    ]
    if K >= 6 * 128:
        configs.extend(
            [
                {
                    "family": "4wave",
                    "block_m": 128,
                    "fold_group_size": 4,
                    "k_loop_unroll": 6,
                },
                {
                    "family": "4wave",
                    "block_m": 128,
                    "fold_group_size": 6,
                    "k_loop_unroll": 6,
                },
            ]
        )
    if M >= 192:
        configs.append(
            {
                "family": "4wave",
                "block_m": 192,
                "fold_group_size": 4,
                "k_loop_unroll": 2,
            }
        )
    if N >= 128 and K >= 3 * 128:
        deep_k = K >= _DEEP_K_BLOCKS * 128
        configs.extend(
            [
                {
                    "family": "8wave",
                    "fold_group_size": 4,
                    "interleave_width": 1,
                    "wait_delay_thunks": 0,
                    "group_m": 4,
                },
                {
                    "family": "8wave",
                    "fold_group_size": 6,
                    "interleave_width": 1,
                    "wait_delay_thunks": 0,
                    "group_m": 4,
                },
                {
                    "family": "8wave",
                    "fold_group_size": 12,
                    "interleave_width": 1,
                    "wait_delay_thunks": 0,
                    "group_m": 4,
                },
                {
                    "family": "8wave",
                    "fold_group_size": 5 if deep_k else 8,
                    "interleave_width": 2 if deep_k else 1,
                    "wait_delay_thunks": 8 if deep_k else 0,
                    "group_m": 4,
                },
            ]
        )
    if len(configs) > _MAX_AUTOTUNE_CONFIGS:
        raise AssertionError(f"autotune generated {len(configs)} configs; maximum is {_MAX_AUTOTUNE_CONFIGS}")
    return configs


def _wgrad_autotune_configs(output_rows: int) -> list[dict]:
    configs = [
        {
            "family": "4wave",
            "block_m": 128,
            "fold_group_size": 4,
            "k_loop_unroll": 2,
        },
        {
            "family": "4wave",
            "block_m": 128,
            "fold_group_size": 6,
            "k_loop_unroll": 2,
        },
    ]
    if output_rows >= 192:
        configs.append(
            {
                "family": "4wave",
                "block_m": 192,
                "fold_group_size": 4,
                "k_loop_unroll": 2,
            }
        )
    return configs


def _config_key(config: dict) -> tuple:
    return tuple(sorted(config.items()))


def _get_compiled(key: tuple, build, args: tuple):
    entry = _compiled_cache.get(key)
    if entry is None:
        raw = build()
        entry = raw, flyc.compile(raw, *args)
        _compiled_cache[key] = entry
    return entry


def _run_compiled(entry, args: tuple) -> None:
    raw, compiled = entry
    (raw if torch.cuda.is_current_stream_capturing() else compiled)(*args)


def _validation_indices(out: torch.Tensor) -> torch.Tensor | None:
    if out.numel() <= 65536:
        return None
    count = 8192
    indices = torch.arange(count, device=out.device, dtype=torch.int64)
    return indices * (out.numel() - 1) // (count - 1)


def _output_sample(out: torch.Tensor, indices: torch.Tensor | None) -> torch.Tensor | None:
    flat = out.reshape(-1)
    sample = flat if indices is None else flat.index_select(0, indices)
    sample = sample.float()
    if not torch.isfinite(sample).all().item():
        return None
    return sample.clone()


def _autotune_and_run(key: tuple, candidates: list[tuple], out: torch.Tensor) -> None:
    """Validate at most eight candidates against the 4-wave baseline and cache the fastest."""

    if not candidates or len(candidates) > _MAX_AUTOTUNE_CONFIGS:
        raise AssertionError(f"expected 1..{_MAX_AUTOTUNE_CONFIGS} autotune candidates")

    selected_key = _autotune_cache.get(key)
    if torch.cuda.is_current_stream_capturing():
        if selected_key is None:
            raise RuntimeError("blockwise FP8 GEMM must be autotuned before CUDA graph capture")
        for config, _, args in candidates:
            if _config_key(config) == selected_key:
                entry = _compiled_cache.get(key + (selected_key,))
                if entry is None:
                    raise RuntimeError("blockwise FP8 GEMM must be compiled before CUDA graph capture")
                _run_compiled(entry, args)
                return
        raise RuntimeError(f"cached blockwise FP8 GEMM config is unavailable for {key}")

    if selected_key is not None:
        for config, build, args in candidates:
            if _config_key(config) == selected_key:
                entry = _get_compiled(key + (selected_key,), build, args)
                _run_compiled(entry, args)
                return
        _autotune_cache.pop(key)

    indices = _validation_indices(out)
    baseline_config, baseline_build, baseline_args = candidates[0]
    baseline_key = _config_key(baseline_config)
    try:
        baseline_entry = _get_compiled(key + (baseline_key,), baseline_build, baseline_args)
        _, baseline_compiled = baseline_entry
        baseline_compiled(*baseline_args)
        torch.cuda.synchronize()
        reference = _output_sample(out, indices)
        if reference is None:
            raise RuntimeError("non-finite output")
        reference_norm = float((reference * reference).sum().item()) or 1.0
        best_time = _robust_time(baseline_compiled, baseline_args, warmup=2, reps=3, iters=20)
    except Exception as error:
        raise RuntimeError(f"blockwise FP8 GEMM fallback failed for {key}: {error!r}") from error

    best = baseline_key, baseline_entry, baseline_args
    for config, build, args in candidates[1:]:
        config_key = _config_key(config)
        try:
            entry = _get_compiled(key + (config_key,), build, args)
            _, compiled = entry
            compiled(*args)
            torch.cuda.synchronize()
            sample = _output_sample(out, indices)
            if sample is None:
                continue
            error = float(((sample - reference) ** 2).sum().item()) / reference_norm
            if error >= (2e-2) ** 2:
                continue
            elapsed = _robust_time(compiled, args, warmup=2, reps=3, iters=20)
        except Exception:
            continue
        if elapsed < best_time * 0.985:
            best = config_key, entry, args
            best_time = elapsed

    selected_key, entry, args = best
    _autotune_cache[key] = selected_key
    _run_compiled(entry, args)


def _out_dtype_name(out_dtype: torch.dtype) -> str:
    if out_dtype == torch.bfloat16:
        return "bf16"
    if out_dtype == torch.float16:
        return "fp16"
    raise TypeError(f"blockwise FP8 GEMM output must be bf16 or fp16, got {out_dtype}")


def gemm_fp8_blockwise_forward(
    activation_fp8: torch.Tensor,
    weight_fp8: torch.Tensor,
    activation_scale_inv: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    scale_a_k_major: bool | None = None,
) -> torch.Tensor:
    """Run blockwise FP8 forward GEMM and autotune its schedule."""

    M, K = activation_fp8.shape
    N, K_weight = weight_fp8.shape
    if K != K_weight or not flydsl_blockwise_4wave_forward_supported(M, N, K):
        raise ValueError(
            f"unsupported forward shape: activation={activation_fp8.shape}, weight={weight_fp8.shape}"
        )
    if not activation_fp8.is_contiguous() or not weight_fp8.is_contiguous():
        raise ValueError("forward inputs must be contiguous")

    row_major_scale_shape = (M, K // 128)
    k_major_scale_shape = (K // 128, M)
    scale_shape = tuple(activation_scale_inv.shape)
    if scale_a_k_major is None:
        if scale_shape == k_major_scale_shape and scale_shape != row_major_scale_shape:
            scale_a_k_major = True
        elif scale_shape == row_major_scale_shape and scale_shape != k_major_scale_shape:
            scale_a_k_major = False
        elif scale_shape == row_major_scale_shape:
            scale_a_k_major = flydsl_blockwise_forward_scale_a_k_major(K)
        else:
            raise ValueError(f"invalid forward activation-scale shape {activation_scale_inv.shape}")
    else:
        expected_scale_shape = k_major_scale_shape if scale_a_k_major else row_major_scale_shape
        if scale_shape != expected_scale_shape:
            raise ValueError(
                f"invalid forward activation-scale shape {activation_scale_inv.shape}; "
                f"expected {expected_scale_shape} for scale_a_k_major={scale_a_k_major}"
            )
    if tuple(weight_scale_inv.shape) != (ceildiv(N, 128), K // 128):
        raise ValueError(f"invalid forward weight-scale shape {weight_scale_inv.shape}")

    out = torch.empty((M, N), dtype=out_dtype, device=activation_fp8.device)
    stream = torch.cuda.current_stream(activation_fp8.device)
    args = (
        activation_fp8.view(torch.int8),
        weight_fp8.view(torch.int8),
        out,
        activation_scale_inv,
        weight_scale_inv,
        stream,
    )
    out_dtype_name = _out_dtype_name(out_dtype)
    candidates = []
    for config in _autotune_configs(M, N, K):
        if config["family"] == "8wave":
            build = lambda config=config: compile_blockscale_fp8_gemm_8w_3stage(
                K=K,
                M=M,
                N=N,
                out_dtype=out_dtype_name,
                fold_group_size=config["fold_group_size"],
                interleave_width=config["interleave_width"],
                wait_delay_thunks=config["wait_delay_thunks"],
                scale_a_k_major=scale_a_k_major,
                group_m=config["group_m"],
            )
        else:
            build = lambda config=config: compile_blockscale_fp8_gemm_4w(
                K=K,
                M=M,
                N=N,
                BLOCK_M=config["block_m"],
                out_dtype=out_dtype_name,
                scale_a_k_major=scale_a_k_major,
                scale_b_k_major=False,
                fold_group_size=config["fold_group_size"],
                k_loop_unroll=config["k_loop_unroll"],
            )
        candidates.append((config, build, args))

    key = ("forward", str(get_rocm_arch()), M, N, K, out_dtype_name, scale_a_k_major)
    _autotune_and_run(key, candidates, out)
    return out


def gemm_fp8_blockwise_dgrad(
    grad_out_fp8: torch.Tensor,
    weight_fp8: torch.Tensor,
    grad_out_scale_inv: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run physical NN dgrad and autotune its schedule."""

    M, N = grad_out_fp8.shape
    N_weight, K = weight_fp8.shape
    if N != N_weight or not flydsl_blockwise_4wave_dgrad_supported(M, N, K):
        raise ValueError(f"unsupported dgrad shape: grad={grad_out_fp8.shape}, weight={weight_fp8.shape}")
    if not grad_out_fp8.is_contiguous() or not weight_fp8.is_contiguous():
        raise ValueError("dgrad inputs must be contiguous")
    if tuple(grad_out_scale_inv.shape) != (M, N // 128):
        raise ValueError(f"invalid dgrad A-scale shape {grad_out_scale_inv.shape}")
    if tuple(weight_scale_inv.shape) != (N // 128, K // 128):
        raise ValueError(f"invalid dgrad B-scale shape {weight_scale_inv.shape}")

    out = torch.empty((M, K), dtype=out_dtype, device=grad_out_fp8.device)
    stream = torch.cuda.current_stream(grad_out_fp8.device)
    out_dtype_name = _out_dtype_name(out_dtype)
    configs = _autotune_configs(M, K, N)
    key = ("dgrad", str(get_rocm_arch()), M, K, N, out_dtype_name)
    selected_key = _autotune_cache.get(key)
    if selected_key not in {_config_key(config) for config in configs}:
        _autotune_cache.pop(key, None)
        selected_key = None
    selected = dict(selected_key or ())
    need_4wave = not selected or selected.get("family") == "4wave"
    need_8wave = not selected or selected.get("family") == "8wave"
    weight_t = torch.empty((K, N), dtype=weight_fp8.dtype, device=weight_fp8.device) if need_4wave else None
    weight_scale_k_major = (
        weight_scale_inv.T.contiguous()
        if need_8wave and any(config["family"] == "8wave" for config in configs)
        else None
    )
    candidates = []
    for config in configs:
        if config["family"] == "8wave":
            args = (
                grad_out_fp8.view(torch.int8),
                weight_fp8.view(torch.int8),
                out,
                grad_out_scale_inv,
                weight_scale_k_major,
                stream,
            )
            build = lambda config=config: compile_blockscale_fp8_gemm_nn_fused_8w(
                M=M,
                N=K,
                K=N,
                out_dtype=out_dtype_name,
                fold_group_size=config["fold_group_size"],
                interleave_width=config["interleave_width"],
                wait_delay_thunks=config["wait_delay_thunks"],
                group_m=config["group_m"],
            )
        else:
            args = (
                grad_out_fp8.view(torch.int8),
                weight_fp8.view(torch.int8),
                out,
                grad_out_scale_inv,
                weight_scale_inv,
                weight_t.view(torch.int8) if weight_t is not None else None,
                stream,
            )
            build = lambda config=config: compile_blockscale_fp8_gemm_nn_physical_4w(
                M=M,
                N=K,
                K=N,
                BLOCK_M=config["block_m"],
                out_dtype=out_dtype_name,
                fold_group_size=config["fold_group_size"],
                k_loop_unroll=config["k_loop_unroll"],
            )
        candidates.append((config, build, args))

    _autotune_and_run(key, candidates, out)
    return out


def _gemm_fp8_blockwise_wgrad(
    activation_fp8: torch.Tensor,
    grad_out_fp8: torch.Tensor,
    activation_scale_inv: torch.Tensor,
    grad_out_scale_inv: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    normalized: bool,
) -> torch.Tensor:
    M, K = activation_fp8.shape
    M_grad, N = grad_out_fp8.shape
    if M != M_grad or not flydsl_blockwise_4wave_wgrad_supported(
        M,
        N,
        K,
        require_workspace=not normalized,
    ):
        raise ValueError(
            f"unsupported wgrad shape: activation={activation_fp8.shape}, grad={grad_out_fp8.shape}"
        )
    if tuple(activation_scale_inv.shape) != (M // 128, K):
        raise ValueError(f"invalid wgrad activation-scale shape {activation_scale_inv.shape}")
    if tuple(grad_out_scale_inv.shape) != (M // 128, N):
        raise ValueError(f"invalid wgrad gradient-scale shape {grad_out_scale_inv.shape}")

    out = torch.empty((N, K), dtype=out_dtype, device=activation_fp8.device)
    stream = torch.cuda.current_stream(activation_fp8.device)
    out_dtype_name = _out_dtype_name(out_dtype)
    if normalized:
        activation_t = activation_fp8.T
        grad_t = grad_out_fp8.T
        if not activation_t.is_contiguous() or not grad_t.is_contiguous():
            raise ValueError("normalized wgrad requires transpose views over contiguous storage")
        args = (
            grad_t.view(torch.int8),
            activation_t.view(torch.int8),
            out,
            grad_out_scale_inv,
            activation_scale_inv,
            stream,
        )
    else:
        if not activation_fp8.is_contiguous() or not grad_out_fp8.is_contiguous():
            raise ValueError("physical wgrad inputs must be contiguous")
        grad_t = torch.empty((N, M), dtype=grad_out_fp8.dtype, device=grad_out_fp8.device)
        activation_t = torch.empty((K, M), dtype=activation_fp8.dtype, device=activation_fp8.device)
        args = (
            grad_out_fp8.view(torch.int8),
            activation_fp8.view(torch.int8),
            out,
            grad_out_scale_inv,
            activation_scale_inv,
            grad_t.view(torch.int8),
            activation_t.view(torch.int8),
            stream,
        )

    candidates = []
    for config in _wgrad_autotune_configs(N):
        if normalized:
            build = lambda config=config: compile_blockscale_fp8_gemm_tn_4w(
                K=M,
                M=N,
                N=K,
                BLOCK_M=config["block_m"],
                out_dtype=out_dtype_name,
                scale_a_k_major=True,
                scale_b_k_major=True,
                fold_group_size=config["fold_group_size"],
                k_loop_unroll=config["k_loop_unroll"],
            )
        else:
            build = lambda config=config: compile_blockscale_fp8_gemm_tn_physical_4w(
                M=N,
                N=K,
                K=M,
                BLOCK_M=config["block_m"],
                out_dtype=out_dtype_name,
                fold_group_size=config["fold_group_size"],
                k_loop_unroll=config["k_loop_unroll"],
            )
        candidates.append((config, build, args))

    direction = "wgrad_normalized" if normalized else "wgrad"
    key = (direction, str(get_rocm_arch()), N, K, M, out_dtype_name)
    _autotune_and_run(key, candidates, out)
    return out


def gemm_fp8_blockwise_wgrad(
    activation_fp8: torch.Tensor,
    grad_out_fp8: torch.Tensor,
    activation_scale_inv: torch.Tensor,
    grad_out_scale_inv: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run TN wgrad using the available operand storage layout."""

    normalized = activation_fp8.T.is_contiguous() and grad_out_fp8.T.is_contiguous()
    return _gemm_fp8_blockwise_wgrad(
        activation_fp8,
        grad_out_fp8,
        activation_scale_inv,
        grad_out_scale_inv,
        out_dtype,
        normalized=normalized,
    )


__all__ = [
    "compile_blockscale_fp8_gemm_4w",
    "compile_blockscale_fp8_gemm_8w_3stage",
    "compile_blockscale_fp8_gemm_nn_4w",
    "compile_blockscale_fp8_gemm_nn_fused_8w",
    "compile_blockscale_fp8_gemm_nn_physical_4w",
    "compile_blockscale_fp8_gemm_tn_4w",
    "compile_blockscale_fp8_gemm_tn_physical_4w",
    "compile_fp8_transpose_32x32",
    "flydsl_blockwise_4wave_dgrad_supported",
    "flydsl_blockwise_4wave_forward_supported",
    "flydsl_blockwise_4wave_wgrad_supported",
    "flydsl_blockwise_forward_scale_a_k_major",
    "gemm_fp8_blockwise_dgrad",
    "gemm_fp8_blockwise_forward",
    "gemm_fp8_blockwise_wgrad",
]
