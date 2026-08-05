###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL); see LICENSE-APACHE for the Apache-2.0 terms.
#
# See LICENSE for license information.
###############################################################################

"""gfx950 4-wave block-scaled FP8 forward and backward GEMM.

Computes ``C[M, N] = A[M, K] @ B[N, K].T`` for row-major FP8 A and B.
The FP32 scales use layouts ``A_scale[K / 128, M]`` and
``B_scale[ceil(N / 128), K / 128]``.  Every K=128 MFMA result is scaled
before it is accumulated into the vector-register global accumulator.

Direction wrappers cover normalized NN/dgrad and TN/wgrad. Physical-layout
wrappers prepend a tiled FP8 transpose and consume the original scale layouts
directly, so callers may pass ``B[K,N]`` for NN or ``A[K,M], B[K,N]`` for TN.

One 256-thread workgroup computes a 128x128x128 or 192x128x128
tile.  The four waves use the same 2M x 2N ownership as
``fp8_gemm_4wave``: each wave owns four fragments, equivalent to a 64x64
or 96x64 output tile.  A and B use eight LDS buffers (cur/next x
half0/half1) and a two-step rolled K pipeline. BLOCK_M=128 supports
16x16x128 and 32x32x64 MFMA forms; BLOCK_M=192 uses 16x16x128.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr import buffer_ops as _buffer_ops
from flydsl.expr.typing import T as _T

from primus_turbo.flydsl.gemm.blockscale_fp8_gemm.utils import (
    _BlockScaleMfma,
    _flat_accs,
    _flat_frag,
    _g2s_thunks,
    _s2r_thunks,
    _xcd_swizzle,
    divmod,
    make_fp8_buffer_tensor,
)
from primus_turbo.flydsl.utils.gemm_helper import (
    G2SLoader,
    S2RLoader,
    ceildiv,
    compute_global_swizzle,
    pack_i32x4_i32x8,
    swizzle_128,
    wait_barrier,
)


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
    M: int | None = None,
    N: int | None = None,
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
    """Return a launcher for gfx950 forward-NT block-scaled FP8 GEMM.

    The returned launcher has arguments
    ``(A, B, C, A_scale, B_scale, M, N, stream)``.  A and B are contiguous
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
    if (M is None) != (N is None):
        raise ValueError("M and N must either both be provided or both be omitted")
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
    mn_aligned = M is not None and M % block_m == 0 and N % block_n == 0
    large_a = M is not None and M * K > 0xFFFFFFFF
    large_b = N is not None and N * K > 0xFFFFFFFF
    large_output = M is not None and N is not None and M * N * 2 > 0xFFFFFFFF

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
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        f8_ir_type = fx.Float8E4M3FN.ir_type
        m_extent = fx.Int32(M) if M is not None else c_m
        n_extent = fx.Int32(N) if N is not None else c_n

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
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        if const_expr(M is not None):
            grid_x = ceildiv(M, block_m) * ceildiv(N, block_n)
        else:
            grid_x = ceildiv(c_m, block_m) * ceildiv(c_n, block_n)
        kernel_gemm(
            A,
            B,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": "256,256",
            },
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

    from primus_turbo.flydsl.gemm.blockscale_fp8_gemm.utils import compile_fp8_transpose_32x32

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
        gemm(A, B_workspace, C, A_scale, B_scale, fx.Int32(M), fx.Int32(N), stream)

    return launch


def compile_blockscale_fp8_gemm_tn_physical_4w(*, M: int, N: int, K: int, **kwargs):
    """Compile physical TN ``A[K,M].T @ B[K,N]`` plus both transposes.

    The returned launcher takes
    ``(A, B, C, A_scale, B_scale, A_workspace, B_workspace, stream)``.
    Workspaces are contiguous FP8 tensors with shapes ``[M,K]`` and ``[N,K]``.
    Scales remain in their physical ``[K/128,M]`` and ``[K/128,N]`` layouts.
    """

    from primus_turbo.flydsl.gemm.blockscale_fp8_gemm.utils import compile_fp8_transpose_32x32

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
        gemm(A_workspace, B_workspace, C, A_scale, B_scale, fx.Int32(M), fx.Int32(N), stream)

    return launch


__all__ = [
    "compile_blockscale_fp8_gemm_4w",
    "compile_blockscale_fp8_gemm_nn_4w",
    "compile_blockscale_fp8_gemm_nn_physical_4w",
    "compile_blockscale_fp8_gemm_tn_4w",
    "compile_blockscale_fp8_gemm_tn_physical_4w",
]
