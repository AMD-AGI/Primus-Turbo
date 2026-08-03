# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""gfx950 8-wave, three-stage block-scaled FP8 GEMM proof of concept.

Computes ``C[M, N] = A[M, K] @ B[N, K].T`` from contiguous row-major FP8
operands.  One 512-thread workgroup owns a 256x128 output tile.  Its eight
waves form a 4x2 grid, and each wave owns one contiguous 64x64 tile represented
by sixteen 16x16x128 narrow-MFMA fragments.

Each K128 partial is multiplied by its block scales before it is accumulated.
The default A-scale layout is ``[M, K / 128]``; ``scale_a_k_major=True`` selects
``[K / 128, M]``. B scales always use ``[ceil(N / 128), K / 128]``.
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
    make_fp8_buffer_tensor,
)
from primus_turbo.flydsl.utils.gemm_helper import (
    G2SLoader,
    S2RLoader,
    ceildiv,
    compute_global_swizzle,
    pack_i32x4_i32x8,
    wait_barrier,
)


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
):
    """Return a fixed-shape gfx950 block-scale GEMM launcher.

    The launcher signature is
    ``(A, B, C, A_scale, B_scale, c_m, c_n, stream)``.  ``A`` and ``B`` are
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
        c_m: fx.Int32,
        c_n: fx.Int32,
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
        if const_expr(large_b):
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
        gl_off_b = compute_global_swizzle(
            lane_id,
            wave_id,
            K,
            n_lds_rounds_b,
        )
        a_g2s = G2SLoader(ga_div, gl_off_a, n_lds_rounds_a, f8_ir_type, wave_id)
        b_g2s = G2SLoader(gb_div, gl_off_b, n_lds_rounds_b, f8_ir_type, wave_id)
        a_s2r = S2RLoader(wave_m, n_tiles_a)
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
                fx.Int32(b_global_offset) + future_k,
                n_lds_rounds_b,
            )
            next_a_halves = [[None, None] for _ in range(n_tiles_a)]
            next_b_halves = [[None, None] for _ in range(n_tiles_b)]
            s2r_thunks = _s2r_thunks(a_s2r, next_a, next_a_halves, n_tiles_a)
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
            b_g2s.load(b1, b_global_offset + block_k)
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
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(M, block_m) * ceildiv(N, block_n)
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
                "rocdl.flat_work_group_size": "512,512",
            },
        ).launch(
            grid=(grid_x, 1, 1),
            block=(512, 1, 1),
            stream=stream,
        )

    return launch_gemm


__all__ = ["compile_blockscale_fp8_gemm_8w_3stage"]
