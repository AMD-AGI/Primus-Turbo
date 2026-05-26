###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""C9 wrappers: 32x32x64 mfma + corresponding StoreC32.

Clone of kernels.fp8_gemm_utils.Mfma16x16x128 / StoreC adapted to
mfma_scale_f32_32x32x64_f8f6f4 (first-class supported intrinsic per
FlyDSL/lib/Dialect/FlyROCDL/CDNA4/MmaAtom.cpp:157-160 and 251-256).

Per-lane layout (decoded from cdna4::getThrValLayoutC, MmaAtom.cpp:32-41):
  - GroupM = 64/32 = 2     -> lane row-chunk = lane // 32
  - ValM0  = 4             -> inner row stride (within chunk)
  - ValM1  = 32/4/2 = 4    -> outer chunk count
  - Per-lane = 4 chunks (ValM1) * 4 rows/chunk (ValM0) = 16 f32 outputs
  - Val flat-index v = valM1_idx * ValM0 + valM0_idx  (vec[v] in lane register)

Store mapping for one mfma output tile at (base_row, base_col)
(empirically verified by scripts2/_c11_mfma_32x32_probe.py — silicon for
mfma_f32_32x32x64 has lane%32 = ROW, not COL as in mfma_f32_16x16x128):
  for chunk in [0..4):     # = val_m1
    for r in [0..4):       # = val_m0
      row = base_row + (lane % 32)
      col = base_col + (lane // 32) * 4 + chunk * 8 + r
      vec_index = chunk * 4 + r
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import fly as fly_dialect
from flydsl.expr import arith, range_constexpr
from flydsl.expr.typing import Vector as Vec


class Mfma32x32x64:
    """Wraps mfma_scale_f32_32x32x64_f8f6f4. Mirror Mfma16x16x128 interface."""

    def __init__(self, n_tiles_a, n_tiles_b):
        self.atom = fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(32, 32, 64, fx.Float8E4M3FN)
        )
        self.accum_type = Vec.make_type(16, fx.Float32)
        self.zero_value = Vec.filled(16, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _do_mma(self, a, b, c):
        return fly_dialect.mma_atom_call_ssa(
            [self.accum_type], self.atom, a, b, c
        )

    def call(self, a, b, c):
        assert len(a) == self.n_tiles_a
        assert len(b) == self.n_tiles_b
        assert len(c) == self.n_tiles_a * self.n_tiles_b
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                c[self.idx(i, j)] = self._do_mma(
                    a[i], b[j], c[self.idx(i, j)]
                )
        return c

    def call_one(self, a, b, c, i, j):
        assert i < self.n_tiles_a and j < self.n_tiles_b
        return self._do_mma(a[i], b[j], c[self.idx(i, j)])


class StoreC32:
    """C-tile epilog for mfma_f32_32x32x64 output. Mirrors StoreC layout
    (kernels.fp8_gemm_utils.StoreC) but with the 32x32 per-lane mapping:
    4 row-chunks (ValM1) x 4 rows-per-chunk (ValM0) = 16 stores/lane,
    cols across lane%32 = 32 contiguous N cols per tile.

    Scale loading: per chunk (4 rows, stride 1 in M) we load a 4-element
    a_scale vec; per tile we have 4 such chunks -> 4 vec4 a_scale loads
    per (ti, lane) instead of the 16x16's single vec4.
    """

    def __init__(self, A_scale, B_scale, C, c_rows, c_cols, c_idx_fn,
                 n_tiles_a, n_tiles_b):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b

        c_nbytes = c_rows * c_cols * 2  # bf16
        sa_nbytes = c_rows * 4
        sb_nbytes = c_cols * 4
        gC = fx.rocdl.make_buffer_tensor(C, max_size=False, num_records_bytes=c_nbytes)
        gSA = fx.rocdl.make_buffer_tensor(A_scale, max_size=False, num_records_bytes=sa_nbytes)
        gSB = fx.rocdl.make_buffer_tensor(B_scale, max_size=False, num_records_bytes=sb_nbytes)
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.sa_div = fx.logical_divide(gSA, fx.make_layout(1, 1))
        self.sb_div = fx.logical_divide(gSB, fx.make_layout(1, 1))

        self.scale_atom_4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        self.scale_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        self.out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        self.reg_f32_4 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
        self.reg_f32_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        self.reg_bf16_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.BFloat16)

    def _load_scale_vec4(self, row):
        fx.copy(self.scale_atom_4,
                fx.slice(self.sa_div, (None, fx.Int32(row))),
                self.reg_f32_4)
        return Vec(fx.memref_load_vec(self.reg_f32_4))

    def _load_scale_scalar(self, col):
        fx.copy(self.scale_atom_1,
                fx.slice(self.sb_div, (None, fx.Int32(col))),
                self.reg_f32_1)
        return Vec(fx.memref_load_vec(self.reg_f32_1))[0]

    def _store_bf16(self, value_bf16, c_index):
        fx.memref_store_vec(Vec.filled(1, value_bf16, fx.BFloat16), self.reg_bf16_1)
        fx.copy(self.out_atom_1, self.reg_bf16_1,
                fx.slice(self.c_div, (None, fx.Int32(c_index))))

    def store(self, c_frag, base_row, base_col):
        # Silicon mapping (probe-verified): lane%32 -> ROW (single row per
        # lane), (lane//32)*4 + chunk*8 + r -> COL (16 cols per lane).
        row_off = self.lane_id % 32
        gn_col = (self.lane_id // 32) * 4

        # Pre-load a_scale (per-row scalar, one row per lane per ti).
        a_scales = [
            self._load_scale_scalar(base_row + ti * 32 + row_off)
            for ti in range_constexpr(self.n_tiles_a)
        ]
        # Pre-load b_scale per (tj, chunk). 4 chunks per tj, 4 elements each.
        b_scales = [[None] * 4 for _ in range_constexpr(self.n_tiles_b)]
        for tj in range_constexpr(self.n_tiles_b):
            col_base_tj = base_col + tj * 32 + gn_col
            for chunk in range_constexpr(4):
                b_scales[tj][chunk] = self._load_scale_vec4(col_base_tj + chunk * 8)

        oob = fx.Int32(self.c_rows * self.c_cols)
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 32 + row_off
            row_valid = row < self.c_rows
            for tj in range_constexpr(self.n_tiles_b):
                col_base_tj = base_col + tj * 32 + gn_col
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                for chunk in range_constexpr(4):
                    b_sc = b_scales[tj][chunk]
                    for r in range_constexpr(4):
                        v_idx = chunk * 4 + r
                        col = col_base_tj + chunk * 8 + r
                        scaled = (vec_f32[v_idx] * (a_scales[ti] * b_sc[r])).to(fx.BFloat16)
                        c_index = row * self.c_cols + col
                        self._store_bf16(scaled, arith.select(row_valid, c_index, oob))
