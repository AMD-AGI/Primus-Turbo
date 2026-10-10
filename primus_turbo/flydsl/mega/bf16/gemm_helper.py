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

# MegaMoE-owned snapshot of flydsl/utils/gemm_helper.py as of f6d5ab68.
# Kept private to Mega MoE BF16 so the shared helper can evolve (Mfma atom,
# S2R loaders, swizzle) without changing MegaMoE correctness. Only
# mega/bf16/* imports this module.
#
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.buffer_ops import buffer_store
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue

from primus_turbo.flydsl.utils.gemm_helper import (
    _pack_out_pair,
    _permlane16_swap,
    make_row_band_resource,
    swizzle_128,
)
from primus_turbo.flydsl.utils.prims import _lds_ptr_from_i32


def load_row_idx_to_lds(row_idx_resource, lds_byte_address, entry_byte_offset):
    """Each lane copies the i32 at ``entry_byte_offset`` to LDS ``lds_byte_address + 4 * lane``."""
    rocdl.buffer_load_to_lds(
        row_idx_resource, _lds_ptr_from_i32(lds_byte_address), entry_byte_offset, size_bytes=4
    )


def read_row_idx_from_lds(lds_byte_address):
    """One ds_read_b32; opaque asm, so the compiler does not wait on in-flight LDS DMA before it."""
    op = _llvm.InlineAsmOp(
        res=T.i32,
        operands_=[_raw(_lds_ptr_from_i32(lds_byte_address))],
        asm_string="ds_read_b32 $0, $1\ns_waitcnt lgkmcnt(0)",
        constraints="=&v,v,~{memory}",
        has_side_effects=True,
    )
    return ArithValue(op.result, signed=True)


def compute_global_swizzle_bf16(lane_id, wave_id, K, n_rounds, row_step=1, pair_span=0):
    """Per-lane global element offsets of a [rows, K] operand, composed from the (row, column) pairs."""
    row_cols = compute_global_swizzle_bf16_row_col(lane_id, wave_id, n_rounds, pair_span)
    return [g_row * (row_step * K) + col for g_row, col in row_cols]


def compute_global_swizzle_bf16_row_col(lane_id, wave_id, n_rounds, pair_span=0):
    """(row, column) in elements of each lane's 16-byte piece, one pair per round."""
    row_cols = []
    n_waves = fx.block_dim.x // 64
    for r in range_constexpr(n_rounds):
        row = lane_id // 8 + wave_id * 8 + r * (n_waves * 8)
        col_byte = (lane_id % 8) * 16
        _, c = swizzle_128(row, col_byte)
        g_row = row
        if const_expr(pair_span):
            half = pair_span // 2
            t = row % pair_span
            g_row = (row - t) + (t % half) * 2 + t // half
        row_cols.append((g_row, c // 2))
    return row_cols


def compute_global_swizzle_nn_bf16_wide_row_col(lane_id, wave_id, n_steps):
    """(k-row, column) in elements of each lane's 16-byte piece, one pair per step."""
    row_cols = []
    n_waves = fx.block_dim.x // 64
    kloc = (lane_id // 2) % 8
    n_in = (lane_id // 16) * 16 + (lane_id % 2) * 8
    for step in range_constexpr(n_steps):
        idx = wave_id + step * n_waves
        n64 = idx // 8
        ks8 = idx % 8
        row_cols.append((ks8 * 8 + kloc, n64 * 64 + n_in))
    return row_cols


def store_band_merged16(store_c, c_frags, base_row, base_col, col_step, n_tiles_a, n_tiles_b, row_bound):
    """``StoreCBf16.store_band16`` with adjacent column tiles merged by ``v_permlane16_swap`` into 64 B row runs."""
    assert n_tiles_b % 2 == 0, "the merge pairs adjacent column tiles"
    rsrc = make_row_band_resource(store_c.c_base, base_row, row_bound, store_c.c_cols, 2)
    row_bytes = store_c.c_cols * 2
    base_off = ((store_c.lane_id // 32) * 8) * row_bytes + (base_col + store_c.lane_id % 32) * 2
    for q in range_constexpr(len(c_frags)):
        for ti in range_constexpr(n_tiles_a):
            vecs = [Vec(c_frags[q][ti * n_tiles_b + j]) for j in range_constexpr(n_tiles_b)]
            packed = [
                [
                    _pack_out_pair(vecs[j][2 * h], vecs[j][2 * h + 1], store_c.out_ty)
                    for h in range_constexpr(2)
                ]
                for j in range_constexpr(n_tiles_b)
            ]
            # All swaps first, then the store burst, to keep the permlane->store hazard off-path.
            runs = [
                (Vec.from_elements([fx.Int32(v)], fx.Int32).bitcast(store_c.out_ty), p, r, h)
                for h in range_constexpr(2)
                for p in range_constexpr(n_tiles_b // 2)
                for v, r in zip(_permlane16_swap(packed[2 * p][h], packed[2 * p + 1][h]), (0, 4))
            ]
            for pair, p, r, h in runs:
                for e in range_constexpr(2):
                    buffer_store(
                        pair[e],
                        rsrc,
                        base_off + (ti * 16 + r + 2 * h + e) * row_bytes + q * col_step * 2 + p * 64,
                        cache_modifier=store_c.cache_modifier,
                        offset_is_bytes=True,
                    )
