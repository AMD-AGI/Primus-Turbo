/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// hk_attn/tdm.h -- Tensor Data Mover: 2-D row-major tile -> padded LDS image (tensor_load_to_lds,
// clang 23+ 6-operand form, groups 2..4 zero) and s_wait_tensorcnt.
//
// Descriptor (D#), field layout as REF k_dkdv96ds / dqds and HK tdm.cuh build_tdm_descriptor_2d
// (both proven on this GPU) and FlyDSL tdm_ops.make_tensor_descriptor_2d:
//   group 0 (4 SGPRs, per tile): w0 = pred (1 issue, 0 = NULL descriptor: moves nothing;
//   tdm_g0_null),
//     w1 = LDS byte address, w2 = global byte address [31:0], w3 = address [56:32] | type 2 << 30
//   group 1 (8 SGPRs, loop-invariant): w0 = data_size / pad (layout.h tdm_w0),
//     w1 = tensor_dim0[15:0] << 16, w2 = tensor_dim0[31:16] | tensor_dim1[15:0] << 16,
//     w3 = tensor_dim1[31:16] | tile_dim0 << 16, w4 = tile_dim1, w5..w6 = dim-1 stride (elements,
//     48 bits), w7 = 0
// dim0 = columns (contiguous), dim1 = rows. The tensor extents are measured from the tile origin
// (the global address of group 0): an element at (row, col) of the tile with col >= tensor_dim0 or
// row >= tensor_dim1 is not fetched and lands in LDS as zero (FlyDSL oob_outer_bound, validated on
// this card; t_prims t_tdm checks it). TRUE extents: tensor_dim0 = tile columns (the tile never
// crosses its row), tensor_dim1 = the rows that exist below the origin, clamped to the tile.
// The engine writes row t of the tile at LDS lds_addr + t * pitch: tile_cols * elem_b data bytes,
// then (pitch - row bytes) pad bytes it skips (pad_interval = the row, pad_amount = the pad).
//
// BSHD as [B*S, H*D] row-major (row stride H*D elements, head h = column offset h*D): the tile of
// rows [row0, row0 + R) of head h starts at element row0 * H*D + h*D (tdm_gaddr).
//
// All descriptor words must be wave-uniform (they are SGPR operands): build them from kernel
// arguments, block ids and wave_id() only. At most ~3 TDM ops in flight per wave are useful
// (p1_research); ops of one wave complete in order, so s_wait_tensorcnt N retires all but the N
// newest.
#pragma once
#include "types.h"

namespace hk_attn {

// static description of a 2-D tile of TILE_ROWS x TILE_COLS elements of ELEM_B bytes landing in LDS
// with row pitch PITCH_B bytes
template <int TILE_ROWS, int TILE_COLS, int PITCH_B, int ELEM_B = 2> struct tdm_tile {
    static_assert(
        lay::tdm_tile_ok(TILE_COLS, PITCH_B, ELEM_B),
        "TDM tile: row bytes a power of two in [8, 1024], pad a multiple of 4 B <= 512 B");
    static_assert(TILE_ROWS > 0 && TILE_ROWS < 65536, "tile_dim1 is 16 bits");
    static constexpr int      rows = TILE_ROWS, cols = TILE_COLS, pitch = PITCH_B;
    static constexpr int      bytes = TILE_ROWS * PITCH_B; // LDS footprint
    static constexpr uint32_t w0    = lay::tdm_w0(TILE_COLS, PITCH_B, ELEM_B);
};

// group 1 for tile T, row stride `stride_el` elements, tensor_dim1 = `rows_valid` (clamped to
// [0, T::rows]); tensor_dim0 = T::cols
template <typename T> HKA_DEV u32x8 tdm_g1(int64_t stride_el, int rows_valid) {
    const uint32_t d0 = (uint32_t) T::cols;
    const uint32_t d1 = (uint32_t) (rows_valid < 0         ? 0
                                    : rows_valid > T::rows ? T::rows
                                                           : rows_valid);
    const uint64_t st = (uint64_t) stride_el;
    return u32x8{T::w0,
                 (d0 & 0xFFFFu) << 16,
                 (d0 >> 16) | ((d1 & 0xFFFFu) << 16),
                 (d1 >> 16) | ((uint32_t) T::cols << 16),
                 (uint32_t) T::rows,
                 (uint32_t) st,
                 (uint32_t) (st >> 32) & 0xFFFFu,
                 0u};
}
// group 0: one tile at global byte address `gaddr` into LDS byte address `lds`; pred 0 = NULL
HKA_DEV u32x4 tdm_g0(uint32_t lds, uint64_t gaddr, uint32_t pred = 1u) {
    return u32x4{pred, lds, (uint32_t) gaddr,
                 ((uint32_t) (gaddr >> 32) & 0x01FFFFFFu) | (2u << 30)};
}
// NULL descriptor (pred 0): moves no memory (HK tdm.cuh count == 0); the branch-free way to skip a
// tail transfer (a TDM instruction cannot be EXEC-masked). Evidence on this GPU: HK GEMM rung 07
// (count = 0 tail, wait_tdm<0> at the end) and dsv3 x6 m2d (waves 1..3 issue NULL ops; correct,
// completes), both with the real tile address in the descriptor; t_prims t_tdm_null checks it.
// `gaddr` must therefore be the address of a READABLE tile (e.g. the one the slot would have
// loaded, clamped to the tensor) and g1's tensor_dim1 must keep it inside the allocation: an engine
// that ignored pred would then copy valid bytes, never fault. Whether a NULL op occupies a
// TENSORCNT slot is UNVERIFIED (not observable without a race): never let a partial
// s_wait_tensorcnt N depend on it (wait counts must be right whether or not a NULL op counts).
HKA_DEV u32x4 tdm_g0_null(uint32_t lds, uint64_t gaddr) {
    return tdm_g0(lds, gaddr, 0u);
}
// global byte address of element (row, col0) of a row-major [*, row_el] bf16-like tensor (int64)
HKA_DEV uint64_t tdm_gaddr(const void *base, int64_t row, int64_t row_el, int64_t col0,
                           int elem_b = 2) {
    return (uint64_t) (uintptr_t) base + (uint64_t) ((row * row_el + col0) * elem_b);
}
// tensor_load_to_lds s[g0], s[g1] (one tile; increments TENSORCNT)
HKA_DEV void tdm_load(u32x4 g0, u32x8 g1) {
    const i32x4 z4 = {0, 0, 0, 0};
    const i32x8 z8 = {0, 0, 0, 0, 0, 0, 0, 0};
    __builtin_amdgcn_tensor_load_to_lds(__builtin_bit_cast(i32x4, g0),
                                        __builtin_bit_cast(i32x8, g1), z4, z4, z8, 0);
}
// s_wait_tensorcnt N: all but the N newest TDM ops of this wave retired (their LDS writes visible
// to this wave; other waves additionally need a barrier)
template <int N> HKA_DEV void wait_tensorcnt() {
    __builtin_amdgcn_s_wait_tensorcnt((short) N);
}

} // namespace hk_attn
