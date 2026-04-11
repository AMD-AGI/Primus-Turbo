
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "primus_turbo/common.h"
#include "primus_turbo/shuffle.h"

namespace primus_turbo {

using namespace primus_turbo::detail;

// ============================================================================
// MEMORY LAYOUT - Index Computation for Shuffled Layouts
// ============================================================================

/*
 * Scale Shuffle Index Computation
 * --------------------------------
 * Computes the shuffled memory index for scale factors to optimize
 * memory access patterns during GEMM operations.
 *
 * Permutation formula:
 *   i0 = row // 32
 *   i1 = (row % 32) // 16
 *   i2 = row % 16
 *   i3 = col // 8
 *   i4 = (col % 8) // 4
 *   i5 = col % 4
 *   index = i0*(scale_n_pad//8)*256 + i3*256 + i5*64 + i2*4 + i4*2 + i1
 */
PRIMUS_TURBO_DEVICE __forceinline__ int compute_shuffle_scale_index(int row, int col,
                                                                    int scale_n_pad) {
    int i0 = row >> 5;       // row // 32
    int i1 = (row >> 4) & 1; // (row % 32) // 16
    int i2 = row & 15;       // row % 16
    int i3 = col >> 3;       // col // 8
    int i4 = (col >> 2) & 1; // (col % 8) // 4
    int i5 = col & 3;        // col % 4

    return (i0 * (scale_n_pad >> 3) << 8) + (i3 << 8) + (i5 << 6) + (i2 << 2) + (i4 << 1) + i1;
}

/*
 *  Data Shuffle Index Computation
 * -----------------------------------
 * Computes the shuffled memory index for MXFP4/MXFP8 quantized data.
 * This layout is optimized for GEMM performance by improving cache locality.
 *
 * Structure:
 *   - 16xK blocks where K must be multiple of 32
 *   - Each K=32 block is split into two K=16 sub-blocks
 *   - Data is stored in (BN=16, BK=32) tiles
 */
template <typename DType>
PRIMUS_TURBO_DEVICE __forceinline__ int compute_shuffled_index(int row, int col, int K_packed) {
    static_assert(std::is_same_v<DType, dtype::float4x2_e2m1> ||
                      std::is_same_v<DType, dtype::float8_e4m3> ||
                      std::is_same_v<DType, dtype::float8_e5m2>,
                  "compute_shuffled_index: unsupported DType");

    constexpr int k_elem_stride =
        std::is_same_v<DType, dtype::float4x2_e2m1> ? MXFP4_SHUFFLE_K_ELEM : MXFP8_SHUFFLE_K_ELEM;

    int N_block      = row >> 4;          // row // 16
    int row_in_block = row & 15;          // row % 16
    int K_block      = col >> 5;          // col // 32
    int col_in_block = col & 31;          // col % 32
    int sub_block    = col_in_block >> 4; // Which half: [0:15] or [16:31]
    int k_elem       = col_in_block & 15; // Position within sub-block

    return N_block * (K_packed << 4) + K_block * 512 + sub_block * 256 +
           row_in_block * k_elem_stride + k_elem;
}

} // namespace primus_turbo
