/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Fused MXFP6 / MX packers: the per-launch tile-pack parameters, their setters, and the deferred FP4 dgrad copy of a
// gathered MXFP6 tile weight. The packers themselves are in quantization_mxfp6_gfx950_impl.cuh, instantiated per entry
// point and dtype in quantization_mxfp6_*_gfx950.cu.

#include "quantization_mxfp6_gfx950_impl.cuh"

namespace primus_turbo {

namespace mxfp6_detail {
MXTilePack g_ts_row, g_ts_col;
} // namespace mxfp6_detail

// Outside the file's anonymous namespace: the op layer (another translation unit) calls these.
void mx_tile_pack_set(const MXTilePack &row, const MXTilePack &col) {
    g_ts_row = row;
    g_ts_col = col;
}
MXTilePack mx_tile_pack_row() {
    return g_ts_row;
}
MXTilePack mx_tile_pack_col() {
    return g_ts_col;
}

void mxfp6_tile_to_fp4_col_impl(const uint8_t *c0, const uint8_t *c1, const uint8_t *row_scale, const int64_t R,
                                const int64_t K, uint8_t *col_packed, uint8_t *col_scale, const bool sr,
                                const uint32_t seed, hipStream_t stream) {
    PRIMUS_TURBO_CHECK(R % kGroupSize == 0 && K % 256 == 0, "mxfp6_tile_to_fp4_col: R % 32 == 0, K % 256 == 0");
    const auto row_ts = to_args(g_ts_row), col_ts = to_args(g_ts_col);
    // The rows' Hadamard: H32 (the forward rows; un-rotated in the receiver) or none (an unrotated plane).
    PRIMUS_TURBO_CHECK(row_ts.fp4_had == mxfp4_emit::kHadH32 || row_ts.fp4_had == mxfp4_emit::kHadNone,
                       "mxfp6_tile_to_fp4_col: the rows' Hadamard must be H32 or none");
    const bool unrot = row_ts.fp4_had == mxfp4_emit::kHadH32;
    const dim3 grid(static_cast<uint32_t>(K / 256), static_cast<uint32_t>(R / kGroupSize)), block(256);
    auto launch = [&](auto s, auto u) {
        mxfp6_tile_to_fp4_col_kernel<decltype(s)::value, decltype(u)::value>
            <<<grid, block, 0, stream>>>(c0, c1, row_scale, col_packed, col_scale, seed, row_ts, col_ts);
    };
    using T = std::true_type;
    using F = std::false_type;
    if (sr)
        unrot ? launch(T{}, T{}) : launch(T{}, F{});
    else
        unrot ? launch(F{}, T{}) : launch(F{}, F{});
}

} // namespace primus_turbo
