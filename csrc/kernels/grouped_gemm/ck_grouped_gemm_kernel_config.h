// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "ck_tile/core.hpp"

namespace primus_turbo {
// clang-format off

template<
    ck_tile::index_t M_Tile_,
    ck_tile::index_t N_Tile_,
    ck_tile::index_t K_Tile_,
    ck_tile::index_t M_Warp_Tile_,
    ck_tile::index_t N_Warp_Tile_,
    ck_tile::index_t K_Warp_Tile_,
    ck_tile::index_t M_Warp_,
    ck_tile::index_t N_Warp_,
    ck_tile::index_t K_Warp_,
    bool DoubleSmemBuffer_
>
struct CKGroupedGemmTileConfig {
    static constexpr ck_tile::index_t M_Tile = M_Tile_;
    static constexpr ck_tile::index_t N_Tile = N_Tile_;
    static constexpr ck_tile::index_t K_Tile = K_Tile_;

    static constexpr ck_tile::index_t M_Warp = M_Warp_;
    static constexpr ck_tile::index_t N_Warp = N_Warp_;
    static constexpr ck_tile::index_t K_Warp = K_Warp_;

    static constexpr ck_tile::index_t M_Warp_Tile = M_Warp_Tile_;
    static constexpr ck_tile::index_t N_Warp_Tile = N_Warp_Tile_;
    static constexpr ck_tile::index_t K_Warp_Tile = K_Warp_Tile_;

    static constexpr bool DoubleSmemBuffer = DoubleSmemBuffer_;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    // static constexpr int              kBlockPerCu            = 1;
    static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    static constexpr ck_tile::index_t TileParitionerM01      = 4;
};


// Tile Config Specialization
using CKGroupedGemmTileCfg_256x256x64_32x32x16_2x2x1 = CKGroupedGemmTileConfig<
    256, 256, 64, 32, 32, 16, 2, 2, 1, false
>;

// clang-format on
} // namespace primus_turbo
