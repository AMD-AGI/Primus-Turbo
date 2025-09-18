// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "ck_grouped_gemm_kernel.h"

namespace primus_turbo {

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType,
          typename ALayout, typename BLayout, typename CLayout>
std::unique_ptr<CKGroupedGemmRunnerInterFace>
get_ck_grouped_gemm_instance_gfx942(const ck_tile::index_t group_num, const ck_tile::index_t m,
                                    const ck_tile::index_t n, const ck_tile::index_t k) {

    if constexpr (std::is_same_v<ADataType, ck_tile::half_t> ||
                  std::is_same_v<ADataType, ck_tile::bfloat16_t>) {
        if (n % 256 == 0) {
            using TileConfig = CKGroupedGemmTileCfg_256x256x64_32x32x16_2x2x1;
            using Runner = CKGroupedGemmRunner<ADataType, BDataType, CDataType, ALayout, BLayout,
                                               CLayout, TileConfig, AccDataType>;
            return std::make_unique<Runner>();
        } else if (n % 128 == 0) {
            using TileConfig = CKGroupedGemmTileCfg_256x128x64_32x32x16_2x2x1;
            using Runner = CKGroupedGemmRunner<ADataType, BDataType, CDataType, ALayout, BLayout,
                                               CLayout, TileConfig, AccDataType>;
            return std::make_unique<Runner>();
        } else {
            using TileConfig = CKGroupedGemmTileCfg_256x128x64_32x32x16_2x2x1_padding;
            using Runner = CKGroupedGemmRunner<ADataType, BDataType, CDataType, ALayout, BLayout,
                                               CLayout, TileConfig, AccDataType>;
            return std::make_unique<Runner>();
        }
    } else if constexpr (std::is_same_v<ADataType, ck_tile::bf8_t> ||
                         std::is_same_v<ADataType, ck_tile::fp8_t>) {
        if (n % 256 == 0) {
            using TileConfig = CKGroupedGemmTileCfg_256x256x128_32x32x32_2x2x1;
            using Runner     = CKQuantGroupedGemmRunner<ADataType, BDataType, CDataType, ALayout,
                                                        BLayout, CLayout, TileConfig, AccDataType>;
            return std::make_unique<Runner>();
        } else if (n % 128 == 0) {
            using TileConfig = CKGroupedGemmTileCfg_256x128x128_32x32x32_2x2x1;
            using Runner     = CKQuantGroupedGemmRunner<ADataType, BDataType, CDataType, ALayout,
                                                        BLayout, CLayout, TileConfig, AccDataType>;
            return std::make_unique<Runner>();
        } else {
            using TileConfig = CKGroupedGemmTileCfg_256x128x128_32x32x32_2x2x1_padding;
            using Runner     = CKQuantGroupedGemmRunner<ADataType, BDataType, CDataType, ALayout,
                                                        BLayout, CLayout, TileConfig, AccDataType>;
            return std::make_unique<Runner>();
        }
    } else {

        using TileConfig = CKGroupedGemmTileCfg_256x128x64_32x32x16_2x2x1_padding;
        using Runner     = CKGroupedGemmRunner<ADataType, BDataType, CDataType, ALayout, BLayout,
                                               CLayout, TileConfig, AccDataType>;
        return std::make_unique<Runner>();
    }
}

// clang-format off
// **************** Explicit Instantiation ****************
#define DECL_CK_GG_GFX942_INSTANCE(AType, BType, CType, ALayout, BLayout, CLayout)                 \
    template std::unique_ptr<CKGroupedGemmRunnerInterFace>                                         \
    get_ck_grouped_gemm_instance_gfx942<AType, BType, CType, float, ALayout, BLayout, CLayout>(    \
        const ck_tile::index_t, const ck_tile::index_t, const ck_tile::index_t,                    \
        const ck_tile::index_t);

// FP16 * FP16 = FP16
DECL_CK_GG_GFX942_INSTANCE(ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, RowMajor, ColMajor, RowMajor)
DECL_CK_GG_GFX942_INSTANCE(ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, RowMajor, RowMajor, RowMajor)
DECL_CK_GG_GFX942_INSTANCE(ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, ColMajor, RowMajor, RowMajor)

// BF16 * BF16 = BF16
DECL_CK_GG_GFX942_INSTANCE(ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t, RowMajor, ColMajor, RowMajor)
DECL_CK_GG_GFX942_INSTANCE(ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t, RowMajor, RowMajor, RowMajor)
DECL_CK_GG_GFX942_INSTANCE(ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t, ColMajor, RowMajor, RowMajor)

// FP8 * FP8 = FP16
DECL_CK_GG_GFX942_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, RowMajor, ColMajor, RowMajor)
DECL_CK_GG_GFX942_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, RowMajor, RowMajor, RowMajor)
DECL_CK_GG_GFX942_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, ColMajor, RowMajor, RowMajor)

// FP8 * FP8 = BF16
DECL_CK_GG_GFX942_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, RowMajor, ColMajor, RowMajor)
DECL_CK_GG_GFX942_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, RowMajor, RowMajor, RowMajor)
DECL_CK_GG_GFX942_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, ColMajor, RowMajor, RowMajor)

// BF8 * BF8 = FP16
DECL_CK_GG_GFX942_INSTANCE(ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, RowMajor, ColMajor, RowMajor)
DECL_CK_GG_GFX942_INSTANCE(ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, RowMajor, RowMajor, RowMajor)
DECL_CK_GG_GFX942_INSTANCE(ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, ColMajor, RowMajor, RowMajor)

// BF8 * BF8 = BF16
DECL_CK_GG_GFX942_INSTANCE(ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, RowMajor, ColMajor, RowMajor)
DECL_CK_GG_GFX942_INSTANCE(ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, RowMajor, RowMajor, RowMajor)
DECL_CK_GG_GFX942_INSTANCE(ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, ColMajor, RowMajor, RowMajor)

#undef DECL_CK_GG_GFX942_INSTANCE
// clang-format on
} // namespace primus_turbo
