// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "ck_grouped_gemm_kernel_instance_factory.h"

namespace primus_turbo {
// clang-format off

#ifdef PRIMUS_TURBO_GFX942
template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType,
          typename ALayout, typename BLayout, typename CLayout>
std::unique_ptr<CKGroupedGemmRunnerInterFace>
get_ck_grouped_gemm_instance_gfx942(const ck_tile::index_t group_num, const ck_tile::index_t m,
                                    const ck_tile::index_t n, const ck_tile::index_t k) {
    std::unique_ptr<CKGroupedGemmRunnerInterFace> runner = nullptr;
    if (get_current_arch() != GPUArch::GFX942) {
        PRIMUS_TURBO_ERROR("Currently Arch != gfx942");
    }

    if constexpr (std::is_same_v<ADataType, ck_tile::half_t> || std::is_same_v<ADataType, ck_tile::bfloat16_t>) {
        if (n % 256 == 0) {
            using TileConfig = CKGroupedGemmTileCfg_256x256x64_32x32x16_2x2x1;
            using Runner     = CKGroupedGemmRunner<ADataType, BDataType, CDataType, ALayout, BLayout, CLayout, TileConfig, AccDataType>;
            runner           = std::make_unique<Runner>();
        } else if (n % 128 == 0) {
            using TileConfig = CKGroupedGemmTileCfg_256x128x64_32x32x16_2x2x1;
            using Runner     = CKGroupedGemmRunner<ADataType, BDataType, CDataType, ALayout, BLayout, CLayout, TileConfig, AccDataType>;
            runner           = std::make_unique<Runner>();
        } else {
            using TileConfig = CKGroupedGemmTileCfg_256x128x64_32x32x16_2x2x1_padding;
            using Runner     = CKGroupedGemmRunner<ADataType, BDataType, CDataType, ALayout, BLayout, CLayout, TileConfig, AccDataType>;
            runner           = std::make_unique<Runner>();
        }
    } else if constexpr (std::is_same_v<ADataType, ck_tile::bf8_t> || std::is_same_v<ADataType, ck_tile::fp8_t>) {
        if (n % 256 == 0) {
            using TileConfig = GFX942_CKGroupedGemmTileCfg_256x256x128_32x32x32_2x2x1;
            using Runner     = CKQuantGroupedGemmRunnerWithArch<GPUArch::GFX942, ADataType, BDataType, CDataType, ALayout, BLayout, CLayout, TileConfig, AccDataType>;
            runner = std::make_unique<Runner>();
        } else if (n % 128 == 0) {
            using TileConfig = GFX942_CKGroupedGemmTileCfg_256x128x128_32x32x32_2x2x1;
            using Runner     = CKQuantGroupedGemmRunnerWithArch<GPUArch::GFX942, ADataType, BDataType, CDataType, ALayout, BLayout, CLayout, TileConfig, AccDataType>;
            runner = std::make_unique<Runner>();
        } else {
            using TileConfig = GFX942_CKGroupedGemmTileCfg_256x128x128_32x32x32_2x2x1_padding;
            using Runner     = CKQuantGroupedGemmRunnerWithArch<GPUArch::GFX942, ADataType, BDataType, CDataType, ALayout, BLayout, CLayout, TileConfig, AccDataType>;
            runner = std::make_unique<Runner>();
        }
    } else {
        PRIMUS_TURBO_ERROR("Grouped Gemm only support fp16/bf16/fp8/bf8");
    }
    return runner;
}
#endif

#ifdef PRIMUS_TURBO_GFX950
template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType,
          typename ALayout, typename BLayout, typename CLayout>
std::unique_ptr<CKGroupedGemmRunnerInterFace>
get_ck_grouped_gemm_instance_gfx950(const ck_tile::index_t group_num, const ck_tile::index_t m,
                                    const ck_tile::index_t n, const ck_tile::index_t k) {
    std::unique_ptr<CKGroupedGemmRunnerInterFace> runner = nullptr;
    if (get_current_arch() != GPUArch::GFX950) {
        PRIMUS_TURBO_ERROR("Currently Arch != gfx950");
    }

    if constexpr (std::is_same_v<ADataType, ck_tile::half_t> || std::is_same_v<ADataType, ck_tile::bfloat16_t>) {
        if (n % 256 == 0) {
            using TileConfig = CKGroupedGemmTileCfg_256x256x64_32x32x16_2x2x1;
            using Runner     = CKGroupedGemmRunner<ADataType, BDataType, CDataType, ALayout, BLayout, CLayout, TileConfig, AccDataType>;
            runner           = std::make_unique<Runner>();
        } else if (n % 128 == 0) {
            using TileConfig = CKGroupedGemmTileCfg_256x128x64_32x32x16_2x2x1;
            using Runner     = CKGroupedGemmRunner<ADataType, BDataType, CDataType, ALayout, BLayout, CLayout, TileConfig, AccDataType>;
            runner           = std::make_unique<Runner>();
        } else {
            using TileConfig = CKGroupedGemmTileCfg_256x128x64_32x32x16_2x2x1_padding;
            using Runner     = CKGroupedGemmRunner<ADataType, BDataType, CDataType, ALayout, BLayout, CLayout, TileConfig, AccDataType>;
            runner           = std::make_unique<Runner>();
        }
    } else if constexpr (std::is_same_v<ADataType, ck_tile::bf8_t> || std::is_same_v<ADataType, ck_tile::fp8_t>) {
        if (n % 256 == 0) {
            using TileConfig = GFX950_CKGroupedGemmTileCfg_256x256x128_16x16x128_2x2x1;
            using Runner     = CKQuantGroupedGemmRunnerWithArch<GPUArch::GFX950, ADataType, BDataType, CDataType, ALayout, BLayout, CLayout, TileConfig, AccDataType>;
            runner           = std::make_unique<Runner>();
        } else if (n % 128 == 0) {
            using TileConfig = GFX950_CKGroupedGemmTileCfg_256x256x128_16x16x128_2x2x1_padding;
            using Runner     = CKQuantGroupedGemmRunnerWithArch<GPUArch::GFX950, ADataType, BDataType, CDataType, ALayout, BLayout, CLayout, TileConfig, AccDataType>;
            runner           = std::make_unique<Runner>();
        } else {
            using TileConfig = GFX950_CKGroupedGemmTileCfg_128x128x128_32x32x64_2x2x1;
            using Runner     = CKQuantGroupedGemmRunnerWithArch<GPUArch::GFX950, ADataType, BDataType, CDataType, ALayout, BLayout, CLayout, TileConfig, AccDataType>;
            runner           = std::make_unique<Runner>();
        }
    } else {
        PRIMUS_TURBO_ERROR("Grouped Gemm only support fp16/bf16/fp8/bf8");
    }
    return runner;
}
#endif

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType,
          typename ALayout, typename BLayout, typename CLayout>
std::unique_ptr<CKGroupedGemmRunnerInterFace>
get_ck_grouped_gemm_instance(const ck_tile::index_t group_num, const ck_tile::index_t m,
                             const ck_tile::index_t n, const ck_tile::index_t k) {
    const GPUArch arch = get_current_arch();
    switch (arch) {
#ifdef PRIMUS_TURBO_GFX942
    case GPUArch::GFX942: {
        return get_ck_grouped_gemm_instance_gfx942<ADataType, BDataType, CDataType, AccDataType,
                                                   ALayout, BLayout, CLayout>(group_num, m, n, k);
    }
#endif
#ifdef PRIMUS_TURBO_GFX950
    case GPUArch::GFX950: {
        return get_ck_grouped_gemm_instance_gfx950<ADataType, BDataType, CDataType, AccDataType,
                                                   ALayout, BLayout, CLayout>(group_num, m, n, k);
    }
#endif
    default:
        PRIMUS_TURBO_ERROR("Unsupported arch in get_ck_grouped_gemm_instance()");
    }
}

#define DECL_GET_CK_GG_INSTANCE(AType, BType, CType, ALayout, BLayout, CLayout)                    \
    template std::unique_ptr<CKGroupedGemmRunnerInterFace>                                         \
    get_ck_grouped_gemm_instance<AType, BType, CType, float, ALayout, BLayout, CLayout>(           \
        const ck_tile::index_t, const ck_tile::index_t, const ck_tile::index_t,                    \
        const ck_tile::index_t);

// FP16 * FP16 = FP16
APPLY_GET_CK_GG_INSTANCE_ALL_LAYOUT(DECL_GET_CK_GG_INSTANCE, ck_tile::half_t, ck_tile::half_t, ck_tile::half_t)
// BF16 * BF16 = BF16
APPLY_GET_CK_GG_INSTANCE_ALL_LAYOUT(DECL_GET_CK_GG_INSTANCE, ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t)
// FP8_E4M3 * FP8_E4M3 = FP16
APPLY_GET_CK_GG_INSTANCE_ALL_LAYOUT(DECL_GET_CK_GG_INSTANCE, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t)
// FP8_E4M3 * FP8_E4M3 = BF16
APPLY_GET_CK_GG_INSTANCE_ALL_LAYOUT(DECL_GET_CK_GG_INSTANCE, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t)
// FP8_E5M2 * FP8_E5M2 = FP16
APPLY_GET_CK_GG_INSTANCE_ALL_LAYOUT(DECL_GET_CK_GG_INSTANCE, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t)
// FP8_E5M2 * FP8_E5M2 = BF16
APPLY_GET_CK_GG_INSTANCE_ALL_LAYOUT(DECL_GET_CK_GG_INSTANCE, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t)

// clang-format on
} // namespace primus_turbo
