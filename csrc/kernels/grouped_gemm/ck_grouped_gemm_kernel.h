// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include <hip/hip_runtime.h>
#include <string>

#include "ck_grouped_gemm_kernel_config.h"
#include "primus_turbo/arch.h"

namespace primus_turbo {
// clang-format off

using RowMajor = ck_tile::tensor_layout::gemm::RowMajor;
using ColMajor = ck_tile::tensor_layout::gemm::ColumnMajor;


template <typename Kernel>
inline void _launch_ck_grouped_kernel(const ck_tile::stream_config& stream_cfg,
                                      ck_tile::index_t group_num,
                                      void* args_ptr,
                                      uint32_t num_cu) {
    constexpr int kBlockPerCu = 1;
    const dim3 blocks = Kernel::BlockSize();
    dim3       grids  = Kernel::MaxOccupancyGridSize(stream_cfg);
    grids.x           = std::min(grids.x, num_cu);
    ck_tile::launch_kernel(
        stream_cfg, ck_tile::make_kernel<kBlockPerCu>(
                        Kernel{}, grids, blocks, 0,
                        ck_tile::cast_pointer_to_constant_address_space(args_ptr), group_num));
}

class CKGroupedGemmRunnerInterFace {
public:
    virtual ~CKGroupedGemmRunnerInterFace() = default;
    // virtual void init_args() = 0;
    virtual void run(const ck_tile::stream_config &stream_cfg,
                     const ck_tile::index_t group_num,
                     void *args_ptr, const uint32_t num_cu) = 0;
};


template <
    GPUArch arch,
    typename ADataType,
    typename BDataType,
    typename CDataType,
    typename ALayout,
    typename BLayout,
    typename CLayout,
    typename TileConfig,
    typename AccDataType=float
>
class CKGroupedGemmRunner : public CKGroupedGemmRunnerInterFace {
public:
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<
            TileConfig::M_Tile,
            TileConfig::N_Tile,
            TileConfig::K_Tile
        >,
        ck_tile::sequence<
            TileConfig::M_Warp,
            TileConfig::N_Warp,
            TileConfig::K_Warp
        >,
        ck_tile::sequence<
            TileConfig::M_Warp_Tile,
            TileConfig::N_Warp_Tile,
            TileConfig::K_Warp_Tile
        >
    >;

    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
        GemmShape,
        TileConfig::TileParitionerGroupNum,
        TileConfig::TileParitionerM01
    >;

    using Traits = ck_tile::TileGemmTraits<
        TileConfig::kPadM, TileConfig::kPadN, TileConfig::kPadK,
        ALayout, BLayout, CLayout
    >;

    using GemmUniversalTraits = ck_tile::PersistentTileGemmUniversalTraits<
        TileConfig::kPadM, TileConfig::kPadN, TileConfig::kPadK,
        TileConfig::DoubleSmemBuffer,
        ALayout, BLayout, CLayout>;


    static constexpr ck_tile::GemmPipelineScheduler GemmPipelineScheduler = ck_tile::GemmPipelineScheduler::Intrawave;

    using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
        ADataType,
        BDataType,
        AccDataType,
        GemmShape,
        GemmUniversalTraits,
        GemmPipelineScheduler
    >;

    // V3
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;
    // using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3;

    static constexpr ck_tile::memory_operation_enum MemoryOp = ck_tile::memory_operation_enum::set;
    using GemmEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<
            ADataType, BDataType, ck_tile::tuple<>, AccDataType,
            CDataType, ck_tile::tuple<>, CLayout,
            ck_tile::element_wise::PassThrough,
            TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
            TileConfig::M_Warp, TileConfig::N_Warp,
            TileConfig::M_Warp_Tile, TileConfig::N_Warp_Tile, TileConfig::K_Warp_Tile,
            UniversalGemmProblem::TransposeC,
            MemoryOp
        >
    >;

    using Kernel = ck_tile::GroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;


public:
    void run(const ck_tile::stream_config &stream_cfg,
             const ck_tile::index_t group_num,
             void *args_ptr, const uint32_t num_cu) override {
        _launch_ck_grouped_kernel<Kernel>(stream_cfg, group_num, args_ptr, num_cu);
    }
};

template <
    GPUArch arch,
    typename ADataType,
    typename BDataType,
    typename CDataType,
    typename ALayout,
    typename BLayout,
    typename CLayout,
    typename TileConfig,
    typename AccDataType=float
>
class CKQuantGroupedGemmRunner : public CKGroupedGemmRunnerInterFace {
public:
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<
            TileConfig::M_Tile,
            TileConfig::N_Tile,
            TileConfig::K_Tile
        >,
        ck_tile::sequence<
            TileConfig::M_Warp,
            TileConfig::N_Warp,
            TileConfig::K_Warp
        >,
        ck_tile::sequence<
            TileConfig::M_Warp_Tile,
            TileConfig::N_Warp_Tile,
            TileConfig::K_Warp_Tile
        >
    >;

    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
        GemmShape,
        TileConfig::TileParitionerGroupNum,
        TileConfig::TileParitionerM01
    >;

    static constexpr ck_tile::QuantType QuantMode = ck_tile::QuantType::RowColQuant;
    using AQLayout = ck_tile::tensor_layout::gemm::RowMajor;
    using BQLayout = ck_tile::tensor_layout::gemm::ColumnMajor;

    using GemmUniversalTraits = ck_tile::TileGemmQuantTraits<
        TileConfig::kPadM,
        TileConfig::kPadN,
        TileConfig::kPadK,
        false,
        ALayout,
        BLayout,
        CLayout,
        QuantMode,
        AQLayout,
        BQLayout,
        TileConfig::DoubleSmemBuffer,
        true
    >;

    using QuantGemmProblem = ck_tile::GemmRowColTensorQuantPipelineProblem<ADataType,
                                                                     BDataType,
                                                                     AccDataType,
                                                                     AccDataType,
                                                                     GemmShape,
                                                                     GemmUniversalTraits>;

    // V3
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<QuantGemmProblem>;

    static constexpr ck_tile::memory_operation_enum MemoryOp = ck_tile::memory_operation_enum::set;
    using GemmEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<
                                         ADataType,
                                         BDataType,
                                         ck_tile::tuple<>,
                                         AccDataType,
                                         CDataType,
                                         ck_tile::tuple<>,
                                         CLayout,
                                        ck_tile::element_wise::PassThrough,
            TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
            TileConfig::M_Warp, TileConfig::N_Warp,
            TileConfig::M_Warp_Tile, TileConfig::N_Warp_Tile, TileConfig::K_Warp_Tile,
            QuantGemmProblem::TransposeC,
            MemoryOp
        >
    >;

    using Kernel = ck_tile::QuantGroupedGemmKernel<
        TilePartitioner,
        GemmPipeline,
        GemmEpilogue,
        GemmUniversalTraits::kQuantType
    >;

public:
    void run(const ck_tile::stream_config &stream_cfg,
             const ck_tile::index_t group_num,
             void *args_ptr, const uint32_t num_cu) override {
        _launch_ck_grouped_kernel<Kernel>(stream_cfg, group_num, args_ptr, num_cu);
    }
};

#ifdef PRIMUS_TURBO_GFX942
template <
    typename ADataType,
    typename BDataType,
    typename CDataType,
    typename AccDataType,
    typename ALayout,
    typename BLayout,
    typename CLayout
>
std::unique_ptr<CKGroupedGemmRunnerInterFace>
get_ck_grouped_gemm_instance_gfx942(const ck_tile::index_t group_num, const ck_tile::index_t m,
                             const ck_tile::index_t n, const ck_tile::index_t k){
    std::unique_ptr<CKGroupedGemmRunnerInterFace> runner = nullptr;
    if (get_current_arch() != GPUArch::GFX942) {
        PRIMUS_TURBO_ERROR("Currently Arch != gfx942");
    }

    if constexpr (std::is_same_v<ADataType, ck_tile::half_t> ||
                  std::is_same_v<ADataType, ck_tile::bfloat16_t>) {
        if (n % 256 == 0) {
            using TileConfig = CKGroupedGemmTileCfg_256x256x64_32x32x16_2x2x1;
            using Runner = CKGroupedGemmRunner<GPUArch::GFX942, ADataType, BDataType, CDataType, ALayout, BLayout,
                                               CLayout, TileConfig, AccDataType>;
            runner = std::make_unique<Runner>();
        } else if (n % 128 == 0) {
            using TileConfig = CKGroupedGemmTileCfg_256x128x64_32x32x16_2x2x1;
            using Runner = CKGroupedGemmRunner<GPUArch::GFX942, ADataType, BDataType, CDataType, ALayout, BLayout,
                                               CLayout, TileConfig, AccDataType>;
            runner = std::make_unique<Runner>();
        } else {
            using TileConfig = CKGroupedGemmTileCfg_256x128x64_32x32x16_2x2x1_padding;
            using Runner = CKGroupedGemmRunner<GPUArch::GFX942, ADataType, BDataType, CDataType, ALayout, BLayout,
                                               CLayout, TileConfig, AccDataType>;
            runner = std::make_unique<Runner>();
        }
    } else if constexpr (std::is_same_v<ADataType, ck_tile::bf8_t> ||
                         std::is_same_v<ADataType, ck_tile::fp8_t>) {
        if (n % 256 == 0) {
            using TileConfig = CKGroupedGemmTileCfg_256x256x128_32x32x32_2x2x1;
            using Runner     = CKQuantGroupedGemmRunner<GPUArch::GFX942, ADataType, BDataType, CDataType, ALayout,
                                                        BLayout, CLayout, TileConfig, AccDataType>;
            runner = std::make_unique<Runner>();
        } else if (n % 128 == 0) {
            using TileConfig = CKGroupedGemmTileCfg_256x128x128_32x32x32_2x2x1;
            using Runner     = CKQuantGroupedGemmRunner<GPUArch::GFX942, ADataType, BDataType, CDataType, ALayout,
                                                        BLayout, CLayout, TileConfig, AccDataType>;
            runner = std::make_unique<Runner>();
        } else {
            using TileConfig = CKGroupedGemmTileCfg_256x128x128_32x32x32_2x2x1_padding;
            using Runner     = CKQuantGroupedGemmRunner<GPUArch::GFX942, ADataType, BDataType, CDataType, ALayout,
                                                        BLayout, CLayout, TileConfig, AccDataType>;
            runner = std::make_unique<Runner>();
        }
    } else {
        PRIMUS_TURBO_ERROR("Grouped Gemm only support fp16/bf16/fp8/bf8");
    }
    return runner;
}
#endif


#ifdef PRIMUS_TURBO_GFX950
template <
    typename ADataType,
    typename BDataType,
    typename CDataType,
    typename AccDataType,
    typename ALayout,
    typename BLayout,
    typename CLayout
>
std::unique_ptr<CKGroupedGemmRunnerInterFace>
get_ck_grouped_gemm_instance_gfx950(const ck_tile::index_t group_num, const ck_tile::index_t m,
                             const ck_tile::index_t n, const ck_tile::index_t k){
    std::unique_ptr<CKGroupedGemmRunnerInterFace> runner = nullptr;
    if (get_current_arch() != GPUArch::GFX950) {
        PRIMUS_TURBO_ERROR("Currently Arch != gfx950");
    }
    if constexpr (std::is_same_v<ADataType, ck_tile::half_t> ||
                  std::is_same_v<ADataType, ck_tile::bfloat16_t>) {
        if (n % 256 == 0) {
            using TileConfig = CKGroupedGemmTileCfg_256x256x64_32x32x16_2x2x1;
            using Runner = CKGroupedGemmRunner<GPUArch::GFX950, ADataType, BDataType, CDataType, ALayout, BLayout,
                                               CLayout, TileConfig, AccDataType>;
            runner = std::make_unique<Runner>();
        } else if (n % 128 == 0) {
            using TileConfig = CKGroupedGemmTileCfg_256x128x64_32x32x16_2x2x1;
            using Runner = CKGroupedGemmRunner<GPUArch::GFX950, ADataType, BDataType, CDataType, ALayout, BLayout,
                                               CLayout, TileConfig, AccDataType>;
            runner = std::make_unique<Runner>();
        } else {
            using TileConfig = CKGroupedGemmTileCfg_256x128x64_32x32x16_2x2x1_padding;
            using Runner = CKGroupedGemmRunner<GPUArch::GFX950, ADataType, BDataType, CDataType, ALayout, BLayout,
                                               CLayout, TileConfig, AccDataType>;
            runner = std::make_unique<Runner>();
        }
    } else if constexpr (std::is_same_v<ADataType, ck_tile::bf8_t> ||
                         std::is_same_v<ADataType, ck_tile::fp8_t>) {
        if (n % 256 == 0) {
            using TileConfig = CKGroupedGemmTileCfg_256x256x128_16x16x128_2x2x1;
            using Runner     = CKQuantGroupedGemmRunner<GPUArch::GFX950, ADataType, BDataType, CDataType, ALayout,
                                                        BLayout, CLayout, TileConfig, AccDataType>;
            runner = std::make_unique<Runner>();
        } else if (n % 128 == 0) {
            using TileConfig = CKGroupedGemmTileCfg_256x256x128_16x16x128_2x2x1_padding;
            using Runner     = CKQuantGroupedGemmRunner<GPUArch::GFX950, ADataType, BDataType, CDataType, ALayout,
                                                        BLayout, CLayout, TileConfig, AccDataType>;
            runner = std::make_unique<Runner>();
        } else {
            using TileConfig = CKGroupedGemmTileCfg_128x128x128_32x32x64_2x2x1;
            using Runner     = CKQuantGroupedGemmRunner<GPUArch::GFX950, ADataType, BDataType, CDataType, ALayout,
                                                        BLayout, CLayout, TileConfig, AccDataType>;
            runner = std::make_unique<Runner>();
        }
    } else {
        PRIMUS_TURBO_ERROR("Grouped Gemm only support fp16/bf16/fp8/bf8");
    }
    return runner;
}
#endif


template <
    typename ADataType,
    typename BDataType,
    typename CDataType,
    typename AccDataType,
    typename ALayout,
    typename BLayout,
    typename CLayout
>
std::unique_ptr<CKGroupedGemmRunnerInterFace>
get_ck_grouped_gemm_instance(const ck_tile::index_t group_num, const ck_tile::index_t m,
                             const ck_tile::index_t n, const ck_tile::index_t k){
    const GPUArch arch = get_current_arch();
    switch (arch) {
#ifdef PRIMUS_TURBO_GFX942
        case GPUArch::GFX942: {
            return get_ck_grouped_gemm_instance_gfx942<
                ADataType, BDataType, CDataType, AccDataType,
                ALayout, BLayout, CLayout>(group_num, m, n, k);
        }
#endif
#ifdef PRIMUS_TURBO_GFX950
        case GPUArch::GFX950: {
            return get_ck_grouped_gemm_instance_gfx950<
                ADataType, BDataType, CDataType, AccDataType,
                ALayout, BLayout, CLayout>(group_num, m, n, k);
        }
#endif
        default:
            PRIMUS_TURBO_ERROR("Unsupported arch in get_ck_grouped_gemm_instance()");
    }
}


// **************** GFX942 Instantiation ****************
#ifdef PRIMUS_TURBO_GFX942
#define DECL_CK_GG_GFX942_EXTERN_INSTANCE(AType, BType, CType, ALayout, BLayout, CLayout)           \
extern template std::unique_ptr<CKGroupedGemmRunnerInterFace>                                       \
get_ck_grouped_gemm_instance_gfx942<AType, BType, CType, float, ALayout, BLayout, CLayout>(         \
    const ck_tile::index_t, const ck_tile::index_t, const ck_tile::index_t, const ck_tile::index_t);

// FP16 * FP16 = FP16
DECL_CK_GG_GFX942_EXTERN_INSTANCE(ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, RowMajor, ColMajor, RowMajor)
DECL_CK_GG_GFX942_EXTERN_INSTANCE(ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, RowMajor, RowMajor, RowMajor)
DECL_CK_GG_GFX942_EXTERN_INSTANCE(ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, ColMajor, RowMajor, RowMajor)

// BF16 * BF16 = BF16
DECL_CK_GG_GFX942_EXTERN_INSTANCE(ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t, RowMajor, ColMajor, RowMajor)
DECL_CK_GG_GFX942_EXTERN_INSTANCE(ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t, RowMajor, RowMajor, RowMajor)
DECL_CK_GG_GFX942_EXTERN_INSTANCE(ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t, ColMajor, RowMajor, RowMajor)

// FP8_E4M3 * FP8_E4M3 = FP16
DECL_CK_GG_GFX942_EXTERN_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, RowMajor, ColMajor, RowMajor)
DECL_CK_GG_GFX942_EXTERN_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, RowMajor, RowMajor, RowMajor)
DECL_CK_GG_GFX942_EXTERN_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, ColMajor, RowMajor, RowMajor)

// FP8_E4M3 * FP8_E4M3 = BF16
DECL_CK_GG_GFX942_EXTERN_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, RowMajor, ColMajor, RowMajor)
DECL_CK_GG_GFX942_EXTERN_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, RowMajor, RowMajor, RowMajor)
DECL_CK_GG_GFX942_EXTERN_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, ColMajor, RowMajor, RowMajor)

// FP8_E5M2 * FP8_E5M2 = FP16
DECL_CK_GG_GFX942_EXTERN_INSTANCE(ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, RowMajor, ColMajor, RowMajor)
DECL_CK_GG_GFX942_EXTERN_INSTANCE(ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, RowMajor, RowMajor, RowMajor)
DECL_CK_GG_GFX942_EXTERN_INSTANCE(ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, ColMajor, RowMajor, RowMajor)

// FP8_E5M2 * FP8_E5M2 = BF16
DECL_CK_GG_GFX942_EXTERN_INSTANCE(ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, RowMajor, ColMajor, RowMajor)
DECL_CK_GG_GFX942_EXTERN_INSTANCE(ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, RowMajor, RowMajor, RowMajor)
DECL_CK_GG_GFX942_EXTERN_INSTANCE(ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, ColMajor, RowMajor, RowMajor)

#undef DECL_CK_GG_GFX942_EXTERN_INSTANCE

#define DECL_CK_GG_GFX942_INSTANCE(AType, BType, CType, ALayout, BLayout, CLayout)                  \
template std::unique_ptr<CKGroupedGemmRunnerInterFace>                                              \
get_ck_grouped_gemm_instance_gfx942<AType, BType, CType, float, ALayout, BLayout, CLayout>(         \
    const ck_tile::index_t, const ck_tile::index_t, const ck_tile::index_t, const ck_tile::index_t);
#endif
// ******************************************************


// **************** GFX950 Instantiation ****************
#ifdef PRIMUS_TURBO_GFX950
#define DECL_CK_GG_GFX950_EXTERN_INSTANCE(AType, BType, CType, ALayout, BLayout, CLayout)           \
extern template std::unique_ptr<CKGroupedGemmRunnerInterFace>                                       \
get_ck_grouped_gemm_instance_gfx950<AType, BType, CType, float, ALayout, BLayout, CLayout>(         \
    const ck_tile::index_t, const ck_tile::index_t, const ck_tile::index_t, const ck_tile::index_t);

// FP16 * FP16 = FP16
DECL_CK_GG_GFX950_EXTERN_INSTANCE(ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, RowMajor, ColMajor, RowMajor)
DECL_CK_GG_GFX950_EXTERN_INSTANCE(ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, RowMajor, RowMajor, RowMajor)
DECL_CK_GG_GFX950_EXTERN_INSTANCE(ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, ColMajor, RowMajor, RowMajor)

// BF16 * BF16 = BF16
DECL_CK_GG_GFX950_EXTERN_INSTANCE(ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t, RowMajor, ColMajor, RowMajor)
DECL_CK_GG_GFX950_EXTERN_INSTANCE(ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t, RowMajor, RowMajor, RowMajor)
DECL_CK_GG_GFX950_EXTERN_INSTANCE(ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t, ColMajor, RowMajor, RowMajor)

// FP8_E4M3 * FP8_E4M3 = FP16
DECL_CK_GG_GFX950_EXTERN_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, RowMajor, ColMajor, RowMajor)
DECL_CK_GG_GFX950_EXTERN_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, RowMajor, RowMajor, RowMajor)
DECL_CK_GG_GFX950_EXTERN_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, ColMajor, RowMajor, RowMajor)

// FP8_E4M3 * FP8_E4M3 = BF16
DECL_CK_GG_GFX950_EXTERN_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, RowMajor, ColMajor, RowMajor)
DECL_CK_GG_GFX950_EXTERN_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, RowMajor, RowMajor, RowMajor)
DECL_CK_GG_GFX950_EXTERN_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, ColMajor, RowMajor, RowMajor)

// FP8_E5M2 * FP8_E5M2 = FP16
DECL_CK_GG_GFX950_EXTERN_INSTANCE(ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, RowMajor, ColMajor, RowMajor)
DECL_CK_GG_GFX950_EXTERN_INSTANCE(ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, RowMajor, RowMajor, RowMajor)
DECL_CK_GG_GFX950_EXTERN_INSTANCE(ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, ColMajor, RowMajor, RowMajor)

// FP8_E5M2 * FP8_E5M2 = BF16
DECL_CK_GG_GFX950_EXTERN_INSTANCE(ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, RowMajor, ColMajor, RowMajor)
DECL_CK_GG_GFX950_EXTERN_INSTANCE(ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, RowMajor, RowMajor, RowMajor)
DECL_CK_GG_GFX950_EXTERN_INSTANCE(ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, ColMajor, RowMajor, RowMajor)

#undef DECL_CK_GG_GFX950_EXTERN_INSTANCE

#define DECL_CK_GG_GFX950_INSTANCE(AType, BType, CType, ALayout, BLayout, CLayout)                  \
template std::unique_ptr<CKGroupedGemmRunnerInterFace>                                              \
get_ck_grouped_gemm_instance_gfx950<AType, BType, CType, float, ALayout, BLayout, CLayout>(         \
    const ck_tile::index_t, const ck_tile::index_t, const ck_tile::index_t, const ck_tile::index_t);
#endif
// ******************************************************

// clang-format on
} // namespace primus_turbo
