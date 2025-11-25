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
                                      void* args_ptr, uint32_t num_cu) {
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
        ALayout, BLayout, CLayout
    >;


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
    typename ADataType,
    typename BDataType,
    typename CDataType,
    typename ALayout,
    typename BLayout,
    typename CLayout,
    typename TileConfig,
    ck_tile::QuantType QuantMode,
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

    using AQLayout = ck_tile::tensor_layout::gemm::RowMajor;
    using BQLayout = ck_tile::tensor_layout::gemm::ColumnMajor;

    using GemmUniversalTraits = ck_tile::TileGemmQuantTraits<
        TileConfig::kPadM,
        TileConfig::kPadN,
        TileConfig::kPadK,
        false,
        false,
        ALayout,
        BLayout,
        CLayout,
        QuantMode,
        AQLayout,
        BQLayout,
        false,
        TileConfig::DoubleSmemBuffer,
        true
    >;

    using QuantGemmProblem = ck_tile::GemmRowColTensorQuantPipelineProblem<
        ADataType,
        BDataType,
        AccDataType,
        AccDataType,
        GemmShape,
        GemmUniversalTraits
    >;

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


template <
    GPUArch arch,
    typename ADataType,
    typename BDataType,
    typename CDataType,
    typename ALayout,
    typename BLayout,
    typename CLayout,
    typename TileConfig,
    ck_tile::QuantType QuantMode,
    typename AccDataType=float
>
class CKQuantGroupedGemmRunnerWithArch : public CKQuantGroupedGemmRunner<
        ADataType, BDataType, CDataType,
        ALayout, BLayout, CLayout,
        TileConfig, QuantMode, AccDataType> {
    static_assert(TileConfig::arch == arch, "Tile arch mismatch with Runner arch");
};

// ***********************************************************************************
#define DECL_CK_GG_RUNNER_EXTERN(A, B, C, AL, BL, CL, TileCfg)                      \
    extern template class CKGroupedGemmRunner<A, B, C, AL, BL, CL, TileCfg, float>;

#define DECL_CK_GG_RUNNER(A, B, C, AL, BL, CL, TileCfg)                             \
    template class CKGroupedGemmRunner<A, B, C, AL, BL, CL, TileCfg, float>;

#define APPLY_CK_GG_ALL_LAYOUT(MACRO, A, B, C, TileCfg)   \
    MACRO(A, B, C, RowMajor, ColMajor, RowMajor, TileCfg) \
    MACRO(A, B, C, RowMajor, RowMajor, RowMajor, TileCfg) \
    MACRO(A, B, C, ColMajor, RowMajor, RowMajor, TileCfg)

// ***********************************************************************************
#define DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN(ARCH, A, B, C, AL, BL, CL, TileCfg, QuantMode)                         \
    extern template class CKQuantGroupedGemmRunner<A, B, C, AL, BL, CL, TileCfg, QuantMode, float>;                \
    extern template class CKQuantGroupedGemmRunnerWithArch<ARCH, A, B, C, AL, BL, CL, TileCfg, QuantMode, float>;

#define DECL_CK_QGG_RUNNER_WITH_ARCH(ARCH, A, B, C, AL, BL, CL, TileCfg, QuantMode)                        \
    template class CKQuantGroupedGemmRunner<A, B, C, AL, BL, CL, TileCfg, QuantMode, float>;               \
    template class CKQuantGroupedGemmRunnerWithArch<ARCH, A, B, C, AL, BL, CL, TileCfg, QuantMode, float>;

#define APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(MACRO, ARCH, A, B, C, TileCfg)   \
    MACRO(ARCH, A, B, C, RowMajor, ColMajor, RowMajor, TileCfg, ck_tile::QuantType::RowColQuant)   \
    MACRO(ARCH, A, B, C, RowMajor, ColMajor, RowMajor, TileCfg, ck_tile::QuantType::TensorQuant)   \
    MACRO(ARCH, A, B, C, RowMajor, RowMajor, RowMajor, TileCfg, ck_tile::QuantType::RowColQuant)   \
    MACRO(ARCH, A, B, C, RowMajor, RowMajor, RowMajor, TileCfg, ck_tile::QuantType::TensorQuant)   \
    MACRO(ARCH, A, B, C, ColMajor, RowMajor, RowMajor, TileCfg, ck_tile::QuantType::RowColQuant)   \
    MACRO(ARCH, A, B, C, ColMajor, RowMajor, RowMajor, TileCfg, ck_tile::QuantType::TensorQuant)

// ***********************************************************************************
#if defined(PRIMUS_TURBO_GFX942) || defined(PRIMUS_TURBO_GFX950)
// FP16 * FP16 = FP16
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER_EXTERN, ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, CKGroupedGemmTileCfg_256x256x64_32x32x16_2x2x1)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER_EXTERN, ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, CKGroupedGemmTileCfg_256x128x64_32x32x16_2x2x1)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER_EXTERN, ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, CKGroupedGemmTileCfg_256x128x64_32x32x16_2x2x1_padding)

// BF16 * BF16 = BF16
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER_EXTERN, ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t, CKGroupedGemmTileCfg_256x256x64_32x32x16_2x2x1)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER_EXTERN, ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t, CKGroupedGemmTileCfg_256x128x64_32x32x16_2x2x1)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER_EXTERN, ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t, CKGroupedGemmTileCfg_256x128x64_32x32x16_2x2x1_padding)
#endif

// ***********************************************************************************
#ifdef PRIMUS_TURBO_GFX942
// FP8_E4M3 * FP8_E4M3 = FP16
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX942_CKGroupedGemmTileCfg_256x256x128_32x32x32_2x2x1)
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX942_CKGroupedGemmTileCfg_256x128x128_16x16x32_2x2x1)
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX942_CKGroupedGemmTileCfg_256x128x128_16x16x32_2x2x1_padding)

// FP8_E4M3 * FP8_E4M3 = BF16
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX942_CKGroupedGemmTileCfg_256x256x128_32x32x32_2x2x1)
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX942_CKGroupedGemmTileCfg_256x128x128_16x16x32_2x2x1)
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX942_CKGroupedGemmTileCfg_256x128x128_16x16x32_2x2x1_padding)

// FP8_E5M2 * FP8_E5M2 = FP16
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX942_CKGroupedGemmTileCfg_256x256x128_32x32x32_2x2x1)
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX942_CKGroupedGemmTileCfg_256x128x128_16x16x32_2x2x1)
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX942_CKGroupedGemmTileCfg_256x128x128_16x16x32_2x2x1_padding)

// FP8_E5M2 * FP8_E5M2 = BF16
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX942_CKGroupedGemmTileCfg_256x256x128_32x32x32_2x2x1)
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX942_CKGroupedGemmTileCfg_256x128x128_16x16x32_2x2x1)
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX942_CKGroupedGemmTileCfg_256x128x128_16x16x32_2x2x1_padding)
#endif

// ***********************************************************************************
#ifdef PRIMUS_TURBO_GFX950
// FP8_E4M3 * FP8_E4M3 = FP16
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX950_CKGroupedGemmTileCfg_256x256x128_16x16x128_2x2x1)
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX950_CKGroupedGemmTileCfg_256x256x128_16x16x128_2x2x1_padding)
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX950_CKGroupedGemmTileCfg_128x128x128_32x32x64_2x2x1)

// FP8_E4M3 * FP8_E4M3 = BF16
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX950_CKGroupedGemmTileCfg_256x256x128_16x16x128_2x2x1)
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX950_CKGroupedGemmTileCfg_256x256x128_16x16x128_2x2x1_padding)
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX950_CKGroupedGemmTileCfg_128x128x128_32x32x64_2x2x1)

// FP8_E5M2 * FP8_E5M2 = FP16
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX950_CKGroupedGemmTileCfg_256x256x128_16x16x128_2x2x1)
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX950_CKGroupedGemmTileCfg_256x256x128_16x16x128_2x2x1_padding)
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX950_CKGroupedGemmTileCfg_128x128x128_32x32x64_2x2x1)

// FP8_E5M2 * FP8_E5M2 = BF16
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX950_CKGroupedGemmTileCfg_256x256x128_16x16x128_2x2x1)
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX950_CKGroupedGemmTileCfg_256x256x128_16x16x128_2x2x1_padding)
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX950_CKGroupedGemmTileCfg_128x128x128_32x32x64_2x2x1)
#endif

// clang-format on
} // namespace primus_turbo
