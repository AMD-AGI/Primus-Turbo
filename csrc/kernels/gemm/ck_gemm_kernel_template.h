// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include <hip/hip_runtime.h>
#include <string>

#include "ck_gemm_kernel_config.h"
#include "primus_turbo/arch.h"

namespace primus_turbo {

using RowMajor = ck_tile::tensor_layout::gemm::RowMajor;
using ColMajor = ck_tile::tensor_layout::gemm::ColumnMajor;

template <typename Kernel>
inline void _launch_ck_gemm_kernel(const ck_tile::stream_config &stream_cfg,
                                   ck_tile::QuantGemmKernelArgs &kargs, uint32_t num_cu) {
    constexpr int kBlockPerCu = 1;
    const dim3    blocks      = Kernel::BlockSize();
    dim3          grids       = Kernel::GridSize(kargs.M, kargs.N, kargs.k_batch);
    ck_tile::launch_kernel(stream_cfg,
                           ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
}

class CKGemmRunnerInterFace {
public:
    virtual ~CKGemmRunnerInterFace()  = default;
    virtual void run(const ck_tile::stream_config &stream_cfg, ck_tile::QuantGemmKernelArgs &kargs,
                     uint32_t num_cu) = 0;
};

template <typename ADataType, typename BDataType, typename CDataType, typename ALayout,
          typename BLayout, typename CLayout, typename TileConfig, ck_tile::QuantType QuantMode,
          typename AccDataType = float>
class CKQuantGemmRunner : public CKGemmRunnerInterFace {
public:
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileConfig::M_Tile, TileConfig::N_Tile, TileConfig::K_Tile>,
        ck_tile::sequence<TileConfig::M_Warp, TileConfig::N_Warp, TileConfig::K_Warp>,
        ck_tile::sequence<TileConfig::M_Warp_Tile, TileConfig::N_Warp_Tile,
                          TileConfig::K_Warp_Tile>>;
    using TilePartitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;
    using AQLayout        = ck_tile::tensor_layout::gemm::RowMajor;
    using BQLayout        = ck_tile::tensor_layout::gemm::ColumnMajor;

    using GemmUniversalTraits = ck_tile::TileGemmQuantTraits<
        TileConfig::kPadM, TileConfig::kPadN, TileConfig::kPadK, false /*PreshuffleQuant*/,
        false /*PreshuffleB*/, ALayout, BLayout, CLayout, QuantMode, AQLayout, BQLayout,
        false /*TransposeC*/, false /*DoubleSmemBuffer*/, false /*UsePersistentKernel*/>;

    using QuantGemmProblem =
        ck_tile::GemmRowColTensorQuantPipelineProblem<ADataType, BDataType, AccDataType,
                                                      AccDataType, GemmShape, GemmUniversalTraits>;

    // V3
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<QuantGemmProblem>;

    static constexpr ck_tile::memory_operation_enum MemoryOp = ck_tile::memory_operation_enum::set;
    using GemmEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
        ADataType, BDataType, ck_tile::tuple<>, AccDataType, CDataType, ck_tile::tuple<>, CLayout,
        ck_tile::element_wise::PassThrough, TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
        TileConfig::M_Warp, TileConfig::N_Warp, TileConfig::M_Warp_Tile, TileConfig::N_Warp_Tile,
        TileConfig::K_Warp_Tile, false, MemoryOp>>;

    using Kernel = ck_tile::QuantGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue, QuantMode>;

public:
    void run(const ck_tile::stream_config &stream_cfg, ck_tile::QuantGemmKernelArgs &kargs,
             uint32_t num_cu) override {
        _launch_ck_gemm_kernel<Kernel>(stream_cfg, kargs, num_cu);
    }
};

template <GPUArch arch, typename ADataType, typename BDataType, typename CDataType,
          typename ALayout, typename BLayout, typename CLayout, typename TileConfig,
          ck_tile::QuantType QuantMode, typename AccDataType = float>
class CKQuantGemmRunnerWithArch
    : public CKQuantGemmRunner<ADataType, BDataType, CDataType, ALayout, BLayout, CLayout,
                               TileConfig, QuantMode, AccDataType> {
    static_assert(TileConfig::arch == arch, "Tile arch mismatch with Runner arch");
};

// ***********************************************************************************
#define DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN(ARCH, A, B, C, AL, BL, CL, TileCfg, QuantMode)       \
    extern template class CKQuantGemmRunner<A, B, C, AL, BL, CL, TileCfg, QuantMode, float>;       \
    extern template class CKQuantGemmRunnerWithArch<ARCH, A, B, C, AL, BL, CL, TileCfg, QuantMode, \
                                                    float>;

#define DECL_CK_QGEMM_RUNNER_WITH_ARCH(ARCH, A, B, C, AL, BL, CL, TileCfg, QuantMode)              \
    template class CKQuantGemmRunner<A, B, C, AL, BL, CL, TileCfg, QuantMode, float>;              \
    template class CKQuantGemmRunnerWithArch<ARCH, A, B, C, AL, BL, CL, TileCfg, QuantMode, float>;

#define APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(MACRO, ARCH, A, B, C, TileCfg)                          \
    MACRO(ARCH, A, B, C, RowMajor, ColMajor, RowMajor, TileCfg, ck_tile::QuantType::RowColQuant)   \
    MACRO(ARCH, A, B, C, RowMajor, ColMajor, RowMajor, TileCfg, ck_tile::QuantType::TensorQuant)   \
    MACRO(ARCH, A, B, C, RowMajor, RowMajor, RowMajor, TileCfg, ck_tile::QuantType::RowColQuant)   \
    MACRO(ARCH, A, B, C, RowMajor, RowMajor, RowMajor, TileCfg, ck_tile::QuantType::TensorQuant)   \
    MACRO(ARCH, A, B, C, ColMajor, RowMajor, RowMajor, TileCfg, ck_tile::QuantType::RowColQuant)   \
    MACRO(ARCH, A, B, C, ColMajor, RowMajor, RowMajor, TileCfg, ck_tile::QuantType::TensorQuant)

// ***********************************************************************************
// clang-format off
#ifdef PRIMUS_TURBO_GFX942
// FP8_E4M3 * FP8_E4M3 = FP16
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX942_CKGemmTileCfg_256x256x128_32x32x32_2x2x1)
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1)
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1_padding)

// FP8_E4M3 * FP8_E4M3 = BF16
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX942_CKGemmTileCfg_256x256x128_32x32x32_2x2x1)
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1)
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1_padding)

// FP8_E5M2 * FP8_E5M2 = FP16
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX942_CKGemmTileCfg_256x256x128_32x32x32_2x2x1)
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1)
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1_padding)

// FP8_E5M2 * FP8_E5M2 = BF16
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX942_CKGemmTileCfg_256x256x128_32x32x32_2x2x1)
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1)
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1_padding)
#endif

// ***********************************************************************************
#ifdef PRIMUS_TURBO_GFX950
// FP8_E4M3 * FP8_E4M3 = FP16
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX950_CKGemmTileCfg_256x256x128_16x16x128_2x2x1)
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX950_CKGemmTileCfg_256x128x128_16x16x128_2x2x1)
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX950_CKGemmTileCfg_128x128x128_32x32x64_2x2x1_padding)

// FP8_E4M3 * FP8_E4M3 = BF16
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX950_CKGemmTileCfg_256x256x128_16x16x128_2x2x1)
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX950_CKGemmTileCfg_256x128x128_16x16x128_2x2x1)
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX950_CKGemmTileCfg_128x128x128_32x32x64_2x2x1_padding)

// FP8_E5M2 * FP8_E5M2 = FP16
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX950_CKGemmTileCfg_256x256x128_16x16x128_2x2x1)
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX950_CKGemmTileCfg_256x128x128_16x16x128_2x2x1)
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX950_CKGemmTileCfg_128x128x128_32x32x64_2x2x1_padding)

// FP8_E5M2 * FP8_E5M2 = BF16
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX950_CKGemmTileCfg_256x256x128_16x16x128_2x2x1)
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX950_CKGemmTileCfg_256x128x128_16x16x128_2x2x1)
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX950_CKGemmTileCfg_128x128x128_32x32x64_2x2x1_padding)
#endif

// clang-format on
} // namespace primus_turbo
