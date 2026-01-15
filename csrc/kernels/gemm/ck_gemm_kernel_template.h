// // Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
// //
// // See LICENSE for license information.

// #pragma once

// #include "ck_tile/core.hpp"
// #include "ck_tile/host.hpp"
// #include "ck_tile/host/kernel_launch.hpp"
// #include "ck_tile/ops/epilogue.hpp"
// #include "ck_tile/ops/gemm.hpp"
// #include "ck_tile/ops/gemm_quant.hpp"
// #include <hip/hip_runtime.h>
// #include <string>

// #include "ck_gemm_kernel_config.h"
// #include "primus_turbo/arch.h"

// namespace primus_turbo {

// using RowMajor = ck_tile::tensor_layout::gemm::RowMajor;
// using ColMajor = ck_tile::tensor_layout::gemm::ColumnMajor;

// template <typename Kernel>
// inline void _launch_ck_gemm_kernel(const ck_tile::stream_config &stream_cfg,
//                                    ck_tile::QuantGemmKernelArgs &kargs) {
//     constexpr int kBlockPerCu = 1;
//     const dim3    blocks      = Kernel::BlockSize();
//     dim3          grids       = Kernel::GridSize(kargs.M, kargs.N, kargs.k_batch);
//     ck_tile::launch_kernel(stream_cfg,
//                            ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
// }

// class CKGemmRunnerInterFace {
// public:
//     virtual ~CKGemmRunnerInterFace()                      = default;
//     virtual void run(const ck_tile::stream_config &stream_cfg,
//                      ck_tile::QuantGemmKernelArgs &kargs) = 0;
// };

// template <typename ADataType, typename BDataType, typename CDataType, typename ALayout,
//           typename BLayout, typename CLayout, typename TileConfig, ck_tile::QuantType QuantMode,
//           typename AccDataType = float>
// class CKQuantGemmRunner : public CKGemmRunnerInterFace {
// public:
//     using GemmShape = ck_tile::TileGemmShape<
//         ck_tile::sequence<TileConfig::M_Tile, TileConfig::N_Tile, TileConfig::K_Tile>,
//         ck_tile::sequence<TileConfig::M_Warp, TileConfig::N_Warp, TileConfig::K_Warp>,
//         ck_tile::sequence<TileConfig::M_Warp_Tile, TileConfig::N_Warp_Tile,
//                           TileConfig::K_Warp_Tile>>;
//     using TilePartitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;
//     using AQLayout        = ck_tile::tensor_layout::gemm::RowMajor;
//     using BQLayout        = ck_tile::tensor_layout::gemm::ColumnMajor;

//     using GemmUniversalTraits = ck_tile::TileGemmQuantTraits<
//         TileConfig::kPadM, TileConfig::kPadN, TileConfig::kPadK, false /*PreshuffleQuant*/,
//         false /*PreshuffleB*/, ALayout, BLayout, CLayout, QuantMode, AQLayout, BQLayout,
//         false /*TransposeC*/, false /*DoubleSmemBuffer*/, false /*UsePersistentKernel*/>;

//     // Select appropriate quantization group sizes based on A layout
//     // A is always 1D quantization: <1, 1, 128>
//     // B quantization depends on A's layout:
//     //   - When A is RowMajor (R+C or R+R): B uses 2D quantization <1, 128, 128>
//     //   - When A is ColMajor (C+R): B uses 1D quantization <1, 1, 128>
//     using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
//     using BQuantGroupSize = std::conditional_t<
//         std::is_same_v<ALayout, RowMajor>,
//         ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>, // A is RowMajor: B uses 2D
//         ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>>;  // A is ColMajor: B uses 1D

//     // Select appropriate Problem and Pipeline based on QuantMode
//     // For RowColQuant and TensorQuant, use GemmRowColTensorQuantPipelineProblem +
//     // GemmPipelineAgBgCrCompV3 For ABQuantGrouped, use GemmABQuantPipelineProblem +
//     // ABQuantGemmPipelineAgBgCrCompV3
//     using QuantGemmProblem = std::conditional_t<
//         QuantMode == ck_tile::QuantType::RowColQuant ||
//             QuantMode == ck_tile::QuantType::TensorQuant,
//         ck_tile::GemmRowColTensorQuantPipelineProblem<ADataType, BDataType, AccDataType,
//                                                       AccDataType, GemmShape, GemmUniversalTraits>,
//         ck_tile::GemmABQuantPipelineProblem<ADataType,           // ADataType
//                                             AccDataType,         // AQDataType: same as AccDataType
//                                             BDataType,           // BDataType
//                                             AccDataType,         // BQDataType: same as AccDataType
//                                             AccDataType,         // CDataType (accumulator)
//                                             GemmShape,           // BlockGemmShape
//                                             GemmUniversalTraits, // Traits
//                                             AQuantGroupSize,     // AQuantGroupSize: 1D <1, 1, 128>
//                                             BQuantGroupSize, // BQuantGroupSize: 2D <1, 128, 128> or
//                                                              // 1D <1, 1, 128>
//                                             false /*TransposeC*/>>;

//     // V3: Select appropriate Pipeline based on QuantMode
//     using GemmPipeline =
//         std::conditional_t<QuantMode == ck_tile::QuantType::RowColQuant ||
//                                QuantMode == ck_tile::QuantType::TensorQuant,
//                            ck_tile::GemmPipelineAgBgCrCompV3<QuantGemmProblem>,
//                            ck_tile::ABQuantGemmPipelineAgBgCrCompV3<QuantGemmProblem>>;

//     static constexpr ck_tile::memory_operation_enum MemoryOp = ck_tile::memory_operation_enum::set;
//     using GemmEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
//         ADataType, BDataType, ck_tile::tuple<>, AccDataType, CDataType, ck_tile::tuple<>, CLayout,
//         ck_tile::element_wise::PassThrough, TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
//         TileConfig::M_Warp, TileConfig::N_Warp, TileConfig::M_Warp_Tile, TileConfig::N_Warp_Tile,
//         TileConfig::K_Warp_Tile, false, MemoryOp>>;

//     using Kernel = ck_tile::QuantGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue, QuantMode>;

// public:
//     void run(const ck_tile::stream_config &stream_cfg,
//              ck_tile::QuantGemmKernelArgs &kargs) override {
//         _launch_ck_gemm_kernel<Kernel>(stream_cfg, kargs);
//     }
// };

// template <GPUArch arch, typename ADataType, typename BDataType, typename CDataType,
//           typename ALayout, typename BLayout, typename CLayout, typename TileConfig,
//           ck_tile::QuantType QuantMode, typename AccDataType = float>
// class CKQuantGemmRunnerWithArch
//     : public CKQuantGemmRunner<ADataType, BDataType, CDataType, ALayout, BLayout, CLayout,
//                                TileConfig, QuantMode, AccDataType> {
//     static_assert(TileConfig::arch == arch, "Tile arch mismatch with Runner arch");
// };

// // ***********************************************************************************
// #define DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN(ARCH, A, B, C, AL, BL, CL, TileCfg, QuantMode)       \
//     extern template class CKQuantGemmRunner<A, B, C, AL, BL, CL, TileCfg, QuantMode, float>;       \
//     extern template class CKQuantGemmRunnerWithArch<ARCH, A, B, C, AL, BL, CL, TileCfg, QuantMode, \
//                                                     float>;

// #define DECL_CK_QGEMM_RUNNER_WITH_ARCH(ARCH, A, B, C, AL, BL, CL, TileCfg, QuantMode)              \
//     template class CKQuantGemmRunner<A, B, C, AL, BL, CL, TileCfg, QuantMode, float>;              \
//     template class CKQuantGemmRunnerWithArch<ARCH, A, B, C, AL, BL, CL, TileCfg, QuantMode, float>;

// // Macro for RowColQuant and TensorQuant only (for configs with M_Warp=2, N_Warp=2)
// #define APPLY_CK_GEMM_TENSOR_ROW_QUANT_LAYOUT_WITH_ARCH(MACRO, ARCH, A, B, C, TileCfg)             \
//     MACRO(ARCH, A, B, C, RowMajor, ColMajor, RowMajor, TileCfg, ck_tile::QuantType::RowColQuant)   \
//     MACRO(ARCH, A, B, C, RowMajor, ColMajor, RowMajor, TileCfg, ck_tile::QuantType::TensorQuant)   \
//     MACRO(ARCH, A, B, C, RowMajor, RowMajor, RowMajor, TileCfg, ck_tile::QuantType::RowColQuant)   \
//     MACRO(ARCH, A, B, C, RowMajor, RowMajor, RowMajor, TileCfg, ck_tile::QuantType::TensorQuant)   \
//     MACRO(ARCH, A, B, C, ColMajor, RowMajor, RowMajor, TileCfg, ck_tile::QuantType::RowColQuant)   \
//     MACRO(ARCH, A, B, C, ColMajor, RowMajor, RowMajor, TileCfg, ck_tile::QuantType::TensorQuant)

// // Macro for ABQuantGrouped only (for configs with M_Warp=1, N_Warp=4)
// #define APPLY_CK_GEMM_ABQUANT_LAYOUT_WITH_ARCH(MACRO, ARCH, A, B, C, TileCfg)                      \
//     MACRO(ARCH, A, B, C, RowMajor, ColMajor, RowMajor, TileCfg,                                    \
//           ck_tile::QuantType::ABQuantGrouped)                                                      \
//     MACRO(ARCH, A, B, C, ColMajor, RowMajor, RowMajor, TileCfg,                                    \
//           ck_tile::QuantType::ABQuantGrouped)                                                      \
//     MACRO(ARCH, A, B, C, RowMajor, RowMajor, RowMajor, TileCfg, ck_tile::QuantType::ABQuantGrouped)

// // Full macro including all quant modes
// #define APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(MACRO, ARCH, A, B, C, TileCfg)                          \
//     MACRO(ARCH, A, B, C, RowMajor, ColMajor, RowMajor, TileCfg, ck_tile::QuantType::RowColQuant)   \
//     MACRO(ARCH, A, B, C, RowMajor, ColMajor, RowMajor, TileCfg, ck_tile::QuantType::TensorQuant)   \
//     MACRO(ARCH, A, B, C, RowMajor, RowMajor, RowMajor, TileCfg, ck_tile::QuantType::RowColQuant)   \
//     MACRO(ARCH, A, B, C, RowMajor, RowMajor, RowMajor, TileCfg, ck_tile::QuantType::TensorQuant)   \
//     MACRO(ARCH, A, B, C, ColMajor, RowMajor, RowMajor, TileCfg, ck_tile::QuantType::RowColQuant)   \
//     MACRO(ARCH, A, B, C, ColMajor, RowMajor, RowMajor, TileCfg, ck_tile::QuantType::TensorQuant)   \
//     MACRO(ARCH, A, B, C, RowMajor, ColMajor, RowMajor, TileCfg,                                    \
//           ck_tile::QuantType::ABQuantGrouped)                                                      \
//     MACRO(ARCH, A, B, C, ColMajor, RowMajor, RowMajor, TileCfg,                                    \
//           ck_tile::QuantType::ABQuantGrouped)                                                      \
//     MACRO(ARCH, A, B, C, RowMajor, RowMajor, RowMajor, TileCfg, ck_tile::QuantType::ABQuantGrouped)

// // ***********************************************************************************
// // clang-format off
// #ifdef PRIMUS_TURBO_GFX942
// // FP8_E4M3 * FP8_E4M3 = FP16
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX942_CKGemmTileCfg_256x256x128_32x32x32_2x2x1)
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1)
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1_padding)
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1_padK)
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX942_CKGemmTileCfg_128x128x128_32x32x32_2x2x1)
// // FP8_E4M3 * FP8_E4M3 = BF16
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX942_CKGemmTileCfg_256x256x128_32x32x32_2x2x1)
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1)
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1_padding)
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1_padK)
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX942_CKGemmTileCfg_128x128x128_32x32x32_2x2x1)
// // FP8_E5M2 * FP8_E5M2 = FP16
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX942_CKGemmTileCfg_256x256x128_32x32x32_2x2x1)
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1)
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1_padding)
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1_padK)
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX942_CKGemmTileCfg_128x128x128_32x32x32_2x2x1)
// // FP8_E5M2 * FP8_E5M2 = BF16
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX942_CKGemmTileCfg_256x256x128_32x32x32_2x2x1)
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1)
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1_padding)
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1_padK)
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX942, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX942_CKGemmTileCfg_128x128x128_32x32x32_2x2x1)
// #endif

// // ***********************************************************************************
// #ifdef PRIMUS_TURBO_GFX950
// // FP8_E4M3 * FP8_E4M3 = FP16
// // For 2x2x1 configs: RowColQuant and TensorQuant
// APPLY_CK_GEMM_TENSOR_ROW_QUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX950_CKGemmTileCfg_256x256x128_16x16x128_2x2x1)
// APPLY_CK_GEMM_TENSOR_ROW_QUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX950_CKGemmTileCfg_256x128x128_16x16x128_2x2x1)
// APPLY_CK_GEMM_TENSOR_ROW_QUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX950_CKGemmTileCfg_256x128x128_16x16x128_2x2x1_padK)
// APPLY_CK_GEMM_TENSOR_ROW_QUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX950_CKGemmTileCfg_128x128x128_32x32x64_2x2x1_padding)
// // For 1x4x1 config: ABQuantGrouped
// APPLY_CK_GEMM_ABQUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX950_CKGemmTileCfg_128x128x128_16x16x128_1x4x1)
// // FP8_E4M3 * FP8_E4M3 = BF16
// APPLY_CK_GEMM_TENSOR_ROW_QUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX950_CKGemmTileCfg_256x256x128_16x16x128_2x2x1)
// APPLY_CK_GEMM_TENSOR_ROW_QUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX950_CKGemmTileCfg_256x128x128_16x16x128_2x2x1)
// APPLY_CK_GEMM_TENSOR_ROW_QUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX950_CKGemmTileCfg_256x128x128_16x16x128_2x2x1_padK)
// APPLY_CK_GEMM_TENSOR_ROW_QUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX950_CKGemmTileCfg_128x128x128_32x32x64_2x2x1_padding)
// APPLY_CK_GEMM_ABQUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX950_CKGemmTileCfg_128x128x128_16x16x128_1x4x1)
// // FP8_E5M2 * FP8_E5M2 = FP16
// APPLY_CK_GEMM_TENSOR_ROW_QUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX950_CKGemmTileCfg_256x256x128_16x16x128_2x2x1)
// APPLY_CK_GEMM_TENSOR_ROW_QUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX950_CKGemmTileCfg_256x128x128_16x16x128_2x2x1)
// APPLY_CK_GEMM_TENSOR_ROW_QUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX950_CKGemmTileCfg_256x128x128_16x16x128_2x2x1_padK)
// APPLY_CK_GEMM_TENSOR_ROW_QUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX950_CKGemmTileCfg_128x128x128_32x32x64_2x2x1_padding)
// APPLY_CK_GEMM_ABQUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, GFX950_CKGemmTileCfg_128x128x128_16x16x128_1x4x1)
// // FP8_E5M2 * FP8_E5M2 = BF16
// APPLY_CK_GEMM_TENSOR_ROW_QUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX950_CKGemmTileCfg_256x256x128_16x16x128_2x2x1)
// APPLY_CK_GEMM_TENSOR_ROW_QUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX950_CKGemmTileCfg_256x128x128_16x16x128_2x2x1)
// APPLY_CK_GEMM_TENSOR_ROW_QUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX950_CKGemmTileCfg_256x128x128_16x16x128_2x2x1_padK)
// APPLY_CK_GEMM_TENSOR_ROW_QUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX950_CKGemmTileCfg_128x128x128_32x32x64_2x2x1_padding)
// APPLY_CK_GEMM_ABQUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH_EXTERN, GPUArch::GFX950, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, GFX950_CKGemmTileCfg_128x128x128_16x16x128_1x4x1)
// #endif

// // clang-format on
// } // namespace primus_turbo
