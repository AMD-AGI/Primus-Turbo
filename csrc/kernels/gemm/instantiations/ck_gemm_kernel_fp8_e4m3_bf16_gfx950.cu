// // Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
// //
// // See LICENSE for license information.

// #include "kernels/gemm/ck_gemm_kernel_template.h"

// namespace primus_turbo {
// // clang-format off
// #ifdef PRIMUS_TURBO_GFX950

// // FP8_E4M3 * FP8_E4M3 = BF16
// // For 2x2x1 configs: RowColQuant and TensorQuant
// APPLY_CK_GEMM_TENSOR_ROW_QUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX950_CKGemmTileCfg_256x256x128_16x16x128_2x2x1)
// APPLY_CK_GEMM_TENSOR_ROW_QUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX950_CKGemmTileCfg_256x128x128_16x16x128_2x2x1)
// APPLY_CK_GEMM_TENSOR_ROW_QUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX950_CKGemmTileCfg_256x128x128_16x16x128_2x2x1_padK)
// APPLY_CK_GEMM_TENSOR_ROW_QUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX950_CKGemmTileCfg_128x128x128_32x32x64_2x2x1_padding)
// // For 1x4x1 config: ABQuantGrouped
// APPLY_CK_GEMM_ABQUANT_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX950_CKGemmTileCfg_128x128x128_16x16x128_1x4x1)
// #endif
// // clang-format on
// } // namespace primus_turbo
