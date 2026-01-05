// // Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
// //
// // See LICENSE for license information.

// #include "kernels/gemm/ck_gemm_kernel_template.h"

// namespace primus_turbo {
// // clang-format off
// #ifdef PRIMUS_TURBO_GFX942

// // FP8_E5M2 * FP8_E5M2 = BF16
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH, GPUArch::GFX942,
// ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t,
// GFX942_CKGemmTileCfg_256x256x128_32x32x32_2x2x1)
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH, GPUArch::GFX942,
// ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t,
// GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1)
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH, GPUArch::GFX942,
// ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t,
// GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1_padding)
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH, GPUArch::GFX942,
// ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t,
// GFX942_CKGemmTileCfg_256x128x128_32x32x32_2x2x1_padK)
// APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH, GPUArch::GFX942,
// ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t,
// GFX942_CKGemmTileCfg_128x128x128_32x32x32_2x2x1) #endif
// // clang-format on
// } // namespace primus_turbo
