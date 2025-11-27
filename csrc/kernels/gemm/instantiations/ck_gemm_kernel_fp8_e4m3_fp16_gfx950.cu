// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "kernels/gemm/ck_gemm_kernel_template.h"

namespace primus_turbo {
// clang-format off
#ifdef PRIMUS_TURBO_GFX950

// FP8_E4M3 * FP8_E4M3 = FP16
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX950_CKGemmTileCfg_256x256x128_16x16x128_2x2x1)
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX950_CKGemmTileCfg_256x128x128_16x16x128_2x2x1)
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX950_CKGemmTileCfg_128x128x128_32x32x64_2x2x1_padding)
APPLY_CK_GEMM_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGEMM_RUNNER_WITH_ARCH, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX950_CKGemmTileCfg_128x128x128_32x32x64_2x2x1)
#endif
// clang-format on
} // namespace primus_turbo
