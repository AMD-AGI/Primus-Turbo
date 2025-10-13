// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "../ck_grouped_gemm_kernel_template.h"

namespace primus_turbo {
// clang-format off
#ifdef PRIMUS_TURBO_GFX950
// FP8_E4M3 * FP8_E4M3 = FP16
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, GFX950_CKGroupedGemmTileCfg_128x128x128_32x32x64_2x2x1)
// FP8_E4M3 * FP8_E4M3 = BF16
APPLY_CK_GG_ALL_LAYOUT_WITH_ARCH(DECL_CK_QGG_RUNNER_WITH_ARCH, GPUArch::GFX950, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, GFX950_CKGroupedGemmTileCfg_128x128x128_32x32x64_2x2x1)
#endif
// clang-format on
} // namespace primus_turbo
