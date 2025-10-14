// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "kernels/grouped_gemm/ck_grouped_gemm_kernel_template.h"

namespace primus_turbo {
// clang-format off
#if defined(PRIMUS_TURBO_GFX942) || defined(PRIMUS_TURBO_GFX950)
// BF16 * BF16 = BF16
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t, CKGroupedGemmTileCfg_256x256x64_32x32x16_2x2x1)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t, CKGroupedGemmTileCfg_256x128x64_32x32x16_2x2x1)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t, CKGroupedGemmTileCfg_256x128x64_32x32x16_2x2x1_padding)
#endif
// clang-format on
} // namespace primus_turbo
