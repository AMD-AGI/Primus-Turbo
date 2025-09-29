// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "../ck_gemm_kernel.h"

namespace primus_turbo {
// clang-format off
#ifdef PRIMUS_TURBO_GFX950
// FP8_E4M3 * FP8_E4M3 = FP16 - RowColQuant
DECL_CK_GEMM_GFX950_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, RowMajor, ColMajor, RowMajor, ck_tile::QuantType::RowColQuant)
DECL_CK_GEMM_GFX950_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, RowMajor, RowMajor, RowMajor, ck_tile::QuantType::RowColQuant)
DECL_CK_GEMM_GFX950_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, ColMajor, RowMajor, RowMajor, ck_tile::QuantType::RowColQuant)

// FP8_E4M3 * FP8_E4M3 = BF16 - RowColQuant
DECL_CK_GEMM_GFX950_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, RowMajor, ColMajor, RowMajor, ck_tile::QuantType::RowColQuant)
DECL_CK_GEMM_GFX950_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, RowMajor, RowMajor, RowMajor, ck_tile::QuantType::RowColQuant)
DECL_CK_GEMM_GFX950_INSTANCE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, ColMajor, RowMajor, RowMajor, ck_tile::QuantType::RowColQuant)
#endif
// clang-format on
} // namespace primus_turbo
