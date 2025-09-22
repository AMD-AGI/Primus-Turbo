// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "../ck_grouped_gemm_kernel.h"

namespace primus_turbo {
// clang-format off
#ifdef PRIMUS_TURBO_GFX950
// FP16 * FP16 = FP16
DECL_CK_GG_GFX950_INSTANCE(ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, RowMajor, ColMajor, RowMajor)
DECL_CK_GG_GFX950_INSTANCE(ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, RowMajor, RowMajor, RowMajor)
DECL_CK_GG_GFX950_INSTANCE(ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, ColMajor, RowMajor, RowMajor)
#endif
// clang-format on
} // namespace primus_turbo
