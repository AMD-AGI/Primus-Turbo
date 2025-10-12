// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "../ck_grouped_gemm_kernel_instance_factory.h"

namespace primus_turbo {
// clang-format off
#ifdef PRIMUS_TURBO_GFX950
// BF16 * BF16 = BF16
DECL_CK_GG_GFX950_INSTANCE(ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t, RowMajor, ColMajor, RowMajor)
DECL_CK_GG_GFX950_INSTANCE(ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t, RowMajor, RowMajor, RowMajor)
DECL_CK_GG_GFX950_INSTANCE(ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t, ColMajor, RowMajor, RowMajor)
#endif
// clang-format on
} // namespace primus_turbo
