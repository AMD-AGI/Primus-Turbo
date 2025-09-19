// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "../ck_grouped_gemm_kernel.h"

namespace primus_turbo {

// BF16 * BF16 = BF16
DECL_CK_GG_GFX942_INSTANCE(ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t, RowMajor,
                           ColMajor, RowMajor)
DECL_CK_GG_GFX942_INSTANCE(ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t, RowMajor,
                           RowMajor, RowMajor)
DECL_CK_GG_GFX942_INSTANCE(ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t, ColMajor,
                           RowMajor, RowMajor)

} // namespace primus_turbo
