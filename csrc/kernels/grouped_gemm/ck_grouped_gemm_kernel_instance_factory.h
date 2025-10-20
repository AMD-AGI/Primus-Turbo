// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "ck_grouped_gemm_kernel_template.h"

namespace primus_turbo {
// clang-format off

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType,
          typename ALayout, typename BLayout, typename CLayout, ck_tile::QuantType QuantMode>
std::unique_ptr<CKGroupedGemmRunnerInterFace>
get_ck_grouped_gemm_instance(const ck_tile::index_t group_num,
                             const ck_tile::index_t m,
                             const ck_tile::index_t n,
                             const ck_tile::index_t k);

#define DECL_GET_CK_GG_INSTANCE_EXTERN(A, B, C, AL, BL, CL, QM)     \
    extern template std::unique_ptr<CKGroupedGemmRunnerInterFace>   \
    get_ck_grouped_gemm_instance<A, B, C, float, AL, BL, CL, QM>(   \
        const ck_tile::index_t, const ck_tile::index_t,             \
        const ck_tile::index_t, const ck_tile::index_t);

#define APPLY_GET_CK_GG_INSTANCE_ALL_LAYOUT(MACRO, A, B, C) \
    MACRO(A, B, C, RowMajor, ColMajor, RowMajor, ck_tile::QuantType::RowColQuant)                  \
    MACRO(A, B, C, RowMajor, ColMajor, RowMajor, ck_tile::QuantType::TensorQuant)                  \
    MACRO(A, B, C, RowMajor, RowMajor, RowMajor, ck_tile::QuantType::RowColQuant)                  \
    MACRO(A, B, C, RowMajor, RowMajor, RowMajor, ck_tile::QuantType::TensorQuant)                  \
    MACRO(A, B, C, ColMajor, RowMajor, RowMajor, ck_tile::QuantType::RowColQuant)                  \
    MACRO(A, B, C, ColMajor, RowMajor, RowMajor, ck_tile::QuantType::TensorQuant)

// FP16 * FP16 = FP16
APPLY_GET_CK_GG_INSTANCE_ALL_LAYOUT(DECL_GET_CK_GG_INSTANCE_EXTERN, ck_tile::half_t, ck_tile::half_t, ck_tile::half_t)
// BF16 * BF16 = BF16
APPLY_GET_CK_GG_INSTANCE_ALL_LAYOUT(DECL_GET_CK_GG_INSTANCE_EXTERN, ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t)
// FP8_E4M3 * FP8_E4M3 = FP16
APPLY_GET_CK_GG_INSTANCE_ALL_LAYOUT(DECL_GET_CK_GG_INSTANCE_EXTERN, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t)
// FP8_E4M3 * FP8_E4M3 = BF16
APPLY_GET_CK_GG_INSTANCE_ALL_LAYOUT(DECL_GET_CK_GG_INSTANCE_EXTERN, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t)
// FP8_E5M2 * FP8_E5M2 = FP16
APPLY_GET_CK_GG_INSTANCE_ALL_LAYOUT(DECL_GET_CK_GG_INSTANCE_EXTERN, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t)
// FP8_E5M2 * FP8_E5M2 = BF16
APPLY_GET_CK_GG_INSTANCE_ALL_LAYOUT(DECL_GET_CK_GG_INSTANCE_EXTERN, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t)

#undef DECL_GET_CK_GG_INSTANCE_EXTERN

// clang-format on
} // namespace primus_turbo
