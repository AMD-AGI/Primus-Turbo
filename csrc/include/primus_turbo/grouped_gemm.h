// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "ck_tile/core.hpp"
#include <cstdint>
#include <hip/hip_runtime.h>

#include "primus_turbo/common.h"

namespace primus_turbo {

std::int64_t get_ck_grouped_gemm_args_sizes(const int group_num);

template <typename AType, typename BType, typename CType> struct CKGroupedGemmParams {
    void *args_ptr = nullptr;

    const AType *a_ptr = nullptr;
    const BType *b_ptr = nullptr;
    CType       *c_ptr = nullptr;

    const int64_t *group_lens_ptr = nullptr;
    const int64_t *group_offs_ptr = nullptr;

    bool transA = false;
    bool transB = false;

    ck_tile::index_t group_num = 0;
    ck_tile::index_t m         = 0;
    ck_tile::index_t n         = 0;
    ck_tile::index_t k         = 0;

    hipStream_t stream = nullptr;
    uint32_t    num_cu = 0;
};

template <typename AType, typename BType, typename CType, typename ACCType>
struct CKGroupedGemmFP8Params {
    void *args_ptr = nullptr;

    const AType   *a_ptr  = nullptr;
    const BType   *b_ptr  = nullptr;
    CType         *c_ptr  = nullptr;
    const ACCType *aq_ptr = nullptr;
    const ACCType *bq_ptr = nullptr;

    const int64_t *group_lens_ptr = nullptr;
    const int64_t *group_offs_ptr = nullptr;

    bool transA = false;
    bool transB = false;

    ck_tile::index_t group_num = 0;
    ck_tile::index_t m         = 0;
    ck_tile::index_t n         = 0;
    ck_tile::index_t k         = 0;

    hipStream_t stream = nullptr;
    uint32_t    num_cu = 0;
};

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType = float>
void ck_grouped_gemm(const CKGroupedGemmParams<ADataType, BDataType, CDataType> &params);

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType = float>
void ck_grouped_gemm_variable_k(const CKGroupedGemmParams<ADataType, BDataType, CDataType> &params);

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType = float>
void ck_grouped_gemm_fp8(
    const CKGroupedGemmFP8Params<ADataType, BDataType, CDataType, AccDataType> &params);

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType = float>
void ck_grouped_gemm_fp8_variable_k(
    const CKGroupedGemmFP8Params<ADataType, BDataType, CDataType, AccDataType> &params);

template <typename IndexType>
void compute_group_offs(const IndexType *group_lens_ptr, IndexType *group_offs_ptr,
                        const ck_tile::index_t group_num, hipStream_t stream);

} // namespace primus_turbo
