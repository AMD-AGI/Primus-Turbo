#pragma once

#include "ck_tile/core.hpp"
#include <cstdint>
#include <hip/hip_runtime.h>

#include "primus_turbo/common.h"

namespace primus_turbo {

std::int64_t get_ck_grouped_gemm_args_sizes(const int group_num);

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType = float>
void ck_grouped_gemm(void *args_ptr, const ADataType *a_ptr, const BDataType *b_ptr,
                     CDataType *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
                     const bool transA, const bool transB, const ck_tile::index_t group_num,
                     const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t k,
                     hipStream_t stream);

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType = float>
void ck_grouped_gemm_variable_k(void *args_ptr, const ADataType *a_ptr, const BDataType *b_ptr,
                                CDataType *c_ptr, const int64_t *group_lens_ptr,
                                const int64_t *group_offs_ptr, const bool transA, const bool transB,
                                const ck_tile::index_t group_num, const ck_tile::index_t m,
                                const ck_tile::index_t n, const ck_tile::index_t k,
                                hipStream_t stream);

} // namespace primus_turbo
