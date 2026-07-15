// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#ifdef BUILD_CK_BACKEND
#include "ck_tile/ops/gemm_quant/pipeline/tile_gemm_quant_traits.hpp"
#endif
#include <cstdint>
#include <hip/hip_runtime.h>

#include "primus_turbo/common.h"

namespace primus_turbo {

#ifdef BUILD_CK_BACKEND
std::int64_t get_ck_grouped_gemm_args_sizes(const int group_num);
std::int64_t get_ck_grouped_gemm_fp8_args_sizes(const int group_num);
#endif

std::int64_t get_hipblaslt_grouped_gemm_workspace_size();

//==================================================================
//  Grouped GEMM Params
//==================================================================

template <typename AType, typename BType, typename CType> struct GroupedGemmParams {
    const AType *a_ptr = nullptr;
    const BType *b_ptr = nullptr;
    CType       *c_ptr = nullptr;

    const int64_t *group_lens_ptr = nullptr;
    const int64_t *group_offs_ptr = nullptr;

    bool transA = false;
    bool transB = false;

    int32_t group_num = 0;
    int32_t m         = 0;
    int32_t n         = 0;
    int32_t k         = 0;

    hipStream_t stream = nullptr;
    uint32_t    num_cu = 0;

    // Work-stealing toggle. When true, ws_counter_ptr must point to an
    // int32 buffer of size (NUM_XCDS_WS + 2): per-XCD slots [0..NUM_XCDS_WS-1],
    // global slot [NUM_XCDS_WS], done slot [NUM_XCDS_WS + 1]. On MI355X that
    // is 10 ints. The buffer is zeroed once at allocation by the Python wrapper;
    // the kernel self-resets (the last-out CTA writes zeros), so no per-launch
    // zeroing is required at the C++ binding. Persistent CTAs claim tiles via
    // atomicAdd; `ws_local_per_xcd` selects the mode:
    //   = 0                          : global-only (single counter)
    //   = ceil(total_tiles / 8)      : per-XCD-only (8 counters, no global)
    //   = anything in between        : hierarchical (per-XCD + global tail)
    // See vendored grouped_gemm_kernel_ws.hpp for details.
    bool     work_steal       = false;
    int32_t *ws_counter_ptr   = nullptr;
    int32_t  ws_local_per_xcd = 0;
};

#ifdef BUILD_CK_BACKEND
template <typename AType, typename BType, typename CType>
struct CKGroupedGemmParams : public GroupedGemmParams<AType, BType, CType> {
    void *args_ptr = nullptr;
};

template <typename AType, typename BType, typename CType, typename ACCType>
struct CKGroupedGemmFP8Params : public CKGroupedGemmParams<AType, BType, CType> {
    const ACCType *aq_ptr = nullptr;
    const ACCType *bq_ptr = nullptr;
};
#endif // BUILD_CK_BACKEND

struct HipblasltGroupedGemmParams {
    const void          *a_ptr       = nullptr;
    const void          *a_scale_ptr = nullptr;
    hipDataType          a_type;
    std::vector<int64_t> a_shape;

    const void          *b_ptr       = nullptr;
    const void          *b_scale_ptr = nullptr;
    hipDataType          b_type;
    std::vector<int64_t> b_shape;

    void                *c_ptr = nullptr;
    hipDataType          c_type;
    std::vector<int64_t> c_shape;

    const int64_t *group_lens_ptr = nullptr;
    const int64_t *group_offs_ptr = nullptr;
    bool           transA         = false;
    bool           transB         = false;
    int32_t        group_num      = 0;
    hipStream_t    stream         = nullptr;
    void          *workspace      = nullptr;

    bool use_low_precision = false;

    hipblasLtHandle_t            handle     = nullptr;
    hipblasLtMatmulMatrixScale_t scale_mode = HIPBLASLT_MATMUL_MATRIX_SCALE_END;
};

//==================================================================
//  CK Grouped GEMM
//==================================================================
#ifdef BUILD_CK_BACKEND

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType = float>
void ck_grouped_gemm(const CKGroupedGemmParams<ADataType, BDataType, CDataType> &params);

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType = float>
void ck_grouped_gemm_variable_k(const CKGroupedGemmParams<ADataType, BDataType, CDataType> &params);

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType,
          ck_tile::QuantType QuantMode>
void ck_grouped_gemm_fp8(
    const CKGroupedGemmFP8Params<ADataType, BDataType, CDataType, AccDataType> &params);

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType,
          ck_tile::QuantType QuantMode>
void ck_grouped_gemm_fp8_variable_k(
    const CKGroupedGemmFP8Params<ADataType, BDataType, CDataType, AccDataType> &params);

#endif // BUILD_CK_BACKEND

//==================================================================
//  hipBLASLt Grouped GEMM
//==================================================================

void hipblaslt_grouped_gemm(const HipblasltGroupedGemmParams &params, const bool pre_sync);

template <typename IndexType>
void compute_group_offs(const IndexType *group_lens_ptr, IndexType *group_offs_ptr,
                        const int64_t group_num, hipStream_t stream);

} // namespace primus_turbo
