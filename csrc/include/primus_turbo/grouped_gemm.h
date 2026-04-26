// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "ck_tile/ops/gemm_quant/pipeline/tile_gemm_quant_traits.hpp"
#include <cstdint>
#include <hip/hip_runtime.h>

#include "primus_turbo/common.h"
#include "primus_turbo/dtype.h"

namespace primus_turbo {

std::int64_t get_ck_grouped_gemm_args_sizes(const int group_num);
std::int64_t get_ck_grouped_gemm_fp8_args_sizes(const int group_num);

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
};

template <typename AType, typename BType, typename CType>
struct CKGroupedGemmParams : public GroupedGemmParams<AType, BType, CType> {
    void *args_ptr = nullptr;
};

template <typename AType, typename BType, typename CType, typename ACCType>
struct CKGroupedGemmFP8Params : public CKGroupedGemmParams<AType, BType, CType> {
    const ACCType *aq_ptr = nullptr;
    const ACCType *bq_ptr = nullptr;
};

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

//==================================================================
//  Turbo Grouped GEMM (MXFP8, NT layout, GFX950)
//==================================================================

template <typename AType, typename BType, typename CType> struct TurboGroupedGemmMXFP8Params {
    const AType *a_ptr = nullptr; // [total_M, K]
    const BType *b_ptr = nullptr; // [group_num, N, K]
    CType       *c_ptr = nullptr; // [total_M, N]

    const dtype::float8_e8m0 *a_scale_ptr = nullptr; // [total_M, K/32], E8M0
    const dtype::float8_e8m0 *b_scale_ptr = nullptr; // [group_num, N, K/32], E8M0

    const int64_t *group_lens_ptr = nullptr; // [group_num]
    const int64_t *group_offs_ptr = nullptr; // [group_num+1], cumulative row offsets

    int32_t group_num = 0;
    int32_t total_m   = 0;
    int32_t n         = 0;
    int32_t k         = 0;

    // Exact sum of per-group tile counts: sum_g ceil(M_g / 256).  MUST match
    // the kernel's flat-tile dispatcher; launching extra padding workgroups
    // (which early-exit) introduces a ~0.025% intermittent output race.
    // Computed host-side from group_lens by the wrapper.
    int32_t grid_x = 0;

    void       *workspace      = nullptr;
    size_t      workspace_size = 0;
    hipStream_t stream         = nullptr;
};

size_t turbo_grouped_gemm_mxfp8_workspace_size(int32_t total_m, int32_t group_num, int32_t n,
                                               int32_t k);

template <typename AType, typename BType, typename CType>
void turbo_grouped_gemm_mxfp8_impl(const TurboGroupedGemmMXFP8Params<AType, BType, CType> &params);

//==================================================================
//  Turbo Grouped GEMM (MXFP8) — variable-K wgrad path
//==================================================================
//
// Computes per-group dB[g] = LHS[g] @ RHS[g]^T  where:
//   LHS = dC^T (col-quant transposed), shape (N, total_M)
//   RHS = A^T  (col-quant transposed), shape (K, total_M)
//   dB output, shape (group_num, N, K)
// Reduction is along total_M; per-group columns are [group_offs[g], group_offs[g+1]).
//
// Constraints:
//   - n % 16 == 0, k % 16 == 0 (preshuffle row-block alignment)
//   - per-group M_g % 32 == 0 (so group_offs[g] is a multiple of MX_BLOCK_SIZE,
//     ensuring scale tiles do not cross group boundaries)

template <typename AType, typename BType, typename CType> struct TurboGroupedGemmMXFP8WgradParams {
    const AType *lhs_ptr = nullptr;  // dC^T fp8: (N, total_M)
    const BType *rhs_ptr = nullptr;  // A^T  fp8: (K, total_M)
    CType       *db_ptr  = nullptr;  // dB:       (group_num, N, K)

    const dtype::float8_e8m0 *lhs_scale_ptr = nullptr;  // (N, total_M/32) E8M0
    const dtype::float8_e8m0 *rhs_scale_ptr = nullptr;  // (K, total_M/32) E8M0

    const int64_t *group_lens_ptr = nullptr;  // [group_num]
    const int64_t *group_offs_ptr = nullptr;  // [group_num+1]

    int32_t group_num = 0;
    int32_t total_m   = 0;
    int32_t n         = 0;
    int32_t k         = 0;

    void       *workspace      = nullptr;
    size_t      workspace_size = 0;
    hipStream_t stream         = nullptr;
};

size_t turbo_grouped_gemm_mxfp8_wgrad_workspace_size(int32_t total_m, int32_t n, int32_t k);

template <typename AType, typename BType, typename CType>
void turbo_grouped_gemm_mxfp8_wgrad_impl(
    const TurboGroupedGemmMXFP8WgradParams<AType, BType, CType> &params);

//==================================================================
//  hipBLASLt Grouped GEMM
//==================================================================

void hipblaslt_grouped_gemm(const HipblasltGroupedGemmParams &params, const bool pre_sync);

template <typename IndexType>
void compute_group_offs(const IndexType *group_lens_ptr, IndexType *group_offs_ptr,
                        const int64_t group_num, hipStream_t stream);

} // namespace primus_turbo
