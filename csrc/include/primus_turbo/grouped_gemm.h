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
    const AType *a_ptr = nullptr; // [total_M_in,  K] (input layout: may be per-group padded)
    const BType *b_ptr = nullptr; // [group_num, N, K]
    CType       *c_ptr = nullptr; // [total_M_out, N] (output layout: c_group_offs_ptr)

    const dtype::float8_e8m0 *a_scale_ptr = nullptr; // [total_M_in, K/32]
    const dtype::float8_e8m0 *b_scale_ptr = nullptr; // [group_num, N, K/32]

    // Per-group real-row count and starting offset in the input layout.
    // Padding rows beyond ``group_lens_ptr[g]`` are not visited.
    const int64_t *group_lens_ptr = nullptr; // [group_num]
    const int64_t *group_offs_ptr = nullptr; // [group_num+1]

    // Output-row offsets (always non-null; launcher must fall back to
    // group_offs_ptr when the unpadded-input case is intended).  When
    // distinct from group_offs_ptr, the GEMM writes group g starting at
    // ``c_group_offs_ptr[g]`` (unpadded layout), allowing padded inputs to
    // be compressed away as part of the store.
    const int64_t *c_group_offs_ptr = nullptr; // [group_num+1]

    // Bitmask used to round M_g up to the input-layout padding alignment
    // for SRD bound calculation: ``M_g_in = (M_g + mask) & ~mask``.
    //   127 → padded input (per-group regions aligned to 128 rows)
    //     0 → unpadded input (M_g_in == M_g)
    // Branchless on the kernel side; lets the outer kernel resolve
    // c_group_offs_ptr unconditionally, removing a per-tile null-check
    // and the duplicate i64 split-load chain that came with it.
    int32_t group_m_padding_align_size = 0;

    int32_t group_num = 0;
    int32_t total_m   = 0; // total INPUT rows (a's first dim)
    int32_t n         = 0;
    int32_t k         = 0;

    // Tight per-group tile-M upper bound: max_g ceil(M_g / 256).
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
// Per-group dB[g] = LHS[g] @ RHS[g]^T, with LHS (N, total_M), RHS
// (K, total_M), dB (group_num, N, K); reduction is over total_M.
// Constraints: n % 16 == 0, k % 16 == 0, M_g % 32 == 0.

template <typename AType, typename BType, typename CType> struct TurboGroupedGemmMXFP8WgradParams {
    const AType *lhs_ptr = nullptr; // (N, total_M)
    const BType *rhs_ptr = nullptr; // (K, total_M)
    CType       *db_ptr  = nullptr; // (group_num, N, K)

    const dtype::float8_e8m0 *lhs_scale_ptr = nullptr; // (N, total_M/32)
    const dtype::float8_e8m0 *rhs_scale_ptr = nullptr; // (K, total_M/32)

    const int64_t *group_lens_ptr = nullptr; // [group_num]
    const int64_t *group_offs_ptr = nullptr; // [group_num+1]

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
