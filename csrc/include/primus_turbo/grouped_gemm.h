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

    // When true, the caller has already preshuffled `b_scale_ptr` into the
    // 16x4 layout that `turbo_grouped_gemm_mxfp8_*_persistent_kernel` expects
    // (via a previous `turbo_preshuffle_mxfp8_scale_16x4` call).  The launch
    // skips the per-call B-scale preshuffle kernel and the workspace's
    // `b_scale_preshuf` slot is unused.  Used by the Python wrapper to cache
    // the preshuffle output across forward+dgrad calls when B is a weight.
    bool b_scale_preshuffled = false;

    void       *workspace      = nullptr;
    size_t      workspace_size = 0;
    hipStream_t stream         = nullptr;
};

// When `b_scale_preshuffled` is true the caller has already preshuffled B's
// E8M0 scales into the 16x4 uint32 layout the kernel expects (cached across
// fwd+dgrad on the Python side because B is a weight tensor) and the kernel
// will source `params.b_scale_ptr` directly, leaving the workspace's
// `b_scale_preshuf` slot unused.  Skip allocating that slot — for the
// DSv3-GateUP shape (G=16, N=4096, K=7168) this drops `~58.7 MB` of unused
// workspace per call, halving the cudaMalloc/caching-allocator footprint and
// the `at::empty(...)` cost on cache misses.  Default keeps backward-compat:
// callers that don't yet pass the preshuffled flag still allocate the union.
size_t turbo_grouped_gemm_mxfp8_workspace_size(int32_t total_m, int32_t group_num, int32_t n,
                                               int32_t k, bool b_scale_preshuffled = false);

template <typename AType, typename BType, typename CType>
void turbo_grouped_gemm_mxfp8_impl(const TurboGroupedGemmMXFP8Params<AType, BType, CType> &params);

// Standalone launcher for the 16x4 column-major preshuffle that converts an
// E8M0 scale tensor into the layout the persistent grouped GEMM kernel
// consumes via `reinterpret_cast<const uint32_t *>` (see
// `preshuffle_scale_16x4_kernel` in `turbo_gemm_mxfp8_kernel.h`).  Wrapped
// here so callers (e.g. the PyTorch op `turbo_preshuffle_mxfp8_scale_16x4`)
// do not need to include the kernel header from a `.cpp` translation unit.
//
// Layout:
//   - `in_ptr`  : E8M0 scale buffer, length rows * cols bytes
//   - `out_ptr` : preshuffled output buffer, same byte length
// Grid: rows/16 workgroups of 64 threads.
void turbo_preshuffle_mxfp8_scale_16x4_launch(const uint8_t *in_ptr, uint32_t *out_ptr,
                                              int rows, int cols, hipStream_t stream);

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
