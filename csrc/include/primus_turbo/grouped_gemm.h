// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#ifdef PRIMUS_TURBO_BUILD_CK_BACKEND
#include "ck_tile/ops/gemm_quant/pipeline/tile_gemm_quant_traits.hpp"
#endif
#include <cstdint>
#include <hip/hip_runtime.h>

#include "primus_turbo/common.h"

namespace primus_turbo {

#ifdef PRIMUS_TURBO_BUILD_CK_BACKEND
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
};

#ifdef PRIMUS_TURBO_BUILD_CK_BACKEND
template <typename AType, typename BType, typename CType>
struct CKGroupedGemmParams : public GroupedGemmParams<AType, BType, CType> {
    void *args_ptr = nullptr;
};

template <typename AType, typename BType, typename CType, typename ACCType>
struct CKGroupedGemmFP8Params : public CKGroupedGemmParams<AType, BType, CType> {
    const ACCType *aq_ptr = nullptr;
    const ACCType *bq_ptr = nullptr;
};
#endif // PRIMUS_TURBO_BUILD_CK_BACKEND

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
#ifdef PRIMUS_TURBO_BUILD_CK_BACKEND

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

#endif // PRIMUS_TURBO_BUILD_CK_BACKEND

//==================================================================
//  hipBLASLt Grouped MXFP8 GEMM (gfx1250)
//==================================================================

// Per-group MXFP8 grouped GEMM descriptor. All host vectors are sized group_num
// and indexed by the (host-accessible) group id. Device buffers (a_padded,
// a_swz_scale, b_swz_scale, c_data) are pre-allocated/freed by the caller.
// Per group g this runs the NT GEMM  D = A[mpad,K] @ B[npad,K]^T  (contraction K,
// VEC32_UE8M0 swizzled scales), matching the dense MX op.
struct HipblasltMXGroupedGemmParams {
    const void *a_padded    = nullptr; // [sum(a_mpad), K] fp8 (128-padded activation / wgrad lhs)
    const void *a_swz_scale = nullptr; // concatenated per-group groups-of-4 swizzled e8m0
    const void *b_data      = nullptr; // [sum(b_mpad), K] fp8 (weight / wgrad rhs, 128-padded)
    const void *b_swz_scale = nullptr; // concatenated per-group swizzled e8m0
    void       *c_data      = nullptr; // output

    hipDataType ab_type;
    hipDataType c_type;

    int64_t ldc = 0; // output row stride (real N for fwd / OUT_N for wgrad; 128-multiple)

    const int64_t *group_lens_host = nullptr; // host-accessible group lens
    const int64_t *a_row_off       = nullptr; // A row start per group (in a_padded rows)
    const int64_t *a_scale_off     = nullptr; // element offset of group's swizzled A scale
    const int64_t *b_row_off       = nullptr; // B row start per group (in b_data rows)
    const int64_t *b_scale_off     = nullptr; // element offset of group's swizzled B scale
    const int64_t *c_off_bytes     = nullptr; // byte offset of group's output block
    const int64_t *a_mpad          = nullptr; // padded A rows (m) per group
    const int64_t *b_mpad          = nullptr; // padded B rows (n) per group
    const int64_t *kdim            = nullptr; // per-group contraction (fwd: K; wgrad: M_g)

    int               group_num = 0;
    hipStream_t       stream    = nullptr;
    void             *workspace = nullptr;
    hipblasLtHandle_t handle    = nullptr;
};

// Pack per-group real rows from a padded-block (m_pad rows) GEMM output into the
// tight output the MoE forward expects (real rows at group_offs_out). One launch.
void mxfp8_pack_output_grouped(const void *padded, void *tight, int64_t N, int dtype_bytes,
                               const int *src_row_off, const int *dst_row_off, const int *row_len,
                               const int64_t *elem_pref, int group_num, int64_t total_out_elems,
                               hipStream_t stream);

// Batched device kernels (one launch over all groups) used to prepare the MX
// grouped GEMM inputs. Declared here so the binding can drive them.
void mxfp8_swizzle_scale_grouped(const uint8_t *in_scale, uint8_t *out_scale, int64_t ks,
                                 int64_t ks_pad, const int *row_in_off, const int *row_len,
                                 const int *mpad, const int64_t *out_blk_off,
                                 const int64_t *blk_pref, int group_num, int64_t total_out_tiles,
                                 hipStream_t stream);

void mxfp8_pad_data_grouped(const uint8_t *in_data, uint8_t *out_data, int64_t K,
                            const int *row_in_off, const int *row_len, const int64_t *out_row_off,
                            const int64_t *elem_pref, int group_num, int64_t total_out_elems,
                            hipStream_t stream);

void hipblaslt_grouped_gemm_mxfp8(const HipblasltMXGroupedGemmParams &params);

//==================================================================
//  hipBLASLt Grouped GEMM
//==================================================================

void hipblaslt_grouped_gemm(const HipblasltGroupedGemmParams &params, const bool pre_sync);

template <typename IndexType>
void compute_group_offs(const IndexType *group_lens_ptr, IndexType *group_offs_ptr,
                        const int64_t group_num, hipStream_t stream);

} // namespace primus_turbo
