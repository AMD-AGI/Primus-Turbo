// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <hipblaslt/hipblaslt.h>

namespace primus_turbo {

int64_t get_hipblaslt_workspace_size_in_byte();

void hipblaslt_gemm_impl(const void *A, const hipDataType A_type, const int64_t lda,
                         const void *scaleA_inv, hipblasOperation_t transA, const void *B,
                         const hipDataType B_type, const int64_t ldb, const void *scaleB_inv,
                         hipblasOperation_t transB, void *D, const hipDataType D_type,
                         const int64_t ldd, const int64_t m, const int64_t n, const int64_t k,
                         void *workspace, const int64_t workspace_size, const bool use_fp8,
                         const bool use_rowwise, hipblasLtHandle_t handle, hipStream_t stream);

} // namespace primus_turbo
