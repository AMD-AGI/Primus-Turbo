// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <mutex>

#include "primus_turbo/gemm.h"
#include "primus_turbo/grouped_gemm.h"
#include "pytorch/extensions.h"
#include "pytorch/type_traits.h"

namespace primus_turbo::pytorch {

inline HipblasltGroupedGemmParams
make_hipblaslt_grouped_gemm_params(const at::Tensor &a, const at::Tensor &b, at::Tensor &c,
                                   const at::Tensor &group_lens, const at::Tensor &group_offs,
                                   bool transA, bool transB, at::Tensor workspace) {
    HipblasltGroupedGemmParams params;

    params.a_ptr   = reinterpret_cast<void *>(a.data_ptr());
    params.a_type  = get_hipblaslt_dtype(a.scalar_type());
    params.a_shape = a.sizes().vec();

    params.b_ptr   = reinterpret_cast<void *>(b.data_ptr());
    params.b_type  = get_hipblaslt_dtype(b.scalar_type());
    params.b_shape = b.sizes().vec();

    params.c_ptr   = reinterpret_cast<void *>(c.data_ptr());
    params.c_type  = get_hipblaslt_dtype(c.scalar_type());
    params.c_shape = c.sizes().vec();

    params.group_lens_ptr = reinterpret_cast<const int64_t *>(group_lens.data_ptr());
    params.group_offs_ptr = reinterpret_cast<const int64_t *>(group_offs.data_ptr());
    params.transA         = transA;
    params.transB         = transB;
    params.group_num      = group_lens.numel();
    params.stream         = at::cuda::getCurrentCUDAStream();
    params.workspace      = workspace.data_ptr();
    params.handle         = at::cuda::getCurrentCUDABlasLtHandle();
    return params;
}

at::Tensor hipblaslt_grouped_gemm(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                                  at::Tensor &group_offs, const bool transA, const bool transB,
                                  const bool pre_sync) {
    // Check
    PRIMUS_TURBO_CHECK(is_16bit_floating_point_dtype(a.scalar_type()),
                       "hipblaslt_grouped_gemm only supports float16 and bfloat16");
    PRIMUS_TURBO_CHECK(is_16bit_floating_point_dtype(b.scalar_type()),
                       "hipblaslt_grouped_gemm only supports float16 and bfloat16");
    PRIMUS_TURBO_CHECK(group_lens.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(group_offs.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(a.scalar_type() == b.scalar_type(), "a and b dtype mismatch");

    // Create output tensor
    at::Tensor c;
    if (transA) {
        const int64_t bs = group_lens.numel();
        const int64_t m  = a.size(1);
        const int64_t n  = transB ? b.size(0) : b.size(1);
        c                = at::empty({bs, m, n}, a.options());
    } else {
        const int64_t m = a.size(0);
        const int64_t n = transB ? b.size(1) : b.size(2);
        c               = at::empty({m, n}, a.options());
    }

    const int64_t workspace_size =
        primus_turbo::get_hipblaslt_grouped_gemm_workspace_size(group_lens.numel());
    at::Tensor workspace = at::empty(
        {workspace_size}, at::TensorOptions().dtype(at::kByte).device(group_lens.device()));

    auto params = make_hipblaslt_grouped_gemm_params(a, b, c, group_lens, group_offs, transA,
                                                     transB, workspace);
    primus_turbo::hipblaslt_grouped_gemm(params, pre_sync);
    return c;
}

} // namespace primus_turbo::pytorch
