// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

// Backend-agnostic grouped GEMM host utilities. These are always built, regardless of which
// GEMM backends (CK / turbo) are enabled, so that helpers such as group-offset computation
// remain available even when a particular backend is disabled.

#include "primus_turbo/grouped_gemm.h"
#include "pytorch/extensions.h"

namespace primus_turbo::pytorch {

at::Tensor grouped_gemm_compute_offs(at::Tensor &group_lens) {
    PRIMUS_TURBO_CHECK(group_lens.scalar_type() == at::kLong,
                       "group_lens must be of type Long (int64_t)");

    // Output holds an exclusive prefix sum, so it has one more element than the input.
    at::Tensor group_offs = at::empty({group_lens.numel() + 1}, group_lens.options());

    auto stream = at::cuda::getCurrentCUDAStream();
    compute_group_offs<int64_t>(reinterpret_cast<const int64_t *>(group_lens.data_ptr()),
                                reinterpret_cast<int64_t *>(group_offs.data_ptr()),
                                group_lens.numel(), stream);

    return group_offs;
}

} // namespace primus_turbo::pytorch
