// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

// Standalone, CK-free implementation of compute_group_offs. This is the single definition
// of the kernel and is always built, regardless of which GEMM backends are enabled, so the
// symbol is available to both CK and non-CK host bindings.

#include <cstdint>
#include <hip/hip_runtime.h>

namespace primus_turbo {

template <typename IndexType>
__global__ void compute_group_offs_device(const IndexType *group_lens_ptr,
                                          IndexType *group_offs_ptr, const int group_num) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        group_offs_ptr[0] = 0;
    }

    if (idx < group_num) {
        // Exclusive-prefix-sum to produce group offsets.
        IndexType cumsum = 0;
        for (int i = 0; i < idx; i++) {
            cumsum += group_lens_ptr[i];
        }
        group_offs_ptr[idx + 1] = cumsum + group_lens_ptr[idx];
    }
}

template <typename IndexType>
void compute_group_offs(const IndexType *group_lens_ptr, IndexType *group_offs_ptr,
                        const int64_t group_num, hipStream_t stream) {
    const int threads_per_block = 256;
    const int blocks = static_cast<int>((group_num + threads_per_block - 1) / threads_per_block);

    compute_group_offs_device<IndexType><<<blocks, threads_per_block, 0, stream>>>(
        group_lens_ptr, group_offs_ptr, static_cast<int>(group_num));
}

template void compute_group_offs<int64_t>(const int64_t *group_lens_ptr, int64_t *group_offs_ptr,
                                          const int64_t group_num, hipStream_t stream);

} // namespace primus_turbo
