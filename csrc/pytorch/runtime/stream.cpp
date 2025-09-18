// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <vector>

#include "../extensions.h"
#include "primus_turbo/macros.h"

namespace primus_turbo::pytorch {

int64_t create_stream_with_cu_masks(const int device_id, const std::vector<uint32_t> &cu_masks) {
    at::cuda::CUDAGuard guard(device_id);
    hipStream_t         hip_stream;
    PRIMUS_TURBO_CHECK_HIP(
        hipExtStreamCreateWithCUMask(&hip_stream, cu_masks.size(), cu_masks.data()));
    return reinterpret_cast<int64_t>(hip_stream);
}

void destroy_stream(const int device_id, const int64_t stream_ptr) {
    at::cuda::CUDAGuard guard(device_id);
    hipStream_t         hip_stream = reinterpret_cast<hipStream_t>(stream_ptr);
    PRIMUS_TURBO_CHECK_HIP(hipStreamDestroy(hip_stream));
}

} // namespace primus_turbo::pytorch
