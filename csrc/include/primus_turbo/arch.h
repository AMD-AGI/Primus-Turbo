// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "primus_turbo/macros.h"
#include <cstdint>
#include <hip/hip_runtime.h>

namespace primus_turbo {

enum class GPUArch { GFX942, GFX950, GFX1250, UNKNOWN };

inline GPUArch get_current_arch() {
    static GPUArch cached_arch = []() -> GPUArch {
        hipDeviceProp_t prop;
        hipError_t      err = hipGetDeviceProperties(&prop, 0);
        if (err != hipSuccess) {
            return GPUArch::UNKNOWN;
        }
        if (prop.major == 9 && prop.minor == 4)
            return GPUArch::GFX942;
        if (prop.major == 9 && prop.minor == 5)
            return GPUArch::GFX950;
        if (prop.major == 12 && prop.minor == 5)
            return GPUArch::GFX1250;
        return GPUArch::UNKNOWN;
    }();
    return cached_arch;
}

inline bool is_gfx950() {
    return get_current_arch() == GPUArch::GFX950;
}

inline bool is_gfx942() {
    return get_current_arch() == GPUArch::GFX942;
}

inline bool is_gfx1250() {
    return get_current_arch() == GPUArch::GFX1250;
}

// gfx1250 = 32, gfx942 / gfx950 (and others) = 64.
inline int warp_size() {
    if (is_gfx1250()) {
        return 32;
    } else if (is_gfx950() || is_gfx942()) {
        return 64;
    } else {
        PRIMUS_TURBO_ERROR("Unknown architecture");
        return -1;
    }
}

inline int32_t get_multi_processor_count(const int32_t device_id) {
    int32_t num_cu = 0;
    PRIMUS_TURBO_CHECK_HIP(
        hipDeviceGetAttribute(&num_cu, hipDeviceAttributeMultiprocessorCount, device_id));
    return num_cu;
}

inline int32_t get_max_shmem_per_block(const int32_t device_id) {
    int32_t max_shmem = 0;
    PRIMUS_TURBO_CHECK_HIP(
        hipDeviceGetAttribute(&max_shmem, hipDeviceAttributeMaxSharedMemoryPerBlock, device_id));
    return max_shmem;
}

} // namespace primus_turbo
