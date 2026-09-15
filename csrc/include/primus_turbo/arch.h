/***************************************************************************************************
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

#pragma once

#include "primus_turbo/macros.h"
#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>

namespace primus_turbo {

enum class GPUArch { GFX942, GFX950, GFX1250, UNKNOWN };

inline GPUArch arch_from_gcn_name(const char *name) {
    if (name == nullptr || name[0] == '\0') {
        return GPUArch::UNKNOWN;
    }
    // hipDeviceProp_t.gcnArchName is like "gfx942:sramecc+:xnack-".
    if (std::strstr(name, "gfx1250") != nullptr)
        return GPUArch::GFX1250;
    if (std::strstr(name, "gfx950") != nullptr)
        return GPUArch::GFX950;
    if (std::strstr(name, "gfx942") != nullptr)
        return GPUArch::GFX942;
    return GPUArch::UNKNOWN;
}

inline GPUArch arch_from_major_minor(int major, int minor) {
    // HIP reports gfx942 as (9,4) or, on some stacks, (9,42).
    if (major == 9 && (minor == 4 || minor == 42))
        return GPUArch::GFX942;
    if (major == 9 && (minor == 5 || minor == 50))
        return GPUArch::GFX950;
    if (major == 12 && (minor == 5 || minor == 50))
        return GPUArch::GFX1250;
    return GPUArch::UNKNOWN;
}

inline GPUArch detect_current_arch() {
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        return GPUArch::UNKNOWN;
    }
    GPUArch from_name = arch_from_gcn_name(prop.gcnArchName);
    if (from_name != GPUArch::UNKNOWN) {
        return from_name;
    }
    return arch_from_major_minor(prop.major, prop.minor);
}

inline GPUArch get_current_arch() {
    static GPUArch cached_arch = GPUArch::UNKNOWN;
    if (cached_arch == GPUArch::UNKNOWN) {
        cached_arch = detect_current_arch();
    }
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

// gfx1250 = 32, gfx942 / gfx950 (and other CDNA) = 64.
// Host-side callers (JAX abstract eval, workspace sizing) must not throw when
// HIP is not ready: a failed query used to be cached as UNKNOWN forever.
inline int warp_size() {
    GPUArch arch = get_current_arch();
    if (arch == GPUArch::GFX1250) {
        return 32;
    }
    if (arch == GPUArch::GFX942 || arch == GPUArch::GFX950) {
        return 64;
    }
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) == hipSuccess &&
        (prop.warpSize == 32 || prop.warpSize == 64)) {
        return prop.warpSize;
    }
    return 64;
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
