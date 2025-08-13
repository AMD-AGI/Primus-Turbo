// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

namespace primus_turbo {

enum class GPUArch { GFX942, GFX950, UNKNOWN };

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

} // namespace primus_turbo
