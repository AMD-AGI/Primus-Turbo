// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
#pragma once

#include "primus_turbo/platform.h"
#include <cstdint>
#include <cstring>

namespace primus_turbo {

PRIMUS_TURBO_HOST_DEVICE float fp32_from_bits(uint32_t bits) {
#if PRIMUS_TURBO_DEVICE_COMPILE
    return __uint_as_float(bits);
#else
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
#endif
}

PRIMUS_TURBO_HOST_DEVICE uint32_t fp32_to_bits(float f) {
#if PRIMUS_TURBO_DEVICE_COMPILE
    return __float_as_uint(f);
#else
    uint32_t bits;
    memcpy(&bits, &f, sizeof(uint32_t));
    return bits;
#endif
}

} // namespace primus_turbo
