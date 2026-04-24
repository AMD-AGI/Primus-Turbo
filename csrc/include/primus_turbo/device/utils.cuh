// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once
#include "primus_turbo/common.h"

namespace primus_turbo {

using namespace primus_turbo::dtype;

/**
 * Load & Store data utils
 */

// TODO: ASM
template <typename T, const int N> PRIMUS_TURBO_DEVICE void load_data(const T *src, T *dst) {
    constexpr int BYTES = N * sizeof(T);
    static_assert(BYTES == 1 || BYTES == 2 || BYTES == 4 || BYTES == 8 || BYTES == 16 ||
                      BYTES == 32,
                  "Only 1/2/4/8/16/32 bytes are supported.");
    if constexpr (BYTES == 1) {
        *reinterpret_cast<uint8 *>(dst) = *(reinterpret_cast<const uint8 *>(src));
    } else if constexpr (BYTES == 2) {
        *reinterpret_cast<uint16 *>(dst) = *(reinterpret_cast<const uint16 *>(src));
    } else if constexpr (BYTES == 4) {
        *reinterpret_cast<uint32 *>(dst) = *(reinterpret_cast<const uint32 *>(src));
    } else if constexpr (BYTES == 8) {
        *reinterpret_cast<uint64 *>(dst) = *(reinterpret_cast<const uint64 *>(src));
    } else if constexpr (BYTES == 16) {
        *reinterpret_cast<uint4 *>(dst) = *(reinterpret_cast<const uint4 *>(src));
    } else if constexpr (BYTES == 32) {
        // Two consecutive 16B vector loads -> dwordx8 on gfx950.
        reinterpret_cast<uint4 *>(dst)[0] = reinterpret_cast<const uint4 *>(src)[0];
        reinterpret_cast<uint4 *>(dst)[1] = reinterpret_cast<const uint4 *>(src)[1];
    }
}

template <typename T, const int N> PRIMUS_TURBO_DEVICE void store_data(T *dst, const T *src) {
    constexpr int BYTES = N * sizeof(T);
    static_assert(BYTES == 1 || BYTES == 2 || BYTES == 4 || BYTES == 8 || BYTES == 16 ||
                      BYTES == 32,
                  "Only 1/2/4/8/16/32 bytes are supported.");

    if constexpr (BYTES == 1) {
        *reinterpret_cast<uint8 *>(dst) = *reinterpret_cast<const uint8 *>(src);
    } else if constexpr (BYTES == 2) {
        *reinterpret_cast<uint16 *>(dst) = *reinterpret_cast<const uint16 *>(src);
    } else if constexpr (BYTES == 4) {
        *reinterpret_cast<uint32 *>(dst) = *reinterpret_cast<const uint32 *>(src);
    } else if constexpr (BYTES == 8) {
        *reinterpret_cast<uint64 *>(dst) = *reinterpret_cast<const uint64 *>(src);
    } else if constexpr (BYTES == 16) {
        *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src);
    } else if constexpr (BYTES == 32) {
        // Two consecutive 16B vector stores -> dwordx8 on gfx950.
        reinterpret_cast<uint4 *>(dst)[0] = reinterpret_cast<const uint4 *>(src)[0];
        reinterpret_cast<uint4 *>(dst)[1] = reinterpret_cast<const uint4 *>(src)[1];
    }
}

PRIMUS_TURBO_DEVICE __forceinline__ uint32_t float_as_uint(float f) {
    return __float_as_uint(f);
}

PRIMUS_TURBO_DEVICE __forceinline__ float uint_as_float(uint32_t u) {
    return __uint_as_float(u);
}

/*
 * bfloat16 to FP32 Conversion
 * -----------------------
 * bfloat16 is FP32 with the lower 16 bits truncated, so we reconstruct
 * by shifting the 16-bit value left by 16 bits.
 */
PRIMUS_TURBO_DEVICE __forceinline__ void bfloat16x4_to_floatx4(uint64_t packed, float &v0,
                                                               float &v1, float &v2, float &v3) {
    v0 = uint_as_float(((uint32_t) (packed & 0xFFFF)) << 16);
    v1 = uint_as_float(((uint32_t) ((packed >> 16) & 0xFFFF)) << 16);
    v2 = uint_as_float(((uint32_t) ((packed >> 32) & 0xFFFF)) << 16);
    v3 = uint_as_float(((uint32_t) ((packed >> 48) & 0xFFFF)) << 16);
}

/*
 * half to FP32 Conversion
 * -----------------------
 * Convert 4 packed half values (in a uint64_t) to 4 floats using
 * the HIP __half intrinsic.
 */
PRIMUS_TURBO_DEVICE __forceinline__ void halfx4_to_floatx4(uint64_t packed, float &v0, float &v1,
                                                           float &v2, float &v3) {
    uint16_t h0 = (uint16_t) (packed & 0xFFFF);
    uint16_t h1 = (uint16_t) ((packed >> 16) & 0xFFFF);
    uint16_t h2 = (uint16_t) ((packed >> 32) & 0xFFFF);
    uint16_t h3 = (uint16_t) ((packed >> 48) & 0xFFFF);
    v0          = __half2float(*reinterpret_cast<const half *>(&h0));
    v1          = __half2float(*reinterpret_cast<const half *>(&h1));
    v2          = __half2float(*reinterpret_cast<const half *>(&h2));
    v3          = __half2float(*reinterpret_cast<const half *>(&h3));
}

/*
 * Templated conversion helpers dispatching bfloat16 vs half at compile time.
 */
template <bool IS_half>
PRIMUS_TURBO_DEVICE __forceinline__ void
packed_uint16x4_to_floatx4(uint64_t packed, float &v0, float &v1, float &v2, float &v3) {
    if constexpr (IS_half) {
        halfx4_to_floatx4(packed, v0, v1, v2, v3);
    } else {
        bfloat16x4_to_floatx4(packed, v0, v1, v2, v3);
    }
}

template <bool IS_half> PRIMUS_TURBO_DEVICE __forceinline__ float uint16_to_float(uint16_t val) {
    if constexpr (IS_half) {
        return __half2float(*reinterpret_cast<const half *>(&val));
    } else {
        return uint_as_float(((uint32_t) val) << 16);
    }
}

} // namespace primus_turbo
