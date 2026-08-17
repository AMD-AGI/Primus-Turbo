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
        reinterpret_cast<uint4 *>(dst)[0] = reinterpret_cast<const uint4 *>(src)[0];
        reinterpret_cast<uint4 *>(dst)[1] = reinterpret_cast<const uint4 *>(src)[1];
    }
}

template <typename T, const int N> PRIMUS_TURBO_DEVICE void store_data(T *dst, const T *src) {
    constexpr int BYTES = N * sizeof(T);
    static_assert(BYTES == 1 || BYTES == 2 || BYTES == 4 || BYTES == 8 || BYTES == 16,
                  "Only 1/2/4/8/16 bytes are supported.");

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
    }
}

PRIMUS_TURBO_DEVICE uint32_t float_as_uint(float f) {
    return __float_as_uint(f);
}

PRIMUS_TURBO_DEVICE float uint_as_float(uint32_t u) {
    return __uint_as_float(u);
}

/*
 * bfloat16 to FP32 Conversion
 * -----------------------
 * bfloat16 is FP32 with the lower 16 bits truncated, so we reconstruct
 * by shifting the 16-bit value left by 16 bits.
 */
PRIMUS_TURBO_DEVICE void bfloat16x4_to_floatx4(uint64_t packed, float &v0, float &v1, float &v2,
                                               float &v3) {
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
PRIMUS_TURBO_DEVICE void halfx4_to_floatx4(uint64_t packed, float &v0, float &v1, float &v2,
                                           float &v3) {
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
PRIMUS_TURBO_DEVICE void packed_uint16x4_to_floatx4(uint64_t packed, float &v0, float &v1,
                                                    float &v2, float &v3) {
    if constexpr (IS_half) {
        halfx4_to_floatx4(packed, v0, v1, v2, v3);
    } else {
        bfloat16x4_to_floatx4(packed, v0, v1, v2, v3);
    }
}

template <bool IS_half> PRIMUS_TURBO_DEVICE float uint16_to_float(uint16_t val) {
    if constexpr (IS_half) {
        return __half2float(*reinterpret_cast<const half *>(&val));
    } else {
        return uint_as_float(((uint32_t) val) << 16);
    }
}

/*
 * MX block-scaled de-quantization helpers (shared by the MXFP8 / MXFP4 kernels).
 */

// Decode an E8M0 biased exponent into its FP32 power-of-two scale.
PRIMUS_TURBO_DEVICE float e8m0_to_scale(uint8_t e8m0) {
    return uint_as_float(static_cast<uint32_t>(e8m0) << 23);
}

// Decode a CDNA5 E5M3 block scale into FP32. E5M3 is an unsigned 8-bit float
// with a 5-bit exponent (bias 15) and 3 mantissa bits: 0x00 is zero, 0x01..0x07
// are subnormals on the 2^-17 grid, 0x08 is the smallest normal (2^-14) and
// 0xFE is the largest finite value (114688). 0xFF encodes NaN in hardware;
// producers saturate to 0xFE so this maps it to the finite value instead.
PRIMUS_TURBO_DEVICE float e5m3_to_scale(uint8_t e5m3) {
    const uint32_t exp_field = static_cast<uint32_t>(e5m3) >> 3;
    const uint32_t mantissa  = static_cast<uint32_t>(e5m3) & 0x7u;
    if (exp_field == 0) {
        return static_cast<float>(mantissa) * 0x1p-17f;
    }
    // FP32 bias 127 - E5M3 bias 15 = 112; the 3 mantissa bits are the top of
    // the FP32 mantissa field.
    return uint_as_float(((exp_field + 112u) << 23) | (mantissa << 20));
}

// Decode an OCP E4M3 block scale (NVFP4) into FP32: a 4-bit exponent with bias
// 7 and 3 mantissa bits, so 0x00 is zero, 0x01..0x07 are subnormals on the
// 2^-9 grid, 0x08 is the smallest normal (2^-6) and 0x7E is the largest finite
// value (448). 0x7F encodes NaN; producers saturate to 0x7E so this maps it to
// the finite value instead. Block scales are non-negative, so the sign bit is
// ignored and codes >= 0x80 decode to their magnitude.
PRIMUS_TURBO_DEVICE float e4m3_to_scale(uint8_t e4m3) {
    const uint32_t exp_field = (static_cast<uint32_t>(e4m3) >> 3) & 0xFu;
    const uint32_t mantissa  = static_cast<uint32_t>(e4m3) & 0x7u;
    if (exp_field == 0) {
        return static_cast<float>(mantissa) * 0x1p-9f;
    }
    // FP32 bias 127 - E4M3 bias 7 = 120.
    return uint_as_float(((exp_field + 120u) << 23) | (mantissa << 20));
}

} // namespace primus_turbo
