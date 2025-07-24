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
    static_assert(BYTES == 1 || BYTES == 2 || BYTES == 4 || BYTES == 8 || BYTES == 16,
                  "Only 1/2/4/8/16 bytes are supported.");
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

} // namespace primus_turbo
