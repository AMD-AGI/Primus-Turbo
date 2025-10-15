// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace primus_turbo {

namespace pack_config {
constexpr size_t MAX_PACKED_BYTE = 16;
constexpr size_t MAX_PACKED_SIZE = 8;
} // namespace pack_config

template <typename T, int N> constexpr size_t valid_pack() {
    return (N <= pack_config::MAX_PACKED_SIZE) && ((sizeof(T) * N) <= pack_config::MAX_PACKED_BYTE)
               ? N
               : 1;
}

template <typename T> inline size_t get_pack_size(const T *ptr) {
    if (sizeof(T) > pack_config::MAX_PACKED_BYTE)
        return 1;

    size_t pack_size =
        std::min(pack_config::MAX_PACKED_BYTE / sizeof(T), pack_config::MAX_PACKED_SIZE);

    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    while (pack_size > 1 && (addr % (pack_size * sizeof(T)) != 0)) {
        pack_size /= 2;
    }
    return pack_size;
}

} // namespace primus_turbo
