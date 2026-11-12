/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Adapted from attn-a2a-demo (https://github.com/DDEle/attn-a2a-demo) at 7a7343ef47,
// csrc/barrier.hpp.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aa {
constexpr size_t kArrivalBytes = 64;

// arrival[j]: rank j's arrival counter as addressed from this process.
void boundaryBarrier(std::vector<uintptr_t> const &arrival, int rank, int world, uintptr_t stream);
} // namespace aa
