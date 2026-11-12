/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Adapted from attn-a2a-demo (https://github.com/DDEle/attn-a2a-demo) at 7a7343ef47,
// csrc/comm_context.hpp.
#pragma once

#include "3rdparty/SdmaQueue.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace aa {
class CommContext {
public:
    // devices: HIP ordinal of each rank, indexed by rank.
    CommContext(int rank, int world, std::vector<int> const &devices, int sdmaRotate);
    ~CommContext();

    int rank() const { return m_rank; }
    int world() const { return m_world; }

    // ringBase, rptr, wptr, doorbell of this rank's queue toward `peer`.
    std::array<void *, 4> queueFields(int peer) const;

private:
    int                                                          m_rank;
    int                                                          m_world;
    std::vector<std::unique_ptr<TensileLite::Client::SdmaQueue>> m_queues;
};
} // namespace aa
