/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Adapted from attn-a2a-demo (https://github.com/DDEle/attn-a2a-demo) at 7a7343ef47,
// csrc/comm_context.cpp.
#include "comm_context.hpp"

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace aa {
namespace {
// Same ring as TensileLite::Client::sdmaRotateEngine (client/include/DeviceContext.hpp).
uint32_t rotateEngine(uint32_t engine, int steps) {
    if (steps == 0)
        return engine;
    if (engine < 2 || engine > 14 || engine % 2 != 0)
        throw std::runtime_error("SDMA rotation needs an even xGMI engine in 2..14; got " +
                                 std::to_string(engine));
    return 2 + 2 * ((((int) engine - 2) / 2 + steps % 7 + 7) % 7);
}
} // namespace

CommContext::CommContext(int rank, int world, std::vector<int> const &devices, int sdmaRotate)
    : m_rank(rank), m_world(world) {
    if ((int) devices.size() != world || rank < 0 || rank >= world)
        throw std::runtime_error("CommContext: devices must list one HIP ordinal per rank");
    int cur = -1;
    (void) hipGetDevice(&cur);
    if (cur != devices[rank])
        throw std::runtime_error("CommContext: current device " + std::to_string(cur) +
                                 " is not devices[rank]=" + std::to_string(devices[rank]));
    for (int j = 0; j < world; j++) {
        if (j == rank)
            continue;
        int can = 0;
        (void) hipDeviceCanAccessPeer(&can, devices[rank], devices[j]);
        if (!can)
            throw std::runtime_error("CommContext: no peer access to device " +
                                     std::to_string(devices[j]));
        hipError_t e = hipDeviceEnablePeerAccess(devices[j], 0);
        if (e != hipSuccess)
            (void) hipGetLastError();
        if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled)
            throw std::runtime_error(std::string("hipDeviceEnablePeerAccess: ") +
                                     hipGetErrorString(e));
    }
    using namespace TensileLite::Client;
    const uint32_t node = sdmaNodeIdForDevice(devices[rank]);
    for (int j = 0; j < world; j++) {
        const uint32_t engine = sdmaSelectEngine(node, sdmaNodeIdForDevice(devices[j]));
        m_queues.push_back(std::make_unique<SdmaQueue>(
            node, j == rank ? engine : rotateEngine(engine, sdmaRotate)));
    }
}

CommContext::~CommContext() {
    (void) hipDeviceSynchronize();
    (void) hipGetLastError();
}

std::array<void *, 4> CommContext::queueFields(int peer) const {
    auto const &q = m_queues.at(peer)->queueResource();
    return {m_queues.at(peer)->ringBase(), (void *) q.Queue_read_ptr_aql,
            (void *) q.Queue_write_ptr_aql, (void *) q.Queue_DoorBell_aql};
}
} // namespace aa
