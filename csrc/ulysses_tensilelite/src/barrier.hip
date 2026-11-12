/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Adapted from attn-a2a-demo (https://github.com/DDEle/attn-a2a-demo) at 7a7343ef47, csrc/barrier.hip.
#include "barrier.hpp"

#include <Tensile/FusedA2AKernArg.hpp>

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace aa
{
    namespace
    {
        struct ArrivalPeers
        {
            uint32_t* p[TensileLite::FUSED_A2A_MAX_RANKS] = {};
        };

        __global__ void boundaryBarrierKernel(ArrivalPeers peers, int myRank, int world)
        {
            const uint32_t need = (uint32_t)(world - 1);

            for(int j = 0; j < world; j++)
            {
                if(j != myRank)
                    __hip_atomic_fetch_add(
                        peers.p[j], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            }

            while(__hip_atomic_load(peers.p[myRank], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM)
                  < need)
                ;

            __hip_atomic_fetch_sub(
                peers.p[myRank], need, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }

    void boundaryBarrier(std::vector<uintptr_t> const& arrival, int rank, int world, uintptr_t stream)
    {
        if((int)arrival.size() != world || world > TensileLite::FUSED_A2A_MAX_RANKS)
            throw std::runtime_error("boundaryBarrier: need one arrival pointer per rank");
        ArrivalPeers peers;
        for(int j = 0; j < world; j++)
            peers.p[j] = reinterpret_cast<uint32_t*>(arrival[j]);
        hipLaunchKernelGGL(boundaryBarrierKernel,
                           dim3(1),
                           dim3(1),
                           0,
                           reinterpret_cast<hipStream_t>(stream),
                           peers,
                           rank,
                           world);
        hipError_t e = hipGetLastError();
        if(e != hipSuccess)
            throw std::runtime_error(std::string("boundaryBarrier: ") + hipGetErrorString(e));
    }
}
