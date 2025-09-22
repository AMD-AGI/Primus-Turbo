// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <cstdint>

//
// Platform detection
//
#if defined(__HIPCC__)
#define PRIMUS_TURBO_PLATFORM_HIP 1
#else
#define PRIMUS_TURBO_PLATFORM_HIP 0
#endif

//
// Host/Device dispatch macros
//
#if PRIMUS_TURBO_PLATFORM_HIP
#define PRIMUS_TURBO_HOST_DEVICE inline __host__ __device__
#define PRIMUS_TURBO_DEVICE inline __device__
#else
#define PRIMUS_TURBO_HOST_DEVICE inline
#define PRIMUS_TURBO_DEVICE inline
#endif

// Compile-time device-side flag
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#define PRIMUS_TURBO_DEVICE_COMPILE 1
#else
#define PRIMUS_TURBO_DEVICE_COMPILE 0
#endif

//
// Universal warp size constant (AMD = 64)
//
namespace primus_turbo {
constexpr int THREADS_PER_WARP      = 64;
constexpr int MAX_THREADS_PER_BLOCK = 1024; // TODO: ?
} // namespace primus_turbo
