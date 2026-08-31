/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2025 DeepSeek. All rights reserved.
 *
 * Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

#pragma once

#include "primus_turbo/common.h"
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#define NUM_MAX_NVL_PEERS 8
#define NUM_MAX_RDMA_PEERS 20
#define NUM_MAX_FIFO_SLOTS 32768
#define NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
#define NUM_MAX_LOCAL_EXPERTS 1024
#define NUM_BUFFER_ALIGNMENT_BYTES 128

#define FINISHED_SUM_TAG 1024

#define NUM_CPU_TIMEOUT_SECS 300
#define NUM_TIMEOUT_CYCLES 600000000000ll // 600G cycles ~= 300s

#define NUM_WAIT_NANOSECONDS 500

#define NUM_WAIT_CYCLES_TIMES_64 16

#define LOW_LATENCY_SEND_PHASE 1
#define LOW_LATENCY_RECV_PHASE 2

#define DEFAULT_NUM_CU 20
#define DEFAULT_NUM_MAX_XGMI_CHUNKED_SEND_TOKENS 6
#define DEFAULT_NUM_MAX_XGMI_CHUNKED_RECV_TOKENS 256
#define DEFAULT_NUM_MAX_RDMA_CHUNKED_SEND_TOKENS 6
#define DEFAULT_NUM_MAX_RDMA_CHUNKED_RECV_TOKENS 256

static constexpr int32_t kWarpSize = 64;
// For ROCm equals to half the wave size or Nvidia warp size
static constexpr int32_t  kEmulatedWarpSize = kWarpSize / 2;
static constexpr uint64_t kFullWarpMask     = 0xffffffffffffffff;
static constexpr uint64_t kFirstHalfMask    = 0x00000000ffffffff;
static constexpr uint64_t kSecondHalfMask   = 0xffffffff00000000;

// Remove Torch restrictions
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT162_OPERATORS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__
#endif

// Remove Torch restrictions for HIP
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif

static inline int get_num_cpu_timeout_secs() {
    static int timeout = []() {
        const char *env = std::getenv("PRIMUS_TURBO_DEEPEP_CPU_TIMEOUT");
        if (!env || env[0] == '\0') {
            return NUM_CPU_TIMEOUT_SECS;
        }
        try {
            return std::stoi(env);
        } catch (...) {
            return NUM_CPU_TIMEOUT_SECS;
        }
    }();
    return timeout;
}

// The cheap fence replaces the release store's system-scope write-back with a per-wave s_waitcnt
// (st_release_sys_global in the deep_ep utils header). That waitcnt covers only the wave that
// publishes the channel tail, while the payload rows are written by the other waves of the group,
// and the RELAXED store drops the write-back the receiver needs -- so a receiver can see the new
// tail while those rows still sit in the sender's L2 and read the previous round's contents
// instead. Measured on gfx950 (EP=8, DSv3, intranode dispatch): 7 of 8 short runs carried 1-6
// corrupted rows out of 8192, and 50-iter bf16 loss landed at 6.2066 against a 4.5206 DeepEP-off
// reference. The corruption is silent -- it lands in gradients -- so it must not be what a caller
// gets by default. The plain release store costs 2.7% more per step; set
// PRIMUS_TURBO_DEEPEP_DISABLE_CHEAP_FENCE=0 to take the cheap fence back, and its hazard with it.
inline static bool is_enable_cheap_fence() {
    char const *v = std::getenv("PRIMUS_TURBO_DEEPEP_DISABLE_CHEAP_FENCE");
    if (!v || v[0] == '\0')
        return false;
    return std::stoi(v) == 0;
}

// When set to 1, EP dispatch/combine kernels run on the caller's current
// CUDA stream instead of the Buffer's internal comm stream.
// Default: 0.
inline static bool is_ep_force_current_stream() {
    static uint32_t val = []() {
        const char *v = std::getenv("PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM");
        return v ? static_cast<uint32_t>(std::stoi(v)) : 0;
    }();
    return val != 0;
}
