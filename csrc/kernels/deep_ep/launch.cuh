/*
 * Copyright (c) 2025 DeepSeek. All rights reserved.
 *
 * Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 */

#pragma once

#include "primus_turbo/deep_ep/configs.h"
#include "primus_turbo/macros.h"

// ROCm helper functions and structures
namespace rocm::experimental {
typedef struct {
    dim3         num_sms;
    dim3         num_threads;
    unsigned int shared_mem_bytes;
    hipStream_t  stream;
} hipLaunchConfig_t;

// Compile time void** kernelArgs array fill with variadic arguments
template <typename T> void fill_kernel_args(void **f, size_t idx, T &&arg) {
    f[idx] = static_cast<void *>(std::addressof(arg));
}

template <typename Head, typename... Tail>
void fill_kernel_args(void **f, size_t idx, Head &&head, Tail &&...tail) {
    f[idx] = static_cast<void *>(std::addressof(head));
    fill_kernel_args(f, idx + 1, std::forward<Tail>(tail)...);
}
} // namespace rocm::experimental

#ifndef SETUP_LAUNCH_CONFIG
// The code below is a workaround for ROCm. All the proposed overhead
// is to match current macro signatures and should be reworked once
// cudaLaunchKernelExt() hip alternative is live.
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream)                                          \
    rocm::experimental::hipLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream};

#endif // #ifndef SETUP_LAUNCH_CONFIG

#ifndef LAUNCH_KERNEL
template <typename T, typename Kern, typename... Args>
inline void LAUNCH_KERNEL(T &&config, Kern &&kernel, Args &&...args) {
    constexpr size_t k_num_kernel_args = sizeof...(args);
    void            *kernel_args[k_num_kernel_args];
    rocm::experimental::fill_kernel_args(kernel_args, 0, std::forward<Args>(args)...);
    PRIMUS_TURBO_CHECK_HIP(hipLaunchCooperativeKernel(std::forward<Kern>(kernel), config->num_sms,
                                                      config->num_threads, kernel_args,
                                                      config->shared_mem_bytes, config->stream));
}

template <typename T, typename Kern, typename... Args>
inline void LAUNCH_KERNEL_NON_COOPERATIVE(T &&config, Kern &&kernel, Args &&...args) {
    *kernel<<<config->num_sms, config->num_threads, config->shared_mem_bytes, config->stream>>>(
        std::forward<Args>(args)...);
}

#endif // #ifndef LAUNCH_KERNEL

#define SWITCH_RANKS(case_macro)                                                                   \
    switch (num_ranks) {                                                                           \
    case 2:                                                                                        \
        case_macro(2);                                                                             \
    case 4:                                                                                        \
        case_macro(4);                                                                             \
    case 8:                                                                                        \
        case_macro(8);                                                                             \
    default:                                                                                       \
        PRIMUS_TURBO_CHECK(false and "Unsupported ranks");                                         \
    }                                                                                              \
    while (false)

#define SWITCH_RDMA_RANKS(case_macro)                                                              \
    switch (num_ranks / NUM_MAX_NVL_PEERS) {                                                       \
    case 2:                                                                                        \
        case_macro(2);                                                                             \
    case 3:                                                                                        \
        case_macro(3);                                                                             \
    case 4:                                                                                        \
        case_macro(4);                                                                             \
    case 8:                                                                                        \
        case_macro(8);                                                                             \
    case 16:                                                                                       \
        case_macro(16);                                                                            \
    case 18:                                                                                       \
        case_macro(18);                                                                            \
    case 20:                                                                                       \
        case_macro(20);                                                                            \
    default:                                                                                       \
        PRIMUS_TURBO_CHECK(false, "Unsupported RDMA ranks");                                       \
    }                                                                                              \
    while (false)

#define SWITCH_RANKS_WITH_DTYPE(dtype, case_macro)                                                 \
    switch (num_ranks) {                                                                           \
    case 2:                                                                                        \
        case_macro(dtype, 2);                                                                      \
    case 4:                                                                                        \
        case_macro(dtype, 4);                                                                      \
    case 8:                                                                                        \
        case_macro(dtype, 8);                                                                      \
    default:                                                                                       \
        PRIMUS_TURBO_CHECK(false, "Unsupported ranks");                                            \
    }                                                                                              \
    while (false)

#define SWITCH_TYPES(case_macro)                                                                   \
    switch (type) {                                                                                \
    case HIP_R_16BF:                                                                               \
        case_macro(hip_bfloat16);                                                                  \
    case HIP_R_32F:                                                                                \
        case_macro(float);                                                                         \
    default:                                                                                       \
        PRIMUS_TURBO_CHECK(false, "Unsupported type");                                             \
    }                                                                                              \
    while (false)

#define SWITCH_HIDDEN(case_macro)                                                                  \
    switch (hidden) {                                                                              \
    case 2560:                                                                                     \
        case_macro(2560);                                                                          \
    case 5120:                                                                                     \
        case_macro(5120);                                                                          \
    case 4096:                                                                                     \
        case_macro(4096);                                                                          \
    case 7168:                                                                                     \
        case_macro(7168);                                                                          \
    default:                                                                                       \
        PRIMUS_TURBO_CHECK(false, "Unsupported hidden");                                           \
    }                                                                                              \
    while (false)
