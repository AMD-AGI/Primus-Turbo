#pragma once

#include "primus_turbo/deep_ep/configs.cuh"
#include "primus_turbo/deep_ep/exception.cuh"

#ifndef SETUP_LAUNCH_CONFIG
#define SETUP_LAUNCH_CONFIG(sms, threads, stream)                                                  \
    int  __num_sms     = (sms);                                                                    \
    int  __num_threads = (threads);                                                                \
    auto __stream      = (stream)
#endif

#ifndef LAUNCH_KERNEL
#define LAUNCH_KERNEL(config, kernel, ...)                                                         \
    do {                                                                                           \
        kernel<<<__num_sms, __num_threads, 0, __stream>>>(__VA_ARGS__);                            \
        cudaError_t e = cudaGetLastError();                                                        \
        if (e != cudaSuccess) {                                                                    \
            EPException cuda_exception("CUDA", __FILE__, __LINE__, cudaGetErrorString(e));         \
            fprintf(stderr, "%s\n", cuda_exception.what());                                        \
            throw cuda_exception;                                                                  \
        }                                                                                          \
    } while (0)
#endif

#define SET_SHARED_MEMORY_FOR_TMA(kernel) void()

#define SWITCH_RANKS(case_macro)                                                                   \
    switch (num_ranks) {                                                                           \
    case 2:                                                                                        \
        case_macro(2);                                                                             \
    case 4:                                                                                        \
        case_macro(4);                                                                             \
    case 8:                                                                                        \
        case_macro(8);                                                                             \
    default:                                                                                       \
        EP_HOST_ASSERT(false and "Unsupported ranks");                                             \
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
    case 6:                                                                                        \
        case_macro(6);                                                                             \
    case 8:                                                                                        \
        case_macro(8);                                                                             \
    case 12:                                                                                       \
        case_macro(12);                                                                            \
    case 16:                                                                                       \
        case_macro(16);                                                                            \
    case 18:                                                                                       \
        case_macro(18);                                                                            \
    case 20:                                                                                       \
        case_macro(20);                                                                            \
    default:                                                                                       \
        EP_HOST_ASSERT(false and "Unsupported RDMA ranks");                                        \
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
        EP_HOST_ASSERT(false and "Unsupported ranks");                                             \
    }                                                                                              \
    while (false)

#define SWITCH_TYPES(case_macro)                                                                   \
    switch (type) {                                                                                \
    case CUDA_R_16BF:                                                                              \
        case_macro(nv_bfloat16);                                                                   \
    default:                                                                                       \
        EP_HOST_ASSERT(false and "Unsupported type");                                              \
    }                                                                                              \
    while (false)

#define SWITCH_HIDDEN(case_macro)                                                                  \
    switch (hidden) {                                                                              \
    case 2048:                                                                                     \
        case_macro(2048);                                                                          \
    case 2560:                                                                                     \
        case_macro(2560);                                                                          \
    case 3072:                                                                                     \
        case_macro(3072); /* for gpt-oss */                                                        \
    case 4096:                                                                                     \
        case_macro(4096);                                                                          \
    case 5120:                                                                                     \
        case_macro(5120);                                                                          \
    case 6144:                                                                                     \
        case_macro(6144); /* For qwen3 coder */                                                    \
    case 7168:                                                                                     \
        case_macro(7168);                                                                          \
    case 8192:                                                                                     \
        case_macro(8192);                                                                          \
    default:                                                                                       \
        EP_HOST_ASSERT(false and "Unsupported hidden");                                            \
    }                                                                                              \
    while (false)
