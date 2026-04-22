// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "primus_turbo/common.h"
#include "primus_turbo/device/utils.cuh"

#define FINAL_MASK 0xffffffffffffffffULL

namespace primus_turbo {

using namespace primus_turbo::dtype;

// Max
template <typename T> struct MaxOp {
public:
    PRIMUS_TURBO_HOST_DEVICE static T init() {
        return std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity()
                                                    : std::numeric_limits<T>::lowest();
    }

    PRIMUS_TURBO_HOST_DEVICE static T op(const T &x, const T &y) { return x > y ? x : y; }
};

template <> struct MaxOp<float> {
public:
    PRIMUS_TURBO_HOST_DEVICE static float init() { return -std::numeric_limits<float>::infinity(); }

    PRIMUS_TURBO_HOST_DEVICE static float op(const float &x, const float &y) { return fmaxf(x, y); }
};

template <> struct MaxOp<float16> {
public:
    PRIMUS_TURBO_HOST_DEVICE static float16 init() {
        return static_cast<float16>(-std::numeric_limits<float>::infinity());
    }

    PRIMUS_TURBO_HOST_DEVICE static float16 op(const float16 &x, const float16 &y) {
        return x > y ? x : y;
    }
};

template <> struct MaxOp<bfloat16> {
public:
    PRIMUS_TURBO_HOST_DEVICE static bfloat16 init() {
        return static_cast<bfloat16>(-std::numeric_limits<float>::infinity());
    }

    PRIMUS_TURBO_HOST_DEVICE static bfloat16 op(const bfloat16 &x, const bfloat16 &y) {
        return x > y ? x : y;
    }
};

// Min
template <typename T> struct MinOp {
public:
    PRIMUS_TURBO_HOST_DEVICE static T init() {
        return std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity()
                                                    : std::numeric_limits<T>::max();
    }
    PRIMUS_TURBO_HOST_DEVICE static T op(const T &x, const T &y) { return x < y ? x : y; }
};

template <> struct MinOp<float> {
public:
    PRIMUS_TURBO_HOST_DEVICE static float init() { return std::numeric_limits<float>::infinity(); }
    PRIMUS_TURBO_HOST_DEVICE static float op(const float &x, const float &y) { return fminf(x, y); }
};

template <> struct MinOp<float16> {
public:
    PRIMUS_TURBO_HOST_DEVICE static float16 init() {
        return static_cast<float16>(std::numeric_limits<float>::infinity());
    }
    PRIMUS_TURBO_HOST_DEVICE static float16 op(const float16 &x, const float16 &y) {
        return x < y ? x : y;
    }
};

template <> struct MinOp<bfloat16> {
public:
    PRIMUS_TURBO_HOST_DEVICE static bfloat16 init() {
        return static_cast<bfloat16>(std::numeric_limits<float>::infinity());
    }
    PRIMUS_TURBO_HOST_DEVICE static bfloat16 op(const bfloat16 &x, const bfloat16 &y) {
        return x < y ? x : y;
    }
};

// Sum
template <typename T> struct SumOp {
    PRIMUS_TURBO_HOST_DEVICE static T init() { return T(0); }
    PRIMUS_TURBO_HOST_DEVICE static T op(const T &x, const T &y) { return x + y; }
};

// AbsMax
template <typename T> struct AbsMaxOp {
public:
    PRIMUS_TURBO_HOST_DEVICE static T init() { return T(0); }
    PRIMUS_TURBO_HOST_DEVICE static T op(const T &x, const T &y) {
        const T ax = (x < T(0)) ? -x : x;
        const T ay = (y < T(0)) ? -y : y;
        return (ax > ay) ? ax : ay;
    }
};

template <> struct AbsMaxOp<float> {
public:
    PRIMUS_TURBO_HOST_DEVICE static float init() { return 0.0f; }
    PRIMUS_TURBO_HOST_DEVICE static float op(const float &x, const float &y) {
        return fmaxf(fabsf(x), fabsf(y));
    }
};

/**
 * Warp Reduce and Block Reduce
 */
template <template <class> class Func, typename T> PRIMUS_TURBO_DEVICE T WarpReduce(T val) {
#pragma unroll
    for (int offset = THREADS_PER_WARP >> 1; offset > 0; offset >>= 1) {
        T tmp = __shfl_xor_sync(FINAL_MASK, val, offset);
        val   = Func<T>::op(tmp, val);
    }
    return val;
}

template <template <class> class Func, typename T> PRIMUS_TURBO_DEVICE T BlockReduce(const T &val) {
    constexpr int MAX_NUM_WARPS = MAX_THREADS_PER_BLOCK / THREADS_PER_WARP;
    const int     num_warps     = (blockDim.x + THREADS_PER_WARP - 1) / THREADS_PER_WARP;

    __shared__ T smem[MAX_NUM_WARPS];
    const int    warp_id = threadIdx.x / THREADS_PER_WARP;
    const int    lane_id = threadIdx.x % THREADS_PER_WARP;

    T val_reg = Func<T>::init();
    val_reg   = Func<T>::op(val_reg, val);
    val_reg   = WarpReduce<Func, T>(val_reg);
    if (lane_id == 0) {
        smem[warp_id] = val_reg;
    }
    __syncthreads();
    if (warp_id == 0) {
        val_reg = (lane_id < num_warps) ? smem[lane_id] : Func<T>::init();
        val_reg = WarpReduce<Func, T>(val_reg);
        if (lane_id == 0)
            smem[0] = val_reg;
    }
    __syncthreads();
    return smem[0];
}

// ============================================================================
// WARP PRIMITIVES - AMD-Specific DPP/Swizzle Instructions
// ============================================================================

/*
 * ds_swizzle Instructions
 * -----------------------
 * These perform intra-wavefront data exchange without shared memory.
 * The offset parameter encodes the permutation pattern.
 *
 * Format: offset = (AND_mask << 10) | (OR_mask << 5) | XOR_mask
 *
 * Common patterns (BF permute family, XOR_mask = 0x1F):
 *   - 0x041F: AND=1  — combine lanes distance 1
 *   - 0x081F: AND=2  — distance 2
 *   - 0x101F: AND=4  — distance 4 (used inside warp_reduce_max_8_dpp)
 *
 * Reference: AMD CDNA4 ISA, ds_swizzle_b32 (page 480)
 */

PRIMUS_TURBO_DEVICE float ds_swizzle_xor1(float val) {
    float result;
    asm volatile("ds_swizzle_b32 %0, %1 offset:0x041F\n\t"
                 "s_waitcnt lgkmcnt(0)"
                 : "=v"(result)
                 : "v"(val));
    return result;
}

PRIMUS_TURBO_DEVICE float ds_swizzle_xor2(float val) {
    float result;
    asm volatile("ds_swizzle_b32 %0, %1 offset:0x081F\n\t"
                 "s_waitcnt lgkmcnt(0)"
                 : "=v"(result)
                 : "v"(val));
    return result;
}

PRIMUS_TURBO_DEVICE float ds_swizzle_xor8(float val) {
    float result;
    asm volatile("ds_swizzle_b32 %0, %1 offset:0x201F\n\t"
                 "s_waitcnt lgkmcnt(0)"
                 : "=v"(result)
                 : "v"(val));
    return result;
}

PRIMUS_TURBO_DEVICE float ds_swizzle_xor16(float val) {
    float result;
    asm volatile("ds_swizzle_b32 %0, %1 offset:0x401F\n\t"
                 "s_waitcnt lgkmcnt(0)"
                 : "=v"(result)
                 : "v"(val));
    return result;
}

// ============================================================================
// REDUCTION OPERATIONS - Finding Maximum Absolute Value
// ============================================================================

/*
 * Warp Reduction for Max Absolute Value
 * --------------------------------------
 * Reduces 8 values (one per thread in a group) to a single maximum using
 * ds_swizzle for efficient intra-wavefront communication.
 *
 * Pattern:
 *   Step 1: XOR 4 - reduce 8 values to 4 (threads 0-3, 4-7)
 *   Step 2: XOR 2 - reduce 4 values to 2 (threads 0-1, 2-3)
 *   Step 3: XOR 1 - reduce 2 values to 1 (thread 0)
 */
PRIMUS_TURBO_DEVICE float warp_reduce_max_8_dpp(float val) {
    uint32_t v = float_as_uint(val);
    uint32_t tmp;

    // Step 1: Exchange with thread 4 positions away
    asm volatile("ds_swizzle_b32 %0, %1 offset:0x101F" : "=v"(tmp) : "v"(v));
    asm volatile("s_waitcnt lgkmcnt(0)" :::);
    val = fmaxf(val, uint_as_float(tmp));
    v   = float_as_uint(val);

    // Step 2: Exchange with thread 2 positions away
    asm volatile("ds_swizzle_b32 %0, %1 offset:0x081F" : "=v"(tmp) : "v"(v));
    asm volatile("s_waitcnt lgkmcnt(0)" :::);
    val = fmaxf(val, uint_as_float(tmp));
    v   = float_as_uint(val);

    // Step 3: Exchange with adjacent thread
    asm volatile("ds_swizzle_b32 %0, %1 offset:0x041F" : "=v"(tmp) : "v"(v));
    asm volatile("s_waitcnt lgkmcnt(0)" :::);
    val = fmaxf(val, uint_as_float(tmp));

    return val;
}

PRIMUS_TURBO_DEVICE float warp_reduce_max_64_dpp(float val) {
    val = warp_reduce_max_8_dpp(val);
    val = fmaxf(val, ds_swizzle_xor8(val));
    val = fmaxf(val, ds_swizzle_xor16(val));
    val = fmaxf(val, __shfl_xor_sync(FINAL_MASK, val, 32, THREADS_PER_WARP));
    return val;
}

} // namespace primus_turbo
