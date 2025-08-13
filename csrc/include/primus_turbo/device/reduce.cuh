// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "primus_turbo/common.h"

#define FINAL_MASK 0xffffffffffffffffULL

namespace primus_turbo {

/**
 * Warp Reduce and Block Reduce
 */
template <typename T> struct MaxOp {
public:
    static constexpr T           init = std::numeric_limits<T>::min();
    static T PRIMUS_TURBO_DEVICE op(const T &x, const T &y) { return x > y ? x : y; }
};

template <> struct MaxOp<float> {
public:
    static constexpr float           init = -std::numeric_limits<float>::infinity();
    static float PRIMUS_TURBO_DEVICE op(const float &x, const float &y) { return x > y ? x : y; }
};

template <typename T> struct MinOp {
public:
    static constexpr T           init = std::numeric_limits<T>::max();
    static T PRIMUS_TURBO_DEVICE op(const T &x, const T &y) { return x < y ? x : y; }
};

template <> struct MinOp<float> {
public:
    static constexpr float           init = std::numeric_limits<float>::infinity();
    static float PRIMUS_TURBO_DEVICE op(const float &x, const float &y) { return x < y ? x : y; }
};

template <typename T> struct SumOp {
public:
    static constexpr T           init = T(0);
    static T PRIMUS_TURBO_DEVICE op(const T &x, const T &y) { return x + y; }
};

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

    T val_reg = Func<T>::init;
    val_reg   = Func<T>::op(val_reg, val);
    val_reg   = WarpReduce<Func, T>(val_reg);
    if (lane_id == 0) {
        smem[warp_id] = val_reg;
    }
    __syncthreads();
    if (warp_id == 0) {
        val_reg = (lane_id < num_warps) ? smem[lane_id] : Func<T>::init;
        val_reg = WarpReduce<Func, T>(val_reg);
        if (lane_id == 0)
            smem[0] = val_reg;
    }
    __syncthreads();
    return smem[0];
}

} // namespace primus_turbo
