// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "primus_turbo/common.h"

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

} // namespace primus_turbo
