// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "primus_turbo/arch.h"
#include "primus_turbo/dtype.h"
#include "primus_turbo/macros.h"
#include "primus_turbo/platform.h"

namespace primus_turbo {

// Define vector types for BytesToType
struct uint64x4_t {
    uint64_t data[4];
};

struct uint64x8_t {
    uint64_t data[8];
};

template <int BYTES> struct BytesToType {};

template <> struct BytesToType<64> {
    using Type = uint64x8_t;
    static_assert(sizeof(Type) == 64);
};

template <> struct BytesToType<32> {
    using Type = uint64x4_t;
    static_assert(sizeof(Type) == 32);
};

template <> struct BytesToType<16> {
    using Type = uint4;
    static_assert(sizeof(Type) == 16);
};

template <> struct BytesToType<8> {
    using Type = uint64_t;
    static_assert(sizeof(Type) == 8);
};

template <> struct BytesToType<4> {
    using Type = uint32_t;
    static_assert(sizeof(Type) == 4);
};

template <> struct BytesToType<2> {
    using Type = uint16_t;
    static_assert(sizeof(Type) == 2);
};

template <> struct BytesToType<1> {
    using Type = uint8_t;
    static_assert(sizeof(Type) == 1);
};

} // namespace primus_turbo

template <typename T> constexpr PRIMUS_TURBO_HOST_DEVICE T DIVUP(const T &x, const T &y) {
    return (((x) + ((y) -1)) / (y));
}

template <typename T> PRIMUS_TURBO_HOST_DEVICE T ALIGN(T a, T b) {
    return DIVUP<T>(a, b) * b;
}

namespace primus_turbo {

struct PackedEltwiseConfig {
    int64_t nPack;
    int64_t nThread;
    int64_t nBlock;

    PackedEltwiseConfig(int64_t n, int64_t pack_size, int64_t block_size) {
        nPack   = n / pack_size;
        nThread = nPack + n % pack_size;
        nBlock  = DIVUP<int64_t>(nThread, block_size);
    }
};

// Cross device Array
template <typename T, int N> struct Array {
    T data[N];

    Array() = default;

    // construct from host side
    template <typename S> Array(const S *src, int n) {
        PRIMUS_TURBO_CHECK(n <= N, "n size must be <= N");
        for (int i = 0; i < n; ++i) {
            data[i] = static_cast<T>(src[i]);
        }
    }

    PRIMUS_TURBO_HOST_DEVICE
    T &operator[](size_t i) { return data[i]; }

    PRIMUS_TURBO_HOST_DEVICE
    const T &operator[](size_t i) const { return data[i]; }

    PRIMUS_TURBO_HOST_DEVICE
    constexpr int size() const { return N; }
};

// Integer Div/Mod
template <typename T> struct DivModData {
    T div;
    T mod;
    PRIMUS_TURBO_HOST_DEVICE
    DivModData(T d, T m) : div(d), mod(m) {}
};

template <typename T> struct IntDivMod {
    static_assert(std::is_integral_v<T>, "IntDivMod<T>: T must be an integral type");

    T d_;

    PRIMUS_TURBO_HOST_DEVICE
    IntDivMod() = default;

    PRIMUS_TURBO_HOST_DEVICE
    explicit IntDivMod(T d) : d_(d) {}

    PRIMUS_TURBO_HOST_DEVICE T div(T n) const { return n / d_; }

    PRIMUS_TURBO_HOST_DEVICE T mod(T n) const { return n % d_; }

    PRIMUS_TURBO_HOST_DEVICE DivModData<T> div_mod(T n) const {
        return DivModData<T>(n / d_, n % d_);
    }
};

// TODO Fast Integer Div/Mod

} // namespace primus_turbo
