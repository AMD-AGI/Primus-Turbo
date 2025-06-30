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
#include <random>
#include <stdexcept>
#include <vector>

#include "primus_turbo/dtype.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace primus_turbo {

////////////////////////////////////////////////////////////////////////////////////////////////////

struct uint64x8_t {
    uint4 u;
    uint4 v;
    uint4 s;
    uint4 t;
};

struct uint64x4_t {
    uint4 u;
    uint4 v;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr uint32_t THREADS_PER_WARP = 64;

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace primus_turbo {

/*! \brief Convert to C-style or C++-style string */
template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
inline std::string to_string_like(const T &val) {
    return std::to_string(val);
}

inline const std::string &to_string_like(const std::string &val) noexcept {
    return val;
}

constexpr const char *to_string_like(const char *val) noexcept {
    return val;
}

/*! \brief Convert arguments to strings and concatenate */
template <typename... Ts> inline std::string concat_strings(const Ts &...args) {
    std::string str;
    str.reserve(1024); // Assume strings are <1 KB
    (..., (str += to_string_like(args)));
    return str;
}

} // namespace primus_turbo

#define PRIMUS_ERROR(...)                                                                          \
    do {                                                                                           \
        throw ::std::runtime_error(                                                                \
            ::primus_turbo::concat_strings(__FILE__ ":", __LINE__, " in function ", __func__,      \
                                           ": ", ::primus_turbo::concat_strings(__VA_ARGS__)));    \
    } while (false)

#define PRIMUS_CHECK(expr, ...)                                                                    \
    do {                                                                                           \
        if (!(expr)) {                                                                             \
            PRIMUS_ERROR("Assertion failed: " #expr ". ",                                          \
                         ::primus_turbo::concat_strings(__VA_ARGS__));                             \
        }                                                                                          \
    } while (false)

#define PRIMUS_CHECK_HIP(expr)                                                                     \
    do {                                                                                           \
        const hipError_t status_PRIMUS_CHECK_HIP = (expr);                                         \
        if (status_PRIMUS_CHECK_HIP != hipSuccess) {                                               \
            PRIMUS_ERROR("HIP Error: ", hipGetErrorString(status_PRIMUS_CHECK_HIP));               \
        }                                                                                          \
    } while (false)

#define PRIMUS_CHECK_HIPBLASLT(expr)                                                               \
    do {                                                                                           \
        const hipblasStatus_t status_PRIMUS_CHECK_HIPBLAS = (expr);                                \
        if (status_PRIMUS_CHECK_HIPBLAS != HIPBLAS_STATUS_SUCCESS) {                               \
            PRIMUS_ERROR("HIPBLASLT Error: ", std::to_string((int) status_PRIMUS_CHECK_HIPBLAS));  \
        }                                                                                          \
    } while (false)

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> constexpr T DIVUP(const T &x, const T &y) {
    return (((x) + ((y) -1)) / (y));
}
