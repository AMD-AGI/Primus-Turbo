// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <exception>
#include <iostream>
#include <string>
#include <type_traits>

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

inline std::string hipblas_status_to_string(hipblasStatus_t status) {
    switch (status) {
    case HIPBLAS_STATUS_SUCCESS:
        return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED:
        return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED:
        return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE:
        return "HIPBLAS_STATUS_INVALID_VALUE";
    case HIPBLAS_STATUS_MAPPING_ERROR:
        return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED:
        return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR:
        return "HIPBLAS_STATUS_INTERNAL_ERROR";
    case HIPBLAS_STATUS_NOT_SUPPORTED:
        return "HIPBLAS_STATUS_NOT_SUPPORTED";
    case HIPBLAS_STATUS_ARCH_MISMATCH:
        return "HIPBLAS_STATUS_ARCH_MISMATCH";
    case HIPBLAS_STATUS_INVALID_ENUM:
        return "HIPBLAS_STATUS_INVALID_ENUM";
    case HIPBLAS_STATUS_UNKNOWN:
        return "HIPBLAS_STATUS_UNKNOWN";
    case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:
        return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";
    }
    return "<undefined hipblasStatus_t value>";
}

} // namespace primus_turbo

#define PRIMUS_TURBO_ERROR(...)                                                                    \
    do {                                                                                           \
        std::cerr << primus_turbo::concat_strings(__FILE__ ":", __LINE__, " Error in function ",   \
                                                  __func__, ": ",                                  \
                                                  ::primus_turbo::concat_strings(__VA_ARGS__))     \
                  << std::endl;                                                                    \
        std::terminate();                                                                          \
    } while (false)

#define PRIMUS_TURBO_CHECK(expr, ...)                                                              \
    do {                                                                                           \
        if (!(expr)) {                                                                             \
            PRIMUS_TURBO_ERROR("Assertion failed: " #expr ". ",                                    \
                               ::primus_turbo::concat_strings(__VA_ARGS__));                       \
        }                                                                                          \
    } while (false)

#define PRIMUS_TURBO_DEVICE_CHECK(expr)                                                            \
    do {                                                                                           \
        if (!(expr)) {                                                                             \
            printf("Assertion failed: %s:%d, expression: %s\n", __FILE__, __LINE__, #expr);        \
            abort();                                                                               \
        }                                                                                          \
    } while (false)

#define PRIMUS_TURBO_CHECK_HIP(expr)                                                               \
    do {                                                                                           \
        const hipError_t status_PRIMUS_TURBO_CHECK_HIP = (expr);                                   \
        if (status_PRIMUS_TURBO_CHECK_HIP != hipSuccess) {                                         \
            PRIMUS_TURBO_ERROR("HIP Error: ", hipGetErrorString(status_PRIMUS_TURBO_CHECK_HIP));   \
        }                                                                                          \
    } while (false)

#define PRIMUS_TURBO_CHECK_HIPBLAS(expr)                                                           \
    do {                                                                                           \
        const hipblasStatus_t status_PRIMUS_CHECK_HIPBLAS = (expr);                                \
        if (status_PRIMUS_CHECK_HIPBLAS != HIPBLAS_STATUS_SUCCESS) {                               \
            PRIMUS_TURBO_ERROR(                                                                    \
                "HIPBLASLT Error: ",                                                               \
                primus_turbo::hipblas_status_to_string(status_PRIMUS_CHECK_HIPBLAS),               \
                " (code=", std::to_string((int) status_PRIMUS_CHECK_HIPBLAS), ")");                \
        }                                                                                          \
    } while (false)

#define PRIMUS_TURBO_STATIC_CHECK(cond, reason) static_assert(cond, reason)

#define PRIMUS_TURBO_CHECK_ROCSHMEM(expr)                                                          \
    do {                                                                                           \
        const auto status_PRIMUS_TURBO_CHECK_ROCSHMEM = (expr);                                    \
        if (status_PRIMUS_TURBO_CHECK_ROCSHMEM != rocshmem::ROCSHMEM_SUCCESS) {                    \
            PRIMUS_TURBO_ERROR("rocSHMEM Error: ",                                                 \
                               std::to_string((int) status_PRIMUS_TURBO_CHECK_ROCSHMEM));          \
        }                                                                                          \
    } while (false)
