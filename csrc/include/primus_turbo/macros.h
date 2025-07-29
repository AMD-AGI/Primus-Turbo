#pragma once

#include <hip/hip_runtime.h>
#include <stdexcept>
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

} // namespace primus_turbo

#define PRIMUS_TURBO_ERROR(...)                                                                    \
    do {                                                                                           \
        throw ::std::runtime_error(                                                                \
            ::primus_turbo::concat_strings(__FILE__ ":", __LINE__, " in function ", __func__,      \
                                           ": ", ::primus_turbo::concat_strings(__VA_ARGS__)));    \
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
            PRIMUS_TURBO_ERROR("HIPBLASLT Error: ",                                                \
                               std::to_string((int) status_PRIMUS_CHECK_HIPBLAS));                 \
        }                                                                                          \
    } while (false)

#define PRIMUS_TURBO_STATIC_CHECK(cond, reason) static_assert(cond, reason)
