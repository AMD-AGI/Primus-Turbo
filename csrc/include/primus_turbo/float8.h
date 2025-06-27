#ifdef 0

#pragma once
#include <optional>
#include <stdexcept>
#include <string>

#include <hip/hip_version.h>

#if HIP_VERSION >= 60200000
#include <hip/hip_fp8.h>
#endif

namespace primus_turbo {

enum class Float8Format { FNUZ, OCP };

//-----------------------------
// Device Path (Compile)
//-----------------------------
#if defined(__HIP_DEVICE_COMPILE__)

#if defined(__gfx950__)
using __float8_e4m3 = __hip_fp8_e4m3;
using __float8_e5m2 = __hip_fp8_e5m2;
static inline __device__ __host__ bool is_fp8_fnuz() {
    return false;
}
#else
using __float8_e4m3 = __hip_fp8_e4m3_fnuz;
using __float8_e5m2 = __hip_fp8_e5m2_fnuz;
static inline __device__ __host__ bool is_fp8_fnuz() {
    return true;
}
#endif

#endif // __HIP_DEVICE_COMPILE__

//-----------------------------
// Host Path (Runtime)
//-----------------------------
#if !defined(__HIP_DEVICE_COMPILE__)

inline Float8Format get_current_fp8_format() {
    static Float8Format fmt = [] {
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, 0) != hipSuccess)
            throw std::runtime_error("hipGetDeviceProperties failed.");
        if (prop.major == 9 && prop.minor == 5)
            return Float8Format::OCP;
        return Float8Format::FNUZ;
    }();
    return fmt;
}

inline bool is_fp8_fnuz() {
    return get_current_fp8_format() == Float8Format::FNUZ;
}

template <typename FNUZ, typename OCP> union __float8_union {
    FNUZ fnuz;
    OCP  ocp;

    __host__ __device__ __float8_union() = default;

    __host__ operator float() const {
        return is_fp8_fnuz() ? fnuz.operator float() : ocp.operator float();
    }

    __device__ operator float() const;

    __host__ __float8_union(const float &v) {
        if (is_fp8_fnuz())
            fnuz = v;
        else
            ocp = v;
    }

    __device__ __float8_union(const float &v);
};

struct __float8_e4m3 {
    __float8_union<__hip_fp8_e4m3_fnuz, __hip_fp8_e4m3> data;

    __host__ __device__ __float8_e4m3() = default;
    __host__ __device__ operator float() const { return data.operator float(); }
    __host__ __device__ __float8_e4m3(const float &v) : data(v) {}
};

struct __float8_e5m2 {
    __float8_union<__hip_fp8_e5m2_fnuz, __hip_fp8_e5m2> data;

    __host__ __device__ __float8_e5m2() = default;
    __host__ __device__ operator float() const { return data.operator float(); }
    __host__ __device__ __float8_e5m2(const float &v) { data = v; }
};

static_assert(sizeof(__float8_e4m3) == 1, "FP8 type must be 1 byte");
static_assert(sizeof(__float8_e5m2) == 1, "FP8 type must be 1 byte");

#endif // end host

} // namespace primus_turbo

#endif // 0
