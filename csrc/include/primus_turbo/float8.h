// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <optional>
#include <stdexcept>
#include <string>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_version.h>

#include "primus_turbo/arch.h"
#include "primus_turbo/platform.h"

namespace primus_turbo {

enum class Float8Format { FNUZ, OCP };

inline Float8Format current_fp8_format() {
#if PRIMUS_TURBO_DEVICE_COMPILE
    return Float8Format::FNUZ; // dummy
#else
    static Float8Format fmt = [] { return is_gfx950() ? Float8Format::OCP : Float8Format::FNUZ; }();
    return fmt;
#endif
}

PRIMUS_TURBO_HOST_DEVICE bool is_fp8_fnuz() {
#if PRIMUS_TURBO_DEVICE_COMPILE
#if defined(__gfx950__)
    return false; // gfx950 OCP
#else
    return true;
#endif
#else
    return current_fp8_format() == Float8Format::FNUZ;
#endif
}

struct float8_e4m3_t {

#if PRIMUS_TURBO_DEVICE_COMPILE
#if defined(__gfx950__)
    using storage_t = __hip_fp8_e4m3; // OCP on gfx950
#else
    using storage_t = __hip_fp8_e4m3_fnuz; // FNUZ on others
#endif
    storage_t val;
#else // host side – keep both encodings
    union {
        __hip_fp8_e4m3_fnuz fnuz;
        __hip_fp8_e4m3      ocp;
    } u{};
#endif

    PRIMUS_TURBO_HOST_DEVICE float8_e4m3_t() = default;
    //------------------------------------------------------------------
    // converters
    //------------------------------------------------------------------

    //---------------  from bits  -----------------
    PRIMUS_TURBO_HOST_DEVICE static float8_e4m3_t from_bits(uint8_t bits) {
        float8_e4m3_t x;
#if PRIMUS_TURBO_DEVICE_COMPILE
        *reinterpret_cast<uint8_t *>(&x.val) = bits;
#else
        if (is_fp8_fnuz())
            *reinterpret_cast<uint8_t *>(&x.u.fnuz) = bits;
        else
            *reinterpret_cast<uint8_t *>(&x.u.ocp) = bits;
#endif
        return x;
    }

    //---------------  float32  -----------------
    PRIMUS_TURBO_HOST_DEVICE float8_e4m3_t(float f) { *this = f; }

    PRIMUS_TURBO_HOST_DEVICE float8_e4m3_t &operator=(float f) {
#if PRIMUS_TURBO_DEVICE_COMPILE
        val = static_cast<storage_t>(f);
#else
        if (is_fp8_fnuz())
            u.fnuz = static_cast<__hip_fp8_e4m3_fnuz>(f);
        else
            u.ocp = static_cast<__hip_fp8_e4m3>(f);
#endif
        return *this;
    }

    PRIMUS_TURBO_HOST_DEVICE operator float() const {
#if PRIMUS_TURBO_DEVICE_COMPILE
        return static_cast<float>(val);
#else
        return is_fp8_fnuz() ? static_cast<float>(u.fnuz) : static_cast<float>(u.ocp);
#endif
    }

    //---------------  half  -----------------
    // TODO: Opt CVT
    PRIMUS_TURBO_HOST_DEVICE
    float8_e4m3_t(const half h) { *this = static_cast<float>(h); }

    PRIMUS_TURBO_HOST_DEVICE float8_e4m3_t &operator=(const half h) {
        *this = static_cast<float>(h);
        return *this;
    }

    PRIMUS_TURBO_HOST_DEVICE
    operator half() const { return half(float(*this)); }

    //---------------  bfloat16  -----------------
    // TODO: Opt CVT
    PRIMUS_TURBO_HOST_DEVICE
    float8_e4m3_t(const hip_bfloat16 bf) { *this = static_cast<float>(bf); }

    PRIMUS_TURBO_HOST_DEVICE
    float8_e4m3_t &operator=(const hip_bfloat16 bf) {
        *this = static_cast<float>(bf);
        return *this;
    }

    PRIMUS_TURBO_HOST_DEVICE
    operator hip_bfloat16() const { return hip_bfloat16(float(*this)); }

    //------------------------------------------------------------------
    // Basic arithmetic
    //------------------------------------------------------------------
    PRIMUS_TURBO_HOST_DEVICE friend float8_e4m3_t operator+(const float8_e4m3_t &lhs,
                                                            const float8_e4m3_t &rhs) {
        return float8_e4m3_t(float(lhs) + float(rhs));
    }

    PRIMUS_TURBO_HOST_DEVICE friend float8_e4m3_t operator-(const float8_e4m3_t &lhs,
                                                            const float8_e4m3_t &rhs) {
        return float8_e4m3_t(float(lhs) - float(rhs));
    }

    PRIMUS_TURBO_HOST_DEVICE friend float8_e4m3_t operator*(const float8_e4m3_t &lhs,
                                                            const float8_e4m3_t &rhs) {
        return float8_e4m3_t(float(lhs) * float(rhs));
    }

    PRIMUS_TURBO_HOST_DEVICE friend float8_e4m3_t operator/(const float8_e4m3_t &lhs,
                                                            const float8_e4m3_t &rhs) {
        return float8_e4m3_t(float(lhs) / float(rhs));
    }

    //------------------------------------------------------------------
    // In-place basic arithmetic
    //------------------------------------------------------------------
    PRIMUS_TURBO_HOST_DEVICE float8_e4m3_t &operator+=(const float8_e4m3_t &rhs) {
        *this = *this + rhs;
        return *this;
    }

    PRIMUS_TURBO_HOST_DEVICE float8_e4m3_t &operator-=(const float8_e4m3_t &rhs) {
        *this = *this - rhs;
        return *this;
    }

    PRIMUS_TURBO_HOST_DEVICE float8_e4m3_t &operator*=(const float8_e4m3_t &rhs) {
        *this = *this * rhs;
        return *this;
    }

    PRIMUS_TURBO_HOST_DEVICE float8_e4m3_t &operator/=(const float8_e4m3_t &rhs) {
        *this = *this / rhs;
        return *this;
    }
};
static_assert(sizeof(float8_e4m3_t) == 1, "float8_e4m3_t must be 1 byte");
static_assert(alignof(float8_e4m3_t) == 1);
static_assert(std::is_trivially_copyable_v<float8_e4m3_t>);

struct float8_e5m2_t {

#if PRIMUS_TURBO_DEVICE_COMPILE
#if defined(__gfx950__)
    using storage_t = __hip_fp8_e5m2; // OCP on gfx950
#else
    using storage_t = __hip_fp8_e5m2_fnuz; // FNUZ on others
#endif
    storage_t val;
#else // host side – keep both encodings
    union {
        __hip_fp8_e5m2_fnuz fnuz;
        __hip_fp8_e5m2      ocp;
    } u{};
#endif

    PRIMUS_TURBO_HOST_DEVICE float8_e5m2_t() = default;
    //------------------------------------------------------------------
    // converters
    //------------------------------------------------------------------

    //---------------  from bits  -----------------
    PRIMUS_TURBO_HOST_DEVICE static float8_e5m2_t from_bits(uint8_t bits) {
        float8_e5m2_t x;
#if PRIMUS_TURBO_DEVICE_COMPILE
        *reinterpret_cast<uint8_t *>(&x.val) = bits;
#else
        if (is_fp8_fnuz())
            *reinterpret_cast<uint8_t *>(&x.u.fnuz) = bits;
        else
            *reinterpret_cast<uint8_t *>(&x.u.ocp) = bits;
#endif
        return x;
    }

    //---------------  float32  -----------------
    PRIMUS_TURBO_HOST_DEVICE float8_e5m2_t(float f) { *this = f; }

    PRIMUS_TURBO_HOST_DEVICE float8_e5m2_t &operator=(float f) {
#if PRIMUS_TURBO_DEVICE_COMPILE
        val = static_cast<storage_t>(f);
#else
        if (is_fp8_fnuz())
            u.fnuz = static_cast<__hip_fp8_e5m2_fnuz>(f);
        else
            u.ocp = static_cast<__hip_fp8_e5m2>(f);
#endif
        return *this;
    }

    PRIMUS_TURBO_HOST_DEVICE operator float() const {
#if PRIMUS_TURBO_DEVICE_COMPILE
        return static_cast<float>(val);
#else
        return is_fp8_fnuz() ? static_cast<float>(u.fnuz) : static_cast<float>(u.ocp);
#endif
    }

    //---------------  half  -----------------
    // TODO: Opt CVT
    PRIMUS_TURBO_HOST_DEVICE
    float8_e5m2_t(const half h) { *this = static_cast<float>(h); }

    PRIMUS_TURBO_HOST_DEVICE float8_e5m2_t &operator=(const half h) {
        *this = static_cast<float>(h);
        return *this;
    }

    PRIMUS_TURBO_HOST_DEVICE
    operator half() const { return half(float(*this)); }

    //---------------  bfloat16  -----------------
    // TODO: Opt CVT
    PRIMUS_TURBO_HOST_DEVICE
    float8_e5m2_t(const hip_bfloat16 bf) { *this = static_cast<float>(bf); }

    PRIMUS_TURBO_HOST_DEVICE
    float8_e5m2_t &operator=(const hip_bfloat16 bf) {
        *this = static_cast<float>(bf);
        return *this;
    }

    PRIMUS_TURBO_HOST_DEVICE
    operator hip_bfloat16() const { return hip_bfloat16(float(*this)); }

    //------------------------------------------------------------------
    // Basic arithmetic
    //------------------------------------------------------------------
    PRIMUS_TURBO_HOST_DEVICE friend float8_e5m2_t operator+(const float8_e5m2_t &lhs,
                                                            const float8_e5m2_t &rhs) {
        return float8_e5m2_t(float(lhs) + float(rhs));
    }

    PRIMUS_TURBO_HOST_DEVICE friend float8_e5m2_t operator-(const float8_e5m2_t &lhs,
                                                            const float8_e5m2_t &rhs) {
        return float8_e5m2_t(float(lhs) - float(rhs));
    }

    PRIMUS_TURBO_HOST_DEVICE friend float8_e5m2_t operator*(const float8_e5m2_t &lhs,
                                                            const float8_e5m2_t &rhs) {
        return float8_e5m2_t(float(lhs) * float(rhs));
    }

    PRIMUS_TURBO_HOST_DEVICE friend float8_e5m2_t operator/(const float8_e5m2_t &lhs,
                                                            const float8_e5m2_t &rhs) {
        return float8_e5m2_t(float(lhs) / float(rhs));
    }

    //------------------------------------------------------------------
    // In-place basic arithmetic
    //------------------------------------------------------------------
    PRIMUS_TURBO_HOST_DEVICE float8_e5m2_t &operator+=(const float8_e5m2_t &rhs) {
        *this = *this + rhs;
        return *this;
    }

    PRIMUS_TURBO_HOST_DEVICE float8_e5m2_t &operator-=(const float8_e5m2_t &rhs) {
        *this = *this - rhs;
        return *this;
    }

    PRIMUS_TURBO_HOST_DEVICE float8_e5m2_t &operator*=(const float8_e5m2_t &rhs) {
        *this = *this * rhs;
        return *this;
    }

    PRIMUS_TURBO_HOST_DEVICE float8_e5m2_t &operator/=(const float8_e5m2_t &rhs) {
        *this = *this / rhs;
        return *this;
    }
};
static_assert(sizeof(float8_e5m2_t) == 1, "float8_e5m2_t must be 1 byte");
static_assert(alignof(float8_e5m2_t) == 1);
static_assert(std::is_trivially_copyable_v<float8_e5m2_t>);

} // namespace primus_turbo

namespace std {

using primus_turbo::float8_e4m3_t;
using primus_turbo::float8_e5m2_t;

template <> class numeric_limits<float8_e4m3_t> {
public:
    static constexpr bool is_specialized = true;
    static constexpr bool has_infinity   = false;
    static constexpr bool has_quiet_NaN  = true;

    PRIMUS_TURBO_HOST_DEVICE static float8_e4m3_t min() { return float8_e4m3_t::from_bits(0x08); }

    PRIMUS_TURBO_HOST_DEVICE static float8_e4m3_t max() {
#if PRIMUS_TURBO_DEVICE_COMPILE
#if defined(__gfx950__)
        return float8_e4m3_t::from_bits(0x7E);
#else
        return float8_e4m3_t::from_bits(0x7F);
#endif
#else
        return primus_turbo::is_fp8_fnuz() ? float8_e4m3_t::from_bits(0x7F)
                                           : float8_e4m3_t::from_bits(0x7E);
#endif
    }

    PRIMUS_TURBO_HOST_DEVICE static float8_e4m3_t lowest() { return float8_e4m3_t(-float(max())); }

    // E4M3 has no INF: by specification, has_infinity = false;
    // Here we defensively return max()
    PRIMUS_TURBO_HOST_DEVICE static float8_e4m3_t infinity() { return max(); }

    // NaN
    // OCP : 0x7F
    // FNUZ: 0x80
    PRIMUS_TURBO_HOST_DEVICE static float8_e4m3_t quiet_NaN() {
#if PRIMUS_TURBO_DEVICE_COMPILE
#if defined(__gfx950__)
        return float8_e4m3_t::from_bits(0x7F);
#else
        return float8_e4m3_t::from_bits(0x80);
#endif
#else
        return primus_turbo::is_fp8_fnuz() ? float8_e4m3_t::from_bits(0x80)
                                           : float8_e4m3_t::from_bits(0x7F);
#endif
    }
};

template <> class numeric_limits<float8_e5m2_t> {
public:
    static constexpr bool is_specialized = true;
#if PRIMUS_TURBO_DEVICE_COMPILE
#if defined(__gfx950__)
    static constexpr bool has_infinity = true;
#else
    static constexpr bool has_infinity = false;
#endif
#else
    // Host: cannot determine at compile time whether this is OCP or FNUZ.
    // Therefore, conservatively set has_infinity = false.
    // Generic code should not rely on has_infinity; instead call infinity(),
    // which will return a true Inf in OCP and max() in FNUZ.
    static constexpr bool has_infinity = false;
#endif
    static constexpr bool has_quiet_NaN = true;

    PRIMUS_TURBO_HOST_DEVICE static float8_e5m2_t min() { return float8_e5m2_t::from_bits(0x04); }

    PRIMUS_TURBO_HOST_DEVICE static float8_e5m2_t max() {
#if PRIMUS_TURBO_DEVICE_COMPILE
#if defined(__gfx950__)
        return float8_e5m2_t::from_bits(0x7B);
#else
        return float8_e5m2_t::from_bits(0x7F);
#endif
#else
        return primus_turbo::is_fp8_fnuz() ? float8_e5m2_t::from_bits(0x7F)
                                           : float8_e5m2_t::from_bits(0x7B);
#endif
    }

    PRIMUS_TURBO_HOST_DEVICE static float8_e5m2_t lowest() { return float8_e5m2_t(-float(max())); }

    PRIMUS_TURBO_HOST_DEVICE static float8_e5m2_t infinity() {
#if PRIMUS_TURBO_DEVICE_COMPILE
#if defined(__gfx950__)
        return float8_e5m2_t::from_bits(0x7C);
#else
        return max();
#endif
#else
        return primus_turbo::is_fp8_fnuz() ? max() : float8_e5m2_t::from_bits(0x7C);
#endif
    }

    PRIMUS_TURBO_HOST_DEVICE static float8_e5m2_t quiet_NaN() {
#if PRIMUS_TURBO_DEVICE_COMPILE
#if defined(__gfx950__)
        return float8_e5m2_t::from_bits(0x7D);
#else
        return float8_e5m2_t::from_bits(0x80);
#endif
#else
        return primus_turbo::is_fp8_fnuz() ? float8_e5m2_t::from_bits(0x80)
                                           : float8_e5m2_t::from_bits(0x7D);
#endif
    }
};

} // namespace std
