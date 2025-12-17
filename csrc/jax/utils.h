// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "jax/ffi.h"
#include "primus_turbo/common.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

// clang-format off
#define FFI_TYPE_SWITCH_FLOAT_ALL(ffi_dtype, TYPE, ...)                          \
    switch (ffi_dtype) {                                                         \
    case ffi::F32: {                                                             \
        using TYPE = dtype::float32;                                             \
        { __VA_ARGS__ }                                                          \
    } break;                                                                     \
    case ffi::F16: {                                                             \
        using TYPE = dtype::float16;                                             \
        { __VA_ARGS__ }                                                          \
    } break;                                                                     \
    case ffi::BF16: {                                                            \
        using TYPE = dtype::bfloat16;                                            \
        { __VA_ARGS__ }                                                          \
    } break;                                                                     \
    case ffi::F8E4M3FNUZ:                                                        \
    case ffi::F8E4M3FN: {                                                        \
        using TYPE = dtype::float8_e4m3;                                         \
        { __VA_ARGS__ }                                                          \
    } break;                                                                     \
    case ffi::F8E5M2FNUZ:                                                        \
    case ffi::F8E5M2: {                                                          \
        using TYPE = dtype::float8_e5m2;                                         \
        { __VA_ARGS__ }                                                          \
    } break;                                                                     \
    default:                                                                     \
        PRIMUS_TURBO_ERROR("Invalid dtype (only fp32/fp16/bf16/fp8).");          \
    }

#define FFI_TYPE_SWITCH_FP16_BF16_FP32(ffi_dtype, TYPE, ...)                     \
    switch (ffi_dtype) {                                                         \
    case ffi::F32: {                                                             \
        using TYPE = dtype::float32;                                             \
        { __VA_ARGS__ }                                                          \
    } break;                                                                     \
    case ffi::F16: {                                                             \
        using TYPE = dtype::float16;                                             \
        { __VA_ARGS__ }                                                          \
    } break;                                                                     \
    case ffi::BF16: {                                                            \
        using TYPE = dtype::bfloat16;                                            \
        { __VA_ARGS__ }                                                          \
    } break;                                                                     \
    default:                                                                     \
        PRIMUS_TURBO_ERROR("Invalid dtype (only fp32/fp16/bf16).");              \
    }

#define FFI_TYPE_SWITCH_FP8(ffi_dtype, TYPE, ...)                                \
    switch (ffi_dtype) {                                                         \
    case ffi::F8E4M3FNUZ:                                                        \
    case ffi::F8E4M3FN: {                                                        \
        using TYPE = dtype::float8_e4m3;                                         \
        { __VA_ARGS__ }                                                          \
    } break;                                                                     \
    case ffi::F8E5M2FNUZ:                                                        \
    case ffi::F8E5M2: {                                                          \
        using TYPE = dtype::float8_e5m2;                                         \
        { __VA_ARGS__ }                                                          \
    } break;                                                                     \
    default:                                                                     \
        PRIMUS_TURBO_ERROR("Invalid dtype (only fp8).");                         \
    }

// clang-format on
