// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <torch/extension.h>

#include "primus_turbo/common.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

// clang-format off
#define TORCH_TYPE_SWITCH_FLOAT_ALL(scalar_type, TYPE, ...)                       \
    switch (scalar_type) {                                                        \
    case at::kFloat: {                                                            \
        using TYPE = dtype::float32;                                              \
        { __VA_ARGS__ }                                                           \
    } break;                                                                      \
    case at::kHalf: {                                                             \
        using TYPE = dtype::float16;                                              \
        { __VA_ARGS__ }                                                           \
    } break;                                                                      \
    case at::kBFloat16: {                                                         \
        using TYPE = dtype::bfloat16;                                             \
        { __VA_ARGS__ }                                                           \
    } break;                                                                      \
    case at::kFloat8_e4m3fnuz:                                                    \
    case at::kFloat8_e4m3fn: {                                                    \
        using TYPE = dtype::float8_e4m3;                                          \
        { __VA_ARGS__ }                                                           \
    } break;                                                                      \
    case at::kFloat8_e5m2fnuz:                                                    \
    case at::kFloat8_e5m2: {                                                      \
        using TYPE = dtype::float8_e5m2;                                          \
        { __VA_ARGS__ }                                                           \
    } break;                                                                      \
    default:                                                                      \
        TORCH_CHECK(false, "Invalid dtype (only fp32/fp16/bf16/fp8).");           \
    }

#define TORCH_TYPE_SWITCH_FP16_BF16_FP32(scalar_type, TYPE, ...)                  \
    switch (scalar_type) {                                                        \
    case at::kFloat: {                                                            \
        using TYPE = dtype::float32;                                              \
        { __VA_ARGS__ }                                                           \
    } break;                                                                      \
    case at::kHalf: {                                                             \
        using TYPE = dtype::float16;                                              \
        { __VA_ARGS__ }                                                           \
    } break;                                                                      \
    case at::kBFloat16: {                                                         \
        using TYPE = dtype::bfloat16;                                             \
        { __VA_ARGS__ }                                                           \
    } break;                                                                      \
    default:                                                                      \
        TORCH_CHECK(false, "Invalid dtype (only fp32/fp16/bf16).");               \
    }

#define TORCH_TYPE_SWITCH_FP8(scalar_type, TYPE, ...)                             \
    switch (scalar_type) {                                                        \
    case at::kFloat8_e4m3fnuz:                                                    \
    case at::kFloat8_e4m3fn: {                                                    \
        using TYPE = dtype::float8_e4m3;                                          \
        { __VA_ARGS__ }                                                           \
    } break;                                                                      \
    case at::kFloat8_e5m2fnuz:                                                    \
    case at::kFloat8_e5m2: {                                                      \
        using TYPE = dtype::float8_e5m2;                                          \
        { __VA_ARGS__ }                                                           \
    } break;                                                                      \
    default:                                                                      \
        TORCH_CHECK(false, "Invalid dtype (only fp8).");                          \
    }

#define TORCH_TYPE_SWITCH_FP8ONLY(scalar_type, type, ...)                         \
    [&] {                                                                         \
        using namespace primus_turbo;                                             \
        switch (scalar_type) {                                                    \
        case at::kFloat8_e4m3fnuz:                                                \
        case at::kFloat8_e4m3fn: {                                                \
            using type = dtype::float8_e4m3;                                      \
            { __VA_ARGS__ }                                                       \
        } break;                                                                  \
        case at::kFloat8_e5m2fnuz:                                                \
        case at::kFloat8_e5m2: {                                                  \
            using type = dtype::float8_e5m2;                                      \
            { __VA_ARGS__ }                                                       \
        } break;                                                                  \
        default:                                                                  \
            PRIMUS_TURBO_ERROR("Only float8 types are supported.");               \
        }                                                                         \
    }()

#define TORCH_TYPE_SWITCH_INPUT(scalar_type, type, ...)                      \
    [&] {                                                                    \
        using namespace primus_turbo;                                        \
        switch (scalar_type) {                                               \
        case at::kFloat: {                                                   \
            using type = dtype::float32;                                     \
            { __VA_ARGS__ }                                                  \
        } break;                                                             \
        case at::kHalf: {                                                    \
            using type = dtype::float16;                                     \
            { __VA_ARGS__ }                                                  \
        } break;                                                             \
        case at::kBFloat16: {                                                \
            using type = dtype::bfloat16;                                    \
            { __VA_ARGS__ }                                                  \
        } break;                                                             \
        case at::kFloat8_e4m3fnuz:                                           \
        case at::kFloat8_e4m3fn: {                                           \
            using type = dtype::float8_e4m3;                                 \
            { __VA_ARGS__ }                                                  \
        } break;                                                             \
        case at::kFloat8_e5m2fnuz:                                           \
        case at::kFloat8_e5m2: {                                             \
            using type = dtype::float8_e5m2;                                 \
            { __VA_ARGS__ }                                                  \
        } break;                                                             \
        default:                                                             \
            PRIMUS_TURBO_ERROR("Invalid input type.");                       \
        }                                                                    \
    }()

// clang-format on
