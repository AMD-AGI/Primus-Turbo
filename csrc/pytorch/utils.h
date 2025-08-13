// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <torch/extension.h>

#include "primus_turbo/common.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

// clang-format off

/*
#define TORCH_TYPE_SWITCH_ALL(dtype, type, ...)                                                    \
    switch (dtype) {                                                                               \
        using namespace primus_turbo;                                                              \
    case at::kByte: {                                                                              \
        using type = unsigned char;                                                                \
        {                                                                                          \
            __VA_ARGS__                                                                            \
        }                                                                                          \
    } break;                                                                                       \
    case at::kInt32: {                                                                             \
        using type = int32_t;                                                                      \
        {                                                                                          \
            __VA_ARGS__                                                                            \
        }                                                                                          \
    } break;                                                                                       \
    case at::kInt64: {                                                                             \
        using type = int64_t;                                                                      \
        {                                                                                          \
            __VA_ARGS__                                                                            \
        }                                                                                          \
    } break;                                                                                       \
    case at::kFloat: {                                                                             \
        using type = float;                                                                        \
        {                                                                                          \
            __VA_ARGS__                                                                            \
        }                                                                                          \
    } break;                                                                                       \
    case at::kHalf: {                                                                              \
        using type = primus_turbo::dtype::float16;                                                 \
        {                                                                                          \
            __VA_ARGS__                                                                            \
        }                                                                                          \
    } break;                                                                                       \
    case at::kBFloat16: {                                                                          \
        using type = bf16;                                                                         \
        {                                                                                          \
            __VA_ARGS__                                                                            \
        }                                                                                          \
    } break;                                                                                       \
    case at::kFloat8_e4m3fnuz:                                                                     \
    case at::kFloat8_e4m3fn: {                                                                     \
        using type = fp8e4m3;                                                                      \
        {                                                                                          \
            __VA_ARGS__                                                                            \
        }                                                                                          \
    } break;                                                                                       \
    case at::kFloat8_e5m2fnuz:                                                                     \
    case at::kFloat8_e5m2: {                                                                       \
        using type = fp8e5m2;                                                                      \
        {                                                                                          \
            __VA_ARGS__                                                                            \
        }                                                                                          \
    } break;                                                                                       \
    default:                                                                                       \
        PRIMUS_TURBO_ERROR("Invalid type.");                                                       \
    }
*/

#define TORCH_TYPE_SWITCH_OUTPUT(scalar_type, type, ...)                          \
    [&] {                                                                         \
        using namespace primus_turbo;                                             \
        switch (scalar_type) {                                                    \
        case at::kFloat: {                                                        \
            using type = dtype::float32;                                          \
            { __VA_ARGS__ }                                                       \
        } break;                                                                  \
        case at::kHalf: {                                                         \
            using type = dtype::float16;                                          \
            { __VA_ARGS__ }                                                       \
        } break;                                                                  \
        case at::kBFloat16: {                                                     \
            using type = dtype::bfloat16;                                         \
            { __VA_ARGS__ }                                                       \
        } break;                                                                  \
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
            PRIMUS_TURBO_ERROR("Invalid output type.");                           \
        }                                                                         \
    }()

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
