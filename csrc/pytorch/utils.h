#pragma once

#include <torch/extension.h>

#include "common.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

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
        using type = fp16;                                                                         \
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
        PRIMUS_ERROR("Invalid type.");                                                             \
    }

#define TORCH_TYPE_SWITCH_OUTPUT(dtype, type, ...)                                                 \
    switch (dtype) {                                                                               \
        using namespace primus_turbo;                                                              \
    case at::kFloat: {                                                                             \
        using type = float;                                                                        \
        {                                                                                          \
            __VA_ARGS__                                                                            \
        }                                                                                          \
    } break;                                                                                       \
    case at::kHalf: {                                                                              \
        using type = fp16;                                                                         \
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
        PRIMUS_ERROR("Invalid type.");                                                             \
    }

#define TORCH_TYPE_SWITCH_FP8ONLY(dtype, type, ...)                                                \
    switch (dtype) {                                                                               \
        using namespace primus_turbo;                                                              \
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
        PRIMUS_ERROR("Invalid type.");                                                             \
    }

#define TORCH_TYPE_SWITCH_INPUT(dtype, type, ...)                                                  \
    switch (dtype) {                                                                               \
        using namespace primus_turbo;                                                              \
    case at::kFloat: {                                                                             \
        using type = float;                                                                        \
        {                                                                                          \
            __VA_ARGS__                                                                            \
        }                                                                                          \
    } break;                                                                                       \
    case at::kHalf: {                                                                              \
        using type = fp16;                                                                         \
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
        PRIMUS_ERROR("Invalid type.");                                                             \
    }
