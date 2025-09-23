// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "ck_tile/core.hpp"
#include "primus_turbo/dtype.h"
#include <torch/extension.h>

namespace primus_turbo::pytorch {

using namespace primus_turbo::dtype;

// ************************************************
/**
 *  DataType Mapping : at::ScalarType -> CK-Tile Type
 */
template <at::ScalarType scalar_type> struct TorchToCKTileType;

template <> struct TorchToCKTileType<at::kFloat8_e4m3fnuz> {
    using type = ck_tile::fp8_t;
};

template <> struct TorchToCKTileType<at::kFloat8_e4m3fn> {
    using type = ck_tile::fp8_t;
};

template <> struct TorchToCKTileType<at::kFloat8_e5m2fnuz> {
    using type = ck_tile::bf8_t;
};

template <> struct TorchToCKTileType<at::kFloat8_e5m2> {
    using type = ck_tile::bf8_t;
};

template <> struct TorchToCKTileType<at::kHalf> {
    using type = ck_tile::half_t;
};

template <> struct TorchToCKTileType<at::kBFloat16> {
    using type = ck_tile::bfloat16_t;
};

template <> struct TorchToCKTileType<at::kFloat> {
    using type = float32;
};
// ************************************************

static inline bool is_16bit_floating_point_dtype(at::ScalarType dtype) {
    return dtype == at::kHalf || dtype == at::kBFloat16;
}

static inline bool is_8bit_floating_point_dtype(at::ScalarType dtype) {
    return dtype == at::kFloat8_e4m3fnuz || dtype == at::kFloat8_e4m3fn ||
           dtype == at::kFloat8_e5m2fnuz || dtype == at::kFloat8_e5m2;
}

static inline bool is_floating_point_dtype(at::ScalarType dtype) {
    return dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat;
}
} // namespace primus_turbo::pytorch
