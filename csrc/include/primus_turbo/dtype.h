// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once
#include <cstdint>

#include "primus_turbo/float4.h"
#include "primus_turbo/float8.h"
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

// https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/low_fp_types.html#
namespace primus_turbo {

namespace dtype {

using float64       = double;
using float32       = float;
using float16       = half;
using bfloat16      = hip_bfloat16;
using float8_e4m3   = float8_e4m3_t;
using float8_e5m2   = float8_e5m2_t;
using float8_e8m0   = float8_e8m0_t;
using float4x2_e2m1 = float4x2_e2m1_t;

using int64 = int64_t;
using int32 = int32_t;
using int16 = int16_t;
using int8  = int8_t;

using uint64 = uint64_t;
using uint32 = uint32_t;
using uint16 = uint16_t;
using uint8  = uint8_t;

} // namespace dtype

} // namespace primus_turbo
