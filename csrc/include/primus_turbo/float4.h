// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <optional>
#include <stdexcept>
#include <string>

#include <hip/hip_fp4.h>

#include "primus_turbo/arch.h"
#include "primus_turbo/floating_point_utils.h"
#include "primus_turbo/platform.h"

namespace primus_turbo {

using float4x2_e2m1_t = __hip_fp4x2_e2m1;

} // namespace primus_turbo
