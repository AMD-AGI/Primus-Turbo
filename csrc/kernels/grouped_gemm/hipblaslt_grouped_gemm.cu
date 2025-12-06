// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <mutex>

#include "primus_turbo/grouped_gemm.h"

namespace primus_turbo {

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType = float>
class HipblasltGroupedGemm {};

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType>
void hipblaslt_grouped_gemm(const GroupedGemmParams<ADataType, BDataType, CDataType> &params) {
    // static HipblasltGroupedGemm<ADataType, BDataType, CDataType, AccDataType> instance;
    // instance.run(params);
}

} // namespace primus_turbo
