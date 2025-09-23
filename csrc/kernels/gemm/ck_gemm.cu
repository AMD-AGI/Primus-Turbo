// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "ck_tile/host/hip_check_error.hpp"

#include "ck_gemm_kernel.h"
#include "primus_turbo/grouped_gemm.h"

namespace primus_turbo {

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType,
          typename KernelArgsType>
void ck_gemm_fp8(
    const CKGemmFP8Params<ADataType, BDataType, CDataType, AccDataType, KernelArgsType> &params) {
    const ck_tile::index_t k_batch = 1;
    const bool             splitk  = k_batch > 1;

    const ck_tile::index_t strideA  = params.transA ? params.m : params.k;
    const ck_tile::index_t strideB  = params.transB ? params.k : params.n;
    const ck_tile::index_t strideC  = params.n;
    const ck_tile::index_t strideAQ = 1;
    const ck_tile::index_t strideBQ = 1;

    const auto                             stream_cfg = ck_tile::stream_config{params.stream};
    std::unique_ptr<CKGemmRunnerInterFace> runner;
    using CLayout = RowMajor;
    if (!params.transA && !params.transB) { // NN
        using ALayout = RowMajor;
        using BLayout = RowMajor;
        runner        = get_ck_gemm_instance<ADataType, BDataType, CDataType, AccDataType, ALayout,
                                             BLayout, CLayout>(params.m, params.n, params.k);
    } else if (!params.transA && params.transB) { // NT
        using ALayout = RowMajor;
        using BLayout = ColMajor;
        runner        = get_ck_gemm_instance<ADataType, BDataType, CDataType, AccDataType, ALayout,
                                             BLayout, CLayout>(params.m, params.n, params.k);
    } else {
        PRIMUS_TURBO_CHECK(false, "CKGroupedGemm only support NN and NT");
    }
    runner->run(stream_cfg, params.group_num, params.args_ptr, params.num_cu);
}

} // namespace primus_turbo
