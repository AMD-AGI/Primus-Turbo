// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "ck_grouped_gemm_kernel.h"

namespace primus_turbo {

template <typename ADataType, typename BDataType, typename CDataType, typename ALayout,
          typename BLayout, typename CLayout, typename TileConfig, typename AccDataType>
void CKGroupedGemmRunner<ADataType, BDataType, CDataType, ALayout, BLayout, CLayout, TileConfig,
                         AccDataType>::run(const ck_tile::stream_config &stream_cfg,
                                           const ck_tile::index_t group_num, void *args_ptr) {

    constexpr int kBlockPerCu = 1;

    constexpr dim3 blocks = Kernel::BlockSize();
    const dim3     grids  = Kernel::MaxOccupancyGridSize(stream_cfg);
    ck_tile::launch_kernel(
        stream_cfg, ck_tile::make_kernel<blocks.x, kBlockPerCu>(
                        Kernel{}, grids, blocks, 0,
                        ck_tile::cast_pointer_to_constant_address_space(args_ptr), group_num));
}

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType,
          typename ALayout, typename BLayout, typename CLayout>
std::unique_ptr<CKGroupedGemmRunnerInterFace>
get_ck_grouped_gemm_instance(const ck_tile::index_t group_num, const ck_tile::index_t m,
                             const ck_tile::index_t n, const ck_tile::index_t k) {
    using TileConfig = CKGroupedGemmTileCfg_256x256x64_32x32x16_2x2x1;
    using Runner = CKGroupedGemmRunner<ADataType, BDataType, CDataType, ALayout, BLayout, CLayout,
                                       TileConfig, AccDataType>;

    return std::make_unique<Runner>();
}

// ** FP16 **
// NT
template std::unique_ptr<CKGroupedGemmRunnerInterFace>
    get_ck_grouped_gemm_instance<ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, float, RowMajor,
                                 ColMajor, RowMajor>(ck_tile::index_t, ck_tile::index_t,
                                                     ck_tile::index_t, ck_tile::index_t);
// NN
template std::unique_ptr<CKGroupedGemmRunnerInterFace>
    get_ck_grouped_gemm_instance<ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, float, RowMajor,
                                 RowMajor, RowMajor>(ck_tile::index_t, ck_tile::index_t,
                                                     ck_tile::index_t, ck_tile::index_t);
// TN
template std::unique_ptr<CKGroupedGemmRunnerInterFace>
    get_ck_grouped_gemm_instance<ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, float, ColMajor,
                                 RowMajor, RowMajor>(ck_tile::index_t, ck_tile::index_t,
                                                     ck_tile::index_t, ck_tile::index_t);

// ** BF16 **
// NT
template std::unique_ptr<CKGroupedGemmRunnerInterFace>
    get_ck_grouped_gemm_instance<ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t,
                                 float, RowMajor, ColMajor, RowMajor>(ck_tile::index_t,
                                                                      ck_tile::index_t,
                                                                      ck_tile::index_t,
                                                                      ck_tile::index_t);
// NN
template std::unique_ptr<CKGroupedGemmRunnerInterFace>
    get_ck_grouped_gemm_instance<ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t,
                                 float, RowMajor, RowMajor, RowMajor>(ck_tile::index_t,
                                                                      ck_tile::index_t,
                                                                      ck_tile::index_t,
                                                                      ck_tile::index_t);
// TN
template std::unique_ptr<CKGroupedGemmRunnerInterFace>
    get_ck_grouped_gemm_instance<ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t,
                                 float, ColMajor, RowMajor, RowMajor>(ck_tile::index_t,
                                                                      ck_tile::index_t,
                                                                      ck_tile::index_t,
                                                                      ck_tile::index_t);

} // namespace primus_turbo
