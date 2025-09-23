// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_group_quant.hpp"
#include <hip/hip_runtime.h>
#include <string>

#include "cktile_gemm_kernel_config.h"

namespace primus_turbo {
// clang-format off

using RowMajor = ck_tile::tensor_layout::gemm::RowMajor;
using ColMajor = ck_tile::tensor_layout::gemm::ColumnMajor;


class CKTileGemmRunnerInterFace {
public:
    virtual ~CKTileGemmRunnerInterFace() = default;
    // virtual void init_args() = 0;
    virtual void run(const ck_tile::stream_config &stream_cfg,
                     const ck_tile::index_t group_num,
                     void *args_ptr, const uint32_t num_cu) = 0;
};

template <
    typename ADataType,
    typename BDataType,
    typename CDataType,
    typename ALayout,
    typename BLayout,
    typename CLayout,
    typename TileConfig,
    typename AccDataType=float
>
class CKTileQuantGemmRunner : public CKTileGemmRunnerInterFace {
public:
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<
            TileConfig::M_Tile,
            TileConfig::N_Tile,
            TileConfig::K_Tile
        >,
        ck_tile::sequence<
            TileConfig::M_Warp,
            TileConfig::N_Warp,
            TileConfig::K_Warp
        >,
        ck_tile::sequence<
            TileConfig::M_Warp_Tile,
            TileConfig::N_Warp_Tile,
            TileConfig::K_Warp_Tile
        >
    >;
    static constexpr ck_tile::QuantType QuantMode = ck_tile::QuantType::RowColQuant;
    using TilePartitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;
    using AQLayout = ck_tile::tensor_layout::gemm::RowMajor;
    using BQLayout = ck_tile::tensor_layout::gemm::ColumnMajor;

    using GemmUniversalTraits = ck_tile::TileGemmQuantTraits<
        TileConfig::kPadM,
        TileConfig::kPadN,
        TileConfig::kPadK,
        false,
        ALayout,
        BLayout,
        CLayout,
        QuantMode,
        AQLayout,
        BQLayout,
        false,
        true
    >;

    using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<typename TypeConfig::ADataType,
                                                                 typename TypeConfig::BDataType,
                                                                 typename TypeConfig::AccDataType,
                                                                 GemmShape,
                                                                 GemmUniversalTraits,
                                                                 ComputeDataType>;

    using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>;

    const ck_tile::index_t K_split = 1;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);




    using QuantGemmProblem = ck_tile::GemmRowColQuantPipelineProblem<ADataType,
                                                                     BDataType,
                                                                     AccDataType,
                                                                     AccDataType,
                                                                     GemmShape,
                                                                     GemmUniversalTraits,
                                                                     false,
                                                                     >;

    // V3
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<QuantGemmProblem>;

    static constexpr ck_tile::memory_operation_enum MemoryOp = ck_tile::memory_operation_enum::set;
    using GemmEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<
                                         ADataType,
                                         BDataType,
                                         ck_tile::tuple<>,
                                         AccDataType,
                                         CDataType,
                                         ck_tile::tuple<>,
                                         CLayout,
                                        ck_tile::element_wise::PassThrough,
            TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
            TileConfig::M_Warp, TileConfig::N_Warp,
            TileConfig::M_Warp_Tile, TileConfig::N_Warp_Tile, TileConfig::K_Warp_Tile,
            QuantGemmProblem::TransposeC,
            MemoryOp
        >
    >;

    using Kernel = ck_tile::QuantGroupedGemmKernel<
        TilePartitioner,
        GemmPipeline,
        GemmEpilogue,
        GemmUniversalTraits::kQuantType
    >;

public:
    void run(const ck_tile::stream_config &stream_cfg,
             const ck_tile::index_t group_num,
             void *args_ptr, const uint32_t num_cu) override;
};


template <
    typename ADataType,
    typename BDataType,
    typename CDataType,
    typename AccDataType,
    typename ALayout,
    typename BLayout,
    typename CLayout
>
std::unique_ptr<CKTileGemmRunnerInterFace>
get_cktile_gemm_instance(const ck_tile::index_t group_num, const ck_tile::index_t m,
                             const ck_tile::index_t n, const ck_tile::index_t k);

// clang-format on
} // namespace primus_turbo
