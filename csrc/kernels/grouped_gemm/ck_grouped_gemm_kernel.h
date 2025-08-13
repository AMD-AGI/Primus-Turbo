// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/ops/gemm.hpp"

#include <hip/hip_runtime.h>

#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

#include "ck_grouped_gemm_kernel_config.h"

namespace primus_turbo {
// clang-format off

using RowMajor = ck_tile::tensor_layout::gemm::RowMajor;
using ColMajor = ck_tile::tensor_layout::gemm::ColumnMajor;


class CKGroupedGemmRunnerInterFace {
public:
    virtual ~CKGroupedGemmRunnerInterFace() = default;
    // virtual void init_args() = 0;
    virtual void run(const ck_tile::stream_config &stream_cfg,
                     const ck_tile::index_t group_num,
                     void *args_ptr) = 0;
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
class CKGroupedGemmRunner : public CKGroupedGemmRunnerInterFace {
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

    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
        GemmShape,
        TileConfig::TileParitionerGroupNum,
        TileConfig::TileParitionerM01
    >;

    using Traits = ck_tile::TileGemmTraits<
        TileConfig::kPadM, TileConfig::kPadN, TileConfig::kPadK,
        ALayout, BLayout, CLayout
    >;

    using GemmUniversalTraits = ck_tile::PersistentTileGemmUniversalTraits<
        TileConfig::kPadM, TileConfig::kPadN, TileConfig::kPadK,
        TileConfig::DoubleSmemBuffer,
        ALayout, BLayout, CLayout>;

    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<
        ADataType,
        BDataType,
        AccDataType,
        GemmShape,
        Traits
    >;

    static constexpr ck_tile::GemmPipelineScheduler GemmPipelineScheduler = ck_tile::GemmPipelineScheduler::Intrawave;

    using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
        ADataType,
        BDataType,
        AccDataType,
        GemmShape,
        GemmUniversalTraits,
        GemmPipelineScheduler
    >;

    // V3
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;
    // using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3;

    static constexpr ck_tile::memory_operation_enum MemoryOp = ck_tile::memory_operation_enum::set;
    using GemmEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<
            ADataType, BDataType, ck_tile::tuple<>, AccDataType,
            CDataType, ck_tile::tuple<>, CLayout,
            ck_tile::element_wise::PassThrough,
            GemmPipelineProblem::kBlockSize,
            TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
            TileConfig::M_Warp, TileConfig::N_Warp,
            TileConfig::M_Warp_Tile, TileConfig::N_Warp_Tile, TileConfig::K_Warp_Tile,
            UniversalGemmProblem::TransposeC,
            MemoryOp
        >
    >;

    using Kernel = ck_tile::GroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

public:
    void run(const ck_tile::stream_config &stream_cfg,
             const ck_tile::index_t group_num,
             void *args_ptr) override;
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
std::unique_ptr<CKGroupedGemmRunnerInterFace>
get_ck_grouped_gemm_instance(const ck_tile::index_t group_num, const ck_tile::index_t m,
                             const ck_tile::index_t n, const ck_tile::index_t k);

// clang-format on
} // namespace primus_turbo
