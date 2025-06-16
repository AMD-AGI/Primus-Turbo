#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#include "ck/library/utility/literals.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"

template <ck::index_t... Is> using Seq = ck::Sequence<Is...>;

using FP16 = ck::half_t;
using BF16 = ck::bhalf_t;
using FP8  = ck::f8_t;
using FP32 = float;

using RowMajor = ck::tensor_layout::gemm::RowMajor;
using ColMajor = ck::tensor_layout::gemm::ColumnMajor;

template <typename FP8Type, typename EType, typename ALayout, typename BLayout,
          ck::index_t ScaleBlockM, ck::index_t ScaleBlockN, ck::index_t ScaleBlockK>
struct CKGemmFP8BlockwiseDescriptor {
    using A0DataType       = FP8Type;
    using A1DataType       = FP32;
    using B0DataType       = FP8Type;
    using B1DataType       = FP32;
    using AccDataType      = FP32;
    using CShuffleDataType = FP32;
    using DsDataType       = ck::Tuple<>;
    using EDataType        = EType;
    using ComputeTypeA     = FP8Type;

    using A0Layout = ALayout;
    using B0Layout = BLayout;
    using D0Layout = RowMajor;
    using D1Layout = ColMajor;
    using DsLayout = ck::Tuple<>;
    using ELayout  = RowMajor;

    using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
    using AElementOp   = PassThrough;
    using BElementOp   = PassThrough;
    using CDEElementOp = PassThrough;

    static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

    //--
    static constexpr ck::index_t BlockSize     = 256;
    static constexpr ck::index_t Scale_Block_M = ScaleBlockM;
    static constexpr ck::index_t Scale_Block_N = ScaleBlockN;
    static constexpr ck::index_t Scale_Block_K = ScaleBlockK;

    static constexpr ck::index_t MPerBlock   = 128;
    static constexpr ck::index_t NPerBlock   = 128;
    static constexpr ck::index_t KPerBlock   = 128;
    static constexpr ck::index_t AK1         = 16;
    static constexpr ck::index_t BK1         = 16;
    static constexpr ck::index_t MPerXDL     = 32;
    static constexpr ck::index_t NPerXDL     = 32;
    static constexpr ck::index_t MXdlPerWave = 2;
    static constexpr ck::index_t NXdlPerWave = 2;

    using ABlockTransferThreadClusterLengths_AK0_M_AK1                          = Seq<8, 32, 1>;
    using BBlockTransferThreadClusterLengths_BK0_N_BK1                          = Seq<8, 32, 1>;
    using CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock = Seq<1, 32, 1, 8>;
    //--
};

// Descriptor_<FP8>_<OutType>_<Layout>_ScaleBlkM<SBM>N<SBN>K<SBK>_GemmBlkM<MPB>N<NPB>K<KPB>
using CKGemmFP8BlockwiseDescriptor_E4M3_BF16_NT_ScaleBlkM1N128K128_GemmBlkM128N128K128 =
    CKGemmFP8BlockwiseDescriptor<FP8, BF16, RowMajor, ColMajor, 1, 128, 128>;
