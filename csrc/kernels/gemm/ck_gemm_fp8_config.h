#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#include "ck/library/utility/literals.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"

namespace primus_turbo {

template <ck::index_t... Is> using Seq = ck::Sequence<Is...>;

using FP16 = ck::half_t;
using BF16 = ck::bhalf_t;
using FP8  = ck::f8_t;
using BF8  = ck::bf8_t;
using FP32 = float;

using RowMajor = ck::tensor_layout::gemm::RowMajor;
using ColMajor = ck::tensor_layout::gemm::ColumnMajor;

//******************************************************************//
//**************************** Definition **************************//
//******************************************************************//

//====================== Operator Descriptor =======================
template <typename FP8Type_, typename EType_, typename ALayout_, typename BLayout_,
          ck::index_t ScaleBlockM_, ck::index_t ScaleBlockN_, ck::index_t ScaleBlockK_>
struct CKGemmFP8OperatorDescriptor {
    using A0DataType       = FP8Type_;
    using A1DataType       = FP32;
    using B0DataType       = FP8Type_;
    using B1DataType       = FP32;
    using AccDataType      = FP32;
    using CShuffleDataType = FP32;
    using DsDataType       = ck::Tuple<>;
    using EDataType        = EType_;
    using ComputeTypeA     = FP8Type_;

    using A0Layout = ALayout_;
    using B0Layout = BLayout_;
    using DsLayout = ck::Tuple<>;
    using ELayout  = RowMajor;

    using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
    using AElementOp   = PassThrough;
    using BElementOp   = PassThrough;
    using CDEElementOp = PassThrough;

    static constexpr ck::index_t ScaleBlockM = ScaleBlockM_;
    static constexpr ck::index_t ScaleBlockN = ScaleBlockN_;
    static constexpr ck::index_t ScaleBlockK = ScaleBlockK_;

    static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;
};

//===================== SelectCKGemmFP8OperatorDescriptor ======================
// TODO: TransA and TransB
template <typename AType, typename BType, typename CType> struct SelectCKGemmFP8OperatorDescriptor;

//================================ Block Config ===========-====================
template <ck::index_t BlockSize_, ck::index_t MPerBlock_, ck::index_t NPerBlock_,
          ck::index_t KPerBlock_, ck::index_t AK1_, ck::index_t BK1_, ck::index_t MPerXDL_,
          ck::index_t NPerXDL_, ck::index_t MXdlPerWave_, ck::index_t NXdlPerWave_,
          typename AClusterLengths_, typename BClusterLengths_, typename CShuffleClusterLengths_>
struct CKGemmFP8BlockConfig {
    static constexpr ck::index_t BlockSize = BlockSize_;

    static constexpr ck::index_t MPerBlock = MPerBlock_;
    static constexpr ck::index_t NPerBlock = NPerBlock_;
    static constexpr ck::index_t KPerBlock = KPerBlock_;

    static constexpr ck::index_t AK1 = AK1_;
    static constexpr ck::index_t BK1 = BK1_;

    static constexpr ck::index_t MPerXDL = MPerXDL_;
    static constexpr ck::index_t NPerXDL = NPerXDL_;

    static constexpr ck::index_t MXdlPerWave = MXdlPerWave_;
    static constexpr ck::index_t NXdlPerWave = NXdlPerWave_;

    using ABlockTransferThreadClusterLengths_AK0_M_AK1 = AClusterLengths_;
    using BBlockTransferThreadClusterLengths_BK0_N_BK1 = BClusterLengths_;
    using CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock =
        CShuffleClusterLengths_;
};

//******************************************************************//
//**************************** Specialization **********************//
//******************************************************************//

//====================== Gemm Desc Specialization ==================

using CKGemmFP8Blockwise_E4M3_BF16_NT_ScaleBlkM1N128K128_Desc =
    CKGemmFP8OperatorDescriptor<FP8, BF16, RowMajor, ColMajor, 1, 128, 128>;

using CKGemmFP8Blockwise_E4M3_FP16_NT_ScaleBlkM1N128K128_Desc =
    CKGemmFP8OperatorDescriptor<FP8, FP16, RowMajor, ColMajor, 1, 128, 128>;

using CKGemmFP8Blockwise_E5M2_BF16_NT_ScaleBlkM1N128K128_Desc =
    CKGemmFP8OperatorDescriptor<BF8, BF16, RowMajor, ColMajor, 1, 128, 128>;

using CKGemmFP8Blockwise_E5M2_FP16_NT_ScaleBlkM1N128K128_Desc =
    CKGemmFP8OperatorDescriptor<BF8, FP16, RowMajor, ColMajor, 1, 128, 128>;

// ** Currently CK dont support
// using CKGemmFP8Blockwise_E4M3_BF16_NN_ScaleBlkM1N128K128_Desc =
//     CKGemmFP8OperatorDescriptor<FP8, BF16, RowMajor, RowMajor, 1, 128, 128>;

//====================== Gemm Block Specialization =================

using CKGemmFP8Blockwise_M128N128K128_BlockConfig =
    CKGemmFP8BlockConfig<256, 128, 128, 128, 16, 16, 32, 32, 2, 2, Seq<8, 32, 1>, Seq<8, 32, 1>,
                         Seq<1, 32, 1, 8>>;

// ======= SelectCKGemmFP8OperatorDescriptor Specialization =======

// E4M3 + FP16
template <> struct SelectCKGemmFP8OperatorDescriptor<ck::f8_t, ck::f8_t, ck::half_t> {
    using type = CKGemmFP8Blockwise_E4M3_FP16_NT_ScaleBlkM1N128K128_Desc;
};

// E4M3 + BF16
template <> struct SelectCKGemmFP8OperatorDescriptor<ck::f8_t, ck::f8_t, ck::bhalf_t> {
    using type = CKGemmFP8Blockwise_E4M3_BF16_NT_ScaleBlkM1N128K128_Desc;
};

// E5M2 + FP16
template <> struct SelectCKGemmFP8OperatorDescriptor<ck::bf8_t, ck::bf8_t, ck::half_t> {
    using type = CKGemmFP8Blockwise_E5M2_FP16_NT_ScaleBlkM1N128K128_Desc;
};
// E5M2 + BF16
template <> struct SelectCKGemmFP8OperatorDescriptor<ck::bf8_t, ck::bf8_t, ck::bhalf_t> {
    using type = CKGemmFP8Blockwise_E5M2_BF16_NT_ScaleBlkM1N128K128_Desc;
};

} // namespace primus_turbo
