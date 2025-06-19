#pragma once
#include "ck_gemm_fp8_config.h"

namespace primus_turbo {

template <typename OperatorDesc, typename BlockConfig> struct CKGemmFP8BlockwiseLauncher {
    // clang-format off
    using Kernel = ck::tensor_operation::device::DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3
    <
        typename OperatorDesc::A0Layout,
        typename OperatorDesc::B0Layout,
        typename OperatorDesc::DsLayout,
        typename OperatorDesc::ELayout,

        typename OperatorDesc::A0DataType,
        typename OperatorDesc::A1DataType,
        typename OperatorDesc::B0DataType,
        typename OperatorDesc::B1DataType,
        typename OperatorDesc::DsDataType,
        typename OperatorDesc::EDataType,
        typename OperatorDesc::AccDataType,
        typename OperatorDesc::CShuffleDataType,

        typename OperatorDesc::AElementOp,
        typename OperatorDesc::BElementOp,
        typename OperatorDesc::CDEElementOp,

        OperatorDesc::GemmSpec,

        BlockConfig::BlockSize,

        OperatorDesc::Scale_Block_M,
        OperatorDesc::Scale_Block_N,
        OperatorDesc::Scale_Block_K,

        BlockConfig::MPerBlock,
        BlockConfig::NPerBlock,
        BlockConfig::KPerBlock,
        BlockConfig::AK1,
        BlockConfig::BK1,
        BlockConfig::MPerXDL,
        BlockConfig::NPerXDL,
        BlockConfig::MXdlPerWave,
        BlockConfig::NXdlPerWave,

        typename BlockConfig::ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, 2, BlockConfig::AK1, BlockConfig::AK1, 0,
        typename BlockConfig::BBlockTransferThreadClusterLengths_BK0_N_BK1,
        ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, 2, BlockConfig::BK1, BlockConfig::BK1, 0,
        1, 1,

        typename BlockConfig::CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        Seq<8>,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v1,
        typename OperatorDesc::ComputeTypeA
    >;
    // clang-format on

    static typename Kernel::Argument MakeArgument(void *a_ptr, void *a_scales_ptr, void *b_ptr,
                                                  void *b_scales_ptr, void *c_ptr, const int32_t M,
                                                  const int32_t N, const int32_t K,
                                                  ck::index_t StrideA, ck::index_t StrideB,
                                                  ck::index_t StrideE) {
        // TODO:
        using PassThrough                    = ck::tensor_operation::element_wise::PassThrough;
        auto                  a_element_op   = PassThrough{};
        auto                  b_element_op   = PassThrough{};
        auto                  cde_element_op = PassThrough{};
        constexpr ck::index_t NumDTensor     = OperatorDesc::DsDataType::Size();
        auto                  argument       = Kernel::MakeArgument(
            a_ptr, b_ptr, std::array<const void *, NumDTensor>{}, c_ptr, M, N, K, StrideA, StrideB,
            std::array<ck::index_t, NumDTensor>{}, StrideE, a_scales_ptr, b_scales_ptr,
            a_element_op, b_element_op, cde_element_op);
        if (!Kernel::IsSupportedArgument(argument)) {
            // TODO:
            throw std::runtime_error(
                "wrong! device_gemm with the specified compilation parameters does "
                "not support this GEMM problem");
        }
        return argument;
    }

    static void Run(const typename Kernel::Argument &args, hipStream_t stream) {
        auto invoker = Kernel::MakeInvoker();
        invoker.Run(args, StreamConfig{stream});
    }
};

} // namespace primus_turbo
