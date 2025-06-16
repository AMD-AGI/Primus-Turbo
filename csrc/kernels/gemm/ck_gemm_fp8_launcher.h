#pragma once
#include "ck_gemm_fp8_descriptor.h"

namespace primus_turbo {

template <typename Descriptor> struct CKGemmFP8BlockwiseLauncher {
    // clang-format off
    using Kernel = ck::tensor_operation::device::DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3
    <
        typename Descriptor::A0Layout,
        typename Descriptor::B0Layout,
        typename Descriptor::DsLayout,
        typename Descriptor::ELayout,

        typename Descriptor::A0DataType,
        typename Descriptor::A1DataType,
        typename Descriptor::B0DataType,
        typename Descriptor::B1DataType,
        typename Descriptor::DsDataType,
        typename Descriptor::EDataType,
        typename Descriptor::AccDataType,
        typename Descriptor::CShuffleDataType,

        typename Descriptor::AElementOp,
        typename Descriptor::BElementOp,
        typename Descriptor::CDEElementOp,

        Descriptor::GemmSpec,
        Descriptor::BlockSize,
        Descriptor::Scale_Block_M,
        Descriptor::Scale_Block_N,
        Descriptor::Scale_Block_K,

        Descriptor::MPerBlock,
        Descriptor::NPerBlock,
        Descriptor::KPerBlock,
        Descriptor::AK1,
        Descriptor::BK1,
        Descriptor::MPerXDL,
        Descriptor::NPerXDL,
        Descriptor::MXdlPerWave,
        Descriptor::NXdlPerWave,

        typename Descriptor::ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, 2, Descriptor::AK1, Descriptor::AK1, 0,
        typename Descriptor::BBlockTransferThreadClusterLengths_BK0_N_BK1,
        ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, 2, Descriptor::BK1, Descriptor::BK1, 0,
        1, 1,

        typename Descriptor::CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        Seq<8>,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v1,
        typename Descriptor::ComputeTypeA
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
        constexpr ck::index_t NumDTensor     = Descriptor::DsDataType::Size();
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
