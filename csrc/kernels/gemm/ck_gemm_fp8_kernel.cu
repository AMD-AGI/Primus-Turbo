#include "ck_gemm_fp8_launcher.h"
#include "gemm_fp8.h"
// #include "ck_gemm_fp8_kernel.cuh"

namespace primus_turbo {

void ck_gemm_fp8_blockwise_kernel(void *a_ptr, void *a_scales_ptr, void *b_ptr, void *b_scales_ptr,
                                  void *c_ptr, const int32_t M, const int32_t N, const int32_t K,
                                  const bool transA, const bool transB, hipStream_t stream) {
    // printf("ck_gemm_fp8_blockwise_kernel\n");

    ck::index_t StrideA = K;
    ck::index_t StrideB = K;
    ck::index_t StrideE = N;

    // using Descriptor =
    //     CKGemmFP8BlockwiseDescriptor_E4M3_BF16_NT_ScaleBlkM1N128K128_GemmBlkM128N128K128;

    using OperatorDescriptor  = CKGemmFP8Blockwise_E4M3_BF16_NT_ScaleBlkM1N128K128_Desc;
    using OperatorBlockConfig = CKGemmFP8Blockwise_M128N128K128_BlockConfig;
    using Operator            = CKGemmFP8BlockwiseLauncher<OperatorDescriptor, OperatorBlockConfig>;

    auto args = Operator::MakeArgument(a_ptr, a_scales_ptr, b_ptr, b_scales_ptr, c_ptr, M, N, K,
                                       StrideA, StrideB, StrideE);

    Operator::Run(args, stream);
}

// void ck_gemm_fp8_blockwise_kernel(void *a_ptr, void *a_scales_ptr, void *b_ptr, void
// *b_scales_ptr,
//                                   void *c_ptr, const int32_t M, const int32_t N, const int32_t K,
//                                   const bool transA, const bool transB, hipStream_t stream) {
//     ck::index_t StrideA = K;
//     ck::index_t StrideB = K;
//     ck::index_t StrideE = N;

//     constexpr ck::index_t NumDTensor = DsDataType::Size();
//     auto a_element_op   = AElementOp{};
//     auto b_element_op   = BElementOp{};
//     auto cde_element_op = CDEElementOp{};

//     auto device_op = DeviceOpInstance{};
//     auto invoker   = device_op.MakeInvoker();
//     auto argument  = device_op.MakeArgument(
//             a_ptr, b_ptr, std::array<const void *, NumDTensor>{},
//             c_ptr, M, N, K, StrideA, StrideB,
//             std::array<ck::index_t, NumDTensor>{},
//             StrideE, a_scales_ptr, b_scales_ptr,
//             a_element_op, b_element_op, cde_element_op);
//     if(!device_op.IsSupportedArgument(argument)) {
//         throw std::runtime_error(
//             "wrong! device_gemm with the specified compilation parameters does "
//             "not support this GEMM problem");
//     }

//     invoker.Run(argument, StreamConfig{stream});
// }

} // namespace primus_turbo
