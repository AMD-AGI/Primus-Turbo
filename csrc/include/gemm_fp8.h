#pragma once
#include "../kernels/gemm/ck_gemm_fp8_launcher.h"
#include <cstdint>
#include <hip/hip_runtime.h>
#include <stdexcept>

namespace primus_turbo {

template <typename AType, typename BType, typename CType, bool transA, bool transB>
void ck_gemm_fp8_blockwise_kernel_impl(const AType *a_ptr, const float *a_scales_ptr,
                                       const BType *b_ptr, const float *b_scales_ptr, CType *c_ptr,
                                       const int32_t M, const int32_t N, const int32_t K,
                                       hipStream_t stream) {
    ck::index_t StrideA = transA ? M : K;
    ck::index_t StrideB = transB ? K : N;
    ck::index_t StrideE = N;

    using OperatorDescriptor =
        typename SelectCKGemmFP8OperatorDescriptor<AType, BType, CType, transA, transB>::type;
    using OperatorBlockConfig = CKGemmFP8Blockwise_M128N128K128_BlockConfig;
    using Operator            = CKGemmFP8BlockwiseLauncher<OperatorDescriptor, OperatorBlockConfig>;

    auto args = Operator::MakeArgument(a_ptr, a_scales_ptr, b_ptr, b_scales_ptr, c_ptr, M, N, K,
                                       StrideA, StrideB, StrideE);
    Operator::Run(args, stream);
}

template <typename AType, typename BType, typename CType>
void ck_gemm_fp8_blockwise_kernel(const AType *a_ptr, const float *a_scales_ptr, const BType *b_ptr,
                                  const float *b_scales_ptr, CType *c_ptr, const int32_t M,
                                  const int32_t N, const int32_t K, const bool transA,
                                  const bool transB, hipStream_t stream) {
    if (!transA && transB) {
        ck_gemm_fp8_blockwise_kernel_impl<AType, BType, CType, false, true>(
            a_ptr, a_scales_ptr, b_ptr, b_scales_ptr, c_ptr, M, N, K, stream);
    } else {
        throw std::runtime_error("ck_gemm_fp8_blockwise: Currently Only NT (transA = false, transB "
                                 "= true) is supported.");
    }
}

} // namespace primus_turbo
