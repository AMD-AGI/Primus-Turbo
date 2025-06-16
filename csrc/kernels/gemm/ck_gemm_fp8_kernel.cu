#include "ck_gemm_fp8_launcher.h"
#include "gemm_fp8.h"

namespace primus_turbo {

void ck_gemm_fp8_blockwise_kernel(void *a_ptr, void *a_scales_ptr, void *b_ptr, void *b_scales_ptr,
                                  void *c_ptr, const int32_t M, const int32_t N, const int32_t K,
                                  const bool transA, const bool transB, hipStream_t stream) {
    printf("ck_gemm_fp8_blockwise_kernel\n");

    ck::index_t StrideA = K;
    ck::index_t StrideB = K;
    ck::index_t StrideE = N;

    using Descriptor =
        CKGemmFP8BlockwiseDescriptor_E4M3_BF16_NT_ScaleBlkM1N128K128_GemmBlkM128N128K128;
    auto args = CKGemmFP8BlockwiseLauncher<Descriptor>::MakeArgument(
        a_ptr, a_scales_ptr, b_ptr, b_scales_ptr, c_ptr, M, N, K, StrideA, StrideB, StrideE);

    CKGemmFP8BlockwiseLauncher<Descriptor>::Run(args, stream);
}

} // namespace primus_turbo
