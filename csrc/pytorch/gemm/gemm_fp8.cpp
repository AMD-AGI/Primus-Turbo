#include "gemm_fp8.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
namespace primus_turbo::pytorch {

inline void print_tensor_info(const torch::Tensor &t, const std::string &name) {
    std::cout << name << ".shape=" << t.sizes() << ", dtype=" << t.dtype() << "; \n";
}

// GEMM FP8 Blockwise
torch::Tensor gemm_fp8_blockwise(torch::Tensor &a, torch::Tensor &a_scales, torch::Tensor &b,
                                 torch::Tensor &b_scales, torch::Tensor &c, const bool transA,
                                 const bool transB, const int64_t block_size) {
    TORCH_CHECK(a.is_cuda(), "a must be CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be CUDA tensor");
    TORCH_CHECK(c.is_cuda(), "c must be CUDA tensor");

    // print_tensor_info(a, "a");
    // print_tensor_info(a_scales, "a_scales");
    // print_tensor_info(b, "b");
    // print_tensor_info(b_scales, "b_scales");
    // print_tensor_info(c, "c");
    // std::cout << "transA=" << transA << ", transB=" << transB << ", block_size=" << block_size
    //           << std::endl;

    const int32_t M = transA ? a.size(1) : a.size(0);
    const int32_t K = transA ? a.size(0) : a.size(1);
    const int32_t N = transB ? b.size(0) : b.size(1);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    ck_gemm_fp8_blockwise_kernel(a.data_ptr(), a_scales.data_ptr(), b.data_ptr(),
                                 b_scales.data_ptr(), c.data_ptr(), M, N, K, transA, transB,
                                 stream);
    return c;
}

} // namespace primus_turbo::pytorch
