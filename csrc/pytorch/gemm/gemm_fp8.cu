#include "../type_traits.h"
#include "primus_turbo/gemm_fp8.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace primus_turbo::pytorch {

inline void print_tensor_info(const torch::Tensor &t, const std::string &name) {
    std::cout << name << ".shape=" << t.sizes() << ", dtype=" << t.dtype() << "; \n";
}

#define DISPATCH_OUT_DTYPE(scalar_type, ...)                                                       \
    if ((scalar_type) == torch::kHalf) {                                                           \
        using CType = typename TorchToCKType<torch::kHalf>::type;                                  \
        {                                                                                          \
            __VA_ARGS__                                                                            \
        }                                                                                          \
    } else if ((scalar_type) == torch::kBFloat16) {                                                \
        using CType = typename TorchToCKType<torch::kBFloat16>::type;                              \
        {                                                                                          \
            __VA_ARGS__                                                                            \
        }                                                                                          \
    } else {                                                                                       \
        TORCH_CHECK(false, "Only support output datatype[fp16,bf16]");                             \
    }

// GEMM FP8 Blockwise
torch::Tensor gemm_fp8_blockwise(torch::Tensor &a, torch::Tensor &a_scales, torch::Tensor &b,
                                 torch::Tensor &b_scales, torch::Tensor &c, const bool transA,
                                 const bool transB, const int64_t block_size) {
    // TODO: Support more layout
    TORCH_CHECK(transA == false && transB == true,
                "CK GEMM FP8 Blockwise currently only support NT");

    // TODO: More Check
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "a, b, c must be CUDA tensors");
    TORCH_CHECK(a.device() == b.device() && a.device() == c.device(),
                "a, b, c must be on the same device");

    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(c.is_contiguous(), "c must be contiguous");

    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "a, b, c must be 2D tensors");
    TORCH_CHECK(a_scales.dim() == 2 && b_scales.dim() == 2,
                "a_scales, b_scales must be 2D tensors");

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

    // std::cout << "M: " << M << " N: " << N << " K: " << K << "\n";

    auto       stream    = at::cuda::getCurrentCUDAStream().stream();
    const auto fp8_dtype = a.scalar_type();
    // TOOD: Hybird?
    TORCH_CHECK(a.scalar_type() == b.scalar_type(),
                "FP8 GEMM requires A and B to have the same dtype, but got A = ", a.scalar_type(),
                ", B = ", b.scalar_type());

    if (fp8_dtype == torch::kFloat8_e4m3fnuz || fp8_dtype == torch::kFloat8_e4m3fn) {
        using AType = typename TorchToCKType<torch::kFloat8_e4m3fnuz>::type;
        using BType = AType;
        DISPATCH_OUT_DTYPE(c.scalar_type(), {
            ck_gemm_fp8_blockwise_kernel<AType, BType, CType>(
                reinterpret_cast<const AType *>(a.data_ptr()), a_scales.data_ptr<dtype::float32>(),
                reinterpret_cast<const BType *>(b.data_ptr()), b_scales.data_ptr<dtype::float32>(),
                reinterpret_cast<CType *>(c.data_ptr()), M, N, K, transA, transB, stream);
        });
    } else if (fp8_dtype == torch::kFloat8_e5m2fnuz || fp8_dtype == torch::kFloat8_e5m2) {
        using AType = typename TorchToCKType<torch::kFloat8_e5m2fnuz>::type;
        using BType = AType;
        DISPATCH_OUT_DTYPE(c.scalar_type(), {
            ck_gemm_fp8_blockwise_kernel<AType, BType, CType>(
                reinterpret_cast<const AType *>(a.data_ptr()), a_scales.data_ptr<dtype::float32>(),
                reinterpret_cast<const BType *>(b.data_ptr()), b_scales.data_ptr<dtype::float32>(),
                reinterpret_cast<CType *>(c.data_ptr()), M, N, K, transA, transB, stream);
        });
    } else {
        TORCH_CHECK(false, "Unsupported: A dtype = ", a.scalar_type(),
                    ", B dtype = ", b.scalar_type());
    }

    return c;
}

} // namespace primus_turbo::pytorch
