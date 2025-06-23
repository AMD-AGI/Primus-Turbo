#include <torch/extension.h>

namespace primus_turbo::pytorch {

// This is a demo for testing csrc lib
torch::Tensor gemm_meta(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "gemm expects both tensors to be 2D");
    TORCH_CHECK(a.sizes()[1] == b.sizes()[0], "matrix size mismatch");
    const int64_t m = a.sizes()[0];
    const int64_t n = b.sizes()[1];
    return at::empty({m, n}, a.options().device(at::kMeta));
}

torch::Tensor gemm_fp8_blockwise_meta(torch::Tensor &a, torch::Tensor &a_scales, torch::Tensor &b,
                                      torch::Tensor &b_scales, torch::Tensor &c, const bool transA,
                                      const bool transB, const int64_t block_size) {
    const int32_t m = transA ? a.size(1) : a.size(0);
    const int32_t n = transB ? b.size(0) : b.size(1);
    return at::empty({m, n}, c.options().device(at::kMeta));
}

} // namespace primus_turbo::pytorch
