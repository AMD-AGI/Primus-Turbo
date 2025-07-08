#include <torch/extension.h>

#include "../extensions.h"

namespace primus_turbo::pytorch {

at::Tensor hipblaslt_gemm_meta(at::Tensor A, at::Tensor B, const at::ScalarType out_dtype,
                               bool transA, bool transB, bool transC) {
    const int64_t m = transA ? A.size(1) : A.size(0);
    const int64_t n = transB ? B.size(0) : B.size(1);

    std::vector<int64_t> c_shape = transC ? std::vector<int64_t>{n, m} : std::vector<int64_t>{m, n};
    at::TensorOptions    options = at::TensorOptions().dtype(out_dtype).device(at::kMeta);
    return at::empty(c_shape, options);
}

torch::Tensor gemm_fp8_blockwise_meta(torch::Tensor &a, torch::Tensor &a_scales, torch::Tensor &b,
                                      torch::Tensor &b_scales, torch::Tensor &c, const bool transA,
                                      const bool transB, const int64_t block_size) {
    const int32_t m = transA ? a.size(1) : a.size(0);
    const int32_t n = transB ? b.size(0) : b.size(1);
    return at::empty({m, n}, c.options().device(at::kMeta));
}

} // namespace primus_turbo::pytorch
