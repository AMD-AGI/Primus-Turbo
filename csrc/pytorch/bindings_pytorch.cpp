#include <torch/extension.h>

namespace primus_turbo::pytorch {

/********************************************/

// This is a demo for testing csrc lib
torch::Tensor gemm(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_meta(torch::Tensor a, torch::Tensor b);

torch::Tensor gemm_fp8_blockwise(torch::Tensor &a, torch::Tensor &a_scales, torch::Tensor &b,
                                 torch::Tensor &b_scales, torch::Tensor &c, const bool transA,
                                 const bool transB, const int64_t block_size);

TORCH_LIBRARY(primus_turbo_cpp_extension, m) {
    m.def("gemm(Tensor a, Tensor b) -> Tensor");
    m.def("gemm_fp8_blockwise("
          "Tensor a, Tensor a_scales, "
          "Tensor b, Tensor b_scales, "
          "Tensor c, "
          "bool transA, bool transB, "
          "int block_size"
          ") -> Tensor");
}

TORCH_LIBRARY_IMPL(primus_turbo_cpp_extension, CUDA, m) {
    m.impl("gemm", gemm);
    m.impl("gemm_fp8_blockwise", gemm_fp8_blockwise);
}

TORCH_LIBRARY_IMPL(primus_turbo_cpp_extension, Meta, m) {
    m.impl("gemm", gemm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("gemm", &gemm_forward, "GEMM kernel for ROCm");
}
/********************************************/

} // namespace primus_turbo::pytorch
