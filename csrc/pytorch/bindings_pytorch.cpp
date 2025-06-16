#include <torch/extension.h>

namespace primus_turbo::pytorch {

/********************************************/

// This is a demo for testing csrc lib
torch::Tensor gemm(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_meta(torch::Tensor a, torch::Tensor b);

TORCH_LIBRARY(primus_turbo_cpp_extension, m) {
    m.def("gemm(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(primus_turbo_cpp_extension, CUDA, m) {
    m.impl("gemm", gemm);
}

TORCH_LIBRARY_IMPL(primus_turbo_cpp_extension, Meta, m) {
    m.impl("gemm", gemm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("gemm", &gemm_forward, "GEMM kernel for ROCm");
}
/********************************************/

} // namespace primus_turbo::pytorch
