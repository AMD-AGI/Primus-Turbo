#include <torch/extension.h>

namespace primus_turbo::pytorch {

// This is a demo for testing csrc lib
torch::Tensor gemm(torch::Tensor a, torch::Tensor b) {
    return torch::matmul(a, b);
}

} // primus_turbo::pytorch
