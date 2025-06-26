#pragma once
#include "ck/ck.hpp"
#include "ck/utility/data_type.hpp"
#include <torch/extension.h>

#include "primus_turbo/dtype.h"

namespace primus_turbo::pytorch {

using namespace primus_turbo::dtype;

// ************************************************ //

// CK supported scalar data types.
// https://rocm.docs.amd.com/projects/composable_kernel/en/develop/reference/Composable_Kernel_supported_scalar_types.html

// Map torch::ScalarType -> CK type
template <torch::ScalarType scalar_type> struct TorchToCKType;

// TODO: FP8 Type. OCP
template <> struct TorchToCKType<torch::kFloat8_e4m3fnuz> {
    using type = ck::f8_t;
};

template <> struct TorchToCKType<torch::kFloat8_e5m2fnuz> {
    using type = ck::bf8_t;
};

template <> struct TorchToCKType<torch::kHalf> {
    using type = ck::half_t;
};

template <> struct TorchToCKType<torch::kBFloat16> {
    using type = ck::bhalf_t;
};

template <> struct TorchToCKType<torch::kFloat> {
    using type = float32;
};
// ************************************************ //

} // namespace primus_turbo::pytorch
