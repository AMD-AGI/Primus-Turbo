/***************************************************************************************************
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

#pragma once

#include <ATen/ATen.h>

#include <array>
#include <cstdint>
#include <string>

#include "kittens.cuh"

namespace primus_turbo::hipkittens {

// Build a HipKittens global-layout descriptor from a torch tensor.
//
// This replaces the header-only binding the kernels used upstream
// (kittens::py::from_object<GL>::make, which reads a pybind11::object through the Python
// attribute protocol). Reaching the same fields off at::Tensor keeps the kernels callable
// from a torch custom op, where there is no pybind object to inspect, and moves the shape
// and device checks from Python attribute lookups to the C++ API.
//
// The descriptor holds a raw pointer, so the caller must keep the tensor alive for the
// launch. Every call site here is a launcher that takes the tensor by reference and
// launches before returning, so that holds by construction.
template <typename GL>
GL make_gl_from_tensor(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda(), "hipkittens attention: ", name, " must be on an AMD GPU");
    TORCH_CHECK(t.is_contiguous(), "hipkittens attention: ", name,
                " must be contiguous; the kernel derives its strides from the shape");
    TORCH_CHECK(t.dim() <= 4, "hipkittens attention: ", name, " must have at most 4 dims, got ",
                t.dim());

    // Right-align the shape into four axes, exactly as the upstream binding did, so a tensor
    // with fewer dims lands on the same axes the kernel indexes.
    std::array<int, 4> shape = {1, 1, 1, 1};
    const int64_t dims = t.dim();
    for (int64_t i = 0; i < dims; ++i) {
        shape[4 - dims + i] = static_cast<int>(t.size(i));
    }

    return kittens::make_gl<GL>(reinterpret_cast<uint64_t>(t.data_ptr()), shape[0], shape[1],
                                shape[2], shape[3]);
}

}  // namespace primus_turbo::hipkittens
