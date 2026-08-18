/***************************************************************************************************
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

#pragma once

#include <cstdint>

#include "kittens.cuh"
#include "primus_turbo/hipkittens/attention.h"

namespace primus_turbo::hipkittens {

// Build a HipKittens global-layout descriptor from a plain buffer description.
//
// This replaces the header-only binding the kernels used upstream
// (kittens::py::from_object<GL>, which reads a pybind11::object through the Python attribute
// protocol). That cannot be called from anywhere but Python; make_gl is the real interface
// underneath it and takes exactly what HkTensorDesc carries.
//
// The descriptor holds a raw pointer, so the caller must keep the buffer alive for the
// launch. Every call site is a launcher that takes the descriptor by reference and launches
// before returning, so that holds by construction.
template <typename GL>
GL make_gl_from_desc(const HkTensorDesc &t) {
    return kittens::make_gl<GL>(reinterpret_cast<uint64_t>(t.data), t.d0, t.d1, t.d2, t.d3);
}

}  // namespace primus_turbo::hipkittens
