// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "primus_turbo/common.h"
#include <xla/ffi/api/ffi.h>

namespace ffi = xla::ffi;

namespace primus_turbo::jax {

XLA_FFI_DECLARE_HANDLER_SYMBOL(RMSNormFwdHandler);
XLA_FFI_DECLARE_HANDLER_SYMBOL(RMSNormBwdHandler);

// Grouped GEMM (CK backend)
XLA_FFI_DECLARE_HANDLER_SYMBOL(GroupedGemmHandler);
XLA_FFI_DECLARE_HANDLER_SYMBOL(GroupedGemmVariableKHandler);
XLA_FFI_DECLARE_HANDLER_SYMBOL(ComputeGroupOffsHandler);

XLA_FFI_DECLARE_HANDLER_SYMBOL(GroupedGemmFP8Handler);
XLA_FFI_DECLARE_HANDLER_SYMBOL(GroupedGemmFP8VariableKHandler);

// Grouped GEMM (hipBLASLt backend)
XLA_FFI_DECLARE_HANDLER_SYMBOL(GroupedGemmHipblasltHandler);
XLA_FFI_DECLARE_HANDLER_SYMBOL(GroupedGemmVariableKHipblasltHandler);

XLA_FFI_DECLARE_HANDLER_SYMBOL(QuantizeFP8TensorwiseHandler);
XLA_FFI_DECLARE_HANDLER_SYMBOL(DequantizeFP8TensorwiseHandler);
XLA_FFI_DECLARE_HANDLER_SYMBOL(QuantizeFP8RowwiseHandler);

} // namespace primus_turbo::jax
