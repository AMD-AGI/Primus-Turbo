// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "ffi.h"
#include "primus_turbo/common.h"
#include <vector>
#include <xla/ffi/api/ffi.h>

namespace ffi = xla::ffi;

namespace primus_turbo::jax {

//==================================================================
//  RMSNorm
//==================================================================
XLA_FFI_DECLARE_HANDLER_SYMBOL(RMSNormFwdHandler);
XLA_FFI_DECLARE_HANDLER_SYMBOL(RMSNormBwdHandler);

//==================================================================
//  Grouped GEMM
//==================================================================
XLA_FFI_DECLARE_HANDLER_SYMBOL(GroupedGemmHandler);
XLA_FFI_DECLARE_HANDLER_SYMBOL(GroupedGemmVariableKHandler);
XLA_FFI_DECLARE_HANDLER_SYMBOL(ComputeGroupOffsHandler);

XLA_FFI_DECLARE_HANDLER_SYMBOL(GroupedGemmFP8Handler);
XLA_FFI_DECLARE_HANDLER_SYMBOL(GroupedGemmFP8VariableKHandler);

int64_t GetCKGroupedGemmWorkspaceSize(int32_t group_num);
int64_t GetCKGroupedGemmFP8WorkspaceSize(int32_t group_num);
int64_t GetCKGroupedGemmFP8VariableKWorkspaceSize(int32_t group_num, int32_t m, int32_t n);

//==================================================================
//  Quantization
//==================================================================
XLA_FFI_DECLARE_HANDLER_SYMBOL(QuantizeFP8TensorwiseHandler);
XLA_FFI_DECLARE_HANDLER_SYMBOL(DequantizeFP8TensorwiseHandler);
XLA_FFI_DECLARE_HANDLER_SYMBOL(QuantizeFP8RowwiseHandler);

int64_t GetQuantizeFP8TensorwiseWorkspaceSize(int64_t n);

//==================================================================
//  DeepEP
//==================================================================
XLA_FFI_DECLARE_HANDLER_SYMBOL(MoEDispatchHandler);
XLA_FFI_DECLARE_HANDLER_SYMBOL(MoECachedDispatchHandler);
XLA_FFI_DECLARE_HANDLER_SYMBOL(MoECombineHandler);

} // namespace primus_turbo::jax
