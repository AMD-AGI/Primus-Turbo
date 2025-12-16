// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "extensions.h"
#include "ffi.h"
#include <pybind11/pybind11.h>

#define REGISTER_FFI_HANDLER(dict, name, fn) dict[#name] = ::primus_turbo::jax::EncapsulateFFI(fn);

namespace primus_turbo::jax {

template <typename T> pybind11::capsule EncapsulateFFI(T *fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be an XLA FFI handler");
    return pybind11::capsule(reinterpret_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict Registrations() {
    pybind11::dict dict;

    // RMSNorm
    // dict["rmsnorm_fwd"] = EncapsulateFFI(RMSNormFwdHandler);
    REGISTER_FFI_HANDLER(dict, rmsnorm_fwd, RMSNormFwdHandler);
    REGISTER_FFI_HANDLER(dict, rmsnorm_bwd, RMSNormBwdHandler);

    // DeepEP
    REGISTER_FFI_HANDLER(dict, moe_dispatch, MoEDispatchHandler);
    REGISTER_FFI_HANDLER(dict, moe_cached_dispatch, MoECachedDispatchHandler);
    REGISTER_FFI_HANDLER(dict, moe_combine, MoECombineHandler);

    // Grouped GEMM
    REGISTER_FFI_HANDLER(dict, grouped_gemm, GroupedGemmHandler);
    REGISTER_FFI_HANDLER(dict, grouped_gemm_variable_k, GroupedGemmVariableKHandler);
    REGISTER_FFI_HANDLER(dict, compute_group_offs, ComputeGroupOffsHandler);

    // Grouped GEMM FP8
    REGISTER_FFI_HANDLER(dict, grouped_gemm_fp8, GroupedGemmFP8Handler);
    REGISTER_FFI_HANDLER(dict, grouped_gemm_fp8_variable_k, GroupedGemmFP8VariableKHandler);

    // FP8 Quantization
    REGISTER_FFI_HANDLER(dict, quantize_fp8_tensorwise, QuantizeFP8TensorwiseHandler);
    REGISTER_FFI_HANDLER(dict, dequantize_fp8_tensorwise, DequantizeFP8TensorwiseHandler);
    REGISTER_FFI_HANDLER(dict, quantize_fp8_rowwise, QuantizeFP8RowwiseHandler);

    return dict;
}

PYBIND11_MODULE(_C, m) {
    m.def("registrations", &Registrations);

    // DType enum
    pybind11::enum_<DType>(m, "DType", pybind11::module_local())
        .value("kByte", DType::kByte)
        .value("kInt16", DType::kInt16)
        .value("kInt32", DType::kInt32)
        .value("kInt64", DType::kInt64)
        .value("kFloat32", DType::kFloat32)
        .value("kFloat16", DType::kFloat16)
        .value("kBFloat16", DType::kBFloat16)
        .value("kFloat8E4M3FN", DType::kFloat8E4M3FN)
        .value("kFloat8E4M3FNUZ", DType::kFloat8E4M3FNUZ)
        .value("kFloat8E5M2", DType::kFloat8E5M2)
        .value("kFloat8E5M2FNUZ", DType::kFloat8E5M2FNUZ)
        .value("kFloat8E8M0", DType::kFloat8E8M0);
}

} // namespace primus_turbo::jax
