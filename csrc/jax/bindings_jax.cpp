// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "extensions.h"
#include "ffi.h"
#include "jax/deep_ep/deep_ep.h"
#include "primus_turbo/arch.h"
#include "primus_turbo/deep_ep/config.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

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
    REGISTER_FFI_HANDLER(dict, moe_dispatch_inproc, MoEDispatchHandler);
    REGISTER_FFI_HANDLER(dict, moe_cached_dispatch_inproc, MoECachedDispatchHandler);
    REGISTER_FFI_HANDLER(dict, moe_combine_inproc, MoECombineHandler);
    REGISTER_FFI_HANDLER(dict, moe_dispatch_per_process, MoEDispatchPerProcessHandler);
    REGISTER_FFI_HANDLER(dict, moe_cached_dispatch_per_process, MoECachedDispatchPerProcessHandler);
    REGISTER_FFI_HANDLER(dict, moe_combine_per_process, MoECombinePerProcessHandler);
    REGISTER_FFI_HANDLER(dict, moe_internode_dispatch_per_process,
                         MoEInternodeDispatchPerProcessHandler);
    REGISTER_FFI_HANDLER(dict, moe_internode_cached_dispatch_per_process,
                         MoEInternodeCachedDispatchPerProcessHandler);
    REGISTER_FFI_HANDLER(dict, moe_internode_combine_per_process,
                         MoEInternodeCombinePerProcessHandler);

    // Grouped GEMM
    REGISTER_FFI_HANDLER(dict, ck_grouped_gemm, CKGroupedGemmHandler);
    REGISTER_FFI_HANDLER(dict, ck_grouped_gemm_variable_k, CKGroupedGemmVariableKHandler);
    REGISTER_FFI_HANDLER(dict, compute_group_offs, ComputeGroupOffsHandler);

    // Grouped GEMM FP8
    REGISTER_FFI_HANDLER(dict, ck_grouped_gemm_fp8, CKGroupedGemmFP8Handler);
    REGISTER_FFI_HANDLER(dict, ck_grouped_gemm_fp8_variable_k, CKGroupedGemmFP8VariableKHandler);

    // FP8 Quantization
    REGISTER_FFI_HANDLER(dict, quantize_fp8_tensorwise, QuantizeFP8TensorwiseHandler);
    REGISTER_FFI_HANDLER(dict, dequantize_fp8_tensorwise, DequantizeFP8TensorwiseHandler);
    REGISTER_FFI_HANDLER(dict, quantize_fp8_rowwise, QuantizeFP8RowwiseHandler);

    return dict;
}

PYBIND11_MODULE(_C, m) {
    m.def("registrations", &Registrations);

    // DeepEP Config
    pybind11::class_<primus_turbo::deep_ep::Config>(m, "Config")
        .def(pybind11::init<int, int, int, int, int>(), pybind11::arg("num_sms") = DEFAULT_NUM_CU,
             pybind11::arg("num_max_nvl_chunked_send_tokens") =
                 DEFAULT_NUM_MAX_XGMI_CHUNKED_SEND_TOKENS,
             pybind11::arg("num_max_nvl_chunked_recv_tokens") =
                 DEFAULT_NUM_MAX_XGMI_CHUNKED_RECV_TOKENS,
             pybind11::arg("num_max_rdma_chunked_send_tokens") =
                 DEFAULT_NUM_MAX_RDMA_CHUNKED_SEND_TOKENS,
             pybind11::arg("num_max_rdma_chunked_recv_tokens") =
                 DEFAULT_NUM_MAX_RDMA_CHUNKED_RECV_TOKENS)
        .def_readonly("num_sms", &primus_turbo::deep_ep::Config::num_sms)
        .def_readonly("num_max_nvl_chunked_send_tokens",
                      &primus_turbo::deep_ep::Config::num_max_nvl_chunked_send_tokens)
        .def_readonly("num_max_nvl_chunked_recv_tokens",
                      &primus_turbo::deep_ep::Config::num_max_nvl_chunked_recv_tokens)
        .def_readonly("num_max_rdma_chunked_send_tokens",
                      &primus_turbo::deep_ep::Config::num_max_rdma_chunked_send_tokens)
        .def_readonly("num_max_rdma_chunked_recv_tokens",
                      &primus_turbo::deep_ep::Config::num_max_rdma_chunked_recv_tokens)
        .def("get_nvl_buffer_size_hint", &primus_turbo::deep_ep::Config::get_nvl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint",
             &primus_turbo::deep_ep::Config::get_rdma_buffer_size_hint);

    m.def("get_low_latency_rdma_size_hint", &primus_turbo::deep_ep::get_low_latency_rdma_size_hint);

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

    m.def("get_ck_grouped_gemm_workspace_size", &GetCKGroupedGemmWorkspaceSize);
    m.def("get_ck_grouped_gemm_fp8_workspace_size", &GetCKGroupedGemmFP8WorkspaceSize);
    m.def("get_ck_grouped_gemm_fp8_variable_k_workspace_size",
          &GetCKGroupedGemmFP8VariableKWorkspaceSize);

    m.def("get_quantize_fp8_tensorwise_workspace_size", &GetQuantizeFP8TensorwiseWorkspaceSize);
    m.def("get_quantize_fp8_rowwise_workspace_size", &GetQuantizeFP8RowwiseWorkspaceSize);

    m.def("get_device_compute_capability", [](int32_t device_id) -> std::pair<int, int> {
        hipDeviceProp_t prop;
        hipError_t      err = hipGetDeviceProperties(&prop, device_id);
        if (err != hipSuccess) {
            return {0, 0};
        }
        return {prop.major, prop.minor};
    });
    m.def("is_gfx950", &primus_turbo::is_gfx950);

    // DeepEP per-process buffer management
    namespace dep = deep_ep;
    auto dep_m    = m.def_submodule("deep_ep");
    dep_m.def("create_per_process_buffer", &dep::create_per_process_buffer, py::arg("rank"),
              py::arg("num_ranks"), py::arg("num_nvl_bytes"), py::arg("num_rdma_bytes") = 0);
    dep_m.def("sync_per_process_buffer", &dep::sync_per_process_buffer, py::arg("all_handles"),
              py::arg("root_unique_id") = py::none());
    dep_m.def("destroy_per_process_buffer", &dep::destroy_per_process_buffer);
    dep_m.def("is_per_process_buffer_ready", &dep::is_per_process_buffer_ready);
    dep_m.def("per_process_buffer_nvl_bytes", &dep::per_process_buffer_nvl_bytes);
#ifndef DISABLE_ROCSHMEM
    dep_m.def("get_unique_id", []() -> pybind11::bytes {
        auto uid = primus_turbo::deep_ep::internode::get_unique_id();
        return {reinterpret_cast<const char *>(uid.data()), uid.size()};
    });
    dep_m.def(
        "get_rdma_buffer_size_hint",
        [](int64_t hidden_bytes, int num_ranks, int num_sms, int nvl_send, int nvl_recv,
           int rdma_send, int rdma_recv) -> int64_t {
            auto cfg =
                primus_turbo::deep_ep::Config(num_sms, nvl_send, nvl_recv, rdma_send, rdma_recv);
            return static_cast<int64_t>(
                cfg.get_rdma_buffer_size_hint(static_cast<int64_t>(hidden_bytes), num_ranks));
        },
        py::arg("hidden_bytes"), py::arg("num_ranks"), py::arg("num_sms"), py::arg("nvl_send"),
        py::arg("nvl_recv"), py::arg("rdma_send"), py::arg("rdma_recv"));
    dep_m.def("get_source_meta_bytes", &primus_turbo::deep_ep::internode::get_source_meta_bytes);
    dep_m.def("has_rocshmem", []() { return true; });
#else
    dep_m.def("has_rocshmem", []() { return false; });
#endif

    dep_m.def(
        "get_nvl_buffer_size_hint",
        [](int64_t hidden_bytes, int num_ranks, int num_sms, int nvl_send, int nvl_recv,
           int rdma_send, int rdma_recv) -> int64_t {
            auto cfg =
                primus_turbo::deep_ep::Config(num_sms, nvl_send, nvl_recv, rdma_send, rdma_recv);
            return static_cast<int64_t>(
                cfg.get_nvl_buffer_size_hint(static_cast<size_t>(hidden_bytes), num_ranks));
        },
        py::arg("hidden_bytes"), py::arg("num_ranks"), py::arg("num_sms"), py::arg("nvl_send"),
        py::arg("nvl_recv"), py::arg("rdma_send"), py::arg("rdma_recv"));
}

} // namespace primus_turbo::jax
