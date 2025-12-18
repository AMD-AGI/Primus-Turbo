// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <torch/extension.h>

#include "extensions.h"

namespace primus_turbo::pytorch {

/********************************************/

TORCH_LIBRARY(primus_turbo_cpp_extension, m) {
    // ********* Gemm *********
    m.def("hipblaslt_gemm(Tensor A, Tensor B, "
          "ScalarType out_dtype, bool transA, bool transB, bool transC) -> Tensor");
    m.def(
        "hipblaslt_gemm_fp8(Tensor A, Tensor scaleA_inv, Tensor B, Tensor scaleB_inv,"
        "ScalarType out_dtype, bool transA, bool transB, bool transC, str granularity) -> Tensor");
    m.def(
        "hipblaslt_gemm_fp4(Tensor A, Tensor scaleA_inv, Tensor B, Tensor scaleB_inv,"
        "ScalarType out_dtype, bool transA, bool transB, bool transC, str granularity) -> Tensor");
    m.def("ck_gemm_fp8(Tensor a, Tensor b, Tensor a_scales, Tensor b_scales, bool transA,"
          "bool transB, ScalarType out_dtype, str granularity) -> Tensor");

    // ********* Quantization *********
    m.def("quantize_fp8_tensorwise(Tensor input, ScalarType dest_dtype, Tensor? scale_opt=None) -> "
          "Tensor[]");
    m.def("quantize_fp8_rowwise(Tensor input, ScalarType dest_dtype, int axis, Tensor? "
          "scale_opt=None) -> Tensor[]");

    m.def("dequantize_fp8_tensorwise(Tensor input, Tensor scale_inv, ScalarType dest_dtype) -> "
          "Tensor");

    // ********* RMSNorm *********
    m.def("rmsnorm_fwd(Tensor input, Tensor gamma, float eps) -> Tensor");
    m.def("rmsnorm_bwd(Tensor input, Tensor gamma, Tensor grad_out, float eps) -> Tensor[]");

    // ********* Grouped Gemm *********
    m.def("ck_grouped_gemm(Tensor a, Tensor b, Tensor group_lens, Tensor group_offs, bool transA, "
          "bool transB, int? num_cu=None) -> Tensor");
    m.def("ck_grouped_gemm_variable_k(Tensor a, Tensor b, Tensor group_lens, Tensor group_offs, "
          "bool transA, bool transB, int? num_cu=None) -> Tensor");
    m.def("ck_grouped_gemm_variable_k_2(Tensor a, Tensor b, Tensor group_lens, Tensor group_offs, "
          "bool transA, bool transB, int? num_cu=None) -> Tensor");
    m.def("ck_grouped_gemm_fp8(Tensor a, Tensor b, Tensor a_scales, Tensor b_scales, "
          "Tensor group_lens, Tensor group_offs, bool transA, bool transB, "
          "ScalarType out_dtype, str granularity, int? num_cu) -> Tensor");
    m.def("ck_grouped_gemm_fp8_variable_k(Tensor a, Tensor b, Tensor a_scales, Tensor b_scales, "
          "Tensor group_lens, Tensor group_offs, bool transA, bool transB, "
          "ScalarType out_dtype, str granularity, int? num_cu) -> Tensor");
    m.def("hipblaslt_grouped_gemm(Tensor a, Tensor b, Tensor group_lens, Tensor group_offs, "
          "bool transA, bool transB, bool pre_sync) -> Tensor");
    m.def("grouped_gemm_compute_offs(Tensor group_lens) -> Tensor");
}

TORCH_LIBRARY_IMPL(primus_turbo_cpp_extension, CUDA, m) {
    // ********* Gemm *********
    m.impl("hipblaslt_gemm", hipblaslt_gemm);
    m.impl("hipblaslt_gemm_fp8", hipblaslt_gemm_fp8);
    m.impl("hipblaslt_gemm_fp4", hipblaslt_gemm_fp4);
    m.impl("ck_gemm_fp8", ck_gemm_fp8);
    // ********* Quantization *********
    m.impl("quantize_fp8_tensorwise", quantize_fp8_tensorwise);
    m.impl("quantize_fp8_rowwise", quantize_fp8_rowwise);

    m.impl("dequantize_fp8_tensorwise", dequantize_fp8_tensorwise);

    // ********* RMSNorm *********
    m.impl("rmsnorm_fwd", rmsnorm_fwd);
    m.impl("rmsnorm_bwd", rmsnorm_bwd);

    // ********* Grouped Gemm *********
    m.impl("ck_grouped_gemm", ck_grouped_gemm);
    m.impl("ck_grouped_gemm_variable_k", ck_grouped_gemm_variable_k);
    m.impl("ck_grouped_gemm_variable_k_2", ck_grouped_gemm_variable_k_2);
    m.impl("ck_grouped_gemm_fp8", ck_grouped_gemm_fp8);
    m.impl("ck_grouped_gemm_fp8_variable_k", ck_grouped_gemm_fp8_variable_k);
    m.impl("grouped_gemm_compute_offs", grouped_gemm_compute_offs);
    m.impl("hipblaslt_grouped_gemm", hipblaslt_grouped_gemm);
}

TORCH_LIBRARY_IMPL(primus_turbo_cpp_extension, Meta, m) {
    // ********* Gemm *********
    m.impl("hipblaslt_gemm", hipblaslt_gemm_meta);
    m.impl("hipblaslt_gemm_fp8", hipblaslt_gemm_fp8_meta);
    m.impl("ck_gemm_fp8", ck_gemm_fp8_meta);

    // ********* Quantization *********
    m.impl("quantize_fp8_tensorwise", quantize_fp8_tensorwise_meta);
    m.impl("dequantize_fp8_tensorwise", dequantize_fp8_tensorwise_meta);
    m.impl("quantize_fp8_rowwise", quantize_fp8_rowwise_meta);

    // ********* RMSNorm *********
    m.impl("rmsnorm_fwd", rmsnorm_fwd_meta);
    m.impl("rmsnorm_bwd", rmsnorm_bwd_meta);

    // ********* Grouped Gemm *********
    m.impl("ck_grouped_gemm", ck_grouped_gemm_meta);
    m.impl("ck_grouped_gemm_variable_k", ck_grouped_gemm_variable_k_meta);
    m.impl("ck_grouped_gemm_variable_k_2", ck_grouped_gemm_variable_k_2_meta);
    m.impl("ck_grouped_gemm_fp8", ck_grouped_gemm_fp8_meta);
    m.impl("ck_grouped_gemm_fp8_variable_k", ck_grouped_gemm_fp8_variable_k_meta);
    m.impl("grouped_gemm_compute_offs", grouped_gemm_compute_offs_meta);
    m.impl("hipblaslt_grouped_gemm", hipblaslt_grouped_gemm_meta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "rendezvous_shmem",
        [](const std::string &group_name, const std::vector<int64_t> &shape,
           c10::ScalarType dtype) { return rendezvous_shmem(group_name, shape, dtype); },
        py::arg("group_name"), py::arg("shape"), py::arg("dtype"));

    // ********* DeepEP *********
    auto deep_ep_module =
        m.def_submodule("deep_ep", "DeepEP: an efficient expert-parallel communication library");
    pybind11::class_<primus_turbo::deep_ep::Config>(deep_ep_module, "Config")
        .def(pybind11::init<int, int, int, int, int>(), py::arg("num_sms") = DEFAULT_NUM_CU,
             py::arg("num_max_nvl_chunked_send_tokens")  = DEFAULT_NUM_MAX_XGMI_CHUNKED_SEND_TOKENS,
             py::arg("num_max_nvl_chunked_recv_tokens")  = DEFAULT_NUM_MAX_XGMI_CHUNKED_RECV_TOKENS,
             py::arg("num_max_rdma_chunked_send_tokens") = DEFAULT_NUM_MAX_RDMA_CHUNKED_SEND_TOKENS,
             py::arg("num_max_rdma_chunked_recv_tokens") = DEFAULT_NUM_MAX_RDMA_CHUNKED_RECV_TOKENS)
        .def("get_nvl_buffer_size_hint", &primus_turbo::deep_ep::Config::get_nvl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint",
             &primus_turbo::deep_ep::Config::get_rdma_buffer_size_hint);

    deep_ep_module.def("get_low_latency_rdma_size_hint",
                       &primus_turbo::deep_ep::get_low_latency_rdma_size_hint);

    pybind11::class_<deep_ep::EventHandle>(deep_ep_module, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

    pybind11::class_<deep_ep::Buffer>(deep_ep_module, "Buffer")
        .def(pybind11::init<int, int, int64_t, int64_t, bool, bool, bool>())
        .def("is_available", &deep_ep::Buffer::is_available)
        .def("get_num_rdma_ranks", &deep_ep::Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &deep_ep::Buffer::get_rdma_rank)
        .def("get_root_rdma_rank", &deep_ep::Buffer::get_root_rdma_rank)
        .def("get_local_device_id", &deep_ep::Buffer::get_local_device_id)
        .def("get_local_ipc_handle", &deep_ep::Buffer::get_local_ipc_handle)
        .def("get_local_nvshmem_unique_id", &deep_ep::Buffer::get_local_nvshmem_unique_id)
        .def("get_local_buffer_tensor", &deep_ep::Buffer::get_local_buffer_tensor)
        .def("get_comm_stream", &deep_ep::Buffer::get_comm_stream)
        .def("sync", &deep_ep::Buffer::sync)
        .def("destroy", &deep_ep::Buffer::destroy)
        .def("get_dispatch_layout", &deep_ep::Buffer::get_dispatch_layout)
        .def("intranode_dispatch", &deep_ep::Buffer::intranode_dispatch)
        .def("intranode_combine", &deep_ep::Buffer::intranode_combine)
        .def("internode_dispatch", &deep_ep::Buffer::internode_dispatch)
        .def("internode_combine", &deep_ep::Buffer::internode_combine)
        .def("clean_low_latency_buffer", &deep_ep::Buffer::clean_low_latency_buffer)
        .def("low_latency_dispatch", &deep_ep::Buffer::low_latency_dispatch)
        .def("low_latency_combine", &deep_ep::Buffer::low_latency_combine)
        .def("get_next_low_latency_combine_buffer",
             &deep_ep::Buffer::get_next_low_latency_combine_buffer);

    // ********* Runtime *********
    auto runtime_module = m.def_submodule("runtime", "Runtime utilities");
    runtime_module.def("create_stream_with_cu_masks", &create_stream_with_cu_masks);
    runtime_module.def("destroy_stream", &destroy_stream);
}

/********************************************/

} // namespace primus_turbo::pytorch
