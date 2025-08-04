#include <torch/extension.h>

#include "extensions.h"

namespace primus_turbo::pytorch {

/********************************************/

TORCH_LIBRARY(primus_turbo_cpp_extension, m) {
    m.def("hipblaslt_gemm(Tensor A, Tensor scaleA_inv, Tensor B, Tensor scaleB_inv,"
          "ScalarType out_dtype, bool transA, bool transB, bool "
          "transC) -> "
          "Tensor");
    m.def("gemm_fp8_blockwise("
          "Tensor a, Tensor a_scales, "
          "Tensor b, Tensor b_scales, "
          "Tensor c, "
          "bool transA, bool transB, "
          "int block_size"
          ") -> Tensor");
    m.def("fp8_quantize(Tensor input, Tensor scale, ScalarType dest_dtype) -> Tensor");
    m.def("fp8_dequantize(Tensor input, Tensor scale_inv, ScalarType dest_dtype) -> Tensor");
    m.def("rmsnorm_fwd(Tensor input, Tensor gamma, float eps) -> Tensor");
    m.def("rmsnorm_bwd(Tensor input, Tensor gamma, Tensor grad_out, float eps) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(primus_turbo_cpp_extension, CUDA, m) {
    m.impl("hipblaslt_gemm", hipblaslt_gemm);
    m.impl("gemm_fp8_blockwise", gemm_fp8_blockwise);
    m.impl("fp8_quantize", fp8_quantize);
    m.impl("fp8_dequantize", fp8_dequantize);
    m.impl("rmsnorm_fwd", rmsnorm_fwd);
    m.impl("rmsnorm_bwd", rmsnorm_bwd);
}

TORCH_LIBRARY_IMPL(primus_turbo_cpp_extension, Meta, m) {
    m.impl("hipblaslt_gemm", hipblaslt_gemm_meta);
    m.impl("gemm_fp8_blockwise", gemm_fp8_blockwise_meta);
    m.impl("fp8_quantize", fp8_quantize_meta);
    m.impl("fp8_dequantize", fp8_dequantize_meta);
    m.impl("rmsnorm_fwd", rmsnorm_fwd_meta);
    m.impl("rmsnorm_bwd", rmsnorm_bwd_meta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "rendezvous_shmem",
        [](const std::string &group_name, const std::vector<int64_t> &shape,
           c10::ScalarType dtype) { return rendezvous_shmem(group_name, shape, dtype); },
        py::arg("group_name"), py::arg("shape"), py::arg("dtype"));

    auto deep_ep_module =
        m.def_submodule("deep_ep", "DeepEP: an efficient expert-parallel communication library");
    pybind11::class_<deep_ep::Config>(deep_ep_module, "Config")
        .def(pybind11::init<int, int, int, int, int>(), py::arg("num_sms") = DEFAULT_NUM_CU,
             py::arg("num_max_nvl_chunked_send_tokens")  = DEFAULT_NUM_MAX_XGMI_CHUNKED_SEND_TOKENS,
             py::arg("num_max_nvl_chunked_recv_tokens")  = DEFAULT_NUM_MAX_XGMI_CHUNKED_RECV_TOKENS,
             py::arg("num_max_rdma_chunked_send_tokens") = DEFAULT_NUM_MAX_RDMA_CHUNKED_SEND_TOKENS,
             py::arg("num_max_rdma_chunked_recv_tokens") = DEFAULT_NUM_MAX_RDMA_CHUNKED_RECV_TOKENS)
        .def("get_nvl_buffer_size_hint", &deep_ep::Config::get_nvl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint", &deep_ep::Config::get_rdma_buffer_size_hint);

    deep_ep_module.def("get_low_latency_rdma_size_hint", &deep_ep::get_low_latency_rdma_size_hint);

    pybind11::class_<deep_ep::EventHandle>(deep_ep_module, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

    pybind11::class_<deep_ep::Buffer>(deep_ep_module, "Buffer")
        .def(pybind11::init<int, int, int64_t, int64_t, bool, bool>())
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
}

/********************************************/

} // namespace primus_turbo::pytorch
