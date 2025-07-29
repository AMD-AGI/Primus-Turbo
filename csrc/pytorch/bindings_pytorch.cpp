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

    pybind11::class_<deep_ep::Config>(m, "Config")
        .def(pybind11::init<int, int, int, int, int>(), py::arg("num_sms") = DEFAULT_NUM_CU,
             py::arg("num_max_nvl_chunked_send_tokens")  = DEFAULT_NUM_MAX_XGMI_CHUNKED_SEND_TOKENS,
             py::arg("num_max_nvl_chunked_recv_tokens")  = DEFAULT_NUM_MAX_XGMI_CHUNKED_RECV_TOKENS,
             py::arg("num_max_rdma_chunked_send_tokens") = DEFAULT_NUM_MAX_RDMA_CHUNKED_SEND_TOKENS,
             py::arg("num_max_rdma_chunked_recv_tokens") = DEFAULT_NUM_MAX_RDMA_CHUNKED_RECV_TOKENS)
        .def("get_nvl_buffer_size_hint", &deep_ep::Config::get_nvl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint", &deep_ep::Config::get_rdma_buffer_size_hint);
    pybind11::class_<deep_ep::EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

    pybind11::class_<deep_ep::Buffer>(m, "Buffer")
        .def(pybind11::init<int, int, int64_t, int64_t, bool>())
        .def("is_available", &deep_ep::Buffer::is_available)
        .def("is_low_latency_optimize", &deep_ep::Buffer::is_low_latency_optimize)
        .def("get_num_rdma_ranks", &deep_ep::Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &deep_ep::Buffer::get_rdma_rank)
        .def("get_root_rdma_rank", &deep_ep::Buffer::get_root_rdma_rank)
        .def("get_local_device_id", &deep_ep::Buffer::get_local_device_id)
        .def("get_local_ipc_handle", &deep_ep::Buffer::get_local_ipc_handle)
        .def("get_local_pxn_ipc_handle", &deep_ep::Buffer::get_local_pxn_ipc_handle)
        .def("get_local_nvshmem_unique_id", &deep_ep::Buffer::get_local_nvshmem_unique_id)
        .def("get_local_buffer_tensor", &deep_ep::Buffer::get_local_buffer_tensor)
        .def("sync", &deep_ep::Buffer::sync)
        .def("sync_pxn_handles", &deep_ep::Buffer::sync_pxn_handles)
        .def("get_dispatch_layout", &deep_ep::Buffer::get_dispatch_layout)
        .def("intranode_dispatch", &deep_ep::Buffer::intranode_dispatch)
        .def("intranode_combine", &deep_ep::Buffer::intranode_combine)
        .def("internode_dispatch", &deep_ep::Buffer::internode_dispatch)
        .def("internode_combine", &deep_ep::Buffer::internode_combine);
}

/********************************************/

} // namespace primus_turbo::pytorch
