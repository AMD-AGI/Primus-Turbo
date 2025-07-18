#include "primus_turbo/macros.h"
#include <hip/hip_runtime.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/extension.h>

#include <c10/hip/HIPFunctions.h>

namespace primus_turbo::pytorch {

using ProcessGroupBackend = c10d::ProcessGroup::BackendType;

template <ProcessGroupBackend type>
void all_gather_helper(c10d::ProcessGroup *pg, const void *src, void *dst, int64_t nbytes) {
    PRIMUS_TURBO_CHECK(false, "not support for this group type " + pg->getBackendName());
}

template <>
void all_gather_helper<ProcessGroupBackend::NCCL>(c10d::ProcessGroup *pg, const void *src,
                                                  void *dst, int64_t nbytes) {
    auto option_cpu = at::TensorOptions(torch::kUInt8).device(at::kCPU);
    auto option_gpu =
        at::TensorOptions(torch::kUInt8).device(at::kHIP).device_index(c10::hip::current_device());
    auto dst_tensor     = at::from_blob(dst, {nbytes * pg->getSize()}, option_cpu);
    auto src_tensor     = at::from_blob(const_cast<void *>(src), {nbytes}, option_cpu);
    auto dst_tensor_gpu = dst_tensor.to(option_gpu);
    auto src_tensor_gpu = src_tensor.to(option_gpu);
    pg->_allgather_base(dst_tensor_gpu, src_tensor_gpu)->wait();
    dst_tensor.copy_(dst_tensor_gpu.to(option_cpu));
}

template <>
void all_gather_helper<ProcessGroupBackend::GLOO>(c10d::ProcessGroup *pg, const void *src,
                                                  void *dst, int64_t nbytes) {
    auto option_cpu = at::TensorOptions(torch::kUInt8).device(at::kCPU);
    auto dst_tensor = at::from_blob(dst, {nbytes * pg->getSize()}, option_cpu);
    auto src_tensor = at::from_blob(const_cast<void *>(src), {nbytes}, option_cpu);
    pg->_allgather_base(dst_tensor, src_tensor)->wait();
}

std::vector<torch::Tensor> rendezvous_shmem(const std::string          &group_name,
                                            const std::vector<int64_t> &shape,
                                            c10::ScalarType             dtype) {
    auto group = c10d::resolve_process_group(group_name);
    PRIMUS_TURBO_CHECK(group.get() != nullptr, group_name + " resolved failed!");
    int cur_rank   = group->getRank();
    int world_size = group->getSize();

    // This ring mode behavior is typically used in scenarios where the p2p
    // protocol is not worked any more such as the number of peers exceeds 8.
    auto option_gpu =
        at::TensorOptions(dtype).device(at::kHIP).device_index(c10::hip::current_device());

    PRIMUS_TURBO_CHECK(world_size <= torch::hip::device_count() &&
                       "create_ipc_tensors should only be used intra node");
    auto size = torch::elementSize(dtype) *
                std::accumulate(shape.begin(), shape.end(), (size_t) 1, std::multiplies<>());
    PRIMUS_TURBO_CHECK(size != 0, "requested alloc size must geater than 0.");
    void *ptr = nullptr;
    PRIMUS_TURBO_CHECK_HIP(hipMalloc(&ptr, size));
    PRIMUS_TURBO_CHECK_HIP(hipMemset(ptr, 0, size)); // memset the allocated buffer
    hipIpcMemHandle_t handle;
    PRIMUS_TURBO_CHECK_HIP(hipIpcGetMemHandle(&handle, ptr));
    std::vector<hipIpcMemHandle_t> handles(world_size);

    switch (group->getBackendType()) {
#define _DISPATCH_ALL_GATHER_HELPER_BY_BACKEND(src_type, dst_type)                                 \
    case ProcessGroupBackend::src_type:                                                            \
        all_gather_helper<ProcessGroupBackend::dst_type>(group.get(), &handle, handles.data(),     \
                                                         sizeof(hipIpcMemHandle_t));               \
        break

        _DISPATCH_ALL_GATHER_HELPER_BY_BACKEND(NCCL, NCCL);
        _DISPATCH_ALL_GATHER_HELPER_BY_BACKEND(GLOO, GLOO);
        _DISPATCH_ALL_GATHER_HELPER_BY_BACKEND(MPI, GLOO);

    default:
        all_gather_helper<ProcessGroupBackend::UNDEFINED>(group.get(), &handle, handles.data(),
                                                          sizeof(hipIpcMemHandle_t));

#undef _DISPATCH_ALL_GATHER_HELPER_BY_BACKEND
    }
    std::vector<torch::Tensor> tensors;
    std::vector<void *>        ptrs(world_size);
    for (int i = 0; i < world_size; ++i) {
        if (i != cur_rank) {
            PRIMUS_TURBO_CHECK_HIP(
                hipIpcOpenMemHandle(&ptrs[i], handles[i], hipIpcMemLazyEnablePeerAccess));
        } else {
            ptrs[i] = ptr;
        }
    }

    for (int i = 0; i < world_size; ++i) {
        torch::Tensor tensor;
        if (i == cur_rank) {
            tensor = at::from_blob(
                ptr, shape, [](void *ptr) { PRIMUS_TURBO_CHECK_HIP(hipFree(ptr)); }, option_gpu);
        } else {
            tensor = at::from_blob(
                ptrs[i], shape,
                [](void *ptr) { PRIMUS_TURBO_CHECK_HIP(hipIpcCloseMemHandle(ptr)); }, option_gpu);
        }
        tensors.emplace_back(tensor);
    }

    return tensors;
}

} // namespace primus_turbo::pytorch
