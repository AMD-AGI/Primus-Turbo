#pragma once

// ROCm: port of upstream csrc/utils/shared_memory.hpp. `use_fabric` has no HIP
// counterpart and is rejected, and the allocation size is remembered here because a
// HIP IPC handle carries none. Coherence is the accessor's job (see legacy/utils.cuh).

#include <hip/hip_runtime.h>

#include <unordered_map>

#include <primus_turbo/deep_ep/common/exception.cuh>

// ROCm: deep_ep::shared_memory -> primus_turbo::deep_ep::shared_memory
namespace primus_turbo::deep_ep::shared_memory {

union MemHandleInner {
    hipIpcMemHandle_t cuda_ipc_mem_handle;
};

struct MemHandle {
    MemHandleInner inner;
    size_t size;
};

class SharedMemoryAllocator {
public:
    explicit SharedMemoryAllocator(const bool& use_fabric) : use_fabric(use_fabric) {
        EP_HOST_ASSERT(not use_fabric and "HIP has no fabric memory handle");
    }

    void malloc(void** ptr, size_t size) {
        EP_HOST_ASSERT(size > 0);
        CUDA_RUNTIME_CHECK(hipMalloc(ptr, size));
        sizes[*ptr] = size;
    }

    void free(void* ptr) {
        sizes.erase(ptr);
        CUDA_RUNTIME_CHECK(hipFree(ptr));
    }

    // The size upstream reads back from the driver is remembered at malloc() instead.
    void get_mem_handle(MemHandle* mem_handle, void* ptr) const {
        const auto it = sizes.find(ptr);
        EP_HOST_ASSERT(it != sizes.end());
        mem_handle->size = it->second;
        CUDA_RUNTIME_CHECK(hipIpcGetMemHandle(&mem_handle->inner.cuda_ipc_mem_handle, ptr));
    }

    void open_mem_handle(void** ptr, MemHandle* mem_handle) const {
        CUDA_RUNTIME_CHECK(
            hipIpcOpenMemHandle(ptr, mem_handle->inner.cuda_ipc_mem_handle, hipIpcMemLazyEnablePeerAccess));
    }

    void close_mem_handle(void* ptr) const { CUDA_RUNTIME_CHECK(hipIpcCloseMemHandle(ptr)); }

private:
    bool use_fabric;
    std::unordered_map<void*, size_t> sizes;
};

}  // namespace primus_turbo::deep_ep::shared_memory
