#pragma once

// [agent modifed]: port of upstream `csrc/utils/shared_memory.hpp`. Two deviations,
// both forced by the driver surface:
//   - the `use_fabric` path (CUmemFabricHandle + the cuMem* VMM API through
//     lazy_driver.hpp) has no HIP counterpart, so only upstream's own IPC branch
//     is kept and `use_fabric` is rejected at construction.
//   - the allocation size is remembered here, because a HIP IPC handle carries no
//     size field for `get_mem_handle` to fill in from the driver.
//
// The buffer is ordinary cached device memory, as upstream. Peers do read it over
// xGMI, which CDNA's L2 does not snoop, but coherence is the *accessor's* job on
// this arch, not the allocation's: see the `coherent` family in
// kernels/deep_ep/legacy/utils.cuh. An uncached allocation cannot substitute for
// it -- MTYPE applies to the owner's page tables only, so a peer's IPC mapping
// does not inherit it and the peer's own writes would still sit in its L2.

#include <hip/hip_runtime.h>

#include <unordered_map>

#include <primus_turbo/deep_ep/common/exception.cuh>

// [agent modifed]: deep_ep::shared_memory -> primus_turbo::deep_ep::shared_memory
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
