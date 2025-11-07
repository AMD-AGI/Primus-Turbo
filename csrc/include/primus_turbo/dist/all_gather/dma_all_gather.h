#pragma once

#include "primus_turbo/dist/shmem.h"
#include <hip/hip_runtime.h>

#include <atomic>

namespace primus_turbo::pytorch {

constexpr std::size_t MAX_DEVICES = 8;

struct AllGatherShmStruct {
    std::atomic<int> barrier;
    std::atomic<int> sense;

    bool   is_first_run[MAX_DEVICES];
    size_t group_world_size;

    hipIpcMemHandle_t output_mem_handles[MAX_DEVICES];

    // set by each rank
    hipIpcMemHandle_t remote_base_mem_handles[MAX_DEVICES];
    size_t            remote_base_offsets[MAX_DEVICES];
    // opened by each rank, saved for future handle close
    void *remote_base_ptrs[MAX_DEVICES][MAX_DEVICES];
    // copy event
    hipEvent_t          local_exit_events[MAX_DEVICES];
    hipIpcEventHandle_t local_exit_event_handles[MAX_DEVICES];
    hipEvent_t          remote_exit_events[MAX_DEVICES][MAX_DEVICES];

    hipEvent_t entry_events[MAX_DEVICES];
    hipEvent_t exit_events[MAX_DEVICES];

    AllGatherShmStruct() = default;
};

uintptr_t create_all_gather_handle(const std::string &shm_name, size_t group_rank,
                                   size_t group_world_size);
void      wait_all_gather_handle(uintptr_t handle_ptr, size_t group_rank, size_t group_world_size);
void stream_wait_all_gather_handle(uintptr_t handle_ptr, uintptr_t stream_ptr, size_t group_rank,
                                   size_t group_world_size);
void destroy_all_gather_handle(uintptr_t handle_ptr, const std::string &shm_name, size_t group_rank,
                               size_t group_world_size);
void run_dma_all_gather_into_tensor(dmaHandle_t handle, void *output, void *input,
                                    size_t size_bytes, size_t group_rank, size_t group_world_size,
                                    hipStream_t stream);
void run_dma_all_gather_into_tensor_nobuffer(dmaHandle_t handle, void *output, void *input,
                                             size_t size_bytes, size_t group_rank,
                                             size_t group_world_size, hipStream_t stream);

} // namespace primus_turbo::pytorch