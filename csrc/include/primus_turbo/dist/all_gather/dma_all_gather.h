#pragma once

#include "primus_turbo/dist/shmem.h"
#include "primus_turbo/macros.h"
#include <hip/hip_runtime.h>

#include <atomic>
#include <memory>

namespace primus_turbo::pytorch::dist {

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

class DMAHandle final {
    static std::unordered_map<std::string, std::unique_ptr<DMAHandle>> dma_handles_;

public:
    static DMAHandle *GetHandle(const std::string &group_tag, size_t group_rank,
                                size_t group_size) {
        // TODO (limou)
        // multiple-threads cases
        auto it = dma_handles_.find(group_tag);
        if (it == dma_handles_.end()) {
            uintptr_t new_handle_ptr = create_all_gather_handle(group_tag, group_rank, group_size);
            auto      uptr           = std::unique_ptr<DMAHandle>(
                new DMAHandle(new_handle_ptr, group_tag, group_rank, group_size));

            auto result = dma_handles_.emplace(group_tag, std::move(uptr));
            PRIMUS_TURBO_CHECK(result.second, "emplace DMAHandle failed");
            it = result.first;
        }
        return it->second.get();
    }
    ~DMAHandle() {
        if (handle_ptr_ != 0) {
            destroy_all_gather_handle(handle_ptr_, shm_tag_, group_rank_, group_size_);
            handle_ptr_ = 0;
        }
        for (size_t i = 0; i < MAX_DEVICES; i++) {
            if (copy_events_[i] != nullptr) {
                PRIMUS_TURBO_CHECK_HIP(hipEventDestroy(copy_events_[i]));
                copy_events_[i] = nullptr;
            }
            if (copy_streams_[i] != nullptr) {
                PRIMUS_TURBO_CHECK_HIP(hipStreamDestroy(copy_streams_[i]));
                copy_streams_[i] = nullptr;
            }
        }
    }
    DMAHandle(const DMAHandle &)            = delete;
    DMAHandle(DMAHandle &&)                 = delete;
    DMAHandle &operator=(const DMAHandle &) = delete;
    DMAHandle &operator=(DMAHandle &&)      = delete;

    uintptr_t    GetPtr() const { return handle_ptr_; }
    size_t       GetGroupRank() const { return group_rank_; }
    size_t       GetGroupSize() const { return group_size_; }
    hipStream_t *GetCopyStreams() { return copy_streams_; }
    hipEvent_t  *GetCopyEvents() { return copy_events_; }

private:
    DMAHandle(uintptr_t handle_ptr, const std::string &shm_tag, size_t group_rank,
              size_t group_size)
        : handle_ptr_(handle_ptr), shm_tag_(shm_tag), group_rank_(group_rank),
          group_size_(group_size) {
        for (size_t i = 0; i < MAX_DEVICES; i++) {

            hipStream_t copy_stream = nullptr;
            int         leastPriority, greatestPriority;
            PRIMUS_TURBO_CHECK_HIP(
                hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
            PRIMUS_TURBO_CHECK_HIP(
                hipStreamCreateWithPriority(&copy_stream, hipStreamNonBlocking, greatestPriority));
            copy_streams_[i] = copy_stream;

            hipEvent_t copy_event = nullptr;
            PRIMUS_TURBO_CHECK_HIP(hipEventCreateWithFlags(&copy_event, hipEventDisableTiming));
            copy_events_[i] = copy_event;
        }
    }

private:
    uintptr_t         handle_ptr_{0};
    const std::string shm_tag_;
    size_t            group_rank_{0};
    size_t            group_size_{0};

private:
    hipStream_t copy_streams_[MAX_DEVICES];
    hipEvent_t  copy_events_[MAX_DEVICES];
};

void run_dma_all_gather_into_tensor_nobuffer(DMAHandle *dma_handle, void *output, void *input,
                                             size_t size_bytes, hipStream_t stream);

} // namespace primus_turbo::pytorch::dist