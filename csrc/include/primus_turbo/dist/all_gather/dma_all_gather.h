// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "primus_turbo/macros.h"

#include <hip/hip_runtime.h>

#include <atomic>
#include <memory>

namespace primus_turbo::pytorch::dist {

constexpr std::size_t MAX_DEVICES_PER_NODE = 8;

struct AllGatherShmStruct {
    std::atomic<int> barrier;
    std::atomic<int> sense;

    bool is_first_run[MAX_DEVICES_PER_NODE]; // 可以去掉，统一初始化

    // set by each rank
    hipIpcMemHandle_t remote_base_mem_handles[MAX_DEVICES_PER_NODE];
    size_t            remote_base_offsets[MAX_DEVICES_PER_NODE];
    // opened by each rank, saved for future handle close
    void *remote_base_ptrs[MAX_DEVICES_PER_NODE][MAX_DEVICES_PER_NODE]; // 不需要shm，改成1维
    // copy event
    hipEvent_t local_exit_events[MAX_DEVICES_PER_NODE]; // 不需要shm，改成scalar
    hipIpcEventHandle_t
        local_exit_event_handles[MAX_DEVICES_PER_NODE]; // 改成exit_events, 不区分local/remote
    hipEvent_t remote_exit_events[MAX_DEVICES_PER_NODE]
                                 [MAX_DEVICES_PER_NODE]; // 不需要shm，改成1维?

    hipEvent_t entry_events[MAX_DEVICES_PER_NODE]; // 不需要shm 改成scalar
    hipEvent_t exit_events[MAX_DEVICES_PER_NODE];  // 不需要shm

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

    DMAHandle(uintptr_t handle_ptr, const std::string &shm_tag, size_t group_rank,
              size_t group_size)
        : handle_ptr_(handle_ptr), rankinfo_(shm_tag, group_rank, group_size) {
        rankinfo_.initialize();
    }

public:
    static DMAHandle *get_handle(const std::string &group_tag, size_t group_rank,
                                 size_t group_size) {
        // TODO (limou)
        // multiple-threads cases
        auto it = DMAHandle::dma_handles_.find(group_tag);
        if (it == DMAHandle::dma_handles_.end()) {
            uintptr_t new_handle_ptr = create_all_gather_handle(group_tag, group_rank, group_size);
            auto      new_dma_handle = std::unique_ptr<DMAHandle>(
                new DMAHandle(new_handle_ptr, group_tag, group_rank, group_size));

            auto result = DMAHandle::dma_handles_.emplace(group_tag, std::move(new_dma_handle));
            PRIMUS_TURBO_CHECK(result.second, "emplace new_dma_handle failed");
            it = result.first;
        }
        return it->second.get();
    }
    ~DMAHandle() {
        if (handle_ptr_ != 0) {
            destroy_all_gather_handle(handle_ptr_, rankinfo_.shm_tag, rankinfo_.group_rank,
                                      rankinfo_.group_size);
            handle_ptr_ = 0;
        }

        // TODO (limou)
        // primus_turbo currently uses check macros like PRIMUS_TURBO_CHECK which throws exceptions
        // however, destructors are not allowed to throw exceptions
        // during stack unwinding, this will directly trigger std::terminate()
        rankinfo_.finalize();
    }
    DMAHandle(const DMAHandle &)            = delete;
    DMAHandle(DMAHandle &&)                 = delete;
    DMAHandle &operator=(const DMAHandle &) = delete;
    DMAHandle &operator=(DMAHandle &&)      = delete;

    struct RankInfo final {
        RankInfo(const std::string &shm_tag_in, size_t group_rank_in, size_t group_size_in)
            : shm_tag(shm_tag_in), group_rank(group_rank_in), group_size(group_size_in) {}
        void initialize() {
            for (size_t i = 0; i < MAX_DEVICES_PER_NODE; i++) {

                hipStream_t copy_stream = nullptr;
                int         leastPriority, greatestPriority;
                PRIMUS_TURBO_CHECK_HIP(
                    hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
                PRIMUS_TURBO_CHECK_HIP(hipStreamCreateWithPriority(
                    &copy_stream, hipStreamNonBlocking, greatestPriority));
                copy_streams[i] = copy_stream;

                hipEvent_t copy_event = nullptr;
                PRIMUS_TURBO_CHECK_HIP(hipEventCreateWithFlags(&copy_event, hipEventDisableTiming));
                copy_events[i] = copy_event;
            }
        }
        void finalize() {
            for (size_t i = 0; i < MAX_DEVICES_PER_NODE; i++) {
                if (copy_events[i] != nullptr) {
                    PRIMUS_TURBO_CHECK_HIP(hipEventDestroy(copy_events[i]));
                    copy_events[i] = nullptr;
                }
                if (copy_streams[i] != nullptr) {
                    PRIMUS_TURBO_CHECK_HIP(hipStreamDestroy(copy_streams[i]));
                    copy_streams[i] = nullptr;
                }
            }
        }
        const std::string shm_tag;
        size_t            group_rank;
        size_t            group_size;
        hipStream_t       copy_streams[MAX_DEVICES_PER_NODE];
        hipEvent_t        copy_events[MAX_DEVICES_PER_NODE];
    };

    uintptr_t get_ptr() const { return handle_ptr_; }
    RankInfo *get_rankinfo() { return &rankinfo_; }

private:
    uintptr_t handle_ptr_{0};
    RankInfo  rankinfo_;
};

void run_dma_all_gather_into_tensor_nobuffer(DMAHandle *dma_handle, void *output, void *input,
                                             size_t size_bytes, hipStream_t stream);

} // namespace primus_turbo::pytorch::dist
