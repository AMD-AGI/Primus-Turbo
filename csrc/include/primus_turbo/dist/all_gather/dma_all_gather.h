// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "primus_turbo/dist/shmem.h"
#include "primus_turbo/macros.h"

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <thread>

namespace {
void reusable_barrier(std::atomic<int> &barrier, std::atomic<int> &sense, unsigned int n) {

    static_assert(std::atomic<int>::is_always_lock_free);

    // Check-in
    int count = barrier.fetch_add(1, std::memory_order_acq_rel);
    if (count + 1 == static_cast<int>(n)) {
        sense.store(1, std::memory_order_release); // Last thread sets the sense
    } else {
        while (sense.load(std::memory_order_acquire) == 0) {
            std::this_thread::yield();
        }
    }

    // Check-out
    count = barrier.fetch_sub(1, std::memory_order_acq_rel);
    if (count - 1 == 0) {
        sense.store(0, std::memory_order_release); // Last thread resets the sense
    } else {
        while (sense.load(std::memory_order_acquire) == 1) {
            std::this_thread::yield();
        }
    }
}
} // namespace

namespace primus_turbo::pytorch::dist {

constexpr std::size_t MAX_DEVICES_PER_NODE = 8;

struct AllGatherShmStruct {
    std::atomic<int> barrier;
    std::atomic<int> sense;

    hipIpcMemHandle_t   base_mem_handles[MAX_DEVICES_PER_NODE];
    size_t              base_mem_offsets[MAX_DEVICES_PER_NODE];
    hipIpcEventHandle_t exit_event_handles[MAX_DEVICES_PER_NODE];
};

uintptr_t create_allgather_shared_handle(const std::string &shm_name, size_t group_rank,
                                         size_t group_world_size);

void destroy_allgather_shared_handle(uintptr_t allgather_shared_handle, const std::string &shm_name,
                                     size_t group_rank, size_t group_world_size);

class DMAHandle final {
    static std::unordered_map<std::string, std::unique_ptr<DMAHandle>> dma_handles_;

    DMAHandle(uintptr_t allgather_shared_handle, const std::string &shm_tag, size_t group_rank,
              size_t group_size)
        : allgather_shared_handle_(allgather_shared_handle),
          rankinfo_(shm_tag, group_rank, group_size) {
        SharedMemoryInfo   *info = reinterpret_cast<SharedMemoryInfo *>(allgather_shared_handle_);
        AllGatherShmStruct *shm  = (AllGatherShmStruct *) info->addr;
        rankinfo_.initialize(shm);
    }

public:
    static DMAHandle *get_handle(const std::string &group_tag, size_t group_rank,
                                 size_t group_size) {
        // TODO (limou)
        // multiple-threads cases
        auto it = DMAHandle::dma_handles_.find(group_tag);
        if (it == DMAHandle::dma_handles_.end()) {
            uintptr_t allgather_shared_handle =
                create_allgather_shared_handle(group_tag, group_rank, group_size);
            auto new_dma_handle = std::unique_ptr<DMAHandle>(
                new DMAHandle(allgather_shared_handle, group_tag, group_rank, group_size));

            auto result = DMAHandle::dma_handles_.emplace(group_tag, std::move(new_dma_handle));
            PRIMUS_TURBO_CHECK(result.second, "emplace new_dma_handle failed");
            it = result.first;
        }
        return it->second.get();
    }
    ~DMAHandle() {
        SharedMemoryInfo   *info = reinterpret_cast<SharedMemoryInfo *>(allgather_shared_handle_);
        AllGatherShmStruct *shm  = (AllGatherShmStruct *) info->addr;
        // TODO (limou)
        // primus_turbo currently uses check macros like PRIMUS_TURBO_CHECK which throws exceptions
        // however, destructors are not allowed to throw exceptions
        // during stack unwinding, this will directly trigger std::terminate()
        rankinfo_.finalize(shm);

        if (allgather_shared_handle_ != 0) {
            destroy_allgather_shared_handle(allgather_shared_handle_, rankinfo_.shm_tag,
                                            rankinfo_.group_rank, rankinfo_.group_size);
            allgather_shared_handle_ = 0;
        }
    }
    DMAHandle(const DMAHandle &)            = delete;
    DMAHandle(DMAHandle &&)                 = delete;
    DMAHandle &operator=(const DMAHandle &) = delete;
    DMAHandle &operator=(DMAHandle &&)      = delete;

    struct RankInfo final {
        RankInfo(const std::string &shm_tag_in, size_t group_rank_in, size_t group_size_in)
            : shm_tag(shm_tag_in), group_rank(group_rank_in), group_size(group_size_in) {}

        // TODO : for to group_size
        void initialize(AllGatherShmStruct *shm) {
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

            hipEvent_t local_exit_event = nullptr;
            PRIMUS_TURBO_CHECK_HIP(hipEventCreateWithFlags(
                &local_exit_event, hipEventDisableTiming | hipEventInterprocess));

            hipIpcEventHandle_t local_exit_event_handle;
            PRIMUS_TURBO_CHECK_HIP(
                hipIpcGetEventHandle(&local_exit_event_handle, local_exit_event));
            memcpy((void *) &shm->exit_event_handles[group_rank], &local_exit_event_handle,
                   sizeof(hipIpcEventHandle_t));

            reusable_barrier(shm->barrier, shm->sense, group_size);

            for (size_t i = 0; i < MAX_DEVICES_PER_NODE; ++i) {
                if (i == group_rank) {
                    exit_events[i] = local_exit_event;
                } else {
                    PRIMUS_TURBO_CHECK_HIP(hipIpcOpenEventHandle(
                        &exit_events[i], *(hipIpcEventHandle_t *) &shm->exit_event_handles[i]));
                }
            }
            PRIMUS_TURBO_CHECK_HIP(hipEventCreateWithFlags(&entry_event, hipEventDisableTiming));
        }
        void finalize(AllGatherShmStruct *shm) {
            reusable_barrier(shm->barrier, shm->sense, group_size);
            if (entry_event != nullptr) {
                PRIMUS_TURBO_CHECK_HIP(hipEventDestroy(entry_event));
                entry_event = nullptr;
            }

            for (size_t i = 0; i < MAX_DEVICES_PER_NODE; i++) {
                if (exit_events[i] != nullptr) {
                    PRIMUS_TURBO_CHECK_HIP(hipEventDestroy(exit_events[i]));
                    exit_events[i] = nullptr;
                }
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
        hipStream_t       copy_streams[MAX_DEVICES_PER_NODE]{};
        hipEvent_t        copy_events[MAX_DEVICES_PER_NODE]{};

        // TODO : destroy
        hipEvent_t exit_events[MAX_DEVICES_PER_NODE]{};
        // TODO : destroy
        hipEvent_t entry_event{};
        void      *base_mem_ptrs[MAX_DEVICES_PER_NODE]{};
    };

    uintptr_t get_allgather_shared_handle() const { return allgather_shared_handle_; }
    RankInfo *get_rankinfo() { return &rankinfo_; }

private:
    uintptr_t allgather_shared_handle_{0};
    RankInfo  rankinfo_;
};

void run_dma_all_gather_into_tensor_nobuffer(DMAHandle *dma_handle, void *output, void *input,
                                             size_t size_bytes, hipStream_t stream);

} // namespace primus_turbo::pytorch::dist
