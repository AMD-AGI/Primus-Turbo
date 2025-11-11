// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/dist/all_gather/dma_all_gather.h"
#include "primus_turbo/macros.h"

#include <fcntl.h>

#include <thread>
#include <vector>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

namespace {

class FileGuard final {
public:
    static void WaitFile(const std::string &path) {
        while (!std::filesystem::exists(path)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    static void CreateFile(const std::string &path) {
        int fd = open(path.c_str(), O_CREAT | O_WRONLY, 0666);
        PRIMUS_TURBO_CHECK(fd >= 0);
        close(fd);
    }

public:
    explicit FileGuard(const std::string &path) : file_path_(path) {
        FileGuard::CreateFile(file_path_);
    }
    ~FileGuard() { std::filesystem::remove(file_path_); }

private:
    const std::string &file_path_;
};

} // namespace

namespace primus_turbo::pytorch::dist {

static void reusable_barrier(volatile std::atomic<int> &barrier, volatile std::atomic<int> &sense,
                             unsigned int n) {

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

uintptr_t create_all_gather_handle(const std::string &shm_name, size_t group_rank,
                                   size_t group_world_size) {
    SharedMemoryInfo *info = new SharedMemoryInfo();

    const std::string barrier_path = "/tmp/barrier_" + shm_name;

    if (group_rank == 0) {
        if (sharedMemoryCreate(shm_name.c_str(), sizeof(AllGatherShmStruct), info) != 0) {
            throw std::runtime_error("Failed to create allgather handle.");
        }
        volatile AllGatherShmStruct *shm = nullptr;
        shm                              = (volatile AllGatherShmStruct *) info->addr;

        // init shm
        memset((void *) shm, 0, sizeof(*shm));
        new (const_cast<std::atomic<int> *>(&shm->barrier)) std::atomic<int>(0);
        new (const_cast<std::atomic<int> *>(&shm->sense)) std::atomic<int>(0);

        for (size_t i = 0; i < MAX_DEVICES; ++i) {
            shm->is_first_run[i] = true;
            shm->entry_events[i] = nullptr;
            shm->exit_events[i]  = nullptr;

            shm->remote_base_offsets[i] = 0;
            shm->local_exit_events[i]   = nullptr;

            for (size_t j = 0; j < MAX_DEVICES; ++j) {
                shm->remote_base_ptrs[i][j]   = nullptr;
                shm->remote_exit_events[i][j] = nullptr;
            }
        }

        {
            FileGuard file_guard(barrier_path);
            reusable_barrier(shm->barrier, shm->sense, group_world_size);
        }
    } else {
        int ret = 1;
        do {
            ret = sharedMemoryOpen(shm_name.c_str(), sizeof(AllGatherShmStruct), info);
        } while (ret != 0);

        volatile AllGatherShmStruct *shm = nullptr;
        shm                              = (volatile AllGatherShmStruct *) info->addr;

        FileGuard::WaitFile(barrier_path);
        reusable_barrier(shm->barrier, shm->sense, group_world_size);
    }
    return reinterpret_cast<uintptr_t>(info);
}

void destroy_all_gather_handle(uintptr_t handle_ptr, const std::string &shm_name, size_t group_rank,
                               size_t group_world_size) {
    SharedMemoryInfo *info = reinterpret_cast<SharedMemoryInfo *>(handle_ptr);

    AllGatherShmStruct *shm = (AllGatherShmStruct *) info->addr;

    if (shm->entry_events[group_rank] != nullptr) {
        PRIMUS_TURBO_CHECK_HIP(hipEventDestroy(shm->entry_events[group_rank]));
        shm->entry_events[group_rank] = nullptr;
    }

    if (shm->exit_events[group_rank] != nullptr) {
        PRIMUS_TURBO_CHECK_HIP(hipEventDestroy(shm->exit_events[group_rank]));
        shm->exit_events[group_rank] = nullptr;
    }

    if (group_rank == 0) {
        sharedMemoryClose(info);
        shareMemoryDelete(shm_name.c_str());
    }

    delete info;
}

void run_dma_all_gather_into_tensor_nobuffer(DMAHandle *dma_handle, void *output, void *input,
                                             size_t size_bytes, hipStream_t stream) {

    SharedMemoryInfo *info             = reinterpret_cast<SharedMemoryInfo *>(dma_handle->GetPtr());
    size_t            group_rank       = dma_handle->GetGroupRank();
    size_t            group_world_size = dma_handle->GetGroupSize();
    hipStream_t      *copy_streams     = dma_handle->GetCopyStreams();
    hipEvent_t       *copy_events      = dma_handle->GetCopyEvents();

    volatile AllGatherShmStruct *shm = nullptr;
    shm                              = (volatile AllGatherShmStruct *) info->addr;

    if (shm->is_first_run[group_rank]) {
        hipEvent_t entry_event = nullptr;
        PRIMUS_TURBO_CHECK_HIP(hipEventCreateWithFlags(&entry_event, hipEventDisableTiming));
        shm->entry_events[group_rank] = entry_event;

        hipEvent_t local_exit_event = nullptr;
        PRIMUS_TURBO_CHECK_HIP(hipEventCreateWithFlags(
            &local_exit_event, hipEventDisableTiming | hipEventInterprocess));
        shm->local_exit_events[group_rank] = local_exit_event;

        hipIpcEventHandle_t local_exit_event_handle;
        PRIMUS_TURBO_CHECK_HIP(hipIpcGetEventHandle(&local_exit_event_handle, local_exit_event));
        memcpy((void *) &shm->local_exit_event_handles[group_rank], &local_exit_event_handle,
               sizeof(hipIpcEventHandle_t));

        // wait for all event handles ready
        reusable_barrier(shm->barrier, shm->sense, group_world_size);

        // remote events
        for (size_t i = 0; i < MAX_DEVICES; ++i) {
            if (i == group_rank) {
                continue;
            }

            hipEvent_t remote_exit_event;
            PRIMUS_TURBO_CHECK_HIP(hipIpcOpenEventHandle(
                &remote_exit_event, *(hipIpcEventHandle_t *) &shm->local_exit_event_handles[i]));
            shm->remote_exit_events[group_rank][i] = remote_exit_event;
        }

        shm->group_world_size         = group_world_size;
        shm->is_first_run[group_rank] = false;
    } else {
        if (shm->group_world_size != group_world_size) {
            throw std::runtime_error("Must keep group world size the same.");
        }
    }

    // wait for all mem handle ready
    reusable_barrier(shm->barrier, shm->sense, group_world_size);

    // get output handle, and copy it to shm
    void  *output_base_ptr = nullptr;
    size_t alloc_size      = 0;
    PRIMUS_TURBO_CHECK_HIP(hipMemGetAddressRange(&output_base_ptr, &alloc_size, output));

    hipIpcMemHandle_t base_mem_handle;
    PRIMUS_TURBO_CHECK_HIP(hipIpcGetMemHandle(&base_mem_handle, output_base_ptr));
    memcpy((void *) &shm->remote_base_mem_handles[group_rank], &base_mem_handle,
           sizeof(hipIpcMemHandle_t));
    shm->remote_base_offsets[group_rank] =
        static_cast<char *>(output) - static_cast<char *>(output_base_ptr);

    // wait for all mem handle ready
    reusable_barrier(shm->barrier, shm->sense, group_world_size);

    for (size_t i = 0; i < group_world_size; ++i) {
        if (i != group_rank) {
            void *remote_base_ptr = nullptr;
            PRIMUS_TURBO_CHECK_HIP(hipIpcOpenMemHandle(
                &remote_base_ptr, *(hipIpcMemHandle_t *) &shm->remote_base_mem_handles[i],
                hipIpcMemLazyEnablePeerAccess));
            // save for future close
            shm->remote_base_ptrs[group_rank][i] = remote_base_ptr;
        } else {
            shm->remote_base_ptrs[group_rank][i] = output_base_ptr;
        }
    }

    // record op stream
    PRIMUS_TURBO_CHECK_HIP(hipEventRecord(shm->entry_events[group_rank], stream));

    for (size_t i = 0; i < group_world_size; ++i) {
        hipStream_t copy_stream = copy_streams[i];
        PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(copy_stream, shm->entry_events[group_rank], 0));
    }

    for (size_t i = 0; i < group_world_size; ++i) {
        size_t      i_split     = group_rank;
        size_t      remote_rank = (group_rank + i) % group_world_size;
        hipStream_t copy_stream = copy_streams[remote_rank];

        void *src        = input;
        void *remote_ptr = static_cast<void *>(
            static_cast<char *>(shm->remote_base_ptrs[group_rank][remote_rank]) +
            shm->remote_base_offsets[remote_rank]);
        void *dst = static_cast<void *>(static_cast<char *>(remote_ptr) + size_bytes * i_split);

        PRIMUS_TURBO_CHECK_HIP(
            hipMemcpyAsync(dst, src, size_bytes, hipMemcpyDeviceToDeviceNoCU, copy_stream));
    }

    for (size_t i = 0; i < group_world_size; ++i) {
        size_t      remote_rank = (group_rank + i) % group_world_size;
        hipStream_t copy_stream = copy_streams[remote_rank];
        hipEvent_t  copy_event  = copy_events[remote_rank];
        PRIMUS_TURBO_CHECK_HIP(hipEventRecord(copy_event, copy_stream));
        PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(stream, copy_event, 0));

        hipEvent_t local_exit_event = shm->local_exit_events[group_rank];
        hipEventRecord(local_exit_event, stream);

        // wait all ranks finish event recording
        reusable_barrier(shm->barrier, shm->sense, group_world_size);

        for (size_t i = 0; i < MAX_DEVICES; ++i) {
            if (i == group_rank) {
                continue;
            }

            hipEvent_t remote_exit_event;
            remote_exit_event = shm->remote_exit_events[group_rank][i];
            PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(stream, remote_exit_event, 0));
        }
    }
}

} // namespace primus_turbo::pytorch::dist
