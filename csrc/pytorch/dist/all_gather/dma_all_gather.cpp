// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/dist/all_gather/dma_all_gather.h"
#include "primus_turbo/macros.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

namespace {

class FileGuard final {
public:
    static void wait_file(const std::string &path) {
        while (!std::filesystem::exists(path)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    static void create_file(const std::string &path) {
        std::ofstream file(path.c_str(), std::ios::app);
    }

public:
    explicit FileGuard(const std::string &path) : file_path_(path) {
        FileGuard::create_file(file_path_);
    }
    ~FileGuard() { std::filesystem::remove(file_path_); }

private:
    const std::string &file_path_;
};

#if SYNC_WITH_IPC_EVENT != 1
void sync_callback(hipStream_t stream, hipError_t status, void *userData) {
    using primus_turbo::pytorch::dist::AllGatherShmStruct;
    using primus_turbo::pytorch::dist::DMAHandle;
    using primus_turbo::pytorch::dist::SharedMemoryInfo;

    DMAHandle *dma_handle = static_cast<DMAHandle *>(userData);

    SharedMemoryInfo *info =
        reinterpret_cast<SharedMemoryInfo *>(dma_handle->get_allgather_shared_handle());
    AllGatherShmStruct *shm = static_cast<AllGatherShmStruct *>(info->addr);

    DMAHandle::RankInfo *rankinfo = dma_handle->get_rankinfo();
    reusable_barrier(shm->barrier, shm->sense, rankinfo->group_size);
}
#endif

} // namespace

namespace primus_turbo::pytorch::dist {

uintptr_t create_allgather_shared_handle(const std::string &shm_name, size_t group_rank,
                                         size_t group_size) {
    SharedMemoryInfo *info = new SharedMemoryInfo();

    const std::string barrier_path = "/tmp/barrier_" + shm_name;

    if (group_rank == 0) {
        PRIMUS_TURBO_CHECK(
            shared_memory_create(shm_name.c_str(), sizeof(AllGatherShmStruct), info) == 0,
            "failed to create allgather handle");
        AllGatherShmStruct *shm = static_cast<AllGatherShmStruct *>(info->addr);

        // init shm
        memset(shm, 0, sizeof(*shm));
        new (const_cast<std::atomic<int> *>(&shm->barrier)) std::atomic<int>(0);
        new (const_cast<std::atomic<int> *>(&shm->sense)) std::atomic<int>(0);

        // for (size_t i = 0; i < group_size; ++i) {
        //     shm->buffer_mem_handles[i] = 0;
        // }

        {
            const FileGuard file_guard(barrier_path);
            reusable_barrier(shm->barrier, shm->sense, group_size);
        }
    } else {
        int ret = 1;
        do {
            ret = shared_memory_open(shm_name.c_str(), sizeof(AllGatherShmStruct), info);
        } while (ret != 0);

        AllGatherShmStruct *shm = static_cast<AllGatherShmStruct *>(info->addr);

        FileGuard::wait_file(barrier_path);
        reusable_barrier(shm->barrier, shm->sense, group_size);
    }
    return reinterpret_cast<uintptr_t>(info);
}

void destroy_allgather_shared_handle(uintptr_t allgather_shared_handle, const std::string &shm_name,
                                     size_t group_rank, size_t group_size) {
    SharedMemoryInfo *info = reinterpret_cast<SharedMemoryInfo *>(allgather_shared_handle);

    if (group_rank == 0) {
        shared_memory_close(info);
        shared_memory_delete(shm_name.c_str());
    }

    delete info;
}

void run_dma_all_gather_into_tensor(DMAHandle *dma_handle, void *output, void *input,
                                    size_t size_bytes, hipStream_t stream) {

    SharedMemoryInfo *info =
        reinterpret_cast<SharedMemoryInfo *>(dma_handle->get_allgather_shared_handle());
    AllGatherShmStruct *shm = static_cast<AllGatherShmStruct *>(info->addr);

    DMAHandle::RankInfo *rankinfo        = dma_handle->get_rankinfo();
    size_t               group_rank      = rankinfo->group_rank;
    size_t               group_size      = rankinfo->group_size;
    hipStream_t         *copy_streams    = rankinfo->copy_streams;
    hipEvent_t          *copy_events     = rankinfo->copy_events;
    void               **buffer_mem_ptrs = rankinfo->buffer_mem_ptrs;
#if SYNC_WITH_IPC_EVENT == 1
    hipEvent_t *sync_events = rankinfo->sync_events;
#endif
    if (rankinfo->source_buffer_size < size_bytes) {
        if (rankinfo->source_buffer != nullptr) {
            // TODO : remove this ?
            PRIMUS_TURBO_CHECK_HIP(hipDeviceSynchronize());
            // TODO : remove this ?
            reusable_barrier(shm->barrier, shm->sense, group_size);

            for (size_t i = 0; i < group_size; i++) {
                if (i != group_rank && buffer_mem_ptrs[i] != nullptr) {
                    PRIMUS_TURBO_CHECK_HIP(hipIpcCloseMemHandle(buffer_mem_ptrs[i]));
                }
                buffer_mem_ptrs[i] = nullptr;
            }
            PRIMUS_TURBO_CHECK_HIP(hipFree(rankinfo->source_buffer));
            rankinfo->source_buffer      = nullptr;
            rankinfo->source_buffer_size = 0;
        }
        PRIMUS_TURBO_CHECK_HIP(hipMalloc(&rankinfo->source_buffer, size_bytes));
        rankinfo->source_buffer_size = size_bytes;

        hipIpcMemHandle_t mem_handle;
        PRIMUS_TURBO_CHECK_HIP(hipIpcGetMemHandle(&mem_handle, rankinfo->source_buffer));
        memcpy(&shm->buffer_mem_handles[group_rank], &mem_handle, sizeof(hipIpcMemHandle_t));

        reusable_barrier(shm->barrier, shm->sense, group_size);

        for (size_t i = 0; i < group_size; i++) {
            if (i != group_rank) {
                PRIMUS_TURBO_CHECK_HIP(hipIpcOpenMemHandle(&buffer_mem_ptrs[i],
                                                           shm->buffer_mem_handles[i],
                                                           hipIpcMemLazyEnablePeerAccess));
            } else {
                buffer_mem_ptrs[i] = rankinfo->source_buffer;
            }
        }
    }

    PRIMUS_TURBO_CHECK_HIP(hipEventRecord(rankinfo->entry_event, stream));
    PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(copy_streams[group_rank], rankinfo->entry_event, 0));

    PRIMUS_TURBO_CHECK_HIP(hipMemcpyAsync(buffer_mem_ptrs[group_rank], input, size_bytes,
                                          hipMemcpyDeviceToDeviceNoCU, copy_streams[group_rank]));

#if SYNC_WITH_IPC_EVENT == 1
    PRIMUS_TURBO_CHECK_HIP(hipEventRecord(sync_events[group_rank], copy_streams[group_rank]));
    reusable_barrier(shm->barrier, shm->sense, group_size);

    for (size_t i = 0; i < group_size; i++) {
        if (i != group_rank) {
            PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(copy_streams[i], sync_events[i], 0));
        }
    }
#else
    PRIMUS_TURBO_CHECK_HIP(
        hipStreamAddCallback(copy_streams[group_rank], sync_callback, dma_handle, 0));
    PRIMUS_TURBO_CHECK_HIP(hipEventRecord(rankinfo->sync_event, copy_streams[group_rank]));
    for (size_t i = 0; i < group_size; i++) {
        if (i != group_rank) {
            PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(copy_streams[i], rankinfo->sync_event, 0));
        }
    }
#endif

    // TODO : copy within GPU can start early using a separate stream
    for (size_t i = 0; i < group_size; i++) {
        void *src = buffer_mem_ptrs[i];
        void *dst = static_cast<uint8_t *>(output) + size_bytes * i;

        PRIMUS_TURBO_CHECK_HIP(
            hipMemcpyAsync(dst, src, size_bytes, hipMemcpyDeviceToDeviceNoCU, copy_streams[i]));
        PRIMUS_TURBO_CHECK_HIP(hipEventRecord(copy_events[i], copy_streams[i]));
    }
}

void run_dma_stream_wait(DMAHandle *dma_handle, hipStream_t stream) {
    SharedMemoryInfo *info =
        reinterpret_cast<SharedMemoryInfo *>(dma_handle->get_allgather_shared_handle());
    AllGatherShmStruct  *shm         = static_cast<AllGatherShmStruct *>(info->addr);
    DMAHandle::RankInfo *rankinfo    = dma_handle->get_rankinfo();
    size_t               group_rank  = rankinfo->group_rank;
    size_t               group_size  = rankinfo->group_size;
    hipEvent_t          *copy_events = rankinfo->copy_events;

#if SYNC_WITH_IPC_EVENT == 1
    hipEvent_t *exit_events = rankinfo->exit_events;
    for (size_t i = 0; i < group_size; i++) {
        PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(stream, copy_events[i], 0));
    }
    PRIMUS_TURBO_CHECK_HIP(hipEventRecord(exit_events[group_rank], stream));
    reusable_barrier(shm->barrier, shm->sense, group_size);
    for (size_t i = 0; i < group_size; i++) {
        if (i != group_rank) {
            PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(stream, exit_events[i], 0));
        }
    }
#else
    for (size_t i = 0; i < group_size; i++) {
        PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(stream, copy_events[i], 0));
    }
    // PRIMUS_TURBO_CHECK_HIP(hipStreamAddCallback(stream, sync_callback, dma_handle, 0));
    // PRIMUS_TURBO_CHECK_HIP(hipEventRecord(rankinfo->exit_event, stream));
    // PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(stream, rankinfo->exit_event, 0));
#endif
}

} // namespace primus_turbo::pytorch::dist
