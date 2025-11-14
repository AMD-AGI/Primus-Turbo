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

void close_handle_callback(hipStream_t stream, hipError_t status, void *userData) {
    void *ptr = userData;
    printf("in close handle callback\n");
    fflush(stdout);
    if (ptr != nullptr) {
        PRIMUS_TURBO_CHECK_HIP(hipIpcCloseMemHandle(ptr));
    }
}

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
        AllGatherShmStruct *shm = (AllGatherShmStruct *) info->addr;

        // init shm
        memset((void *) shm, 0, sizeof(*shm));
        new (const_cast<std::atomic<int> *>(&shm->barrier)) std::atomic<int>(0);
        new (const_cast<std::atomic<int> *>(&shm->sense)) std::atomic<int>(0);

        for (size_t i = 0; i < MAX_DEVICES_PER_NODE; ++i) {
            shm->base_mem_offsets[i] = 0;
        }

        {
            FileGuard file_guard(barrier_path);
            reusable_barrier(shm->barrier, shm->sense, group_size);
        }
    } else {
        int ret = 1;
        do {
            ret = shared_memory_open(shm_name.c_str(), sizeof(AllGatherShmStruct), info);
        } while (ret != 0);

        AllGatherShmStruct *shm = (AllGatherShmStruct *) info->addr;

        FileGuard::wait_file(barrier_path);
        reusable_barrier(shm->barrier, shm->sense, group_size);
    }
    return reinterpret_cast<uintptr_t>(info);
}

void destroy_allgather_shared_handle(uintptr_t allgather_shared_handle, const std::string &shm_name,
                                     size_t group_rank, size_t group_world_size) {
    SharedMemoryInfo *info = reinterpret_cast<SharedMemoryInfo *>(allgather_shared_handle);

    if (group_rank == 0) {
        shared_memory_close(info);
        shared_memory_delete(shm_name.c_str());
    }

    delete info;
}

void run_dma_all_gather_into_tensor_nobuffer(DMAHandle *dma_handle, void *output, void *input,
                                             size_t size_bytes, hipStream_t stream) {

    SharedMemoryInfo *info =
        reinterpret_cast<SharedMemoryInfo *>(dma_handle->get_allgather_shared_handle());
    DMAHandle::RankInfo *rankinfo         = dma_handle->get_rankinfo();
    size_t               group_rank       = rankinfo->group_rank;
    size_t               group_world_size = rankinfo->group_size;
    hipStream_t         *copy_streams     = rankinfo->copy_streams;
    hipEvent_t          *copy_events      = rankinfo->copy_events;
    hipEvent_t          *exit_events      = rankinfo->exit_events;
    void               **base_mem_ptrs    = rankinfo->base_mem_ptrs;

    AllGatherShmStruct *shm = (AllGatherShmStruct *) info->addr;

    // wait for all mem handle ready
    reusable_barrier(shm->barrier, shm->sense, group_world_size);

    // get output handle, and copy it to shm
    void  *output_base_ptr = nullptr;
    size_t alloc_size      = 0;
    PRIMUS_TURBO_CHECK_HIP(hipMemGetAddressRange(&output_base_ptr, &alloc_size, output));

    hipIpcMemHandle_t base_mem_handle;
    PRIMUS_TURBO_CHECK_HIP(hipIpcGetMemHandle(&base_mem_handle, output_base_ptr));
    memcpy((void *) &shm->base_mem_handles[group_rank], &base_mem_handle,
           sizeof(hipIpcMemHandle_t));
    shm->base_mem_offsets[group_rank] =
        static_cast<char *>(output) - static_cast<char *>(output_base_ptr);

    // wait for all mem handle ready
    reusable_barrier(shm->barrier, shm->sense, group_world_size);

    for (size_t i = 0; i < group_world_size; ++i) {
        if (i != group_rank) {
            void *remote_base_ptr = nullptr;
            // TODO : hipIpcCloseMemHandle
            PRIMUS_TURBO_CHECK_HIP(hipIpcOpenMemHandle(
                &remote_base_ptr, *(hipIpcMemHandle_t *) &shm->base_mem_handles[i],
                hipIpcMemLazyEnablePeerAccess));

            base_mem_ptrs[i] = remote_base_ptr;
        } else {
            base_mem_ptrs[i] = output_base_ptr;
        }
    }

    // record op stream
    PRIMUS_TURBO_CHECK_HIP(hipEventRecord(rankinfo->entry_event, stream));

    for (size_t i = 0; i < group_world_size; ++i) {
        hipStream_t copy_stream = copy_streams[i];
        PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(copy_stream, rankinfo->entry_event, 0));
    }

    for (size_t i = 0; i < group_world_size; ++i) {
        size_t      i_split     = group_rank;
        size_t      remote_rank = (group_rank + i) % group_world_size;
        hipStream_t copy_stream = copy_streams[remote_rank];

        void *src        = input;
        void *remote_ptr = static_cast<void *>(static_cast<char *>(base_mem_ptrs[remote_rank]) +
                                               shm->base_mem_offsets[remote_rank]);
        void *dst = static_cast<void *>(static_cast<char *>(remote_ptr) + size_bytes * i_split);

        PRIMUS_TURBO_CHECK_HIP(
            hipMemcpyAsync(dst, src, size_bytes, hipMemcpyDeviceToDeviceNoCU, copy_stream));
        // if (remote_rank != group_rank) {
        //     PRIMUS_TURBO_CHECK_HIP(hipStreamAddCallback(copy_stream, close_handle_callback,
        //                                                 base_mem_ptrs[remote_rank], 0));
        // }
    }

    for (size_t i = 0; i < group_world_size; ++i) {
        size_t      remote_rank = (group_rank + i) % group_world_size;
        hipStream_t copy_stream = copy_streams[remote_rank];
        hipEvent_t  copy_event  = copy_events[remote_rank];
        PRIMUS_TURBO_CHECK_HIP(hipEventRecord(copy_event, copy_stream));
        PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(stream, copy_event, 0));

        hipEventRecord(exit_events[group_rank], stream);

        // wait all ranks finish event recording
        reusable_barrier(shm->barrier, shm->sense, group_world_size);

        for (size_t i = 0; i < MAX_DEVICES_PER_NODE; ++i) {
            if (i != group_rank) {
                PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(stream, exit_events[i], 0));
            }
        }
    }
    for (size_t i = 0; i < group_world_size; i++) {
        if (i != group_rank) {
            PRIMUS_TURBO_CHECK_HIP(
                hipStreamAddCallback(stream, close_handle_callback, base_mem_ptrs[i], 0));
        }
    }
}

} // namespace primus_turbo::pytorch::dist

/*
只需要handle在shm中
copy_stream wait entry_event

*/
