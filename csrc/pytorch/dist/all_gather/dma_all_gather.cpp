#include "primus_turbo/dist/all_gather/dma_all_gather.h"
#include "primus_turbo/macros.h"

#include <fcntl.h>
#include <iostream>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <chrono>
#include <iostream>
#include <thread>

namespace primus_turbo::pytorch {

static hipStream_t copy_streams_g[MAX_DEVICES];
static hipEvent_t  copy_events_g[MAX_DEVICES];
static void       *recv_buffer     = nullptr;
static size_t      max_buffer_size = 0;
static void       *remote_ptrs[MAX_DEVICES][MAX_DEVICES];

static size_t group_rank_g = 0;

void barrier(std::atomic<int> &barrier, int rank, int world_size) {
    barrier.fetch_add(1, std::memory_order_acq_rel);

    while (barrier.load(std::memory_order_acquire) < world_size) {
        // 10 us
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

static void reusable_barrier(volatile std::atomic<int> &barrier, volatile std::atomic<int> &sense,
                             unsigned int n) {
    // Check-in
    int count = barrier.fetch_add(1, std::memory_order_acq_rel);
    if (count + 1 == static_cast<int>(n)) {
        sense.store(1, std::memory_order_release); // Last thread sets the sense
    } else {
        while (sense.load(std::memory_order_acquire) == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }

    // Check-out
    count = barrier.fetch_sub(1, std::memory_order_acq_rel);
    if (count - 1 == 0) {
        sense.store(0, std::memory_order_release); // Last thread resets the sense
    } else {
        while (sense.load(std::memory_order_acquire) == 1) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }
}

bool file_exists(const char *path) {
    struct stat buffer;
    return (stat(path, &buffer) == 0);
}

void create_barrier_file(const char *path) {
    int fd = open(path, O_CREAT | O_WRONLY, 0666);
    if (fd < 0) {
        perror("open");
        throw std::runtime_error("Failed to create barrier file");
    }
    close(fd);
}

void wait_for_barrier_file(const char *path) {
    while (!file_exists(path)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void remove_barrier_file(const char *path) {
    if (unlink(path) != 0) {
        perror("unlink");
        throw std::runtime_error("Failed to remove barrier file");
    }
}

uintptr_t create_all_gather_handle(const std::string &shm_name, size_t group_rank,
                                   size_t group_world_size) {
    SharedMemoryInfo *info = new SharedMemoryInfo();

    std::string barrier_path = "/tmp/barrier_" + shm_name;

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
            copy_streams_g[i]    = nullptr;

            // shm->output_mem_handles[i] = nullptr;
            // shm->remote_base_mem_handles[i] = nullptr;
            shm->remote_base_offsets[i] = 0;
            shm->local_exit_events[i]   = nullptr;

            for (size_t j = 0; j < MAX_DEVICES; ++j) {
                shm->remote_base_ptrs[i][j]   = nullptr;
                shm->remote_exit_events[i][j] = nullptr;
            }
        }

        // barrier by file
        create_barrier_file(barrier_path.c_str());
        // atomic barrier
        reusable_barrier(shm->barrier, shm->sense, group_world_size);
        // clear barrier file
        remove_barrier_file(barrier_path.c_str());
    } else {
        int ret = 1;
        do {
            ret = sharedMemoryOpen(shm_name.c_str(), sizeof(AllGatherShmStruct), info);
        } while (ret != 0);

        volatile AllGatherShmStruct *shm = nullptr;
        shm                              = (volatile AllGatherShmStruct *) info->addr;

        wait_for_barrier_file(barrier_path.c_str());

        reusable_barrier(shm->barrier, shm->sense, group_world_size);
    }
    return reinterpret_cast<uintptr_t>(info);
}

void wait_all_gather_handle(uintptr_t handle_ptr, size_t group_rank, size_t group_world_size) {
    SharedMemoryInfo   *info = reinterpret_cast<SharedMemoryInfo *>(handle_ptr);
    AllGatherShmStruct *shm  = (AllGatherShmStruct *) info->addr;

    PRIMUS_TURBO_CHECK_HIP(hipEventSynchronize(shm->exit_events[group_rank]));

    // make sure all processes synced the stream
    // reusable_barrier(shm->barrier, shm->sense, group_world_size);
}

void stream_wait_all_gather_handle(uintptr_t handle_ptr, uintptr_t stream_ptr, size_t group_rank,
                                   size_t group_world_size) {
    SharedMemoryInfo   *info   = reinterpret_cast<SharedMemoryInfo *>(handle_ptr);
    AllGatherShmStruct *shm    = (AllGatherShmStruct *) info->addr;
    hipStream_t         stream = reinterpret_cast<hipStream_t>(stream_ptr);

    if (shm->exit_events[group_rank] != nullptr) {
        PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(stream, shm->exit_events[group_rank], 0));
    }

    // make sure all processes synced the stream
    // reusable_barrier(shm->barrier, shm->sense, group_world_size);
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

    for (size_t i = 0; i < MAX_DEVICES; ++i) {
        if (remote_ptrs[group_rank][i] != nullptr) {
            if (i != group_rank) {
                PRIMUS_TURBO_CHECK_HIP(hipIpcCloseMemHandle(remote_ptrs[group_rank][i]));
            }
            remote_ptrs[group_rank][i] = nullptr;
        }

        if (copy_events_g[i] != nullptr) {
            PRIMUS_TURBO_CHECK_HIP(hipEventDestroy(copy_events_g[i]));
            copy_events_g[i] = nullptr;
        }

        if (copy_streams_g[i] != nullptr) {
            PRIMUS_TURBO_CHECK_HIP(hipStreamDestroy(copy_streams_g[i]));
            copy_streams_g[i] = nullptr;
        }
    }

    if (recv_buffer != nullptr) {
        PRIMUS_TURBO_CHECK_HIP(hipFree(recv_buffer));
    }

    // reusable_barrier(shm->barrier, shm->sense, group_world_size);

    if (group_rank == 0) {
        sharedMemoryClose(info);
        shareMemoryDelete(shm_name.c_str());
    }

    delete info;
}

static void stream_callback(hipStream_t stream, hipError_t status, void *userData) {
    volatile AllGatherShmStruct *shm = (volatile AllGatherShmStruct *) userData;
    reusable_barrier(shm->barrier, shm->sense, shm->group_world_size);
}

static void close_handle_stream_callback(hipStream_t stream, hipError_t status, void *userData) {
    volatile AllGatherShmStruct *shm = (volatile AllGatherShmStruct *) userData;

    // TODO: if pointer is hipfree by pytorch?
    // close all remote handles opened by this rank
    for (size_t i = 0; i < shm->group_world_size; ++i) {
        if (i != group_rank_g && shm->remote_base_ptrs[i] != nullptr) {
            PRIMUS_TURBO_CHECK_HIP(hipIpcCloseMemHandle(shm->remote_base_ptrs[group_rank_g][i]));
        }
        shm->remote_base_ptrs[group_rank_g][i] = nullptr;
        shm->remote_base_offsets[i]            = 0;
    }
    // std::cout << "Second run, RANK[" << group_rank_g << "], line=" << __LINE__
    // << "close handle" << std::endl;

    // reusable_barrier(shm->barrier, shm->sense, shm->group_world_size);
}

void run_dma_all_gather_into_tensor_nobuffer(dmaHandle_t handle, void *output, void *input,
                                             size_t size_bytes, size_t group_rank,
                                             size_t group_world_size, hipStream_t stream) {
    // make sure stream has at least one kernel running
    // launch_emtpy_kernel(stream);

    SharedMemoryInfo *info = handle;

    volatile AllGatherShmStruct *shm = nullptr;
    shm                              = (volatile AllGatherShmStruct *) info->addr;

    if (shm->is_first_run[group_rank]) {
        // std::cout << "First run, RANK[" << group_rank << "], line=" << __LINE__
        // << std::endl;
        group_rank_g = group_rank;
        // first run will create following items:
        // shm: entry_event, exit_event, copy_events,
        // global: copy_streams_g
        // each allgather engine instance should have own events
        hipEvent_t entry_event = nullptr;
        PRIMUS_TURBO_CHECK_HIP(hipEventCreateWithFlags(&entry_event, hipEventDisableTiming));
        shm->entry_events[group_rank] = entry_event;

        hipEvent_t local_exit_event = nullptr;
        PRIMUS_TURBO_CHECK_HIP(hipEventCreateWithFlags(
            &local_exit_event, hipEventDisableTiming | hipEventInterprocess));
        shm->local_exit_events[group_rank] = local_exit_event;

        hipIpcEventHandle_t local_exit_event_handle;
        PRIMUS_TURBO_CHECK_HIP(hipIpcGetEventHandle(&local_exit_event_handle, local_exit_event));
        // shm->local_exit_event_handles[group_rank] = local_exit_event_handle;
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
            // hipIpcEventHandle_t temp_handle = shm->local_exit_event_handles[i];
            PRIMUS_TURBO_CHECK_HIP(hipIpcOpenEventHandle(
                &remote_exit_event, *(hipIpcEventHandle_t *) &shm->local_exit_event_handles[i]));
            shm->remote_exit_events[group_rank][i] = remote_exit_event;
        }

        for (size_t i = 0; i < MAX_DEVICES; ++i) {
            hipEvent_t copy_event = nullptr;
            PRIMUS_TURBO_CHECK_HIP(hipEventCreateWithFlags(&copy_event, hipEventDisableTiming));
            copy_events_g[i] = copy_event;

            hipStream_t copy_stream = nullptr;
            int         leastPriority, greatestPriority;
            PRIMUS_TURBO_CHECK_HIP(
                hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
            PRIMUS_TURBO_CHECK_HIP(
                hipStreamCreateWithPriority(&copy_stream, hipStreamNonBlocking, greatestPriority));
            copy_streams_g[i] = copy_stream;
        }

        shm->group_world_size         = group_world_size;
        shm->is_first_run[group_rank] = false;
    } else {
        if (shm->group_world_size != group_world_size) {
            throw std::runtime_error("Must keep group world size the same.");
        }

        if (group_rank_g != group_rank) {
            throw std::runtime_error("Must keep group rank the same.");
        }
        // std::cout << "Second run, RANK[" << group_rank << "], line=" << __LINE__
        // << std::endl;

        // close mem handle of previous run
        // PRIMUS_TURBO_CHECK_HIP(hipStreamAddCallback(
        //     stream, close_handle_stream_callback,
        //     const_cast<void*>(reinterpret_cast<volatile void*>(shm)), 0));
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
    // shm->remote_base_offsets[group_rank] = output_base_offset;
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
        hipStream_t copy_stream = copy_streams_g[i];
        PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(copy_stream, shm->entry_events[group_rank], 0));
    }

    for (size_t i = 0; i < group_world_size; ++i) {
        size_t      i_split     = group_rank;
        size_t      remote_rank = (group_rank + i) % group_world_size;
        hipStream_t copy_stream = copy_streams_g[remote_rank];

        void *src        = input;
        void *remote_ptr = static_cast<void *>(
            static_cast<char *>(shm->remote_base_ptrs[group_rank][remote_rank]) +
            shm->remote_base_offsets[remote_rank]);
        void *dst = static_cast<void *>(static_cast<char *>(remote_ptr) + size_bytes * i_split);

        // std::cout << "RANK[" << group_rank << "], line=" << __LINE__ << "i=" << i
        // << ", src=" << src << ", remote=" << remote_ptr << ", dst=" << dst <<
        // std::endl;
        PRIMUS_TURBO_CHECK_HIP(
            hipMemcpyAsync(dst, src, size_bytes, hipMemcpyDeviceToDeviceNoCU, copy_stream));
    }

    for (size_t i = 0; i < group_world_size; ++i) {
        size_t      remote_rank = (group_rank + i) % group_world_size;
        hipStream_t copy_stream = copy_streams_g[remote_rank];
        hipEvent_t  copy_event  = copy_events_g[remote_rank];
        PRIMUS_TURBO_CHECK_HIP(hipEventRecord(copy_event, copy_stream));
        PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(stream, copy_event, 0));

        // std::cout << "RANK[" << group_rank << "], line=" << __LINE__ << ", i=" <<
        // i << std::endl; PRIMUS_TURBO_CHECK_HIP(hipStreamSynchronize(stream)); std::cout <<
        // "RANK[" << group_rank << "], line=" << __LINE__ << ", i=" << i <<
        // std::endl;
    }

    // launch_emtpy_kernel(stream);

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

} // namespace primus_turbo::pytorch