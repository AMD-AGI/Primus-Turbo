// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/dist/shmem.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace primus_turbo::pytorch::dist {

int shared_memory_create(const char *name, size_t sz, SharedMemoryInfo *info) {
    info->size = sz;

    info->shmFd = shm_open(name, O_RDWR | O_CREAT, 0777);
    if (info->shmFd < 0) {
        return errno;
    }

    int status = ftruncate(info->shmFd, sz);
    if (status != 0) {
        return status;
    }

    info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
    if (info->addr == nullptr) {
        return errno;
    }

    return 0;
}

int shared_memory_open(const char *name, size_t sz, SharedMemoryInfo *info) {
    info->size = sz;

    info->shmFd = shm_open(name, O_RDWR, 0777);
    if (info->shmFd < 0) {
        return errno;
    }

    info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
    if (info->addr == nullptr) {
        return errno;
    }

    return 0;
}

void shared_memory_close(SharedMemoryInfo *info) {
    if (info->addr) {
        munmap(info->addr, info->size);
    }
    if (info->shmFd) {
        close(info->shmFd);
    }
}

void shared_memory_delete(const char *name) {
    shm_unlink(name);
}

} // namespace primus_turbo::pytorch::dist
