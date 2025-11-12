// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <cstddef>

namespace primus_turbo::pytorch::dist {

struct SharedMemoryInfo {
    void  *addr  = nullptr;
    size_t size  = 0;
    int    shmFd = -1;

    SharedMemoryInfo() = default;

    SharedMemoryInfo(void *addr_, size_t size_, int shmFd_)
        : addr(addr_), size(size_), shmFd(shmFd_) {}
};

int  shared_memory_create(const char *name, size_t sz, SharedMemoryInfo *info);
int  shared_memory_open(const char *name, size_t sz, SharedMemoryInfo *info);
void shared_memory_close(SharedMemoryInfo *info);
void shared_memory_delete(const char *name);

} // namespace primus_turbo::pytorch::dist
