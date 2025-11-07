#pragma once

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstddef>

#if defined(__linux__)
#define cpu_atomic_add32(a, x) __sync_add_and_fetch(a, x)
#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define cpu_atomic_add32(a, x) InterlockedAdd((volatile LONG *) a, x)
#else
#error Unsupported system
#endif

namespace primus_turbo::pytorch {

struct SharedMemoryInfo {
    void  *addr  = nullptr;
    size_t size  = 0;
    int    shmFd = -1;

    SharedMemoryInfo() = default;

    SharedMemoryInfo(void *addr_, size_t size_, int shmFd_)
        : addr(addr_), size(size_), shmFd(shmFd_) {}
};

using dmaHandle_t = SharedMemoryInfo *;

int  sharedMemoryCreate(const char *name, size_t sz, SharedMemoryInfo *info);
int  sharedMemoryOpen(const char *name, size_t sz, SharedMemoryInfo *info);
void sharedMemoryClose(SharedMemoryInfo *info);
void shareMemoryDelete(const char *name);

} // namespace primus_turbo::pytorch