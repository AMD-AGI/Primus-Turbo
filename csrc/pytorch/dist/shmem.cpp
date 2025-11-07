#include "primus_turbo/dist/shmem.h"

namespace primus_turbo::pytorch {

int sharedMemoryCreate(const char *name, size_t sz, SharedMemoryInfo *info) {
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

int sharedMemoryOpen(const char *name, size_t sz, SharedMemoryInfo *info) {
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

void sharedMemoryClose(SharedMemoryInfo *info) {
    if (info->addr) {
        munmap(info->addr, info->size);
    }
    if (info->shmFd) {
        close(info->shmFd);
    }
}

void shareMemoryDelete(const char *name) {
    shm_unlink(name);
}

} // namespace primus_turbo::pytorch