#pragma once

namespace primus_turbo::pytorch::cco::ep {
namespace shared_memory {

struct SharedMemoryAllocator {
    SharedMemoryAllocator();
    void malloc(void **ptr, size_t size);
    void free(void *ptr);
    void get_mem_handle(cudaIpcMemHandle_t *mem_handle, void *ptr);
    void open_mem_handle(void **ptr, cudaIpcMemHandle_t *mem_handle);
    void close_mem_handle(void *ptr);
};
} // namespace shared_memory

} // namespace primus_turbo::pytorch::cco::ep