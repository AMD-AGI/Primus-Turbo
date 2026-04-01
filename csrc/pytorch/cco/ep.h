// #pragma once
// #include "../kernels/cco/configs.h"
// #include <cuda_runtime.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/pytypes.h>
// #include <torch/types.h>
// #include <tuple>
// #include <vector>

// namespace primus_turbo::pytorch::cco::ep {
// namespace symmetric_memory {

// struct SharedMemoryAllocator {
//     SharedMemoryAllocator() = default;
//     void malloc(void **ptr, size_t size);
//     void free(void *ptr);
//     void get_mem_handle(cudaIpcMemHandle_t *mem_handle, void *ptr);
//     void open_mem_handle(void **ptr, cudaIpcMemHandle_t *mem_handle);
//     void close_mem_handle(void *ptr);
// };
// } // namespace shared_memory

// } // namespace primus_turbo::pytorch::cco::ep