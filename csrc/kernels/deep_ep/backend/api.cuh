#pragma once

// [agent modifed]: cut-down port of upstream `csrc/kernels/backend/api.cuh`. Upstream also
// declares the NCCL symmetric-memory context and the CUDA-driver batched write/wait helpers
// there; the legacy buffer only ever calls the six entry points below, and the rest would drag
// in <nccl_device.h> plus the JIT runtime. The `nvshmem` spelling is kept so the legacy host
// sources stay verbatim -- the bodies live in ../runtime.cu and talk to rocSHMEM.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "../legacy/compiled.cuh"

namespace primus_turbo::deep_ep::nvshmem {

std::vector<uint8_t> get_unique_id();

int init(const std::vector<uint8_t>& root_unique_id_val,
         const int& rank,
         const int& num_ranks,
         const int& team_split_stride);

void* alloc(const size_t& size, const size_t& alignment);

void free(void* ptr);

void barrier(const bool& with_cpu_sync, const std::optional<cudaStream_t>& stream_opt = std::nullopt);

void finalize();

}  // namespace primus_turbo::deep_ep::nvshmem
