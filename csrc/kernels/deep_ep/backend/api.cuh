#pragma once

// ROCm: cut-down port of upstream backend/api.cuh, only the six entry points the
// legacy buffer calls. The `nvshmem` spelling stays, the bodies are in ../runtime.cu.

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
