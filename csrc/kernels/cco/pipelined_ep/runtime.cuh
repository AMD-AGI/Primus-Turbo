#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector>

namespace primus_turbo::cco::pipelined_ep::intranode {
template <int kNumRanks> __global__ void barrier(int **barrier_signal_ptrs, int rank);

void barrier(int **barrier_signal_ptrs, int rank, int num_ranks, cudaStream_t stream);
} // namespace primus_turbo::cco::pipelined_ep::intranode