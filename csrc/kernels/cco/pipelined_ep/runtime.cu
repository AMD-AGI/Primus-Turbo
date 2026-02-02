#include <cstring>
#include <vector>

#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

namespace primus_turbo::cco::pipelined_ep {

namespace intranode {

template <int kNumRanks> __global__ void barrier(int **barrier_signal_ptrs, int rank) {
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

void barrier(int **barrier_signal_ptrs, int rank, int num_ranks, cudaStream_t stream) {
#define BARRIER_LAUNCH_CASE(ranks)                                                                 \
    LAUNCH_KERNEL(&cfg, barrier<ranks>, barrier_signal_ptrs, rank);                                \
    break

    SETUP_LAUNCH_CONFIG(1, WARP_SIZE, stream);
    SWITCH_RANKS(BARRIER_LAUNCH_CASE);
#undef BARRIER_LAUNCH_CASE
}

} // namespace intranode

} // namespace primus_turbo::cco::pipelined_ep
