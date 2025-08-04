/*
 * Copyright (c) 2025 DeepSeek. All rights reserved.
 *
 * Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 */

#include <cstring>

#include "configs.h"
#include "launch.h"
#include "primus_turbo/common.h"
#include "utils.h"

namespace primus_turbo::deep_ep {

namespace intranode {

template <int kNumRanks> __global__ void barrier(int **barrier_signal_ptrs, int rank) {
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

void barrier(int **barrier_signal_ptrs, int rank, int num_ranks, hipStream_t stream) {
#define BARRIER_LAUNCH_CASE(ranks)                                                                 \
    LAUNCH_KERNEL(&cfg, barrier<ranks>, barrier_signal_ptrs, rank);                                \
    break

    SETUP_LAUNCH_CONFIG(1, kWarpSize, stream);
    SWITCH_RANKS(BARRIER_LAUNCH_CASE);
#undef BARRIER_LAUNCH_CASE
}

} // namespace intranode

namespace internode {

std::vector<uint8_t> get_unique_id() {
    PRIMUS_TURBO_CHECK(false, "not support");
    return {};
}

#
int init(const std::vector<uint8_t> &root_unique_id_val, int rank, int num_ranks,
         bool low_latency_mode) {
    PRIMUS_TURBO_CHECK(false, "not support");
    return 0;
}

void *alloc(size_t size, size_t alignment) {
    PRIMUS_TURBO_CHECK(false, "not support");
}

void free(void *ptr) {
    PRIMUS_TURBO_CHECK(false, "not support");
}

void barrier() {
    PRIMUS_TURBO_CHECK(false, "not support");
}

void finalize() {
    PRIMUS_TURBO_CHECK(false, "not support");
}
} // namespace internode

} // namespace primus_turbo::deep_ep
