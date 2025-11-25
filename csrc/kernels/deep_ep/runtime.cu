/*
 * Copyright (c) 2025 DeepSeek. All rights reserved.
 *
 * Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 */

#include <cstring>

#include "launch.cuh"
#include "primus_turbo/common.h"
#include "primus_turbo/deep_ep/api.h"
#include "primus_turbo/deep_ep/configs.h"
#include "utils.cuh"

#ifndef DISABLE_ROCSHMEM
#include <rocshmem/rocshmem.hpp>
#endif
namespace primus_turbo::deep_ep {

namespace intranode {

template <int kNumRanks> __global__ void barrier(int **barrier_signal_ptrs, int rank) {
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

void barrier(int **barrier_signal_ptrs, int rank, int num_ranks, hipStream_t stream) {
#define BARRIER_LAUNCH_CASE(ranks)                                                                 \
    LAUNCH_KERNEL_NON_COOPERATIVE(&cfg, barrier<ranks>, barrier_signal_ptrs, rank);                \
    break

    SETUP_LAUNCH_CONFIG(1, kWarpSize, stream);
    SWITCH_RANKS(BARRIER_LAUNCH_CASE);
#undef BARRIER_LAUNCH_CASE
}

} // namespace intranode

namespace internode {

#ifndef DISABLE_ROCSHMEM
rocshmem::rocshmem_team_t        cpu_rdma_team = rocshmem::ROCSHMEM_TEAM_INVALID;
rocshmem::rocshmem_team_config_t cpu_rdma_team_config;

std::vector<uint8_t> get_unique_id() {
    rocshmem::rocshmem_uniqueid_t unique_id;
    PRIMUS_TURBO_CHECK_ROCSHMEM(rocshmem::rocshmem_get_uniqueid(&unique_id));
    std::vector<uint8_t> result(sizeof(rocshmem::rocshmem_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(rocshmem::rocshmem_uniqueid_t));
    return result;
}

int init(const std::vector<uint8_t> &root_unique_id_val, int rank, int num_ranks,
         bool low_latency_mode) {
    rocshmem::rocshmem_uniqueid_t  root_unique_id;
    rocshmem::rocshmem_init_attr_t attr;
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(rocshmem::rocshmem_uniqueid_t));
    PRIMUS_TURBO_CHECK_ROCSHMEM(
        rocshmem::rocshmem_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr));
    PRIMUS_TURBO_CHECK_ROCSHMEM(
        rocshmem::rocshmem_init_attr(rocshmem::ROCSHMEM_INIT_WITH_UNIQUEID, &attr));

    // Create sub-RDMA teams
    // NOTES: if `num_ranks <= NUM_MAX_NVL_PEERS` then only low-latency kernels are used
    if (low_latency_mode and num_ranks > NUM_MAX_NVL_PEERS) {
        PRIMUS_TURBO_CHECK(cpu_rdma_team == rocshmem::ROCSHMEM_TEAM_INVALID);
        PRIMUS_TURBO_CHECK(num_ranks % NUM_MAX_NVL_PEERS == 0);
        PRIMUS_TURBO_CHECK(rocshmem::rocshmem_team_split_strided(
                               rocshmem::ROCSHMEM_TEAM_WORLD, rank % NUM_MAX_NVL_PEERS,
                               NUM_MAX_NVL_PEERS, num_ranks / NUM_MAX_NVL_PEERS,
                               &cpu_rdma_team_config, 0, &cpu_rdma_team) == 0);
        PRIMUS_TURBO_CHECK(cpu_rdma_team != rocshmem::ROCSHMEM_TEAM_INVALID);
    }

    rocshmem::rocshmem_barrier_all();
    return rocshmem::rocshmem_my_pe();
}

void *alloc(size_t size, size_t alignment) {
    auto alloc_size = ALIGN(size, alignment);
    return rocshmem::rocshmem_malloc(alloc_size);
}

void free(void *ptr) {
    rocshmem::rocshmem_free(ptr);
}

void barrier() {
    rocshmem::rocshmem_barrier_all();
}

void finalize() {
    if (cpu_rdma_team != rocshmem::ROCSHMEM_TEAM_INVALID) {
        rocshmem::rocshmem_team_destroy(cpu_rdma_team);
        cpu_rdma_team = rocshmem::ROCSHMEM_TEAM_INVALID;
    }
    rocshmem::rocshmem_finalize();
}
#endif
} // namespace internode

} // namespace primus_turbo::deep_ep
