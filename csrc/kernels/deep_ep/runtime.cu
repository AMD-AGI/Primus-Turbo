#include <cstring>
#include <vector>

#include "launch.cuh"
#include "primus_turbo/deep_ep/configs.cuh"
#include "primus_turbo/deep_ep/exception.cuh"
#include "utils.cuh"

#ifndef DISABLE_ROCSHMEM
#include "ibgda_device.cuh"
#endif

namespace primus_turbo::deep_ep {

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

namespace internode {

#ifndef DISABLE_ROCSHMEM
rocshmem_team_t       cpu_rdma_team = ROCSHMEM_TEAM_INVALID;
nvshmem_team_config_t cpu_rdma_team_config;

std::vector<uint8_t> get_unique_id() {
    rocshmem_uniqueid_t unique_id;
    rocshmem_get_uniqueid(&unique_id);
    std::vector<uint8_t> result(sizeof(rocshmem_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(rocshmem_uniqueid_t));
    return result;
}

int init(const std::vector<uint8_t> &root_unique_id_val, int rank, int num_ranks,
         bool low_latency_mode) {
    rocshmem_uniqueid_t  root_unique_id;
    rocshmem_init_attr_t attr;
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(rocshmem_uniqueid_t));
    rocshmem_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr);
    rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr);

    // Create sub-RDMA teams
    // NOTES: if `num_ranks <= NUM_MAX_NVL_PEERS` then only low-latency kernels are used
    if (low_latency_mode and num_ranks > NUM_MAX_NVL_PEERS) {
        EP_HOST_ASSERT(cpu_rdma_team == ROCSHMEM_TEAM_INVALID);
        EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
        EP_HOST_ASSERT(rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, rank % NUM_MAX_NVL_PEERS,
                                                   NUM_MAX_NVL_PEERS, num_ranks / NUM_MAX_NVL_PEERS,
                                                   &cpu_rdma_team_config, 0, &cpu_rdma_team) == 0);
        EP_HOST_ASSERT(cpu_rdma_team != ROCSHMEM_TEAM_INVALID);
    }

    rocshmem_barrier_all();
    return rocshmem_my_pe();
}

void *alloc(size_t size, size_t alignment) {
    return rocshmem_malloc(align_up(size, alignment));
}

void free(void *ptr) {
    rocshmem_free(ptr);
}

void barrier() {
    rocshmem_barrier_all();
}

void finalize() {
    if (cpu_rdma_team != ROCSHMEM_TEAM_INVALID) {
        rocshmem_team_destroy(cpu_rdma_team);
        cpu_rdma_team = ROCSHMEM_TEAM_INVALID;
    }
    rocshmem_finalize();
}
#endif

} // namespace internode

} // namespace primus_turbo::deep_ep
