/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2025 DeepSeek. All rights reserved.
 *
 * Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

#include <cstring>

#include "backend/api.cuh"
#include "legacy/api.cuh"
#include "legacy/compiled.cuh"
#include "legacy/launch.cuh"
#include "legacy/utils.cuh"
#include "primus_turbo/common.h"

#ifndef DISABLE_ROCSHMEM
#include <rocshmem/rocshmem.hpp>
#endif

// [agent modifed]: the intranode::barrier kernel that used to live here is gone -- the legacy
// copy brought upstream's own definition in legacy/intranode.cu, and two definitions of the
// same symbol do not link.

// [agent modifed]: namespace internode -> nvshmem, and the signatures now match
// backend/api.cuh (which is upstream's own spelling for this backend).
namespace primus_turbo::deep_ep::nvshmem {

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

// [agent modifed]: `bool low_latency_mode` -> `const int& team_split_stride`. Upstream passes
// the NVL peer count when it wants the sub-RDMA team and 0 otherwise, so the stride carries
// both pieces of information the old boolean did.
int init(const std::vector<uint8_t> &root_unique_id_val, const int &rank, const int &num_ranks,
         const int &team_split_stride) {
    rocshmem::rocshmem_uniqueid_t  root_unique_id;
    rocshmem::rocshmem_init_attr_t attr;
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(rocshmem::rocshmem_uniqueid_t));
    PRIMUS_TURBO_CHECK_ROCSHMEM(
        rocshmem::rocshmem_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr));
    PRIMUS_TURBO_CHECK_ROCSHMEM(
        rocshmem::rocshmem_init_attr(rocshmem::ROCSHMEM_INIT_WITH_UNIQUEID, &attr));

    // Create sub-RDMA teams
    // NOTES: if `num_ranks <= LEGACY_NUM_MAX_NVL_PEERS` then only low-latency kernels are used
    if (team_split_stride > 0 and num_ranks > team_split_stride) {
        PRIMUS_TURBO_CHECK(cpu_rdma_team == rocshmem::ROCSHMEM_TEAM_INVALID);
        PRIMUS_TURBO_CHECK(num_ranks % team_split_stride == 0);
        PRIMUS_TURBO_CHECK(rocshmem::rocshmem_team_split_strided(
                               rocshmem::ROCSHMEM_TEAM_WORLD, rank % team_split_stride,
                               team_split_stride, num_ranks / team_split_stride,
                               &cpu_rdma_team_config, 0, &cpu_rdma_team) == 0);
        PRIMUS_TURBO_CHECK(cpu_rdma_team != rocshmem::ROCSHMEM_TEAM_INVALID);
    }

    rocshmem::rocshmem_barrier_all();
    return rocshmem::rocshmem_my_pe();
}

void *alloc(const size_t &size, const size_t &alignment) {
    auto alloc_size = ALIGN(size, alignment);
    return rocshmem::rocshmem_malloc(alloc_size);
}

void free(void *ptr) {
    rocshmem::rocshmem_free(ptr);
}

// [agent modifed]: rocSHMEM has no on-stream barrier, so both of upstream's flavours collapse
// onto the blocking host one; the caller already synchronizes the stream around it.
void barrier(const bool &with_cpu_sync, const std::optional<cudaStream_t> &stream_opt) {
    if (stream_opt.has_value())
        PRIMUS_TURBO_CHECK_HIP(hipStreamSynchronize(stream_opt.value()));
    rocshmem::rocshmem_barrier_all();
}

void finalize() {
    if (cpu_rdma_team != rocshmem::ROCSHMEM_TEAM_INVALID) {
        rocshmem::rocshmem_team_destroy(cpu_rdma_team);
        cpu_rdma_team = rocshmem::ROCSHMEM_TEAM_INVALID;
    }
    rocshmem::rocshmem_finalize();
}
#else
// [agent modifed]: new -- without rocSHMEM the RDMA path cannot exist, but the legacy buffer
// still references these symbols, so they resolve to a clear runtime error instead of a
// link failure. Intranode (`num_rdma_bytes == 0`) never reaches them.
[[noreturn]] static void no_rocshmem() {
    PRIMUS_TURBO_CHECK(false and "rocSHMEM is disabled at compile time, the RDMA path is unavailable");
    __builtin_unreachable();
}

std::vector<uint8_t> get_unique_id() { no_rocshmem(); }
int init(const std::vector<uint8_t> &, const int &, const int &, const int &) { no_rocshmem(); }
void *alloc(const size_t &, const size_t &) { no_rocshmem(); }
void free(void *) { no_rocshmem(); }
void barrier(const bool &, const std::optional<cudaStream_t> &) { no_rocshmem(); }
void finalize() { no_rocshmem(); }
#endif

} // namespace primus_turbo::deep_ep::nvshmem
