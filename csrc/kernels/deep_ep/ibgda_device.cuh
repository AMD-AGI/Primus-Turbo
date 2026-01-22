#pragma once
#include "utils.cuh"
#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

namespace primus_turbo::deep_ep {

template <bool kAlwaysDoPostSend = false>
__device__ static __forceinline__ void
nvshmemi_ibgda_put_nbi_warp(uint64_t req_rptr, uint64_t req_lptr, size_t bytes, int dst_pe,
                            int qp_id, int lane_id, int message_idx) {
    rocshmem_ctx_putmem_nbi_wg(ROCSHMEM_CTX_DEFAULT, reinterpret_cast<void *>(req_rptr),
                               reinterpret_cast<void *>(req_lptr), bytes, dst_pe);
}

__device__ __forceinline__ void nvshmemi_ibgda_amo_nonfetch_add(void *rptr, const int &value,
                                                                int pe, int qp_id,
                                                                bool is_local_copy = false) {
    if (is_local_copy) {
        atomicAdd(static_cast<unsigned long long *>(rptr), value);
    } else {
        rocshmem_ctx_uint64_atomic_add(ROCSHMEM_CTX_DEFAULT, reinterpret_cast<uint64_t *>(rptr),
                                       static_cast<uint64_t>(value), pe);
    }
}

// Wait until wqe `idx - 1` is completed.
__device__ static __forceinline__ void nvshmemi_ibgda_quiet(int dst_pe, int qp_id) {
    rocshmem_ctx_quiet(ROCSHMEM_CTX_DEFAULT);
}

} // namespace primus_turbo::deep_ep
