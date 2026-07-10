// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

// ODC single-node rocSHMEM host-API backend (kernel-library implementation).
//
// The host-only surface below (uid bootstrap + symmetric heap + peer-ptr resolve)
// originally shipped as librs_host5.so and was loaded via ctypes by ODC's
// primus/core/odc/odc/primitives/_rocshmem_backend.py. The host logic is preserved
// verbatim; it now lives in libprimus_turbo_kernels.so and is declared in
// primus_turbo/odc_rocshmem/api.h so the thin pytorch/jax binding can call it
// without linking rocSHMEM itself. Everything is guarded by DISABLE_ROCSHMEM so
// the kernels library still builds on toolchains without rocSHMEM.

#ifndef DISABLE_ROCSHMEM

#include <hip/hip_runtime.h>

#include <cstring>
#include <rocshmem/rocshmem.hpp>

#include "primus_turbo/odc_rocshmem/api.h"

namespace primus_turbo::odc_rocshmem::host {

using namespace rocshmem;

int rs_uid_bytes() {
    return (int) sizeof(rocshmem_uniqueid_t);
}
void rs_get_uid(char *out) {
    rocshmem_uniqueid_t uid;
    rocshmem_get_uniqueid(&uid);
    memcpy(out, uid.data(), sizeof(rocshmem_uniqueid_t));
}
void rs_init_uid(int rank, int nranks, const char *bytes) {
    rocshmem_uniqueid_t uid;
    memcpy(uid.data(), bytes, sizeof(rocshmem_uniqueid_t));
    rocshmem_init_attr_t attr;
    rocshmem_set_attr_uniqueid_args(rank, nranks, &uid, &attr);
    rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr);
}
void rs_get_ctx_fields(long long *a, long long *b) {
    void          *dctx = rocshmem_get_device_ctx();
    rocshmem_ctx_t h;
    hipMemcpy(&h, dctx, sizeof(rocshmem_ctx_t), hipMemcpyDeviceToHost);
    *a = (long long) h.ctx_opaque;
    *b = (long long) h.team_opaque;
}
int rs_my_pe() {
    return rocshmem_my_pe();
}
int rs_n_pes() {
    return rocshmem_n_pes();
}
long long rs_malloc(size_t n) {
    return (long long) rocshmem_malloc(n);
}
long long rs_ptr(long long p, int pe) {
    return (long long) rocshmem_ptr((void *) p, pe);
}
void rs_barrier() {
    rocshmem_barrier_all();
}
void rs_finalize() {
    rocshmem_finalize();
}

} // namespace primus_turbo::odc_rocshmem::host

#endif // DISABLE_ROCSHMEM
