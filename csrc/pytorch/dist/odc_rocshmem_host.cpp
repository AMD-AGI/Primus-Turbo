// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

// ODC single-node rocSHMEM host-API backend, migrated into Primus-Turbo.
//
// The host-only surface below (uid bootstrap + symmetric heap + peer-ptr resolve)
// originally shipped as librs_host5.so and was loaded via ctypes by ODC's
// primus/core/odc/odc/primitives/_rocshmem_backend.py. The host logic is preserved
// verbatim; only the linkage changed: the original `extern "C"` block became a C++
// namespace because Turbo globs every csrc/pytorch/*.cpp into a single _C extension
// and the GDA backend (odc_rocshmem_gda.cpp) defines the identical rs_* symbols, so
// C linkage would collide at link time. The functions are re-exposed as the
// `odc_rocshmem_host` pybind submodule. Everything is guarded by DISABLE_ROCSHMEM so
// the extension still builds on toolchains without rocSHMEM.

#ifndef DISABLE_ROCSHMEM

#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cstring>
#include <rocshmem/rocshmem.hpp>

#include "../extensions.h"

namespace primus_turbo::pytorch::odc_rs_host {

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

} // namespace primus_turbo::pytorch::odc_rs_host

namespace primus_turbo::pytorch {

// Thin pybind wrapper exposing the verbatim host surface as the `odc_rocshmem_host`
// submodule. Char-buffer / bytes marshalling is the only glue over the raw functions.
void register_odc_rocshmem_host(pybind11::module_ &m) {
    namespace rs = primus_turbo::pytorch::odc_rs_host;
    auto sub     = m.def_submodule(
        "odc_rocshmem_host",
        "ODC single-node rocSHMEM host-API backend (uid bootstrap + symmetric heap)");
    sub.def("rs_uid_bytes", &rs::rs_uid_bytes);
    sub.def("rs_get_uid", []() {
        std::string buf((size_t) rs::rs_uid_bytes(), '\0');
        rs::rs_get_uid(buf.data());
        return pybind11::bytes(buf);
    });
    sub.def("rs_init_uid", [](int rank, int nranks, pybind11::bytes uid) {
        std::string s = uid;
        rs::rs_init_uid(rank, nranks, s.data());
    });
    sub.def("rs_my_pe", &rs::rs_my_pe);
    sub.def("rs_n_pes", &rs::rs_n_pes);
    sub.def("rs_malloc", &rs::rs_malloc);
    sub.def("rs_ptr", &rs::rs_ptr);
    sub.def("rs_barrier", &rs::rs_barrier);
    sub.def("rs_finalize", &rs::rs_finalize);
}

} // namespace primus_turbo::pytorch

#endif // DISABLE_ROCSHMEM
