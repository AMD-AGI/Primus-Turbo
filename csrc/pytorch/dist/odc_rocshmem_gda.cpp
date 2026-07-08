// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

// Thin pytorch binding for the ODC multi-node GPU-Direct Async (GDA) rocSHMEM
// backend.
//
// The GDA surface (uid bootstrap, symmetric heap, and the device gather /
// reduce-scatter launchers) is implemented in
// csrc/kernels/odc_rocshmem/odc_rocshmem_gda.cu and lives in
// libprimus_turbo_kernels.so. This TU only does pointer-array / bytes marshalling
// and re-exposes the surface as the `odc_rocshmem_gda` pybind submodule; it links
// the kernels library and therefore needs no rocSHMEM include/link flags itself.
// Guarded by DISABLE_ROCSHMEM so the extension still builds without rocSHMEM.

#ifndef DISABLE_ROCSHMEM

#include <torch/extension.h>

#include <cstddef>
#include <string>
#include <vector>

#include "primus_turbo/odc_rocshmem/api.h"

#include "../extensions.h"

namespace primus_turbo::pytorch {

void register_odc_rocshmem_gda(pybind11::module_ &m) {
    namespace gda = primus_turbo::odc_rocshmem::gda;
    auto sub      = m.def_submodule(
        "odc_rocshmem_gda",
        "ODC multi-node GPU-direct (GDA) rocSHMEM backend: uid bootstrap, symmetric "
             "heap, and device gather / reduce-scatter kernels");

    // Single-node-compatible host surface (mirrors odc_rocshmem_host).
    sub.def("rs_uid_bytes", &gda::rs_uid_bytes);
    sub.def("rs_get_uid", []() {
        std::string buf((size_t) gda::rs_uid_bytes(), '\0');
        gda::rs_get_uid(buf.data());
        return pybind11::bytes(buf);
    });
    sub.def("rs_init_uid", [](int rank, int nranks, pybind11::bytes uid) {
        std::string s = uid;
        gda::rs_init_uid(rank, nranks, s.data());
    });
    sub.def("rs_my_pe", &gda::rs_my_pe);
    sub.def("rs_n_pes", &gda::rs_n_pes);
    sub.def("rs_malloc", &gda::rs_malloc);
    sub.def("rs_ptr", &gda::rs_ptr);
    sub.def("rs_is_remote", &gda::rs_is_remote);
    sub.def("rs_barrier", &gda::rs_barrier);
    sub.def("rs_finalize", &gda::rs_finalize);

    // GPU-direct device-kernel launchers.
    sub.def("gda_gather", [](long long target, long long src, size_t nbytes, std::vector<int> peers,
                             size_t stride_bytes) {
        return gda::gda_gather(target, src, nbytes, peers.data(), (int) peers.size(), stride_bytes);
    });
    sub.def("gda_gather_async", [](long long target, long long src, size_t nbytes,
                                   std::vector<int> peers, size_t stride_bytes, long long stream) {
        return gda::gda_gather_async(target, src, nbytes, peers.data(), (int) peers.size(),
                                     stride_bytes, stream);
    });
    sub.def("gda_reduce_scatter_acc", &gda::gda_reduce_scatter_acc);
    sub.def("gda_reduce_scatter_acc_async", &gda::gda_reduce_scatter_acc_async);
    sub.def("gda_rs_overlap_sync", &gda::gda_rs_overlap_sync);
    sub.def("gda_stage_fence", &gda::gda_stage_fence);
    sub.def("gda_strided_touch", &gda::gda_strided_touch);
    sub.def("gda_hdp_init", &gda::gda_hdp_init);
    sub.def("gda_hdp_flush", &gda::gda_hdp_flush);
    sub.def("gda_microbench", &gda::gda_microbench);
}

} // namespace primus_turbo::pytorch

#endif // DISABLE_ROCSHMEM
