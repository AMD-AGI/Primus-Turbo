// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

// Thin pytorch binding for the ODC single-node rocSHMEM host-API backend.
//
// The host surface (uid bootstrap + symmetric heap + peer-ptr resolve) is
// implemented in csrc/kernels/odc_rocshmem/odc_rocshmem_host.cu and lives in
// libprimus_turbo_kernels.so. This TU only marshals bytes/char-buffers and
// re-exposes the surface as the `odc_rocshmem_host` pybind submodule; it links
// the kernels library and therefore needs no rocSHMEM include/link flags itself.
// Guarded by DISABLE_ROCSHMEM so the extension still builds without rocSHMEM.

#ifndef DISABLE_ROCSHMEM

#include <torch/extension.h>

#include <string>

#include "primus_turbo/odc_rocshmem/api.h"

#include "../extensions.h"

namespace primus_turbo::pytorch {

void register_odc_rocshmem_host(pybind11::module_ &m) {
    namespace rs = primus_turbo::odc_rocshmem::host;
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
