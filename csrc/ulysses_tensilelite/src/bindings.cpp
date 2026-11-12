/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Adapted from attn-a2a-demo (https://github.com/DDEle/attn-a2a-demo) at 7a7343ef47,
// csrc/bindings.cpp.
#include "barrier.hpp"
#include "comm_context.hpp"
#include "device_mem.hpp"
#include "fused_hop.hpp"

#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <climits>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace py = pybind11;

static std::string hipRuntimePath() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&hipGetDevice), &info) == 0 || info.dli_fname == nullptr)
        throw std::runtime_error("dladdr(hipGetDevice) failed");
    char buf[PATH_MAX];
    return realpath(info.dli_fname, buf) ? std::string(buf) : std::string(info.dli_fname);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hip_runtime_path", &hipRuntimePath);

    m.def("alloc", &aa::alloc, py::arg("nbytes"), py::arg("fine_grained"));
    m.def("ipc_handle", [](torch::Tensor const &t) { return py::bytes(aa::ipcHandle(t)); });
    m.def("ipc_open", [](py::bytes h) { return aa::ipcOpen(std::string(h)); });
    m.def("ipc_close", &aa::ipcClose);
    m.def("memcpy_d2d", &aa::memcpyD2D);

    py::class_<aa::CommContext>(m, "CommContext")
        .def(py::init<int, int, std::vector<int> const &, int>(), py::arg("rank"), py::arg("world"),
             py::arg("devices"), py::arg("sdma_rotate"))
        .def("queue_fields", [](aa::CommContext const &c, int peer) {
            auto f = c.queueFields(peer);
            return py::make_tuple((uintptr_t) f[0], (uintptr_t) f[1], (uintptr_t) f[2],
                                  (uintptr_t) f[3]);
        });

    py::class_<aa::FusedHop>(m, "FusedHop")
        .def_static(
            "for_selection",
            [](int rank, int world, std::string const &mode, size_t m, size_t n, size_t k,
               size_t extent, std::string const &dat, std::string const &coDir,
               std::string const &co, double beta, bool trans_a) {
                return std::make_unique<aa::FusedHop>(
                    aa::HopShape{mode, m, n, k, extent, rank, world, beta, trans_a}, dat, coDir,
                    co);
            },
            py::arg("rank"), py::arg("world"), py::arg("mode"), py::arg("m"), py::arg("n"),
            py::arg("k"), py::arg("extent"), py::arg("dat"), py::arg("co_dir"), py::arg("co"),
            py::arg("beta") = 0.0, py::arg("trans_a") = true)
        .def("kernel_name", &aa::FusedHop::kernelName)
        .def("counter_bytes", &aa::FusedHop::counterBytes)
        .def("macro_tile", &aa::FusedHop::macroTile)
        .def("world", &aa::FusedHop::world)
        .def("mode", &aa::FusedHop::modeName)
        .def(
            "bind",
            [](aa::FusedHop &h, int ch, std::vector<uintptr_t> const &flags,
               std::vector<uintptr_t> const &bufs, std::vector<std::array<uintptr_t, 4>> const &qs,
               uintptr_t a, uintptr_t b, uintptr_t d, uintptr_t counter, uintptr_t c) {
                if (flags.size() != bufs.size() || flags.size() != qs.size())
                    throw std::runtime_error("bind: flag/buf/queue lists differ in length");
                std::vector<aa::FusedHop::PeerFields> peers(flags.size());
                for (size_t j = 0; j < flags.size(); j++)
                    peers[j] = {(void *) flags[j], (void *) bufs[j],  (void *) qs[j][0],
                                (void *) qs[j][1], (void *) qs[j][2], (void *) qs[j][3]};
                h.bind(ch, peers, (void const *) a, (void const *) b, (void *) d, (void *) counter,
                       (void const *) c);
            },
            py::arg("ch"), py::arg("flags"), py::arg("bufs"), py::arg("queues"), py::arg("a"),
            py::arg("b"), py::arg("d"), py::arg("counter"), py::arg("c") = 0)
        .def("launch", &aa::FusedHop::launch);

    m.attr("FLAG_BLOCK_BYTES") = aa::kFlagBlockBytes;

    m.def("boundary_barrier", &aa::boundaryBarrier);
    m.attr("ARRIVAL_BYTES") = aa::kArrivalBytes;
}
