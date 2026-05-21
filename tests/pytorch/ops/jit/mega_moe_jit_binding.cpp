// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// JIT pybind module for the DG-aligned mega-MoE layout helpers.
//
// Loaded by ``torch.utils.cpp_extension.load`` from
// ``test_mega_moe_jit_perf.py``.  Exposes the kernel-level layout
// helpers (``layout::Workspace`` / ``layout::Buffer`` / pool-token
// formulas) so the Python parity test can confirm the host-side
// layout math matches the Python reference byte-for-byte.

#include <torch/extension.h>

#include <cstdint>
#include <tuple>
#include <vector>

// Implementations from ``mega_moe_jit_launch.cu`` (HIP TU).
extern "C" void mega_moe_jit_workspace_probe(int num_ranks, int num_experts,
                                             int num_max_tokens_per_rank, int num_topk,
                                             int64_t  *out_workspace_bytes,
                                             uint32_t *out_num_max_pool_tokens,
                                             uint32_t *out_num_max_pool_blocks);

extern "C" int mega_moe_jit_num_max_pool_tokens(int num_ranks, int num_max_tokens_per_rank,
                                                int num_topk, int num_experts_per_rank);

extern "C" int mega_moe_jit_num_padded_sf_pool_tokens(int num_max_pool_tokens, int block_m);

extern "C" void mega_moe_jit_compute_layout(int num_ranks, int num_experts,
                                            int num_max_tokens_per_rank, int num_topk, int hidden,
                                            int intermediate_hidden, int64_t *out_offsets,
                                            int64_t *out_total_bytes, int *out_num_max_pool_tokens,
                                            int *out_num_padded_sf_pool_tokens);

extern "C" int mega_moe_jit_run_stub();

namespace {

// (workspace_bytes, num_max_pool_tokens, num_max_pool_blocks)
std::tuple<int64_t, int64_t, int64_t> workspace_probe(int64_t num_ranks, int64_t num_experts,
                                                      int64_t num_max_tokens_per_rank,
                                                      int64_t num_topk) {
    int64_t  workspace_bytes     = 0;
    uint32_t num_max_pool_tokens = 0;
    uint32_t num_max_pool_blocks = 0;
    mega_moe_jit_workspace_probe(static_cast<int>(num_ranks), static_cast<int>(num_experts),
                                 static_cast<int>(num_max_tokens_per_rank),
                                 static_cast<int>(num_topk), &workspace_bytes, &num_max_pool_tokens,
                                 &num_max_pool_blocks);
    return {workspace_bytes, static_cast<int64_t>(num_max_pool_tokens),
            static_cast<int64_t>(num_max_pool_blocks)};
}

int64_t num_max_pool_tokens(int64_t num_ranks, int64_t num_max_tokens_per_rank, int64_t num_topk,
                            int64_t num_experts_per_rank) {
    return mega_moe_jit_num_max_pool_tokens(
        static_cast<int>(num_ranks), static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_topk), static_cast<int>(num_experts_per_rank));
}

int64_t num_padded_sf_pool_tokens(int64_t num_max_pool_tokens, int64_t block_m) {
    return mega_moe_jit_num_padded_sf_pool_tokens(static_cast<int>(num_max_pool_tokens),
                                                  static_cast<int>(block_m));
}

// (offsets[11], total_bytes, num_max_pool_tokens, num_padded_sf_pool_tokens)
std::tuple<std::vector<int64_t>, int64_t, int64_t, int64_t>
compute_layout(int64_t num_ranks, int64_t num_experts, int64_t num_max_tokens_per_rank,
               int64_t num_topk, int64_t hidden, int64_t intermediate_hidden) {
    int64_t offsets[11]               = {0};
    int64_t total_bytes               = 0;
    int     num_max_pool_tokens       = 0;
    int     num_padded_sf_pool_tokens = 0;
    mega_moe_jit_compute_layout(static_cast<int>(num_ranks), static_cast<int>(num_experts),
                                static_cast<int>(num_max_tokens_per_rank),
                                static_cast<int>(num_topk), static_cast<int>(hidden),
                                static_cast<int>(intermediate_hidden), offsets, &total_bytes,
                                &num_max_pool_tokens, &num_padded_sf_pool_tokens);
    return {std::vector<int64_t>(offsets, offsets + 11), total_bytes,
            static_cast<int64_t>(num_max_pool_tokens),
            static_cast<int64_t>(num_padded_sf_pool_tokens)};
}

int64_t run_stub() {
    return static_cast<int64_t>(mega_moe_jit_run_stub());
}

} // anonymous namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("workspace_probe", &workspace_probe,
          "Returns (workspace_bytes, num_max_pool_tokens, num_max_pool_blocks)",
          py::arg("num_ranks"), py::arg("num_experts"), py::arg("num_max_tokens_per_rank"),
          py::arg("num_topk"));
    m.def("num_max_pool_tokens", &num_max_pool_tokens,
          "Returns layout::get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, "
          "num_topk, num_experts_per_rank)",
          py::arg("num_ranks"), py::arg("num_max_tokens_per_rank"), py::arg("num_topk"),
          py::arg("num_experts_per_rank"));
    m.def("num_padded_sf_pool_tokens", &num_padded_sf_pool_tokens,
          "Returns layout::get_num_padded_sf_pool_tokens(num_max_pool_tokens, block_m)",
          py::arg("num_max_pool_tokens"), py::arg("block_m"));
    m.def("compute_layout", &compute_layout,
          "Returns (offsets[11], total_bytes, num_max_pool_tokens, num_padded_sf_pool_tokens) "
          "for the DG-aligned symmetric buffer layout",
          py::arg("num_ranks"), py::arg("num_experts"), py::arg("num_max_tokens_per_rank"),
          py::arg("num_topk"), py::arg("hidden"), py::arg("intermediate_hidden"));
    m.def("run_stub", &run_stub,
          "Sentinel that returns 1 to indicate the DG-aligned device kernel body is not yet "
          "implemented; instantiates the impls/ launcher template to confirm it parses under "
          "hipcc.");
}
