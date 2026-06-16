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
#include <optional>
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

extern "C" int mega_moe_jit_get_token_alignment_for_mega_moe();

extern "C" void mega_moe_jit_compute_layout(int num_ranks, int num_experts,
                                            int num_max_tokens_per_rank, int num_topk, int hidden,
                                            int intermediate_hidden, int64_t *out_offsets,
                                            int64_t *out_total_bytes, int *out_num_max_pool_tokens,
                                            int *out_num_padded_sf_pool_tokens);

extern "C" int mega_moe_jit_run_stub();

extern "C" int mega_moe_jit_run_mega_moe(const int64_t *sym_buffer_bases, int num_sym_buffer_bases,
                                         int rank_idx, int num_tokens, int num_max_tokens_per_rank,
                                         int hidden, int intermediate_hidden, int num_experts,
                                         int num_topk, int num_ranks, int64_t y_ptr,
                                         int64_t l1_weights_ptr, int64_t l1_weights_sf_ptr,
                                         int64_t l2_weights_ptr, int64_t l2_weights_sf_ptr,
                                         int64_t recv_stats_ptr, float activation_clamp,
                                         int fast_math);

extern "C" int  mega_moe_jit_prof_enabled();
extern "C" int  mega_moe_jit_prof_num_stages();
extern "C" void mega_moe_jit_prof_reset();
extern "C" void mega_moe_jit_prof_read(unsigned long long *acc_out, unsigned long long *cnt_out);
extern "C" int  mega_moe_jit_prof_wallclock_khz();

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

int64_t get_token_alignment_for_mega_moe() {
    return static_cast<int64_t>(mega_moe_jit_get_token_alignment_for_mega_moe());
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

inline int64_t tensor_ptr(const std::optional<at::Tensor> &t) {
    if (!t.has_value() || !t->defined())
        return 0;
    return reinterpret_cast<int64_t>(t->data_ptr());
}

// DG-aligned runtime entry point.  Accepts torch tensors for all inputs
// + outputs and dispatches the gfx950 mega-MoE kernel through the JIT
// launcher template.  Currently only the compile-time "smoke" shape is
// supported; non-smoke shapes return -1.
int64_t run_mega_moe(std::vector<int64_t> sym_buffer_bases, int64_t rank_idx, int64_t num_tokens,
                     int64_t num_max_tokens_per_rank, int64_t hidden, int64_t intermediate_hidden,
                     int64_t num_experts, int64_t num_topk, int64_t num_ranks, at::Tensor y,
                     at::Tensor l1_weights, at::Tensor l1_weights_sf, at::Tensor l2_weights,
                     at::Tensor l2_weights_sf, std::optional<at::Tensor> recv_stats,
                     double activation_clamp, bool fast_math) {
    TORCH_CHECK(!sym_buffer_bases.empty(), "sym_buffer_bases must be non-empty");
    TORCH_CHECK(y.is_cuda(), "y must be on CUDA");
    return static_cast<int64_t>(mega_moe_jit_run_mega_moe(
        sym_buffer_bases.data(), static_cast<int>(sym_buffer_bases.size()),
        static_cast<int>(rank_idx), static_cast<int>(num_tokens),
        static_cast<int>(num_max_tokens_per_rank), static_cast<int>(hidden),
        static_cast<int>(intermediate_hidden), static_cast<int>(num_experts),
        static_cast<int>(num_topk), static_cast<int>(num_ranks),
        reinterpret_cast<int64_t>(y.data_ptr()), reinterpret_cast<int64_t>(l1_weights.data_ptr()),
        reinterpret_cast<int64_t>(l1_weights_sf.data_ptr()),
        reinterpret_cast<int64_t>(l2_weights.data_ptr()),
        reinterpret_cast<int64_t>(l2_weights_sf.data_ptr()), tensor_ptr(recv_stats),
        static_cast<float>(activation_clamp), fast_math ? 1 : 0));
}

// ── In-kernel per-stage profiler wrappers (-DMEGA_MOE_PROFILE=1) ──
bool prof_enabled() {
    return mega_moe_jit_prof_enabled() != 0;
}
int prof_wallclock_khz() {
    return mega_moe_jit_prof_wallclock_khz();
}
void prof_reset() {
    mega_moe_jit_prof_reset();
}

// Returns (acc[kNumStages], cnt[kNumStages]) of s_memrealtime ticks / counts.
std::pair<std::vector<int64_t>, std::vector<int64_t>> prof_read() {
    const int                       n = mega_moe_jit_prof_num_stages();
    std::vector<unsigned long long> acc(n), cnt(n);
    if (n > 0)
        mega_moe_jit_prof_read(acc.data(), cnt.data());
    std::vector<int64_t> acc_i(acc.begin(), acc.end()), cnt_i(cnt.begin(), cnt.end());
    return {acc_i, cnt_i};
}

} // anonymous namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prof_enabled", &prof_enabled, "True if built with -DMEGA_MOE_PROFILE=1");
    m.def("prof_wallclock_khz", &prof_wallclock_khz,
          "s_memrealtime rate in kHz (0 if unavailable)");
    m.def("prof_reset", &prof_reset, "Zero the per-stage tick/count accumulators");
    m.def("prof_read", &prof_read,
          "Returns (acc_ticks[NSTAGES], counts[NSTAGES]) for the per-stage profiler");
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
    m.def("get_token_alignment_for_mega_moe", &get_token_alignment_for_mega_moe,
          "Returns primus_turbo::mega_moe::kTokenAlignment (LCM of kCandidateBlockM)");
    m.def("compute_layout", &compute_layout,
          "Returns (offsets[11], total_bytes, num_max_pool_tokens, num_padded_sf_pool_tokens) "
          "for the DG-aligned symmetric buffer layout",
          py::arg("num_ranks"), py::arg("num_experts"), py::arg("num_max_tokens_per_rank"),
          py::arg("num_topk"), py::arg("hidden"), py::arg("intermediate_hidden"));
    m.def("run_stub", &run_stub,
          "Sentinel that returns 1 to indicate the DG-aligned device kernel body is not yet "
          "implemented; instantiates the impls/ launcher template to confirm it parses under "
          "hipcc.");
    m.def("run_mega_moe", &run_mega_moe,
          "DG-aligned runtime entry: dispatches the gfx950 mega-MoE kernel with the given "
          "torch tensors. Returns 0 on success, -1 if the requested shape doesn't match the "
          "JIT-instantiated 'smoke' template, or a positive hipError_t otherwise.",
          py::arg("sym_buffer_bases"), py::arg("rank_idx"), py::arg("num_tokens"),
          py::arg("num_max_tokens_per_rank"), py::arg("hidden"), py::arg("intermediate_hidden"),
          py::arg("num_experts"), py::arg("num_topk"), py::arg("num_ranks"), py::arg("y"),
          py::arg("l1_weights"), py::arg("l1_weights_sf"), py::arg("l2_weights"),
          py::arg("l2_weights_sf"), py::arg("recv_stats"), py::arg("activation_clamp"),
          py::arg("fast_math"));
}
