// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Thin C++ adapters around the HipKittens BF16 grouped GEMM dispatchers.
// The HK BF16 kernel translation unit declares ``grouped_layout_globals``
// at global scope, which collides with the FP8 kernel's same-named struct.
// We pre-include the kittens headers at global scope (so kittens.cuh's
// own include guards make the in-cpp include a no-op) and then wrap the
// kernel cpp inside a named namespace ``hk_bf16_kernel`` to isolate it.

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#define PRIMUS_TURBO_HK_INTEGRATION
namespace hk_bf16_kernel {
#include "kernel_bf16_dynamic.cpp"
}  // namespace hk_bf16_kernel

namespace primus_turbo {
namespace hk {

namespace {
using BFGL = hk_bf16_kernel::_gl;
inline BFGL make_bf16_gl(void* ptr, int b, int d, int r, int c) {
    return BFGL(reinterpret_cast<kittens::bf16*>(ptr), b, d, r, c);
}
}  // namespace

// BF16 grouped RCR (forward + dgrad-via-RCR reroute).
// a: [M_total, K] bf16, b: [G, N, K] bf16, c: [M_total, N] bf16.
void hk_grouped_rcr_bf16(
    void* a_ptr, int M_total, int aK,
    void* b_ptr, int G_b, int bN, int bK,
    void* c_ptr, int cM, int cN,
    const int64_t* group_offs_ptr, int G,
    int group_m, int m_per_group, int num_xcds,
    hipStream_t stream)
{
    hk_bf16_kernel::grouped_layout_globals g{
        make_bf16_gl(a_ptr, 1, 1, M_total, aK),
        make_bf16_gl(b_ptr, 1, G_b, bN, bK),
        make_bf16_gl(c_ptr, 1, 1, cM, cN),
        group_offs_ptr, stream,
        G, /*n*/0, /*k*/0, /*ki*/0, /*bpc*/0,
        group_m, num_xcds, /*M_total*/0,
        /*fast_n*/0, /*fast_k*/0,
        m_per_group,
        /*tile_counter*/nullptr,
    };
    hk_bf16_kernel::dispatch_grouped<hk_bf16_kernel::Layout::RCR>(g);
}

// BF16 grouped RRR (alternative dgrad path).
void hk_grouped_rrr_bf16(
    void* a_ptr, int M_total, int aK,
    void* b_ptr, int G_b, int bK_, int bN,
    void* c_ptr, int cM, int cN,
    const int64_t* group_offs_ptr, int G,
    int group_m, int m_per_group, int num_xcds,
    hipStream_t stream)
{
    hk_bf16_kernel::grouped_layout_globals g{
        make_bf16_gl(a_ptr, 1, 1, M_total, aK),
        make_bf16_gl(b_ptr, 1, G_b, bK_, bN),
        make_bf16_gl(c_ptr, 1, 1, cM, cN),
        group_offs_ptr, stream,
        G, /*n*/0, /*k*/0, /*ki*/0, /*bpc*/0,
        group_m, num_xcds, /*M_total*/0,
        /*fast_n*/0, /*fast_k*/0,
        m_per_group,
        /*tile_counter*/nullptr,
    };
    hk_bf16_kernel::dispatch_grouped<hk_bf16_kernel::Layout::RRR>(g);
}

// BF16 grouped variable-K CRR (wgrad).
// a: [M_total, N] bf16 (grad_out), b: [M_total, K] bf16 (x),
// c: [G, N, K] bf16 (grad_w).
void hk_grouped_var_k_crr_bf16(
    void* a_ptr, int M_total, int aN,
    void* b_ptr, int bM_, int bK,
    void* c_ptr, int G_c, int cN, int cK,
    const int64_t* group_offs_ptr, int G,
    int group_m, int num_xcds,
    hipStream_t stream)
{
    hk_bf16_kernel::grouped_var_k_layout_globals g{
        make_bf16_gl(a_ptr, 1, 1, M_total, aN),
        make_bf16_gl(b_ptr, 1, 1, bM_, bK),
        make_bf16_gl(c_ptr, 1, G_c, cN, cK),
        group_offs_ptr, stream,
        G, /*M_total*/0, /*n*/0, /*k*/0,
        group_m, num_xcds,
        /*bpr*/0, /*bpc*/0,
        /*fast_n*/0, /*fast_k*/0,
    };
    hk_bf16_kernel::dispatch_grouped_var_k(g);
}

// BF16 dense GEMM. layout_id: 0=RCR, 1=RRR, 2=CRR.
void hk_gemm_bf16(
    void* a_ptr, int aR, int aC,
    void* b_ptr, int bR, int bC,
    void* c_ptr, int cR, int cC,
    int layout_id, int group_m, int num_xcds,
    hipStream_t stream)
{
    hk_bf16_kernel::layout_globals g{
        make_bf16_gl(a_ptr, 1, 1, aR, aC),
        make_bf16_gl(b_ptr, 1, 1, bR, bC),
        make_bf16_gl(c_ptr, 1, 1, cR, cC),
        stream,
        /*m,n,k,ki,bpr,bpc*/0, 0, 0, 0, 0, 0,
        group_m, num_xcds,
        /*fast_m,fast_n,fast_k*/0, 0, 0,
    };
    using L = hk_bf16_kernel::Layout;
    switch (layout_id) {
        case 0: hk_bf16_kernel::dispatch_gemm<L::RCR>(g); break;
        case 1: hk_bf16_kernel::dispatch_gemm<L::RRR>(g); break;
        default: hk_bf16_kernel::dispatch_gemm<L::CRR>(g); break;
    }
}

}  // namespace hk
}  // namespace primus_turbo
