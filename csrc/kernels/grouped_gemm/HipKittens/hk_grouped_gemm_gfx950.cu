// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Thin C++ adapters around the HipKittens FP8 + BF16 grouped GEMM
// dispatchers. Pulls the kernel templates in directly via #include of the
// two HipKittens kernel translation units; the PYBIND11_MODULE blocks at
// the bottom of each kernel .cpp are gated behind PRIMUS_TURBO_HK_INTEGRATION
// so they are omitted from this build.
//
// Both HK kernel .cpp files declare helper functions over identical types
// (kittens::gl<bf16, ...>), so with a single-TU dual include those helpers
// collide via ADL. Each .cpp is wrapped in its own named namespace
// (hk_fp8_kernel / hk_bf16_kernel). The dense adapter file
// (csrc/kernels/gemm/HipKittens/hk_gemm_gfx950.cu) mirrors this layout.

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#define PRIMUS_TURBO_HK_INTEGRATION

// Outer anonymous namespace gives each TU its own private copy of the HK
// kernel symbols — see hk_gemm_gfx950.cu for the explanation.
namespace {
namespace hk_fp8_kernel {
#include "kernel_fp8_layouts.cpp"   // fp8 v1: dispatch_grouped_*, grouped_layout_globals
}  // namespace hk_fp8_kernel

// v2 path — fresh mxfp8-pattern rewrite for RCR only (PR #330 adapt 8-warp).
// Provides dispatch_grouped_rcr_v2 + compat grouped_layout_globals_v2.
namespace hk_fp8_kernel_v2 {
#include "kernel_fp8_layouts2.cpp"
}  // namespace hk_fp8_kernel_v2

namespace hk_bf16_kernel {
#include "kernel_bf16_dynamic.cpp"  // bf16: dispatch_grouped, dispatch_grouped_var_k
}  // namespace hk_bf16_kernel
}  // anonymous

namespace primus_turbo {
namespace hk {

namespace {
inline hk_fp8_kernel::_gl_fp8 make_fp8_gl(const void* ptr, int b, int d, int r, int c) {
    return hk_fp8_kernel::_gl_fp8(
        reinterpret_cast<hk_fp8_kernel::fp8e4m3*>(const_cast<void*>(ptr)), b, d, r, c);
}
inline hk_fp8_kernel::_gl_bf16 make_bf16_gl_for_fp8(void* ptr, int b, int d, int r, int c) {
    return hk_fp8_kernel::_gl_bf16(
        reinterpret_cast<hk_fp8_kernel::bf16*>(ptr), b, d, r, c);
}
using BFGL = hk_bf16_kernel::_gl;
inline BFGL make_bf16_gl(void* ptr, int b, int d, int r, int c) {
    return BFGL(reinterpret_cast<kittens::bf16*>(ptr), b, d, r, c);
}
}  // namespace

// ============================================================================
// FP8 grouped adapters.
// ============================================================================

// FP8 grouped RCR (forward + dgrad-via-H4 reroute).
// a: [M_total, K] fp8e4m3, b: [G, N, K] fp8e4m3, c: [M_total, N] bf16.
void hk_grouped_rcr_fp8(
    const void* a_ptr, int M_total, int aK,
    const void* b_ptr, int G_b, int bN, int bK,
    void* c_ptr,       int cM, int cN,
    const float* sa_ptr, const float* sb_ptr,
    const int64_t* group_offs_ptr, int G,
    int group_m, int m_per_group, int num_xcds,
    int num_slots, int chunk_size, int fuse_ktail_off,
    int bn_block,
    hipStream_t stream)
{
    // R444 INTEGRATION: production path forwards to v2 (kernel_fp8_layouts2.cpp,
    // R167 +8pp baseline + R429/R433 P1.2 progress). Previously v2 was opt-in
    // via `hk_grouped_rcr_fp8_new`, leaving production on v1 — official bench
    // 0/24 ≥ 1.15× reflected v1 perf, not v2.
    hk_fp8_kernel_v2::grouped_layout_globals_v2 g_v2{
        a_ptr, b_ptr, c_ptr,
        M_total, G_b, bN, bK, cM, cN,
        sa_ptr, sb_ptr,
        group_offs_ptr, stream,
        G, group_m, m_per_group, num_xcds,
        num_slots, chunk_size,
    };
    hk_fp8_kernel_v2::dispatch_grouped_rcr_v2(g_v2);
}

// Campaign D v2 entry — fresh mxfp8-pattern RCR kernel (see
// kernel_fp8_layouts2.cpp). Same call signature as hk_grouped_rcr_fp8.
void hk_grouped_rcr_fp8_new(
    const void* a_ptr, int M_total, int aK,
    const void* b_ptr, int G_b, int bN, int bK,
    void* c_ptr,       int cM, int cN,
    const float* sa_ptr, const float* sb_ptr,
    const int64_t* group_offs_ptr, int G,
    int group_m, int m_per_group, int num_xcds,
    int num_slots, int chunk_size, int fuse_ktail_off,
    int bn_block,
    hipStream_t stream)
{
    hk_fp8_kernel_v2::grouped_layout_globals_v2 g_v2{
        a_ptr, b_ptr, c_ptr,
        M_total, G_b, bN, bK, cM, cN,
        sa_ptr, sb_ptr,
        group_offs_ptr, stream,
        G, group_m, m_per_group, num_xcds,
        num_slots, chunk_size,
    };
    hk_fp8_kernel_v2::dispatch_grouped_rcr_v2(g_v2);
}

// R473: v2 RRR entry — mirrors hk_grouped_rrr_fp8 signature but routes through
// v2 dispatcher (kernel_fp8_layouts2.cpp::dispatch_grouped_rrr_v2). Initially
// forwards to v1; future commits replace with vacc body for +8pp R167-pattern.
void hk_grouped_rrr_fp8_new(
    const void* a_ptr, int M_total, int aK,
    const void* b_ptr, int G_b, int bK_, int bN,
    void* c_ptr,       int cM, int cN,
    const float* sa_ptr, const float* sb_ptr,
    const int64_t* group_offs_ptr, int G,
    int group_m, int m_per_group, int num_xcds,
    int bn_block,
    hipStream_t stream)
{
    hk_fp8_kernel_v2::grouped_layout_globals_v2_rrr g_v2{
        a_ptr, b_ptr, c_ptr,
        M_total, G_b, bK_, bN, cM, cN,
        sa_ptr, sb_ptr,
        group_offs_ptr, stream,
        G, group_m, m_per_group, num_xcds,
        /*num_slots*/0, /*chunk_size*/0,
        bn_block,
    };
    hk_fp8_kernel_v2::dispatch_grouped_rrr_v2(g_v2);
}

// R556: production RRR routes through v2 (mirror R444 RCR routing).
// v2 dispatch handles bn=128 fallback to v1 internally (R486).
void hk_grouped_rrr_fp8(
    const void* a_ptr, int M_total, int aK,
    const void* b_ptr, int G_b, int bK_, int bN,
    void* c_ptr,       int cM, int cN,
    const float* sa_ptr, const float* sb_ptr,
    const int64_t* group_offs_ptr, int G,
    int group_m, int m_per_group, int num_xcds,
    int bn_block,
    hipStream_t stream)
{
    hk_fp8_kernel_v2::grouped_layout_globals_v2_rrr g_v2{
        a_ptr, b_ptr, c_ptr,
        M_total, G_b, bK_, bN, cM, cN,
        sa_ptr, sb_ptr,
        group_offs_ptr, stream,
        G, group_m, m_per_group, num_xcds,
        /*num_slots*/0, /*chunk_size*/0,
        bn_block,
    };
    hk_fp8_kernel_v2::dispatch_grouped_rrr_v2(g_v2);
}

// FP8 grouped variable-K CRR (wgrad).
// a: [M_total, K] fp8e4m3, b: [M_total, N] fp8e4m3, c: [G, N, K] bf16.
void hk_grouped_var_k_crr_fp8(
    const void* a_ptr, int M_total, int aK,
    const void* b_ptr, int bM_, int bN,
    void* c_ptr,       int G_c, int cN, int cK,
    const float* sa_ptr, const float* sb_ptr,
    const int64_t* group_offs_ptr, int G,
    int group_m, int num_xcds,
    hipStream_t stream)
{
    hk_fp8_kernel::grouped_var_k_layout_globals_fp8 g{
        make_fp8_gl(a_ptr, 1, 1, M_total, aK),
        make_fp8_gl(b_ptr, 1, 1, bM_, bN),
        make_bf16_gl_for_fp8(c_ptr, 1, G_c, cN, cK),
        0.f, 0.f,
        sa_ptr, sb_ptr,
        group_offs_ptr, stream,
        /*G,M_total,n,k,group_m,bpr,bpc,fast_n,fast_k,num_xcds,num_slots,chunk_size*/
        G, /*M_total*/0, /*n*/0, /*k*/0, group_m,
        /*bpr*/0, /*bpc*/0, /*fast_n*/0, /*fast_k*/0,
        num_xcds, /*num_slots*/0, /*chunk_size*/0,
    };
    hk_fp8_kernel::dispatch_grouped_var_k_fp8(g);
}

// ============================================================================
// BF16 grouped adapters (moved from hk_grouped_gemm_bf16_gfx950.cu).
// ============================================================================

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

}  // namespace hk
}  // namespace primus_turbo
