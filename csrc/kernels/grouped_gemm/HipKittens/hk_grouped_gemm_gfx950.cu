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
#include "kernel_fp8_layouts.cpp"   // fp8: dispatch_grouped_*, grouped_layout_globals
}  // namespace hk_fp8_kernel

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
    hk_fp8_kernel::grouped_layout_globals g{
        make_fp8_gl(a_ptr, 1, 1, M_total, aK),
        make_fp8_gl(b_ptr, 1, G_b, bN, bK),
        make_bf16_gl_for_fp8(c_ptr, 1, 1, cM, cN),
        0.f, 0.f,
        sa_ptr, sb_ptr,
        group_offs_ptr, stream,
        G, /*n*/0, /*k*/0, /*ki*/0, /*bpc*/0,
        group_m, num_xcds, /*M_total*/0,
        /*fast_n*/0, /*fast_k*/0,
        m_per_group, num_slots, chunk_size, fuse_ktail_off,
        /*sk_split_n*/0, /*sk_partial_buf*/nullptr,
        bn_block,
    };
    hk_fp8_kernel::dispatch_grouped_rcr(g);
}

// FP8 grouped RRR (dense-style, alternative dgrad path).
// bn_block: 0 (default) = BLK_M=BLK_N=256; 128 = bn128 variant (BLK_M=256, BLK_N=128)
// — fewer accumulators + zero spill; see HK 9590230d.
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
    hk_fp8_kernel::grouped_layout_globals g{
        make_fp8_gl(a_ptr, 1, 1, M_total, aK),
        make_fp8_gl(b_ptr, 1, G_b, bK_, bN),
        make_bf16_gl_for_fp8(c_ptr, 1, 1, cM, cN),
        0.f, 0.f,
        sa_ptr, sb_ptr,
        group_offs_ptr, stream,
        G, /*n*/0, /*k*/0, /*ki*/0, /*bpc*/0,
        group_m, num_xcds, /*M_total*/0,
        /*fast_n*/0, /*fast_k*/0,
        m_per_group, /*num_slots*/0, /*chunk_size*/0, /*fuse_ktail_off*/0,
        /*sk_split_n*/0, nullptr,
        bn_block,
    };
    hk_fp8_kernel::dispatch_grouped_rrr(g);
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
