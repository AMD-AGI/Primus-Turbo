// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Thin C++ adapters around the HipKittens FP8 grouped GEMM dispatchers.
// Pulls the kernel templates in directly via #include of the HipKittens
// kernel translation unit; the PYBIND11_MODULE block at the bottom of
// kernel_fp8_layouts.cpp is gated behind PRIMUS_TURBO_HK_INTEGRATION so
// it is omitted from this build.

#define PRIMUS_TURBO_HK_INTEGRATION
#include "kernel_fp8_layouts.cpp"  // brings in dispatch_grouped_rcr, dispatch_grouped_var_k_fp8

namespace primus_turbo {
namespace hk {

namespace {
inline _gl_fp8 make_fp8_gl(const void* ptr, int b, int d, int r, int c) {
    return _gl_fp8(reinterpret_cast<fp8e4m3*>(const_cast<void*>(ptr)), b, d, r, c);
}
inline _gl_bf16 make_bf16_gl(void* ptr, int b, int d, int r, int c) {
    return _gl_bf16(reinterpret_cast<bf16*>(ptr), b, d, r, c);
}
}  // namespace

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
    grouped_layout_globals g{
        make_fp8_gl(a_ptr, 1, 1, M_total, aK),
        make_fp8_gl(b_ptr, 1, G_b, bN, bK),
        make_bf16_gl(c_ptr, 1, 1, cM, cN),
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
    dispatch_grouped_rcr(g);
}

// FP8 grouped RRR (dense-style, alternative dgrad path).
void hk_grouped_rrr_fp8(
    const void* a_ptr, int M_total, int aK,
    const void* b_ptr, int G_b, int bK_, int bN,
    void* c_ptr,       int cM, int cN,
    const float* sa_ptr, const float* sb_ptr,
    const int64_t* group_offs_ptr, int G,
    int group_m, int m_per_group, int num_xcds,
    hipStream_t stream)
{
    grouped_layout_globals g{
        make_fp8_gl(a_ptr, 1, 1, M_total, aK),
        make_fp8_gl(b_ptr, 1, G_b, bK_, bN),
        make_bf16_gl(c_ptr, 1, 1, cM, cN),
        0.f, 0.f,
        sa_ptr, sb_ptr,
        group_offs_ptr, stream,
        G, /*n*/0, /*k*/0, /*ki*/0, /*bpc*/0,
        group_m, num_xcds, /*M_total*/0,
        /*fast_n*/0, /*fast_k*/0,
        m_per_group, /*num_slots*/0, /*chunk_size*/0, /*fuse_ktail_off*/0,
        /*sk_split_n*/0, nullptr,
    };
    dispatch_grouped_rrr(g);
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
    grouped_var_k_layout_globals_fp8 g{
        make_fp8_gl(a_ptr, 1, 1, M_total, aK),
        make_fp8_gl(b_ptr, 1, 1, bM_, bN),
        make_bf16_gl(c_ptr, 1, G_c, cN, cK),
        0.f, 0.f,
        sa_ptr, sb_ptr,
        group_offs_ptr, stream,
        /*G,M_total,n,k,group_m,bpr,bpc,fast_n,fast_k,num_xcds,num_slots,chunk_size*/
        G, /*M_total*/0, /*n*/0, /*k*/0, group_m,
        /*bpr*/0, /*bpc*/0, /*fast_n*/0, /*fast_k*/0,
        num_xcds, /*num_slots*/0, /*chunk_size*/0,
    };
    dispatch_grouped_var_k_fp8(g);
}

// FP8 dense GEMM. layout_id: 0=RCR, 1=RRR, 2=CRR. Scales may be host
// floats (sa/sb) OR device pointers (sa_dev/sb_dev non-null) — when
// device-side, the kernel epilogue reads them and we skip the .item()
// stream sync that would otherwise cost ~18us per call on small shapes.
void hk_gemm_fp8(
    void* a_ptr, int aR, int aC,
    void* b_ptr, int bR, int bC,
    void* c_ptr, int cR, int cC,
    float sa, float sb,
    const float* sa_dev, const float* sb_dev,
    int layout_id, int group_m,
    hipStream_t stream)
{
    layout_globals g{
        make_fp8_gl(a_ptr, 1, 1, aR, aC),
        make_fp8_gl(b_ptr, 1, 1, bR, bC),
        make_bf16_gl(c_ptr, 1, 1, cR, cC),
        sa, sb,
        stream,
        /*m,n,k,bpr,bpc,ki,fast_m,fast_n,fast_k*/
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        group_m,
        sa_dev, sb_dev,
    };
    switch (layout_id) {
        case 0: dispatch<Layout::RCR>(g); break;
        case 1: dispatch<Layout::RRR>(g); break;
        default: dispatch<Layout::CRR>(g); break;
    }
}

}  // namespace hk
}  // namespace primus_turbo
