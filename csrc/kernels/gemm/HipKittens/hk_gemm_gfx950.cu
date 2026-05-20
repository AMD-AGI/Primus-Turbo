// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Thin C++ adapters around the HipKittens FP8 + BF16 dense GEMM dispatchers.
// Pulls the kernel templates in directly via #include of the two HipKittens
// kernel translation units; the PYBIND11_MODULE blocks at the bottom of
// each kernel .cpp are gated behind PRIMUS_TURBO_HK_INTEGRATION so they are
// omitted from this build.
//
// Both HK kernel .cpp files declare helper functions (e.g. load_bf16_scalar)
// over identical types (kittens::gl<bf16, ...>). With a single-TU dual
// include those helpers collide via ADL — so wrap each .cpp in its own
// named namespace (hk_fp8_kernel / hk_bf16_kernel). The grouped adapter
// file (hk_grouped_gemm_gfx950.cu) mirrors this layout.

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#define PRIMUS_TURBO_HK_INTEGRATION

// Outer anonymous namespace gives each TU its own private copy of the HK
// kernel symbols — both the dense and grouped .cu's include the same .cpp
// files, so without per-TU internal linkage we'd get duplicate-symbol link
// errors. Each .cpp defines BOTH dense + grouped dispatchers + __global__
// kernels so we cannot split include responsibilities between dense and
// grouped TUs without rewriting HK source.
namespace {
namespace hk_fp8_kernel {
#include "kernel_fp8_layouts.cpp"   // fp8: dispatch<Layout::*>, layout_globals
}  // namespace hk_fp8_kernel

namespace hk_bf16_kernel {
#include "kernel_bf16_dynamic.cpp"  // bf16: dispatch_gemm<Layout::*>, layout_globals
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
    hk_fp8_kernel::layout_globals g{
        make_fp8_gl(a_ptr, 1, 1, aR, aC),
        make_fp8_gl(b_ptr, 1, 1, bR, bC),
        make_bf16_gl_for_fp8(c_ptr, 1, 1, cR, cC),
        sa, sb,
        stream,
        /*m,n,k,bpr,bpc,ki,fast_m,fast_n,fast_k*/
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        group_m,
        sa_dev, sb_dev,
    };
    using L = hk_fp8_kernel::Layout;
    switch (layout_id) {
        case 0: hk_fp8_kernel::dispatch<L::RCR>(g); break;
        case 1: hk_fp8_kernel::dispatch<L::RRR>(g); break;
        default: hk_fp8_kernel::dispatch<L::CRR>(g); break;
    }
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
