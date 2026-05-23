// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// PyTorch ATen ops for the HipKittens FP8 grouped GEMM adapters declared
// in csrc/kernels/grouped_gemm/HipKittens/hk_grouped_gemm_gfx950.cu.

#include "pytorch/extensions.h"
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>

namespace primus_turbo { namespace hk {
// Forward declarations of the gfx950 kernel-level wrappers.
void hk_grouped_rcr_fp8(
    const void* a_ptr, int M_total, int aK,
    const void* b_ptr, int G_b, int bN, int bK,
    void* c_ptr,       int cM, int cN,
    const float* sa_ptr, const float* sb_ptr,
    const int64_t* group_offs_ptr, int G,
    int group_m, int m_per_group, int num_xcds,
    int num_slots, int chunk_size, int fuse_ktail_off,
    int bn_block,
    hipStream_t stream);

void hk_grouped_rcr_fp8_new(
    const void* a_ptr, int M_total, int aK,
    const void* b_ptr, int G_b, int bN, int bK,
    void* c_ptr,       int cM, int cN,
    const float* sa_ptr, const float* sb_ptr,
    const int64_t* group_offs_ptr, int G,
    int group_m, int m_per_group, int num_xcds,
    int num_slots, int chunk_size, int fuse_ktail_off,
    int bn_block,
    hipStream_t stream);

void hk_grouped_var_k_crr_fp8(
    const void* a_ptr, int M_total, int aK,
    const void* b_ptr, int bM_, int bN,
    void* c_ptr,       int G_c, int cN, int cK,
    const float* sa_ptr, const float* sb_ptr,
    const int64_t* group_offs_ptr, int G,
    int group_m, int num_xcds,
    hipStream_t stream);

void hk_grouped_rrr_fp8(
    const void* a_ptr, int M_total, int aK,
    const void* b_ptr, int G_b, int bK_, int bN,
    void* c_ptr,       int cM, int cN,
    const float* sa_ptr, const float* sb_ptr,
    const int64_t* group_offs_ptr, int G,
    int group_m, int m_per_group, int num_xcds,
    int bn_block,
    hipStream_t stream);

void hk_grouped_rcr_bf16(
    void* a_ptr, int M_total, int aK,
    void* b_ptr, int G_b, int bN, int bK,
    void* c_ptr, int cM, int cN,
    const int64_t* group_offs_ptr, int G,
    int group_m, int m_per_group, int num_xcds,
    hipStream_t stream);

void hk_grouped_rrr_bf16(
    void* a_ptr, int M_total, int aK,
    void* b_ptr, int G_b, int bK_, int bN,
    void* c_ptr, int cM, int cN,
    const int64_t* group_offs_ptr, int G,
    int group_m, int m_per_group, int num_xcds,
    hipStream_t stream);

void hk_grouped_var_k_crr_bf16(
    void* a_ptr, int M_total, int aN,
    void* b_ptr, int bM_, int bK,
    void* c_ptr, int G_c, int cN, int cK,
    const int64_t* group_offs_ptr, int G,
    int group_m, int num_xcds,
    hipStream_t stream);
}}  // namespace primus_turbo::hk

namespace {
inline hipStream_t current_stream() {
    return c10::hip::getCurrentHIPStream().stream();
}
}

namespace primus_turbo::pytorch {

at::Tensor hk_grouped_rcr_fp8(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                              at::Tensor &b_scales, at::Tensor &group_offs,
                              int64_t group_m, int64_t m_per_group, int64_t num_xcds,
                              at::ScalarType out_dtype, int64_t bn_block) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && a_scales.is_cuda() && b_scales.is_cuda(),
                "hk_grouped_rcr_fp8: tensors must be on cuda");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && group_offs.is_contiguous(),
                "hk_grouped_rcr_fp8: a, b, group_offs must be contiguous");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 3,
                "hk_grouped_rcr_fp8: a [M,K] / b [G,N,K] required");
    const int M_total = static_cast<int>(a.size(0));
    const int aK      = static_cast<int>(a.size(1));
    const int G_b     = static_cast<int>(b.size(0));
    const int bN      = static_cast<int>(b.size(1));
    const int bK      = static_cast<int>(b.size(2));
    const int G       = static_cast<int>(group_offs.numel()) - 1;
    auto out = at::empty({M_total, bN}, a.options().dtype(out_dtype));
    primus_turbo::hk::hk_grouped_rcr_fp8(
        a.data_ptr(), M_total, aK,
        b.data_ptr(), G_b, bN, bK,
        out.data_ptr(), M_total, bN,
        a_scales.data_ptr<float>(), b_scales.data_ptr<float>(),
        group_offs.data_ptr<int64_t>(), G,
        static_cast<int>(group_m), static_cast<int>(m_per_group),
        static_cast<int>(num_xcds),
        /*num_slots*/0, /*chunk_size*/0, /*fuse_ktail_off*/0,
        static_cast<int>(bn_block),
        current_stream());
    return out;
}

// Campaign D v2 entry point. Mirrors hk_grouped_rcr_fp8; routes through
// hk_fp8_kernel_v2 namespace (kernel_fp8_layouts2.cpp).
at::Tensor hk_grouped_rcr_fp8_new(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                  at::Tensor &b_scales, at::Tensor &group_offs,
                                  int64_t group_m, int64_t m_per_group, int64_t num_xcds,
                                  at::ScalarType out_dtype, int64_t bn_block) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && a_scales.is_cuda() && b_scales.is_cuda(),
                "hk_grouped_rcr_fp8_new: tensors must be on cuda");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && group_offs.is_contiguous(),
                "hk_grouped_rcr_fp8_new: a, b, group_offs must be contiguous");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 3,
                "hk_grouped_rcr_fp8_new: a [M,K] / b [G,N,K] required");
    const int M_total = static_cast<int>(a.size(0));
    const int aK      = static_cast<int>(a.size(1));
    const int G_b     = static_cast<int>(b.size(0));
    const int bN      = static_cast<int>(b.size(1));
    const int bK      = static_cast<int>(b.size(2));
    const int G       = static_cast<int>(group_offs.numel()) - 1;
    auto out = at::empty({M_total, bN}, a.options().dtype(out_dtype));
    primus_turbo::hk::hk_grouped_rcr_fp8_new(
        a.data_ptr(), M_total, aK,
        b.data_ptr(), G_b, bN, bK,
        out.data_ptr(), M_total, bN,
        a_scales.data_ptr<float>(), b_scales.data_ptr<float>(),
        group_offs.data_ptr<int64_t>(), G,
        static_cast<int>(group_m), static_cast<int>(m_per_group),
        static_cast<int>(num_xcds),
        /*num_slots*/0, /*chunk_size*/0, /*fuse_ktail_off*/0,
        static_cast<int>(bn_block),
        current_stream());
    return out;
}

at::Tensor hk_grouped_var_k_crr_fp8(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                    at::Tensor &b_scales, at::Tensor &group_offs,
                                    int64_t group_m, int64_t num_xcds,
                                    at::ScalarType out_dtype) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(),
                "hk_grouped_var_k_crr_fp8: tensors must be on cuda");
    const int M_total = static_cast<int>(a.size(0));
    const int aK      = static_cast<int>(a.size(1));
    const int bM_     = static_cast<int>(b.size(0));
    const int bN      = static_cast<int>(b.size(1));
    const int G       = static_cast<int>(group_offs.numel()) - 1;
    // c shape is [G, N, K] where N=aK and K=bN here? Actually for var-K wgrad:
    //   a (grad_out): [M_total, N_fwd], b (x): [M_total, K_fwd], c (grad_w): [G, N_fwd, K_fwd]
    // Caller passes a=grad_out, b=x, so c = [G, a.size(1), b.size(1)].
    auto out = at::empty({G, aK, bN}, a.options().dtype(out_dtype));
    primus_turbo::hk::hk_grouped_var_k_crr_fp8(
        a.data_ptr(), M_total, aK,
        b.data_ptr(), bM_, bN,
        out.data_ptr(), G, aK, bN,
        a_scales.data_ptr<float>(), b_scales.data_ptr<float>(),
        group_offs.data_ptr<int64_t>(), G,
        static_cast<int>(group_m), static_cast<int>(num_xcds),
        current_stream());
    return out;
}

// FP8 grouped RRR (dgrad direct path — no transpose reroute).
// a (grad_out): [M_total, N_inner] fp8e4m3
// b (weight):   [G, N_inner, K_out] fp8e4m3 (already in row-major K-then-N order
//                                              from the kernel's perspective)
// c (grad_a):   [M_total, K_out] out_dtype
at::Tensor hk_grouped_rrr_fp8(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                              at::Tensor &b_scales, at::Tensor &group_offs,
                              int64_t group_m, int64_t m_per_group, int64_t num_xcds,
                              at::ScalarType out_dtype, int64_t bn_block) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && group_offs.is_cuda(),
                "hk_grouped_rrr_fp8: tensors must be on cuda");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && group_offs.is_contiguous(),
                "hk_grouped_rrr_fp8: a, b, group_offs must be contiguous");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 3,
                "hk_grouped_rrr_fp8: a [M,K] / b [G,K,N] required");
    const int M_total = static_cast<int>(a.size(0));
    const int aK      = static_cast<int>(a.size(1));
    const int G_b     = static_cast<int>(b.size(0));
    const int bK_     = static_cast<int>(b.size(1));
    const int bN      = static_cast<int>(b.size(2));
    const int G       = static_cast<int>(group_offs.numel()) - 1;
    auto out = at::empty({M_total, bN}, a.options().dtype(out_dtype));
    primus_turbo::hk::hk_grouped_rrr_fp8(
        a.data_ptr(), M_total, aK,
        b.data_ptr(), G_b, bK_, bN,
        out.data_ptr(), M_total, bN,
        a_scales.data_ptr<float>(), b_scales.data_ptr<float>(),
        group_offs.data_ptr<int64_t>(), G,
        static_cast<int>(group_m), static_cast<int>(m_per_group),
        static_cast<int>(num_xcds),
        static_cast<int>(bn_block),
        current_stream());
    return out;
}

at::Tensor hk_grouped_rcr_bf16(at::Tensor &a, at::Tensor &b, at::Tensor &group_offs,
                               int64_t group_m, int64_t m_per_group, int64_t num_xcds) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && group_offs.is_cuda(),
                "hk_grouped_rcr_bf16: tensors must be on cuda");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && group_offs.is_contiguous(),
                "hk_grouped_rcr_bf16: a, b, group_offs must be contiguous");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 3,
                "hk_grouped_rcr_bf16: a [M,K] / b [G,N,K] required");
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16,
                "hk_grouped_rcr_bf16: a, b must be bf16");
    const int M_total = static_cast<int>(a.size(0));
    const int aK      = static_cast<int>(a.size(1));
    const int G_b     = static_cast<int>(b.size(0));
    const int bN      = static_cast<int>(b.size(1));
    const int bK      = static_cast<int>(b.size(2));
    const int G       = static_cast<int>(group_offs.numel()) - 1;
    auto out = at::empty({M_total, bN}, a.options());
    primus_turbo::hk::hk_grouped_rcr_bf16(
        a.data_ptr(), M_total, aK,
        b.data_ptr(), G_b, bN, bK,
        out.data_ptr(), M_total, bN,
        group_offs.data_ptr<int64_t>(), G,
        static_cast<int>(group_m), static_cast<int>(m_per_group),
        static_cast<int>(num_xcds),
        current_stream());
    return out;
}

at::Tensor hk_grouped_rrr_bf16(at::Tensor &a, at::Tensor &b, at::Tensor &group_offs,
                               int64_t group_m, int64_t m_per_group, int64_t num_xcds) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && group_offs.is_cuda(),
                "hk_grouped_rrr_bf16: tensors must be on cuda");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && group_offs.is_contiguous(),
                "hk_grouped_rrr_bf16: a, b, group_offs must be contiguous");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 3,
                "hk_grouped_rrr_bf16: a [M,K] / b [G,K,N] required");
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16,
                "hk_grouped_rrr_bf16: a, b must be bf16");
    const int M_total = static_cast<int>(a.size(0));
    const int aK      = static_cast<int>(a.size(1));
    const int G_b     = static_cast<int>(b.size(0));
    const int bK_     = static_cast<int>(b.size(1));
    const int bN      = static_cast<int>(b.size(2));
    const int G       = static_cast<int>(group_offs.numel()) - 1;
    auto out = at::empty({M_total, bN}, a.options());
    primus_turbo::hk::hk_grouped_rrr_bf16(
        a.data_ptr(), M_total, aK,
        b.data_ptr(), G_b, bK_, bN,
        out.data_ptr(), M_total, bN,
        group_offs.data_ptr<int64_t>(), G,
        static_cast<int>(group_m), static_cast<int>(m_per_group),
        static_cast<int>(num_xcds),
        current_stream());
    return out;
}

at::Tensor hk_grouped_var_k_crr_bf16(at::Tensor &a, at::Tensor &b, at::Tensor &group_offs,
                                     int64_t group_m, int64_t num_xcds) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(),
                "hk_grouped_var_k_crr_bf16: tensors must be on cuda");
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16,
                "hk_grouped_var_k_crr_bf16: a, b must be bf16");
    const int M_total = static_cast<int>(a.size(0));
    const int aN      = static_cast<int>(a.size(1));
    const int bM_     = static_cast<int>(b.size(0));
    const int bK      = static_cast<int>(b.size(1));
    const int G       = static_cast<int>(group_offs.numel()) - 1;
    // Output [G, N_fwd, K_fwd] = [G, a.size(1), b.size(1)].
    auto out = at::empty({G, aN, bK}, a.options());
    primus_turbo::hk::hk_grouped_var_k_crr_bf16(
        a.data_ptr(), M_total, aN,
        b.data_ptr(), bM_, bK,
        out.data_ptr(), G, aN, bK,
        group_offs.data_ptr<int64_t>(), G,
        static_cast<int>(group_m), static_cast<int>(num_xcds),
        current_stream());
    return out;
}
}  // namespace primus_turbo::pytorch
