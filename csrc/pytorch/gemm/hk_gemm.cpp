// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// PyTorch ATen ops for the HipKittens dense GEMM adapters declared in
// csrc/kernels/grouped_gemm/HipKittens/hk_grouped_gemm{,_bf16}_gfx950.cu
// (the FP8 / BF16 dense entry points were appended there to share the
// kernel translation unit with the grouped path).

#include "pytorch/extensions.h"
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>

namespace primus_turbo { namespace hk {
void hk_gemm_fp8(
    void* a_ptr, int aR, int aC,
    void* b_ptr, int bR, int bC,
    void* c_ptr, int cR, int cC,
    float sa, float sb,
    const float* sa_dev, const float* sb_dev,
    int layout_id, int group_m,
    hipStream_t stream);

void hk_gemm_bf16(
    void* a_ptr, int aR, int aC,
    void* b_ptr, int bR, int bC,
    void* c_ptr, int cR, int cC,
    int layout_id, int group_m, int num_xcds,
    hipStream_t stream);
}}  // namespace primus_turbo::hk

namespace {
inline hipStream_t current_stream() {
    return c10::hip::getCurrentHIPStream().stream();
}

// layout strings come from the BackendType-side config (rcr/rrr/crr).
// Map to the kernel-side enum values: RCR=0, RRR=1, CRR=2.
inline int layout_to_id(const std::string& layout) {
    if (layout == "rcr") return 0;
    if (layout == "rrr") return 1;
    return 2;  // crr
}
}  // namespace

namespace primus_turbo::pytorch {

at::Tensor hk_gemm_bf16(at::Tensor &a, at::Tensor &b,
                        const std::string &layout,
                        int64_t group_m, int64_t num_xcds) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(),
                "hk_gemm_bf16: tensors must be on cuda");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "hk_gemm_bf16: a, b must be contiguous");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
                "hk_gemm_bf16: a / b must be 2-D");
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16,
                "hk_gemm_bf16: a, b must be bf16");
    const int aR = static_cast<int>(a.size(0));
    const int aC = static_cast<int>(a.size(1));
    const int bR = static_cast<int>(b.size(0));
    const int bC = static_cast<int>(b.size(1));
    // Output (M,N): inferred from layout — RCR/RRR: M=aR, N=(rcr?bR:bC);
    // CRR: M=aC, N=bC.
    const int id = layout_to_id(layout);
    int cR, cC;
    if (id == 0)      { cR = aR; cC = bR; }   // RCR
    else if (id == 1) { cR = aR; cC = bC; }   // RRR
    else              { cR = aC; cC = bC; }   // CRR
    auto out = at::empty({cR, cC}, a.options());
    primus_turbo::hk::hk_gemm_bf16(
        a.data_ptr(), aR, aC,
        b.data_ptr(), bR, bC,
        out.data_ptr(), cR, cC,
        id, static_cast<int>(group_m), static_cast<int>(num_xcds),
        current_stream());
    return out;
}

at::Tensor hk_gemm_fp8(at::Tensor &a, at::Tensor &b,
                       at::Tensor &a_scales, at::Tensor &b_scales,
                       const std::string &layout, int64_t group_m,
                       at::ScalarType out_dtype) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && a_scales.is_cuda() && b_scales.is_cuda(),
                "hk_gemm_fp8: tensors must be on cuda");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "hk_gemm_fp8: a, b must be contiguous");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
                "hk_gemm_fp8: a / b must be 2-D");
    TORCH_CHECK(a_scales.numel() == 1 && b_scales.numel() == 1,
                "hk_gemm_fp8: tensorwise scales (numel==1) only");
    const int aR = static_cast<int>(a.size(0));
    const int aC = static_cast<int>(a.size(1));
    const int bR = static_cast<int>(b.size(0));
    const int bC = static_cast<int>(b.size(1));
    const int id = layout_to_id(layout);
    int cR, cC;
    if (id == 0)      { cR = aR; cC = bR; }
    else if (id == 1) { cR = aR; cC = bC; }
    else              { cR = aC; cC = bC; }
    auto out = at::empty({cR, cC}, a.options().dtype(out_dtype));
    primus_turbo::hk::hk_gemm_fp8(
        a.data_ptr(), aR, aC,
        b.data_ptr(), bR, bC,
        out.data_ptr(), cR, cC,
        /*sa*/0.f, /*sb*/0.f,
        a_scales.data_ptr<float>(), b_scales.data_ptr<float>(),
        id, static_cast<int>(group_m),
        current_stream());
    return out;
}

}  // namespace primus_turbo::pytorch
