// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// Self-contained backend that launches hand-tuned AMDGCN assembly grouped-GEMM
// kernels shipped as prebuilt code objects (.co). The kernels come from
// mawad-amd/ggemm-asm (persistent-fwd-asm-optimization, wgrad-asm-optimization)
// and are specialized for the gpt-oss MoE FP8 call sites on MI355X (gfx950):
//
//   FWD/DGRAD: lhs=(M,K) rhs=(E,N,K) out=(M,N)   kernel _grouped_fp8_persistent_gemm_kernel
//   WGRAD:     lhs=(M,OUT_M) rhs=(M,OUT_N) out=(E,OUT_M,OUT_N)
//              kernel _grouped_variable_k_gemm_kernel
//
// Both kernels use the same 96-byte flat kernarg layout and are tensorwise-scaled
// (one fp32 scalar per operand). Selection of the .co file is by shape.
//
// This file registers two ops into the existing primus_turbo_cpp_extension library
// via TORCH_LIBRARY_FRAGMENT, so no edits to bindings_pytorch.cpp are required.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#include "pytorch/extensions.h"

namespace primus_turbo::pytorch {

namespace {

#define ASM_CO_HIP_CHECK(call)                                                                     \
    do {                                                                                           \
        hipError_t _err = (call);                                                                  \
        TORCH_CHECK(_err == hipSuccess, "asm_co grouped_gemm HIP error ", _err, " at ", __FILE__,  \
                    ":", __LINE__, ": ", hipGetErrorString(_err));                                 \
    } while (0)

// Flat kernarg buffer layout (96 bytes), identical for both kernels:
//   0  ptr lhs        8  ptr rhs        16 ptr out
//   24 ptr lhs_scale  32 ptr rhs_scale  40 ptr group_offs
//   48 i32 G          52 i32 dim1       56 i32 dim2
//   60 i32 stride_lhs 64 i32 stride_rhs 68 i32 stride_out0
//   72 i32 stride_out1 76 i32 stride_out2
//   80 ptr global_scratch (null)  88 ptr profile_scratch (null)
struct alignas(8) KernArgs {
    const void *lhs;
    const void *rhs;
    void       *out;
    const void *lhs_scale;
    const void *rhs_scale;
    const void *group_offs;
    int32_t     G;
    int32_t     dim1;
    int32_t     dim2;
    int32_t     stride_lhs;
    int32_t     stride_rhs;
    int32_t     stride_out0;
    int32_t     stride_out1;
    int32_t     stride_out2;
    const void *global_scratch;
    const void *profile_scratch;
};
static_assert(sizeof(KernArgs) == 96, "asm_co kernarg layout must be 96 bytes");

const char *asm_co_dir() {
    const char *d = std::getenv("PRIMUS_TURBO_ASM_CO_DIR");
    TORCH_CHECK(d != nullptr,
                "PRIMUS_TURBO_ASM_CO_DIR is not set; cannot locate prebuilt ASM .co kernels");
    return d;
}

// Lazily loads a module + function and caches by "<path>::<kernel>".
hipFunction_t get_function(const std::string &co_path, const char *kernel_name) {
    static std::mutex                            mtx;
    static std::map<std::string, hipFunction_t>  fn_cache;
    static std::map<std::string, hipModule_t>    mod_cache;

    std::string key = co_path + "::" + kernel_name;
    std::lock_guard<std::mutex> lock(mtx);
    auto it = fn_cache.find(key);
    if (it != fn_cache.end()) {
        return it->second;
    }
    hipModule_t mod;
    auto        mit = mod_cache.find(co_path);
    if (mit != mod_cache.end()) {
        mod = mit->second;
    } else {
        ASM_CO_HIP_CHECK(hipModuleLoad(&mod, co_path.c_str()));
        mod_cache[co_path] = mod;
    }
    hipFunction_t fn;
    ASM_CO_HIP_CHECK(hipModuleGetFunction(&fn, mod, kernel_name));
    fn_cache[key] = fn;
    return fn;
}

int default_num_cu() {
    int dev = 0;
    ASM_CO_HIP_CHECK(hipGetDevice(&dev));
    hipDeviceProp_t props;
    ASM_CO_HIP_CHECK(hipGetDeviceProperties(&props, dev));
    return props.multiProcessorCount;
}

void launch(hipFunction_t fn, const KernArgs &args, int grid_cu, int block) {
    size_t arg_size = sizeof(KernArgs);
    void  *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, (void *)&args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                       &arg_size, HIP_LAUNCH_PARAM_END};
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    ASM_CO_HIP_CHECK(hipModuleLaunchKernel(fn, grid_cu, 1, 1, block, 1, 1,
                                           /*sharedMemBytes=*/65536, stream, nullptr, config));
}

} // namespace

// FWD / DGRAD: a=(M,K) fp8, b=(E,N,K) fp8, out=(M,N). transB is expected true.
at::Tensor asm_co_grouped_gemm_fp8(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                   at::Tensor &b_scales, at::Tensor &group_lens,
                                   at::Tensor &group_offs, const bool transA, const bool transB,
                                   at::ScalarType out_dtype, const std::string &granularity,
                                   c10::optional<int64_t> num_cu) {
    TORCH_CHECK(!transA, "asm_co fwd kernel requires transA=False");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 3, "asm_co fwd expects a=2D, b=3D");
    TORCH_CHECK(granularity == "TENSORWISE", "asm_co kernels are tensorwise-scaled only");

    const int64_t M = a.size(0);
    const int64_t K = a.size(1);
    const int64_t E = b.size(0);
    const int64_t N = transB ? b.size(1) : b.size(2);

    // Shape-gated kernel selection. combined_v5.co is specialized for K=2880,
    // N in {5760 (gate_up), 2880 (down)}.
    std::string co_path = std::string(asm_co_dir()) + "/combined_v5.co";

    auto out = at::empty({M, N}, a.options().dtype(out_dtype));

    KernArgs args;
    std::memset(&args, 0, sizeof(args));
    args.lhs         = a.data_ptr();
    args.rhs         = b.data_ptr();
    args.out         = out.data_ptr();
    args.lhs_scale   = a_scales.data_ptr();
    args.rhs_scale   = b_scales.data_ptr();
    args.group_offs  = group_offs.data_ptr();
    args.G           = static_cast<int32_t>(E);
    args.dim1        = static_cast<int32_t>(N);
    args.dim2        = static_cast<int32_t>(K);
    args.stride_lhs  = static_cast<int32_t>(K);
    args.stride_rhs  = static_cast<int32_t>(N * K);
    args.stride_out0 = static_cast<int32_t>(K);
    args.stride_out1 = static_cast<int32_t>(N);
    args.stride_out2 = 1;

    int grid_cu = num_cu.has_value() ? static_cast<int>(*num_cu) : default_num_cu();
    hipFunction_t fn = get_function(co_path, "_grouped_fp8_persistent_gemm_kernel");
    launch(fn, args, grid_cu, /*block=*/512);
    return out;
}

at::Tensor asm_co_grouped_gemm_fp8_meta(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                        at::Tensor &b_scales, at::Tensor &group_lens,
                                        at::Tensor &group_offs, const bool transA,
                                        const bool transB, at::ScalarType out_dtype,
                                        const std::string &granularity,
                                        c10::optional<int64_t> num_cu) {
    const int64_t M = a.size(0);
    const int64_t N = transB ? b.size(1) : b.size(2);
    return at::empty({M, N}, a.options().dtype(out_dtype));
}

// WGRAD (variable-K): a=(M,OUT_M) fp8, b=(M,OUT_N) fp8, out=(E,OUT_M,OUT_N).
at::Tensor asm_co_grouped_gemm_fp8_variable_k(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                              at::Tensor &b_scales, at::Tensor &group_lens,
                                              at::Tensor &group_offs, const bool transA,
                                              const bool transB, at::ScalarType out_dtype,
                                              const std::string &granularity,
                                              c10::optional<int64_t> num_cu) {
    TORCH_CHECK(transA && !transB, "asm_co wgrad kernel requires transA=True, transB=False");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "asm_co wgrad expects a=2D, b=2D");
    TORCH_CHECK(granularity == "TENSORWISE", "asm_co kernels are tensorwise-scaled only");

    const int64_t OUT_M = a.size(1);
    const int64_t OUT_N = b.size(1);
    const int64_t E     = group_lens.size(0);

    std::string co_path = std::string(asm_co_dir()) + "/variable_k_wgrad_mega.co";

    auto out = at::empty({E, OUT_M, OUT_N}, a.options().dtype(out_dtype));

    KernArgs args;
    std::memset(&args, 0, sizeof(args));
    args.lhs         = a.data_ptr();
    args.rhs         = b.data_ptr();
    args.out         = out.data_ptr();
    args.lhs_scale   = a_scales.data_ptr();
    args.rhs_scale   = b_scales.data_ptr();
    args.group_offs  = group_offs.data_ptr();
    args.G           = static_cast<int32_t>(E);
    args.dim1        = static_cast<int32_t>(OUT_M);
    args.dim2        = static_cast<int32_t>(OUT_N);
    args.stride_lhs  = static_cast<int32_t>(OUT_M);
    args.stride_rhs  = static_cast<int32_t>(OUT_N);
    args.stride_out0 = static_cast<int32_t>(OUT_M * OUT_N);
    args.stride_out1 = static_cast<int32_t>(OUT_N);
    args.stride_out2 = 1;

    int grid_cu = num_cu.has_value() ? static_cast<int>(*num_cu) : default_num_cu();
    hipFunction_t fn = get_function(co_path, "_grouped_variable_k_gemm_kernel");
    launch(fn, args, grid_cu, /*block=*/512);
    return out;
}

at::Tensor asm_co_grouped_gemm_fp8_variable_k_meta(at::Tensor &a, at::Tensor &b,
                                                   at::Tensor &a_scales, at::Tensor &b_scales,
                                                   at::Tensor &group_lens, at::Tensor &group_offs,
                                                   const bool transA, const bool transB,
                                                   at::ScalarType     out_dtype,
                                                   const std::string &granularity,
                                                   c10::optional<int64_t> num_cu) {
    const int64_t OUT_M = a.size(1);
    const int64_t OUT_N = b.size(1);
    const int64_t E     = group_lens.size(0);
    return at::empty({E, OUT_M, OUT_N}, a.options().dtype(out_dtype));
}

TORCH_LIBRARY_FRAGMENT(primus_turbo_cpp_extension, m) {
    m.def("asm_co_grouped_gemm_fp8(Tensor a, Tensor b, Tensor a_scales, Tensor b_scales, "
          "Tensor group_lens, Tensor group_offs, bool transA, bool transB, "
          "ScalarType out_dtype, str granularity, int? num_cu) -> Tensor");
    m.def("asm_co_grouped_gemm_fp8_variable_k(Tensor a, Tensor b, Tensor a_scales, Tensor b_scales, "
          "Tensor group_lens, Tensor group_offs, bool transA, bool transB, "
          "ScalarType out_dtype, str granularity, int? num_cu) -> Tensor");
}

TORCH_LIBRARY_IMPL(primus_turbo_cpp_extension, CUDA, m) {
    m.impl("asm_co_grouped_gemm_fp8", asm_co_grouped_gemm_fp8);
    m.impl("asm_co_grouped_gemm_fp8_variable_k", asm_co_grouped_gemm_fp8_variable_k);
}

TORCH_LIBRARY_IMPL(primus_turbo_cpp_extension, Meta, m) {
    m.impl("asm_co_grouped_gemm_fp8", asm_co_grouped_gemm_fp8_meta);
    m.impl("asm_co_grouped_gemm_fp8_variable_k", asm_co_grouped_gemm_fp8_variable_k_meta);
}

} // namespace primus_turbo::pytorch
