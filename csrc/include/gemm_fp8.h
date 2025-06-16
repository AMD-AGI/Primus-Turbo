#pragma once
#include <cstdint>
#include <hip/hip_runtime.h>
namespace primus_turbo {

// TODO: template
void ck_gemm_fp8_blockwise_kernel(void *a_ptr, void *a_scales_ptr, void *b_ptr, void *b_scales_ptr,
                                  void *c_ptr, const int32_t M, const int32_t N, const int32_t K,
                                  const bool transA, const bool transB, hipStream_t stream);

} // namespace primus_turbo
