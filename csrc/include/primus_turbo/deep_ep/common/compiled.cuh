#pragma once

// ROCm: new, force upstream's non-SM90 branch for every legacy TU
#ifndef DISABLE_SM90_FEATURES
#define DISABLE_SM90_FEATURES
#endif

// Make CLion CUDA indexing work
#ifdef __CLION_IDE__
#define __CUDA_ARCH__ 900
#define __CUDACC_RDC__
#define __CUDACC__
#endif

// Remove Torch restrictions
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT162_OPERATORS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__
#endif

#include <cstdint>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

// ROCm: new, alias only the CUDA spellings hipify does not translate
#ifndef PRIMUS_TURBO_DEEPEP_NO_CUDA_ALIASES
using nv_bfloat16  = __hip_bfloat16;
using nv_bfloat162 = __hip_bfloat162;
#endif

// ROCm: keep upstream's DISABLE_SM90_FEATURES fallback typedefs
#define __NV_E4M3 0
#define __NV_E5M2 1
typedef int     __nv_fp8_interpretation_t;
typedef int     __nv_fp8x4_e4m3;
typedef uint8_t __nv_fp8_storage_t;

// ROCm: CUDART_VERSION>=13000 branch is CUDA-only; keep the struct
struct alignas(32) longlong4_t { long long x, y, z, w; };
__device__ __forceinline__ longlong4_t make_longlong4_t(
    const long long& x, const long long& y, const long long& z, const long long& w) {
    return {x, y, z, w};
}

#include <primus_turbo/deep_ep/common/arch.cuh>

#ifndef EP_NUM_TOPK_IDX_BITS
#define EP_NUM_TOPK_IDX_BITS 64
#endif

namespace primus_turbo::deep_ep {

#ifndef DISABLE_SM90_FEATURES
constexpr bool kEnableSM90Features = true;
#else
constexpr bool kEnableSM90Features = false;
#endif

template <int kNumBits> struct int_with_bits;
template <> struct int_with_bits<8>  { using type = int8_t;  };
template <> struct int_with_bits<16> { using type = int16_t; };
template <> struct int_with_bits<32> { using type = int32_t; };
template <> struct int_with_bits<64> { using type = int64_t; };

using topk_idx_t = int_with_bits<EP_NUM_TOPK_IDX_BITS>::type;

union sf_pack_t {
    float fp32;
    int ue8m0x4;
};

constexpr int kNumTMAAlignedBytes = 16;
constexpr int kNumAlignedSFPacks = 16 / sizeof(sf_pack_t);

// Some communication channel settings
constexpr int kNumMaxChannels = 1024;
constexpr int kGinQPDepth = 1024;
constexpr int kGinQPFlushDepth = 768;

} // namespace primus_turbo::deep_ep
