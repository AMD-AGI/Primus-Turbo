/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// hkf.h -- the raw gfx1250 builtins of the HK MLA forward shell (HIP, wave32). Each wrapper lowers to one
// instruction (or to the barrier sequence named). A self-contained subset of gfx1250_mla_bwd/hip/include/hk1.h,
// so the forward does not depend on the backward's headers.
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <utility>

namespace hkf {

typedef __bf16 bf16;
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef int32_t i32x4 __attribute__((ext_vector_type(4)));
typedef int32_t i32x8 __attribute__((ext_vector_type(8)));
typedef int32_t i32x32 __attribute__((ext_vector_type(32)));
typedef __attribute__((address_space(3))) bf16x8 lds_bf16x8;
typedef __attribute__((address_space(3))) unsigned char lds_u8;

#define HKF_INLINE __device__ __forceinline__

template <int N, typename F, int... I>
HKF_INLINE void static_for_impl(F&& f, std::integer_sequence<int, I...>) {
  (f(std::integral_constant<int, I>{}), ...);
}
template <int N, typename F>
HKF_INLINE void static_for(F&& f) {
  static_for_impl<N>(static_cast<F&&>(f), std::make_integer_sequence<int, N>{});
}

HKF_INLINE bf16x16 cat(bf16x8 a, bf16x8 b) {
  return __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
// fptrunc f32 -> bf16 (v_cvt_pk_bf16_f32, round to nearest even), FlyDSL's .to(BFloat16)
HKF_INLINE bf16x8 to_bf16x8(f32x8 v) {
  return bf16x8{(bf16)v[0], (bf16)v[1], (bf16)v[2], (bf16)v[3], (bf16)v[4], (bf16)v[5], (bf16)v[6], (bf16)v[7]};
}
// WMMA 16x16x32 bf16 -> f32 (llvm.amdgcn.wmma.f32.16x16x32.bf16, neg 0, c_mod 0, no reuse)
HKF_INLINE f32x8 wmma(bf16x16 a, bf16x16 b, f32x8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x32_bf16(false, a, false, b, (short)0, c, false, false);
}

HKF_INLINE uint32_t lds_addr(const void* p) { return (uint32_t)(uintptr_t)p; }
HKF_INLINE const lds_u8* lds_ptr(uint32_t a) { return (const lds_u8*)(uintptr_t)a; }
HKF_INLINE bf16x8 lds_load_b128(const lds_u8* p, int off) { return *(const lds_bf16x8*)(p + off); }
HKF_INLINE bf16x8 lds_tr16(const lds_u8* p, int off) {
  return __builtin_amdgcn_ds_load_tr16_b128_v8bf16((lds_bf16x8*)(p + off));
}

// buffer resources (word-3 flags as FlyDSL's make_buffer_tensor on gfx1250) and raw buffer stores
inline constexpr int RSRC_FLAGS = 159744;
HKF_INLINE __amdgpu_buffer_rsrc_t rsrc(const void* base, int nbytes) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(base), (short)0, (long)nbytes, RSRC_FLAGS);
}
HKF_INLINE void buf_store_b128(u32x4 v, __amdgpu_buffer_rsrc_t r, int voff, int soff) {
  __builtin_amdgcn_raw_buffer_store_b128(v, r, voff, soff, 0);
}
HKF_INLINE void buf_store_b32(float v, __amdgpu_buffer_rsrc_t r, int voff, int soff) {
  __builtin_amdgcn_raw_buffer_store_b32(__builtin_bit_cast(unsigned, v), r, voff, soff, 0);
}

// Tensor Data Mover (2-D tensor_load_to_lds)
HKF_INLINE void tdm_load(i32x4 g0, i32x8 g1) {
  const i32x4 z4 = {0, 0, 0, 0};
  const i32x8 z8 = {0, 0, 0, 0, 0, 0, 0, 0};
  __builtin_amdgcn_tensor_load_to_lds(g0, g1, z4, z4, z8, 0);
}
template <int N>
HKF_INLINE void tensor_wait() { __builtin_amdgcn_s_wait_tensorcnt((short)N); }

template <int MASK>
HKF_INLINE void sched_barrier() { __builtin_amdgcn_sched_barrier(MASK); }
inline constexpr int MO_ACQUIRE = 2, MO_RELEASE = 3;
extern "C" __device__ void hkf_llvm_s_wait_dscnt(short) __asm("llvm.amdgcn.s.wait.dscnt");
// this wave's LDS reads retired, release, arrive at the workgroup barrier (no wait)
HKF_INLINE void bar_signal() {
  sched_barrier<0>();
  hkf_llvm_s_wait_dscnt(0);
  __builtin_amdgcn_fence(MO_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier_signal(-1);
  sched_barrier<0>();
}
HKF_INLINE void bar_wait() {
  sched_barrier<0>();
  __builtin_amdgcn_s_barrier_wait(-1);
  __builtin_amdgcn_fence(MO_ACQUIRE, "workgroup", "local");
  sched_barrier<0>();
}
// SCHED_MODE.DISABLE_XDL_ARB_STALL (hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1) = 1), as the backward kernels
HKF_INLINE void lock_simd() { __builtin_amdgcn_s_setreg(154, 1); }
HKF_INLINE int wave_id(int tid) { return __builtin_amdgcn_readfirstlane(tid) >> 5; }
HKF_INLINE float exp2_raw(float x) { return __builtin_amdgcn_exp2f(x); }  // bare v_exp_f32
HKF_INLINE float peer(float x) {  // the value of lane l ^ 16 (v_permlanex16_b32, FlyDSL's selects)
  const unsigned u = __builtin_bit_cast(unsigned, x);
  return __builtin_bit_cast(float, (unsigned)__builtin_amdgcn_permlanex16(u, u, 0x76543210u, 0xFEDCBA98u, false, false));
}

}  // namespace hkf
