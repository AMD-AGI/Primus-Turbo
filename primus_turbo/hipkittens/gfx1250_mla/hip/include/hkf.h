/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// hkf.h -- the gfx1250 builtins of k_mla_fwd (wave32), namespace hkf. Each wrapper lowers to one
// instruction, or to the fence / barrier sequence named. Self-contained: the backward kernels have
// their own (dkdv_prims.h, dq_prims.h).
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <utility>

namespace hkf {

typedef __bf16                                   bf16;
typedef __bf16                                   bf16x8 __attribute__((ext_vector_type(8)));
typedef __bf16                                   bf16x16 __attribute__((ext_vector_type(16)));
typedef float                                    f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t                                 u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t                                 u32x4 __attribute__((ext_vector_type(4)));
typedef int32_t                                  i32x4 __attribute__((ext_vector_type(4)));
typedef int32_t                                  i32x8 __attribute__((ext_vector_type(8)));
typedef int32_t                                  i32x32 __attribute__((ext_vector_type(32)));
typedef __attribute__((address_space(3))) bf16x8 lds_bf16x8;
typedef __attribute__((address_space(3))) unsigned char lds_u8;
typedef __attribute__((address_space(3))) unsigned char lds_u8w;
typedef __attribute__((address_space(3))) u32x4         lds_u32x4;
typedef __attribute__((address_space(3))) u32x2         lds_u32x2;
typedef __attribute__((address_space(3))) uint32_t      lds_u32;

#define HKF_INLINE __device__ __forceinline__

// ---- compile-time loops: f(std::integral_constant<int, I>{}) for I in [0, N)
template <int N, typename F, int... I>
HKF_INLINE void static_for_impl(F &&f, std::integer_sequence<int, I...>) {
    (f(std::integral_constant<int, I>{}), ...);
}
template <int N, typename F> HKF_INLINE void static_for(F &&f) {
    static_for_impl<N>(static_cast<F &&>(f), std::make_integer_sequence<int, N>{});
}

// ---- vectors, WMMA
HKF_INLINE bf16x16 cat(bf16x8 a, bf16x8 b) {
    return __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
// fptrunc f32 -> bf16 (v_cvt_pk_bf16_f32, round to nearest even), FlyDSL's .to(BFloat16)
HKF_INLINE bf16x8 to_bf16x8(f32x8 v) {
    return bf16x8{(bf16) v[0], (bf16) v[1], (bf16) v[2], (bf16) v[3],
                  (bf16) v[4], (bf16) v[5], (bf16) v[6], (bf16) v[7]};
}
// WMMA 16x16x32 bf16 -> f32 (v_wmma_f32_16x16x32_bf16: no negation, no C modifier, no reuse hints)
HKF_INLINE f32x8 wmma(bf16x16 a, bf16x16 b, f32x8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x32_bf16(false, a, false, b, (short) 0, c, false, false);
}

// ---- LDS (addresses are byte offsets in the workgroup's LDS)
HKF_INLINE uint32_t lds_addr(const void *p) {
    return (uint32_t) (uintptr_t) p;
}
HKF_INLINE const lds_u8 *lds_ptr(uint32_t a) {
    return (const lds_u8 *) (uintptr_t) a;
}
// fragment reads: ds_load_b128 and the transposing ds_load_tr16_b128, base + immediate offset
HKF_INLINE bf16x8 lds_load_b128(const lds_u8 *p, int off) {
    return *(const lds_bf16x8 *) (p + off);
}
HKF_INLINE bf16x8 lds_tr16(const lds_u8 *p, int off) {
    return __builtin_amdgcn_ds_load_tr16_b128_v8bf16((lds_bf16x8 *) (p + off));
}
// epilogue staging (ds_store_b128 / _b32, ds_load_b128 / _b64), base + immediate offset
HKF_INLINE lds_u8w *lds_wptr(uint32_t a) {
    return (lds_u8w *) (uintptr_t) a;
}
HKF_INLINE void lds_store_b128(lds_u8w *p, int off, u32x4 v) {
    *(lds_u32x4 *) (p + off) = v;
}
HKF_INLINE void lds_store_b32(lds_u8w *p, int off, uint32_t v) {
    *(lds_u32 *) (p + off) = v;
}
HKF_INLINE u32x4 lds_load_u32x4(const lds_u8 *p, int off) {
    return *(const lds_u32x4 *) (p + off);
}
HKF_INLINE u32x2 lds_load_u32x2(const lds_u8 *p, int off) {
    return *(const lds_u32x2 *) (p + off);
}
// cross-lane LDS data within ONE wave (ds_store by some lanes, ds_load by others): the wave's LDS
// ops execute in order, so no counter wait is needed; this only keeps the compiler from moving the
// loads above the stores (a wavefront-scope fence emits no instruction on gfx1250)
HKF_INLINE void wave_lds_order() {
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
    __builtin_amdgcn_sched_barrier(0);
}

// ---- buffer stores
// buffer resource word 3 as FlyDSL's make_buffer_tensor on gfx1250
inline constexpr int RSRC_FLAGS = 159744;

HKF_INLINE __amdgpu_buffer_rsrc_t rsrc(const void *base, int nbytes) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(base), (short) 0, (long) nbytes,
                                             RSRC_FLAGS);
}
HKF_INLINE void buf_store_b128(u32x4 v, __amdgpu_buffer_rsrc_t r, int voff, int soff) {
    __builtin_amdgcn_raw_buffer_store_b128(v, r, voff, soff, 0);
}
HKF_INLINE void buf_store_b64(u32x2 v, __amdgpu_buffer_rsrc_t r, int voff, int soff) {
    __builtin_amdgcn_raw_buffer_store_b64(v, r, voff, soff, 0);
}

// ---- Tensor Data Mover: one 2-D tensor_load_to_lds (descriptor groups 0 and 1; groups 2..4 zero)
HKF_INLINE void tdm_load(i32x4 g0, i32x8 g1) {
    const i32x4 z4 = {0, 0, 0, 0};
    const i32x8 z8 = {0, 0, 0, 0, 0, 0, 0, 0};
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, z4, z4, z8, 0);
}
// s_wait_tensorcnt N: at most N of this wave's TDM ops in flight
template <int N> HKF_INLINE void tensor_wait() {
    __builtin_amdgcn_s_wait_tensorcnt((short) N);
}

// ---- scheduling, barriers
template <int MASK> HKF_INLINE void sched_barrier() {
    __builtin_amdgcn_sched_barrier(MASK);
}
inline constexpr int MO_ACQUIRE = 2, MO_RELEASE = 3;
// llvm.amdgcn.s.wait.dscnt through an asm label: there is no builtin for it, and the legacy
// __builtin_amdgcn_s_waitcnt cannot be selected on gfx1250
extern "C" __device__ void hkf_llvm_s_wait_dscnt(short) __asm("llvm.amdgcn.s.wait.dscnt");
// retire this wave's LDS reads, release (workgroup, LDS), arrive at the barrier (no wait)
HKF_INLINE void bar_signal() {
    sched_barrier<0>();
    hkf_llvm_s_wait_dscnt(0);
    __builtin_amdgcn_fence(MO_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier_signal(-1);
    sched_barrier<0>();
}
// wait at the workgroup barrier, acquire (workgroup, LDS)
HKF_INLINE void bar_wait() {
    sched_barrier<0>();
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(MO_ACQUIRE, "workgroup", "local");
    sched_barrier<0>();
}

// ---- wave state, lane exchange, transcendentals
// SCHED_MODE.DISABLE_XDL_ARB_STALL = 1, as FlyDSL's rocdl.disable_xdl_arb_stall:
// s_setreg hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1) (encoding 154)
HKF_INLINE void lock_simd() {
    __builtin_amdgcn_s_setreg(154, 1);
}
// wave index in the workgroup as a uniform SGPR
HKF_INLINE int wave_id(int tid) {
    return __builtin_amdgcn_readfirstlane(tid) >> 5;
}
// exp2 as a bare v_exp_f32 (without llvm.exp2.f32's denormal-range scaling)
HKF_INLINE float exp2_raw(float x) {
    return __builtin_amdgcn_exp2f(x);
}
// the value of lane l ^ 16 (v_permlanex16_b32 with FlyDSL's selects)
HKF_INLINE float peer(float x) {
    const unsigned u = __builtin_bit_cast(unsigned, x);
    const unsigned v = __builtin_amdgcn_permlanex16(u, u, 0x76543210u, 0xFEDCBA98u, false, false);
    return __builtin_bit_cast(float, v);
}

} // namespace hkf
